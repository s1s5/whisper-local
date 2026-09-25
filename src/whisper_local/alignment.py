"""alignment_heads correction for kotoba-whisper (plan section 3.1).

The distributed ``config.json`` of ``kotoba-tech/kotoba-whisper-v2.0-faster``
points ``alignment_heads`` at decoder layers 7..25 while ``model.bin`` only has
2 decoder layers. CTranslate2 does not range check
(``TransformerDecoder::set_alignment_heads``) so it writes out of bounds and
the process segfaults as soon as word timestamps are requested.

The fix is to detect the real decoder layer count, and when ``alignment_heads``
is out of range, create a corrected model directory (config copy + ``model.bin``
symlink) and hand that path to :class:`faster_whisper.WhisperModel`.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

#: ``<dtype type id>`` order from ctranslate2 ``model_spec`` / ``types.h``.
_DTYPES = ("float32", "int8", "int16", "int32", "float16", "bfloat16")
_LAYER_RE = re.compile(r"decoder/layer_(\d+)/")

#: Files copied into the corrected model directory (model.bin is symlinked).
_COPY_FILES = (
    "config.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "vocabulary.json",
)


def _read_string(fh) -> str:
    """Read a length prefixed, NUL terminated string from the ctranslate2 header."""
    (length,) = _read_struct(fh, "<H")
    raw = fh.read(length)
    if len(raw) != length or not raw.endswith(b"\x00"):
        raise ValueError("malformed string in model.bin header")
    return raw[:-1].decode("utf-8")


def _read_struct(fh, fmt: str):
    import struct

    size = struct.calcsize(fmt)
    raw = fh.read(size)
    if len(raw) != size:
        raise ValueError("unexpected end of model.bin header")
    return struct.unpack(fmt, raw)


@dataclass(frozen=True)
class DecoderShape:
    """Decoder geometry read from ``model.bin``."""

    n_layers: int
    n_heads: int | None


def inspect_model_bin(model_bin: Path) -> DecoderShape:
    """Read the ctranslate2 model header and return the decoder geometry.

    Only the variable header of ``model.bin`` is parsed; weight payloads are
    skipped, so this stays cheap even for a 1.5 GB file.
    """
    decoder_layers: set[int] = set()
    n_heads: int | None = None
    with open(model_bin, "rb") as fh:
        _read_struct(fh, "<I")  # binary version
        _read_string(fh)  # spec name (WhisperSpec)
        _read_struct(fh, "<I")  # spec revision
        (num_variables,) = _read_struct(fh, "<I")
        for _ in range(num_variables):
            name = _read_string(fh)
            (rank,) = _read_struct(fh, "<B")
            _read_struct(fh, "<" + "I" * rank)  # shape
            (type_id,) = _read_struct(fh, "<B")
            (num_bytes,) = _read_struct(fh, "<I")
            if name == "decoder/num_heads":
                payload = fh.read(num_bytes)
                if len(payload) != num_bytes:
                    raise ValueError("unexpected end of model.bin while reading decoder/num_heads")
                n_heads = int.from_bytes(payload, "little", signed=True)
            else:
                if type_id >= len(_DTYPES):
                    raise ValueError(f"unknown variable dtype id {type_id} for {name!r}")
                fh.seek(num_bytes, os.SEEK_CUR)
            match = _LAYER_RE.match(name)
            if match:
                decoder_layers.add(int(match.group(1)))
    if not decoder_layers:
        raise ValueError("no decoder layers found in model.bin")
    return DecoderShape(n_layers=max(decoder_layers) + 1, n_heads=n_heads)


def _safe_name(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", model).strip("_") or "model"


def _alignment_max_layer(alignment_heads) -> int | None:
    layers = [int(pair[0]) for pair in alignment_heads if len(pair) >= 2]
    return max(layers) if layers else None


def resolve_model_dir(model: str) -> Path:
    """Return the local directory holding the faster-whisper model files.

    ``model`` may be a local path already, otherwise the Hugging Face cache is
    consulted (offline: no download is triggered).
    """
    candidate = Path(model).expanduser()
    if candidate.is_dir():
        return candidate
    from faster_whisper.utils import download_model

    return Path(download_model(model, local_files_only=True))


@dataclass(frozen=True)
class AlignmentResult:
    """Outcome of the alignment_heads check."""

    model_path: Path
    corrected: bool
    n_layers: int
    n_heads: int | None
    original_heads_max_layer: int | None
    message: str


def _build_corrected_dir(source: Path, target: Path, corrected_heads) -> None:
    """Create ``target`` with a corrected config plus symlinked/copied weights."""
    target.mkdir(parents=True, exist_ok=True)
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    config["alignment_heads"] = corrected_heads
    (target / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for name in _COPY_FILES:
        if name == "config.json":
            continue
        src_file = source / name
        if src_file.is_file():
            shutil.copy(src_file, target / name)
    bin_src = source / "model.bin"
    bin_dst = target / "model.bin"
    if bin_dst.is_symlink() or bin_dst.exists():
        bin_dst.unlink()
    os.symlink(bin_src.resolve(), bin_dst)


def ensure_usable_model(
    model: str,
    models_root: Path,
    model_dir: Path | None = None,
    corrected_heads=None,
) -> AlignmentResult:
    """Prepare a model directory whose ``alignment_heads`` are in range.

    Returns an :class:`AlignmentResult`; raises on failure so the caller can
    fall back to a word-timestamp-free model (the server must never crash).
    """
    source = model_dir if model_dir is not None else resolve_model_dir(model)
    source = Path(source)
    config_path = source / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"config.json not found in model dir: {source}")
    if not (source / "model.bin").is_file():
        raise FileNotFoundError(f"model.bin not found in model dir: {source}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    heads = config.get("alignment_heads")
    shape = inspect_model_bin(source / "model.bin")
    original_max = _alignment_max_layer(heads) if heads else None

    if not heads:
        return AlignmentResult(
            model_path=source,
            corrected=False,
            n_layers=shape.n_layers,
            n_heads=shape.n_heads,
            original_heads_max_layer=None,
            message="config has no alignment_heads; word timestamps unavailable",
        )

    if original_max is not None and original_max < shape.n_layers:
        return AlignmentResult(
            model_path=source,
            corrected=False,
            n_layers=shape.n_layers,
            n_heads=shape.n_heads,
            original_heads_max_layer=original_max,
            message=(
                f"alignment_heads max layer {original_max} fits {shape.n_layers} decoder "
                "layers; no correction needed"
            ),
        )

    n_heads = shape.n_heads
    if not n_heads or n_heads <= 0:
        raise ValueError("cannot build corrected alignment_heads without decoder/num_heads")

    if corrected_heads is None:
        # Last decoder layer, all heads (distil-whisper convention; verified on
        # 6 audio files: monotonic / in_range / coverage all healthy).
        corrected_heads = [[shape.n_layers - 1, head] for head in range(n_heads)]

    target = Path(models_root) / _safe_name(model)
    _build_corrected_dir(source, target, corrected_heads)
    return AlignmentResult(
        model_path=target,
        corrected=True,
        n_layers=shape.n_layers,
        n_heads=n_heads,
        original_heads_max_layer=original_max,
        message=(
            f"alignment_heads (max layer {original_max}) out of range for "
            f"{shape.n_layers} decoder layers; corrected to layer "
            f"{shape.n_layers - 1}, all {n_heads} heads -> {target}"
        ),
    )
