"""faster-whisper wrapper: a single model instance shared by the whole process.

The model is loaded once during application startup. Inference is serialized
with a lock so concurrent HTTP requests queue instead of racing on the model.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio
from faster_whisper.transcribe import Segment, TranscriptionInfo

from whisper_local import alignment
from whisper_local.config import Settings

log = logging.getLogger(__name__)


class ModelLoadError(RuntimeError):
    """Raised when the model could not be loaded at all."""


class AudioDecodeError(ValueError):
    """Raised when the uploaded audio cannot be decoded (client error)."""


@dataclass(frozen=True)
class TranscriptionResult:
    """Everything the response formatters need."""

    segments: list[Segment]
    info: TranscriptionInfo
    task: str
    words_available: bool
    word_timestamps_requested: bool


class ModelManager:
    """Owns the single :class:`WhisperModel` instance and the inference lock."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._lock = threading.Lock()
        self._model: WhisperModel | None = None
        self.device: str | None = None
        self.compute_type: str | None = None
        self.words_enabled = False
        self.model_path: Path | None = None
        self.load_notes: list[str] = []

    # -- lifecycle ---------------------------------------------------------

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        """Load the model once. Never raises for recoverable problems."""
        if self._model is not None:
            return

        model_path, words_enabled = self._prepare_model()
        self.model_path = model_path
        self.words_enabled = words_enabled

        device, compute_type = self._initial_device_choice()
        try:
            self._model = self._load_model(model_path, device, compute_type)
        except Exception as exc:  # noqa: BLE001 - startup must survive
            if device == "cuda":
                log.warning(
                    "failed to load model on CUDA (%s: %s); falling back to CPU",
                    type(exc).__name__,
                    exc,
                )
                self.load_notes.append(f"CUDA load failed: {type(exc).__name__}: {exc}")
                device, compute_type = "cpu", self.settings.compute_type or "int8"
                try:
                    self._model = self._load_model(model_path, device, compute_type)
                except Exception as cpu_exc:  # noqa: BLE001
                    raise ModelLoadError(
                        f"failed to load model {self.settings.model!r}: {cpu_exc}"
                    ) from cpu_exc
            else:
                raise ModelLoadError(
                    f"failed to load model {self.settings.model!r}: {exc}"
                ) from exc

        self.device = device
        self.compute_type = compute_type
        log.info(
            "model loaded: model=%s path=%s device=%s compute_type=%s words=%s",
            self.settings.model,
            model_path,
            device,
            compute_type,
            words_enabled,
        )

    def _prepare_model(self) -> tuple[Path, bool]:
        """Resolve the model directory and correct ``alignment_heads`` if needed.

        Returns ``(model_path, words_enabled)``. On failure the original path is
        used with word timestamps disabled, so the server still starts.
        """
        try:
            result = alignment.ensure_usable_model(
                self.settings.model, self.settings.models_root
            )
        except Exception as exc:  # noqa: BLE001 - degrade, do not crash
            log.warning(
                "alignment_heads correction failed (%s: %s); "
                "continuing without word timestamps",
                type(exc).__name__,
                exc,
            )
            self.load_notes.append(f"alignment correction failed: {type(exc).__name__}: {exc}")
            try:
                return alignment.resolve_model_dir(self.settings.model), False
            except Exception as resolve_exc:  # noqa: BLE001
                raise ModelLoadError(
                    f"cannot resolve model directory for {self.settings.model!r}: {resolve_exc}"
                ) from resolve_exc

        log.info("alignment_heads check: %s", result.message)
        if not result.corrected:
            self.load_notes.append(result.message)
        if result.original_heads_max_layer is None:
            log.warning(
                "model config has no alignment_heads; word timestamps disabled"
            )
            return result.model_path, False
        return result.model_path, True

    def _initial_device_choice(self) -> tuple[str, str]:
        requested = self.settings.device
        if requested == "auto":
            device = "cuda" if alignment_cuda_available() else "cpu"
            if device == "cpu":
                log.warning(
                    "CUDA is not available (no usable device); using CPU. "
                    "Set WHISPER_DEVICE to override."
                )
                self.load_notes.append("CUDA unavailable; running on CPU")
        else:
            device = requested
        compute_type = self.settings.compute_type or (
            "float16" if device == "cuda" else "int8"
        )
        return device, compute_type

    def _load_model(self, model_path: Path, device: str, compute_type: str) -> WhisperModel:
        return WhisperModel(
            str(model_path),
            device=device,
            compute_type=compute_type,
            cpu_threads=self.settings.cpu_threads,
            num_workers=1,
            local_files_only=True,
        )

    # -- inference ---------------------------------------------------------

    def transcribe(
        self,
        audio: str,
        *,
        language: str | None = None,
        task: str = "transcribe",
        initial_prompt: str | None = None,
        temperature: float = 0.0,
        want_words: bool = False,
        vad_filter: bool | None = None,
    ) -> TranscriptionResult:
        """Decode then transcribe a file path. Serialized by an internal lock.

        Decoding is separated from inference so undecodable input surfaces as
        :class:`AudioDecodeError` (HTTP 400) instead of a server error.
        """
        samples = self.decode(audio)
        return self.transcribe_array(
            samples,
            language=language,
            task=task,
            initial_prompt=initial_prompt,
            temperature=temperature,
            want_words=want_words,
            vad_filter=vad_filter,
        )

    def decode(self, audio: str):
        """Decode an audio file to a 16 kHz mono float32 array."""
        try:
            return decode_audio(audio)
        except AudioDecodeError:
            raise
        except Exception as exc:  # noqa: BLE001 - PyAV raises many FFmpeg errors
            raise AudioDecodeError(f"cannot decode audio: {exc}") from exc

    def transcribe_array(
        self,
        samples,
        *,
        language: str | None = None,
        task: str = "transcribe",
        initial_prompt: str | None = None,
        temperature: float = 0.0,
        want_words: bool = False,
        vad_filter: bool | None = None,
    ) -> TranscriptionResult:
        """Run one transcription on decoded samples. Serialized by a lock."""
        model = self._model
        if model is None:
            raise ModelLoadError("model is not loaded")

        word_timestamps = bool(want_words and self.words_enabled)
        segments_iter, info = self._run(
            model,
            samples,
            language=language,
            task=task,
            initial_prompt=initial_prompt,
            temperature=temperature,
            word_timestamps=word_timestamps,
            vad_filter=self.settings.vad_filter if vad_filter is None else vad_filter,
        )
        segments = list(segments_iter)
        return TranscriptionResult(
            segments=segments,
            info=info,
            task=task,
            words_available=self.words_enabled,
            word_timestamps_requested=bool(want_words),
        )

    def _run(
        self,
        model: WhisperModel,
        samples,
        *,
        language: str | None,
        task: str,
        initial_prompt: str | None,
        temperature: float,
        word_timestamps: bool,
        vad_filter: bool,
    ) -> tuple[Iterable[Segment], TranscriptionInfo]:
        with self._lock:
            return model.transcribe(
                samples,
                language=language,
                task=task,
                beam_size=5,
                temperature=temperature,
                initial_prompt=initial_prompt,
                word_timestamps=word_timestamps,
                vad_filter=vad_filter,
            )


def alignment_cuda_available() -> bool:
    """True when CTranslate2 reports at least one usable CUDA device."""
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception as exc:  # noqa: BLE001 - absence of CUDA is not fatal
        log.warning("CUDA device probing failed (%s: %s)", type(exc).__name__, exc)
        return False
