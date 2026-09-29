"""Environment driven settings (plan section 5.3)."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

log = logging.getLogger(__name__)

DEFAULT_MODEL = "kotoba-tech/kotoba-whisper-v2.0-faster"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000
DEFAULT_MAX_UPLOAD_MB = 100

# faster-whisper's default is True, but the kotoba model collapses (inference
# stops early and the rest of a long clip is dropped) when it is enabled.
DEFAULT_CONDITION_ON_PREVIOUS_TEXT = False


def repo_root() -> Path:
    """Repository root (``<repo>/src/whisper_local/config.py`` -> ``<repo>``)."""
    return Path(__file__).resolve().parents[2]


DEFAULT_SAVE_AUDIO = True
DEFAULT_SAVE_AUDIO_DIR = repo_root() / "var" / "recordings"


def _int(value: str | None, default: int, name: str) -> int:
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except ValueError:
        log.warning("invalid %s=%r, using default %d", name, value, default)
        return default


def _bool(value: str | None, default: bool, name: str) -> bool:
    if value is None or value.strip() == "":
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    log.warning("invalid %s=%r, using default %s", name, value, default)
    return default


def _str(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


@dataclass(frozen=True)
class Settings:
    """Runtime configuration."""

    model: str = DEFAULT_MODEL
    device: str = "auto"
    compute_type: str | None = None
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    api_key: str | None = None
    max_upload_mb: int = DEFAULT_MAX_UPLOAD_MB
    cpu_threads: int = 0
    vad_filter: bool = False
    condition_on_previous_text: bool = DEFAULT_CONDITION_ON_PREVIOUS_TEXT
    models_root: Path = Path("var/models")
    save_audio: bool = DEFAULT_SAVE_AUDIO
    save_audio_dir: Path = DEFAULT_SAVE_AUDIO_DIR

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        env = os.environ if env is None else env
        raw_save_dir = _str(env.get("WHISPER_SAVE_AUDIO_DIR"))
        return cls(
            model=_str(env.get("WHISPER_MODEL")) or DEFAULT_MODEL,
            device=_str(env.get("WHISPER_DEVICE")) or "auto",
            compute_type=_str(env.get("WHISPER_COMPUTE_TYPE")),
            host=_str(env.get("WHISPER_HOST")) or DEFAULT_HOST,
            port=_int(env.get("WHISPER_PORT"), DEFAULT_PORT, "WHISPER_PORT"),
            api_key=_str(env.get("WHISPER_API_KEY")),
            max_upload_mb=_int(
                env.get("WHISPER_MAX_UPLOAD_MB"),
                DEFAULT_MAX_UPLOAD_MB,
                "WHISPER_MAX_UPLOAD_MB",
            ),
            cpu_threads=_int(env.get("WHISPER_CPU_THREADS"), 0, "WHISPER_CPU_THREADS"),
            vad_filter=_bool(env.get("WHISPER_VAD_FILTER"), False, "WHISPER_VAD_FILTER"),
            condition_on_previous_text=_bool(
                env.get("WHISPER_CONDITION_ON_PREVIOUS_TEXT"),
                DEFAULT_CONDITION_ON_PREVIOUS_TEXT,
                "WHISPER_CONDITION_ON_PREVIOUS_TEXT",
            ),
            models_root=repo_root() / "var" / "models",
            save_audio=_bool(
                env.get("WHISPER_SAVE_AUDIO"), DEFAULT_SAVE_AUDIO, "WHISPER_SAVE_AUDIO"
            ),
            save_audio_dir=Path(raw_save_dir) if raw_save_dir else DEFAULT_SAVE_AUDIO_DIR,
        )
