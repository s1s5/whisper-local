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


def repo_root() -> Path:
    """Repository root (``<repo>/src/whisper_local/config.py`` -> ``<repo>``)."""
    return Path(__file__).resolve().parents[2]


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
    models_root: Path = Path("var/models")

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        env = os.environ if env is None else env
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
            models_root=repo_root() / "var" / "models",
        )
