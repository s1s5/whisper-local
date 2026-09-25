"""HTTP server package: OpenAI compatible Audio API on top of faster-whisper."""

from __future__ import annotations

from whisper_local.server.app import create_app

__all__ = ["create_app"]
