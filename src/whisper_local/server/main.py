"""Uvicorn bootstrap for the OpenAI compatible server."""

from __future__ import annotations

import logging

import uvicorn

from whisper_local.config import Settings

log = logging.getLogger(__name__)


def main() -> None:
    """Start the server on ``WHISPER_HOST``:``WHISPER_PORT`` (default 0.0.0.0:8000)."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    settings = Settings.from_env()
    log.info(
        "starting whisper-local: host=%s port=%s model=%s device=%s",
        settings.host,
        settings.port,
        settings.model,
        settings.device,
    )
    uvicorn.run(
        "whisper_local.server.app:create_app",
        factory=True,
        host=settings.host,
        port=settings.port,
        log_level="info",
    )


if __name__ == "__main__":  # pragma: no cover
    main()
