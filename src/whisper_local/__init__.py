"""whisper-local: local faster-whisper exposed as an OpenAI-compatible HTTP API.

Importing this package has no side effects (no model loading, no inference).

Public API:

* :func:`main` -- start the HTTP server (entry point of the ``whisper-local`` script)
* ``__version__``
"""

from __future__ import annotations

__all__ = ["__version__", "main"]

try:  # pragma: no cover - metadata is always present for an installed package
    from importlib.metadata import version as _version

    __version__ = _version("whisper-local")
except Exception:  # pragma: no cover
    __version__ = "0.0.0"


def main() -> None:
    """Start the OpenAI-compatible HTTP server."""
    from whisper_local.server.main import main as _main

    _main()
