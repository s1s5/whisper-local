"""``uv run python -m server`` shim.

The implementation lives in :mod:`whisper_local.server` (src layout); this root
level package only exists so that the documented launch command keeps working.
"""

from __future__ import annotations

from whisper_local.server.main import main

__all__ = ["main"]

if __name__ == "__main__":
    main()
