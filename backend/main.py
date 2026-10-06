"""Uvicorn entry point.

Kept at the backend root so the external startup contract is unchanged:

    uvicorn main:app

The application itself lives in the `app` package (`app/main.py`).
"""
import sys
from pathlib import Path

# Ensure backend directory is in sys.path for imports
BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.main import app  # noqa: F401,E402

__all__ = ["app"]


if __name__ == "__main__":
    import os
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
