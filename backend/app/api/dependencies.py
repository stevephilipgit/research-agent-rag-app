"""Shared request-scoped dependencies for the API layer."""
import uuid
from typing import Optional

from fastapi import HTTPException

from app.services.security import validate_session_id


def require_session_id(
    header_session: Optional[str],
    payload_session: Optional[str] = None,
    auto_generate: bool = False,
) -> str:
    """
    Resolve and validate session ID.
    Priority: header > payload > auto-generate.
    Raises HTTP 400 if a supplied session ID has an invalid format.
    """
    raw = header_session or payload_session
    if raw:
        validated = validate_session_id(raw)
        if validated is None:
            raise HTTPException(
                status_code=400,
                detail="Invalid X-Session-ID format. Use alphanumeric characters, hyphens, or underscores (max 64 chars).",
            )
        return validated
    if auto_generate:
        return str(uuid.uuid4())
    return "default"
