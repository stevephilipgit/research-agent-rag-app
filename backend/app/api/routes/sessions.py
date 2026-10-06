"""Chat history and session lifecycle endpoints."""
import logging
from typing import Optional

from fastapi import APIRouter, Header

from app.api.dependencies import require_session_id
from app.domain.schemas import HistoryResponse
from app.infrastructure.vector_store.qdrant import delete_session_vectors
from app.services.rag_service import get_history

router = APIRouter(prefix="/api", tags=["api"])
logger = logging.getLogger(__name__)


@router.get("/history", response_model=HistoryResponse)
def history_endpoint(
    session_id: Optional[str] = Header(None, alias="X-Session-ID"),
):
    resolved = require_session_id(session_id)
    return {"messages": get_history(resolved)}


@router.delete("/session")
async def clear_session(
    session_id: Optional[str] = Header(None, alias="X-Session-ID"),
):
    resolved = require_session_id(session_id)
    if resolved and resolved != "default":
        try:
            delete_session_vectors(resolved)
        except Exception:
            logger.exception("Session clear error | session_id=%s", resolved)
    return {"status": "cleared"}
