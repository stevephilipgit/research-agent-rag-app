"""LLM query endpoints (stateless request/response + SSE stream)."""
import json
import logging
from typing import Optional

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.api.dependencies import require_session_id
from app.core.config import REQUESTS_PER_MINUTE, STREAM_REQUESTS_PER_MINUTE
from app.core.exceptions import VectorServiceUnavailable
from app.core.rate_limiter import limiter
from app.domain.schemas import QueryRequest, QueryResponse
from app.infrastructure.vector_store.qdrant import ensure_vector_service_available
from app.services.rag_service import query_agent, stream_query_events

router = APIRouter(prefix="/api", tags=["api"])
logger = logging.getLogger(__name__)


@router.post("/query", response_model=QueryResponse)
@limiter.limit(REQUESTS_PER_MINUTE)
def query_endpoint(
    request: Request,
    payload: QueryRequest,
    session_id: Optional[str] = Header(None, alias="X-Session-ID"),
):
    resolved = require_session_id(session_id, payload.session_id, auto_generate=False)
    ensure_vector_service_available()
    try:
        return query_agent(payload.query, resolved, payload.enable_self_healing)
    except VectorServiceUnavailable:
        raise
    except HTTPException:
        raise
    except Exception:
        logger.exception("Query endpoint error")
        raise HTTPException(status_code=500, detail="Internal server error.")


@router.post("/query/stream")
@limiter.limit(STREAM_REQUESTS_PER_MINUTE)
def query_stream_endpoint(
    request: Request,
    payload: QueryRequest,
    session_id: Optional[str] = Header(None, alias="X-Session-ID"),
):
    resolved = require_session_id(session_id, payload.session_id, auto_generate=False)
    # Must run before the StreamingResponse is created: once streaming starts,
    # the HTTP status is already committed and cannot become a 503.
    ensure_vector_service_available()

    def event_stream():
        try:
            for event in stream_query_events(payload.query, resolved, payload.enable_self_healing):
                yield f"event: {event['type']}\n"
                yield f"data: {json.dumps(event['data'])}\n\n"
        except Exception:
            logger.exception("Streaming query error")
            error_payload = {"detail": "Streaming failed. Please try again."}
            yield "event: error\n"
            yield f"data: {json.dumps(error_payload)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
