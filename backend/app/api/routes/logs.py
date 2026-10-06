"""Structured log endpoints (snapshot + SSE tail)."""
import json
import logging
from typing import Optional

from fastapi import APIRouter, Header, Query
from fastapi.responses import StreamingResponse

from app.core.telemetry import (
    get_logs as get_structured_logs,
    subscribe,
    unsubscribe,
    wait_for_log,
)
from app.domain.schemas import LogResponse
from app.services.rag_service import get_logs
from app.services.security import validate_session_id

router = APIRouter(prefix="/api", tags=["api"])
logger = logging.getLogger(__name__)


@router.get("/logs", response_model=LogResponse)
def logs_endpoint(
    session_id: Optional[str] = Header(None, alias="X-Session-ID"),
):
    # Return all logs (scoped filtering happens at telemetry layer if needed)
    return {"logs": get_logs()}


@router.get("/logs/stream")
def logs_stream_endpoint(
    session_id: Optional[str] = Query(None),
    x_session_id: Optional[str] = Header(None, alias="X-Session-ID"),
):
    s_id = session_id or x_session_id or "default"
    # Validate format even for query-param session IDs (SSE limitation)
    if s_id and s_id != "default":
        validated = validate_session_id(s_id)
        if validated is None:
            s_id = "default"
        else:
            s_id = validated

    def event_stream():
        queue = subscribe()
        logger.info(f"AUDIT: Log stream started for session {s_id}")
        try:
            # Send initial snapshot
            yield "event: snapshot\n"
            yield f"data: {json.dumps(get_structured_logs())}\n\n"

            while True:
                try:
                    entry = wait_for_log(queue, timeout=15.0)
                    if entry is None:
                        # Periodic heartbeat to keep proxy alive
                        yield "event: heartbeat\n"
                        yield "data: {}\n\n"
                        continue

                    yield "event: log\n"
                    yield f"data: {json.dumps(entry)}\n\n"
                except Exception as loop_exc:
                    logger.error(f"AUDIT: Error in log stream loop: {loop_exc}")
                    yield "event: error\n"
                    yield f"data: {json.dumps({'error': str(loop_exc)})}\n\n"
                    break
        except Exception as stream_exc:
            logger.exception("AUDIT: Fatal error in log event_stream")
        finally:
            logger.info(f"AUDIT: Log stream closed for session {s_id}")
            unsubscribe(queue)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable buffering for Nginx/Proxies
            "Access-Control-Allow-Origin": "*",
        }
    )
