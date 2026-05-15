import json
import logging
import uuid
from typing import List, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile, Request, Header, Query
from utils.rate_limiter import limiter
from fastapi.responses import StreamingResponse

from models.schema import (
    DeleteResponse,
    DocumentListResponse,
    HistoryResponse,
    LogResponse,
    QueryRequest,
    QueryResponse,
    UploadResponse,
)
from config.settings import REQUESTS_PER_MINUTE, STREAM_REQUESTS_PER_MINUTE, UPLOAD_REQUESTS_PER_MINUTE
from services.rag_service import (
    delete_registered_document,
    get_documents,
    get_history,
    get_logs,
    query_agent,
    stream_query_events,
    upload_documents,
)
from services.security import validate_session_id
from services.maintenance_service import cleanup_orphan_documents
from core.telemetry import get_logs as get_structured_logs, subscribe, unsubscribe, wait_for_log
from infra.vector_db import delete_session_vectors

router = APIRouter(prefix="/api", tags=["api"])
logger = logging.getLogger(__name__)


def _require_session_id(
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


# ─────────────────────────────────────────────────────────────
# Query endpoints
# ─────────────────────────────────────────────────────────────

@router.post("/query", response_model=QueryResponse)
@limiter.limit(REQUESTS_PER_MINUTE)
def query_endpoint(
    request: Request,
    payload: QueryRequest,
    session_id: Optional[str] = Header(None, alias="X-Session-ID"),
):
    resolved = _require_session_id(session_id, payload.session_id, auto_generate=False)
    try:
        return query_agent(payload.query, resolved, payload.enable_self_healing)
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
    resolved = _require_session_id(session_id, payload.session_id, auto_generate=False)

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


# ─────────────────────────────────────────────────────────────
# Upload endpoint
# ─────────────────────────────────────────────────────────────

@router.post("/upload", response_model=UploadResponse)
@limiter.limit(UPLOAD_REQUESTS_PER_MINUTE)
async def upload_endpoint(
    request: Request,
    files: List[UploadFile] = File(...),
    session_id: Optional[str] = Header(None, alias="X-Session-ID"),
):
    # Uploads always get a session; auto-generate if not supplied
    resolved = _require_session_id(session_id, auto_generate=True)
    try:
        return await upload_documents(files, session_id=resolved)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Upload endpoint error")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


# ─────────────────────────────────────────────────────────────
# History / documents
# ─────────────────────────────────────────────────────────────

@router.get("/history", response_model=HistoryResponse)
def history_endpoint(
    session_id: Optional[str] = Header(None, alias="X-Session-ID"),
):
    resolved = _require_session_id(session_id)
    return {"messages": get_history(resolved)}


@router.get("/documents", response_model=DocumentListResponse)
async def documents_endpoint(
    session_id: Optional[str] = Header(None, alias="X-Session-ID"),
):
    resolved = _require_session_id(session_id)
    return {"documents": await get_documents(resolved)}


# ─────────────────────────────────────────────────────────────
# Delete endpoints — session-scoped (IDOR fix)
# ─────────────────────────────────────────────────────────────

@router.delete("/documents/{doc_id}", response_model=DeleteResponse)
async def delete_document_endpoint(
    doc_id: str,
    session_id: Optional[str] = Header(None, alias="X-Session-ID"),
):
    resolved = _require_session_id(session_id)
    try:
        return await delete_registered_document(doc_id, resolved)
    except HTTPException:
        raise
    except Exception:
        logger.exception("Delete document error | doc_id=%s", doc_id)
        raise HTTPException(status_code=500, detail="Internal server error.")


@router.delete("/document/{doc_id}", response_model=DeleteResponse)
async def delete_document_alias_endpoint(
    doc_id: str,
    session_id: Optional[str] = Header(None, alias="X-Session-ID"),
):
    resolved = _require_session_id(session_id)
    try:
        return await delete_registered_document(doc_id, resolved)
    except HTTPException:
        raise
    except Exception:
        logger.exception("Delete document (alias) error | doc_id=%s", doc_id)
        raise HTTPException(status_code=500, detail="Internal server error.")


# ─────────────────────────────────────────────────────────────
# Session clear
# ─────────────────────────────────────────────────────────────

@router.delete("/session")
async def clear_session(
    session_id: Optional[str] = Header(None, alias="X-Session-ID"),
):
    resolved = _require_session_id(session_id)
    if resolved and resolved != "default":
        try:
            delete_session_vectors(resolved)
        except Exception:
            logger.exception("Session clear error | session_id=%s", resolved)
    return {"status": "cleared"}


# ─────────────────────────────────────────────────────────────
# Logs endpoints
# ─────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────
# Admin audit endpoint
# ─────────────────────────────────────────────────────────────

@router.get("/admin/audit")
async def admin_audit_endpoint(
    dry_run: bool = Query(True, description="When true (default), report only — no deletions. Set false to run live cleanup."),
):
    """
    Run the orphan-document audit.

    - **dry_run=true** (default): Returns orphan counts and full record details
      without modifying any data. Safe to call any time.
    - **dry_run=false**: Executes live cleanup — deletes unrecoverable orphans
      from the registry and marks storage-present / vector-missing records as
      'corrupted'. Use with caution.

    Records created within the last 5 minutes are always skipped regardless of
    mode (grace period protects in-flight ingestions).
    """
    try:
        result = await cleanup_orphan_documents(dry_run=dry_run)
        return {
            "status": "ok",
            "mode": "dry_run" if dry_run else "live",
            **result,
        }
    except Exception:
        logger.exception("Admin audit endpoint error")
        raise HTTPException(status_code=500, detail="Audit job failed. Check server logs.")
