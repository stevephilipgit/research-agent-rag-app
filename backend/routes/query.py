import json
import logging
import re
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
    except Exception:
        logger.exception("Upload endpoint error")
        raise HTTPException(status_code=500, detail="Internal server error.")


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

async def _delete_document_with_session(doc_id: str, session_id: str) -> dict:
    """
    Delete a document only if it belongs to the requesting session.
    Raises HTTP 403 if the document does not belong to the session.
    Raises HTTP 404 if the document is not found.
    """
    from infra.db import db
    from infra.storage import delete_file

    try:
        uuid.UUID(doc_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID format.")

    doc = await db.get_document(doc_id, session_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found or access denied.")

    errors = []

    # Step 1: delete vectors (safe to retry)
    try:
        delete_vectors_by_doc_id(doc_id)
    except Exception as e:
        errors.append(f"Qdrant: {e}")

    # Step 2: delete Supabase file
    try:
        storage_path = doc.get("storage_path")
        if storage_path:
            delete_file(storage_path)
    except Exception as e:
        errors.append(f"Storage: {e}")

    # Step 3: delete registry row only if steps 1+2 succeeded
    if not errors:
        await db.delete("documents", doc_id)
    else:
        await db.update("documents", doc_id, {"status": "delete_failed"})
        raise HTTPException(500, detail={"errors": errors})

    return {
        "status": "success",
        "message": "Document deleted successfully.",
        "documents": await get_documents(session_id),
        "logs": get_logs(),
    }


@router.delete("/documents/{doc_id}", response_model=DeleteResponse)
async def delete_document_endpoint(
    doc_id: str,
    session_id: Optional[str] = Header(None, alias="X-Session-ID"),
):
    resolved = _require_session_id(session_id)
    try:
        return await _delete_document_with_session(doc_id, resolved)
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
        return await _delete_document_with_session(doc_id, resolved)
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
        try:
            yield "event: snapshot\n"
            yield f"data: {json.dumps(get_structured_logs())}\n\n"
            while True:
                entry = wait_for_log(queue, timeout=15.0)
                if entry is None:
                    yield "event: heartbeat\n"
                    yield "data: {}\n\n"
                    continue
                yield "event: log\n"
                yield f"data: {json.dumps(entry)}\n\n"
        finally:
            unsubscribe(queue)

    return StreamingResponse(event_stream(), media_type="text/event-stream")
