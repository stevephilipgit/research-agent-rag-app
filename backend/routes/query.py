import json
import traceback
from typing import List

import uuid
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
from core.telemetry import get_logs as get_structured_logs, subscribe, unsubscribe, wait_for_log
from infra.vector_db import delete_session_vectors

router = APIRouter(prefix="/api", tags=["api"])


@router.post("/query", response_model=QueryResponse)
@limiter.limit(REQUESTS_PER_MINUTE)
def query_endpoint(
    request: Request, 
    payload: QueryRequest,
    session_id: str = Header(None, alias="X-Session-ID")
):
    if not session_id:
        session_id = payload.session_id or "default"
    try:
        return query_agent(payload.query, session_id, payload.enable_self_healing)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/query/stream")
@limiter.limit(STREAM_REQUESTS_PER_MINUTE)
def query_stream_endpoint(
    request: Request, 
    payload: QueryRequest,
    session_id: str = Header(None, alias="X-Session-ID")
):
    if not session_id:
        session_id = payload.session_id or "default"
    def event_stream():
        try:
            for event in stream_query_events(payload.query, session_id, payload.enable_self_healing):
                yield f"event: {event['type']}\n"
                yield f"data: {json.dumps(event['data'])}\n\n"
        except Exception as exc:
            error_payload = {"detail": str(exc)}
            yield "event: error\n"
            yield f"data: {json.dumps(error_payload)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/upload", response_model=UploadResponse)
@limiter.limit(UPLOAD_REQUESTS_PER_MINUTE)
async def upload_endpoint(
    request: Request, 
    files: List[UploadFile] = File(...),
    session_id: str = Header(None, alias="X-Session-ID")
):
    if not session_id:
        session_id = str(uuid.uuid4())
    try:
        return await upload_documents(files, session_id=session_id)
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/history", response_model=HistoryResponse)
def history_endpoint(session_id: str = Header(None, alias="X-Session-ID")):
    return {"messages": get_history(session_id or "default")}


@router.get("/documents", response_model=DocumentListResponse)
def documents_endpoint(session_id: str = Header(None, alias="X-Session-ID")):
    return {"documents": get_documents(session_id or "default")}


@router.delete("/documents/{doc_id}", response_model=DeleteResponse)
def delete_document_endpoint(doc_id: str):
    try:
        return delete_registered_document(doc_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.delete("/document/{doc_id}", response_model=DeleteResponse)
def delete_document_alias_endpoint(doc_id: str):
    try:
        return delete_registered_document(doc_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.delete("/session")
async def clear_session(
    session_id: str = Header(None, alias="X-Session-ID")
):
    if session_id:
        delete_session_vectors(session_id)
    return {"status": "cleared"}


@router.get("/logs", response_model=LogResponse)
def logs_endpoint():
    return {"logs": get_logs()}


@router.get("/logs/stream")
def logs_stream_endpoint(
    session_id: str = Query(None),
    x_session_id: str = Header(None, alias="X-Session-ID")
):
    # Support session_id from query param (SSE limitation) or header
    s_id = session_id or x_session_id or "default"
    
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
