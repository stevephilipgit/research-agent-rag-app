import json
from typing import List

from fastapi import APIRouter, File, HTTPException, UploadFile, Request
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

router = APIRouter(prefix="/api", tags=["api"])


@router.post("/query", response_model=QueryResponse)
@limiter.limit(REQUESTS_PER_MINUTE)
def query_endpoint(request: Request, payload: QueryRequest):
    try:
        return query_agent(payload.query, payload.session_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/query/stream")
@limiter.limit(STREAM_REQUESTS_PER_MINUTE)
def query_stream_endpoint(request: Request, payload: QueryRequest):
    def event_stream():
        try:
            for event in stream_query_events(payload.query, payload.session_id):
                yield f"event: {event['type']}\n"
                yield f"data: {json.dumps(event['data'])}\n\n"
        except Exception as exc:
            error_payload = {"detail": str(exc)}
            yield "event: error\n"
            yield f"data: {json.dumps(error_payload)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/upload", response_model=UploadResponse)
@limiter.limit(UPLOAD_REQUESTS_PER_MINUTE)
async def upload_endpoint(request: Request, files: List[UploadFile] = File(...)):
    try:
        return await upload_documents(files)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/history", response_model=HistoryResponse)
def history_endpoint():
    return {"messages": get_history()}


@router.get("/documents", response_model=DocumentListResponse)
def documents_endpoint():
    return {"documents": get_documents()}


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


@router.get("/logs", response_model=LogResponse)
def logs_endpoint():
    return {"logs": get_logs()}


@router.get("/logs/stream")
def logs_stream_endpoint():
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
