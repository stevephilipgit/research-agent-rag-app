"""Document upload and session-scoped document management endpoints."""
import logging
from typing import List, Optional

from fastapi import APIRouter, File, Header, HTTPException, Request, UploadFile

from app.api.dependencies import require_session_id
from app.core.config import UPLOAD_REQUESTS_PER_MINUTE
from app.core.exceptions import VECTOR_UNAVAILABLE_DETAIL, VectorServiceUnavailable
from app.core.rate_limiter import limiter
from app.domain.schemas import DeleteResponse, DocumentListResponse, UploadResponse
from app.infrastructure.vector_store.qdrant import ensure_vector_service_available
from app.services.rag_service import (
    delete_registered_document,
    get_documents,
    upload_documents,
)

router = APIRouter(prefix="/api", tags=["api"])
logger = logging.getLogger(__name__)


@router.post("/upload", response_model=UploadResponse)
@limiter.limit(UPLOAD_REQUESTS_PER_MINUTE)
async def upload_endpoint(
    request: Request,
    files: List[UploadFile] = File(...),
    session_id: Optional[str] = Header(None, alias="X-Session-ID"),
):
    # Uploads always get a session; auto-generate if not supplied
    resolved = require_session_id(session_id, auto_generate=True)
    ensure_vector_service_available()
    try:
        return await upload_documents(files, session_id=resolved)
    except VectorServiceUnavailable:
        raise
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Upload endpoint error")
        if "Qdrant is unavailable" in str(e):
            # Ingestion refused because the vector store went away mid-request
            raise HTTPException(status_code=503, detail=VECTOR_UNAVAILABLE_DETAIL)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.get("/documents", response_model=DocumentListResponse)
async def documents_endpoint(
    session_id: Optional[str] = Header(None, alias="X-Session-ID"),
):
    resolved = require_session_id(session_id)
    return {"documents": await get_documents(resolved)}


# ─────────────────────────────────────────────────────────────
# Delete endpoints — session-scoped (IDOR fix)
# ─────────────────────────────────────────────────────────────

@router.delete("/documents/{doc_id}", response_model=DeleteResponse)
async def delete_document_endpoint(
    doc_id: str,
    session_id: Optional[str] = Header(None, alias="X-Session-ID"),
):
    resolved = require_session_id(session_id)
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
    resolved = require_session_id(session_id)
    try:
        return await delete_registered_document(doc_id, resolved)
    except HTTPException:
        raise
    except Exception:
        logger.exception("Delete document (alias) error | doc_id=%s", doc_id)
        raise HTTPException(status_code=500, detail="Internal server error.")
