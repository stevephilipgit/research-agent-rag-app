"""
tests/test_upload_empty_pdf.py

Validates FIX 1: empty PDFs are rejected at the extraction stage (HTTP 400)
and no DB registry record is created.

Covers the acceptance criterion from the bug report:
  "Write a test: upload a deliberately empty PDF and assert the endpoint
   returns 400 with no DB record created."

Run with:
    cd backend
    pytest tests/test_upload_empty_pdf.py -v
"""
import io
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

# ── Ensure backend root is on the path ─────────────────────────────────────
BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from fastapi import HTTPException
from fastapi.testclient import TestClient


# ── Minimal PDF bytes (valid header, no extractable text) ──────────────────
_EMPTY_PDF_BYTES = (
    b"%PDF-1.4\n"
    b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
    b"2 0 obj\n<< /Type /Pages /Kids [] /Count 0 >>\nendobj\n"
    b"xref\n0 3\n0000000000 65535 f\n"
    b"trailer\n<< /Size 3 /Root 1 0 R >>\nstartxref\n9\n%%EOF\n"
)

_EMPTY_PDF_HASH = None  # populated at runtime


# ═══════════════════════════════════════════════════════════════════════════
# Helper: build a mock UploadFile
# ═══════════════════════════════════════════════════════════════════════════

def _make_upload_file(filename: str, content: bytes):
    """Return a minimal async-compatible UploadFile-like mock."""
    mock_file = MagicMock()
    mock_file.filename = filename
    mock_file.read = AsyncMock(return_value=content)
    return mock_file


# ═══════════════════════════════════════════════════════════════════════════
# Core unit test — tests upload_documents() directly, no HTTP layer needed
# ═══════════════════════════════════════════════════════════════════════════

class TestUploadEmptyPdfNoDBRecord:
    """
    FIX 1 — Correct write order:
    A storage upload that succeeds but produces 0 extractable chunks must:
      1. Raise HTTPException(400)
      2. Roll back the storage file (delete_file called)
      3. NEVER call db.insert (no DB record created)
    """

    @pytest.mark.asyncio
    async def test_empty_pdf_raises_400(self):
        """An empty PDF triggers HTTP 400 before any DB insert."""
        import hashlib

        upload_file = _make_upload_file("empty.pdf", _EMPTY_PDF_BYTES)

        fake_storage_path = "uploads/test-session/empty.pdf"

        with (
            # security passes
            patch("services.rag_service.validate_file", return_value=True),
            patch("services.rag_service.sanitize_filename", return_value="empty.pdf"),
            # no duplicate in DB
            patch("services.rag_service.get_session_document_count", return_value=0),
            # storage upload succeeds — returns a path
            patch("services.rag_service.upload_file", return_value=fake_storage_path),
            # ingest_documents reports 0 chunks — simulates empty/unreadable PDF
            patch(
                "services.rag_service.ingest_documents",
                return_value={
                    "status": "error",
                    "message": "Chunking resulted in 0 chunks.",
                    "chunks_created": 0,
                    "steps": [],
                    "file_hashes": {},
                },
            ),
            # db.get_by_hash — no existing record
            patch(
                "services.rag_service.db",
                get_by_hash=AsyncMock(return_value=None),
                insert=AsyncMock(side_effect=AssertionError("db.insert must NOT be called for an empty PDF")),
                update=AsyncMock(),
                delete=AsyncMock(),
                query_documents=AsyncMock(return_value=[]),
            ),
            # delete_file spy — should be called for rollback
            patch("services.rag_service.delete_file") as mock_delete_file,
            patch("services.rag_service.get_file_url", return_value="https://example.com/fake"),
            patch("services.rag_service.is_file_hash_indexed_in_qdrant", return_value=False),
            patch("services.rag_service.file_exists", return_value=False),
        ):
            from services.rag_service import upload_documents

            with pytest.raises(HTTPException) as exc_info:
                await upload_documents([upload_file], session_id="test-session")

            assert exc_info.value.status_code == 400, (
                f"Expected HTTP 400 for empty PDF, got {exc_info.value.status_code}"
            )
            assert "text" in exc_info.value.detail.lower() or "chunk" in exc_info.value.detail.lower(), (
                f"Expected detail to mention text/chunk extraction failure, got: {exc_info.value.detail}"
            )

    @pytest.mark.asyncio
    async def test_empty_pdf_rolls_back_storage(self):
        """Storage file is deleted when extraction fails — no orphan left in storage."""
        fake_storage_path = "uploads/test-session/empty.pdf"

        with (
            patch("services.rag_service.validate_file", return_value=True),
            patch("services.rag_service.sanitize_filename", return_value="empty.pdf"),
            patch("services.rag_service.get_session_document_count", return_value=0),
            patch("services.rag_service.upload_file", return_value=fake_storage_path),
            patch(
                "services.rag_service.ingest_documents",
                return_value={
                    "status": "error",
                    "message": "Chunking resulted in 0 chunks.",
                    "chunks_created": 0,
                    "steps": [],
                    "file_hashes": {},
                },
            ),
            patch(
                "services.rag_service.db",
                get_by_hash=AsyncMock(return_value=None),
                insert=AsyncMock(side_effect=AssertionError("db.insert must NOT be called")),
                update=AsyncMock(),
                delete=AsyncMock(),
                query_documents=AsyncMock(return_value=[]),
            ),
            patch("services.rag_service.delete_file") as mock_delete_file,
            patch("services.rag_service.get_file_url", return_value="https://example.com/fake"),
            patch("services.rag_service.is_file_hash_indexed_in_qdrant", return_value=False),
            patch("services.rag_service.file_exists", return_value=False),
        ):
            from services.rag_service import upload_documents

            with pytest.raises(HTTPException):
                await upload_documents(
                    [_make_upload_file("empty.pdf", _EMPTY_PDF_BYTES)],
                    session_id="test-session",
                )

            mock_delete_file.assert_called_once_with(fake_storage_path), (
                "Storage rollback (delete_file) must be called when extraction fails"
            )

    @pytest.mark.asyncio
    async def test_empty_pdf_no_db_insert(self):
        """db.insert is never called when ingestion fails after storage upload."""
        fake_storage_path = "uploads/test-session/empty.pdf"
        db_insert_mock = AsyncMock()

        with (
            patch("services.rag_service.validate_file", return_value=True),
            patch("services.rag_service.sanitize_filename", return_value="empty.pdf"),
            patch("services.rag_service.get_session_document_count", return_value=0),
            patch("services.rag_service.upload_file", return_value=fake_storage_path),
            patch(
                "services.rag_service.ingest_documents",
                return_value={
                    "status": "error",
                    "message": "Chunking resulted in 0 chunks.",
                    "chunks_created": 0,
                    "steps": [],
                    "file_hashes": {},
                },
            ),
            patch(
                "services.rag_service.db",
                get_by_hash=AsyncMock(return_value=None),
                insert=db_insert_mock,
                update=AsyncMock(),
                delete=AsyncMock(),
                query_documents=AsyncMock(return_value=[]),
            ),
            patch("services.rag_service.delete_file"),
            patch("services.rag_service.get_file_url", return_value="https://example.com/fake"),
            patch("services.rag_service.is_file_hash_indexed_in_qdrant", return_value=False),
            patch("services.rag_service.file_exists", return_value=False),
        ):
            from services.rag_service import upload_documents

            with pytest.raises(HTTPException):
                await upload_documents(
                    [_make_upload_file("empty.pdf", _EMPTY_PDF_BYTES)],
                    session_id="test-session",
                )

            db_insert_mock.assert_not_called(), (
                "db.insert must NEVER be called when extraction fails — "
                "this would create an orphan registry record"
            )


# ═══════════════════════════════════════════════════════════════════════════
# Test: storage failure also produces no DB record
# ═══════════════════════════════════════════════════════════════════════════

class TestStorageFailureNoDBRecord:
    """
    FIX 2 — upload_file raises RuntimeError on failure.
    A storage upload failure must:
      1. Raise HTTPException(502)
      2. NEVER call db.insert
    """

    @pytest.mark.asyncio
    async def test_storage_failure_raises_502(self):
        """RuntimeError from upload_file becomes HTTP 502, no DB write."""
        db_insert_mock = AsyncMock()

        with (
            patch("services.rag_service.validate_file", return_value=True),
            patch("services.rag_service.sanitize_filename", return_value="report.pdf"),
            patch("services.rag_service.get_session_document_count", return_value=0),
            patch(
                "services.rag_service.upload_file",
                side_effect=RuntimeError("Supabase returned 400 Bad Request"),
            ),
            patch(
                "services.rag_service.db",
                get_by_hash=AsyncMock(return_value=None),
                insert=db_insert_mock,
                update=AsyncMock(),
                delete=AsyncMock(),
                query_documents=AsyncMock(return_value=[]),
            ),
            patch("services.rag_service.is_file_hash_indexed_in_qdrant", return_value=False),
            patch("services.rag_service.file_exists", return_value=False),
        ):
            from services.rag_service import upload_documents

            with pytest.raises(HTTPException) as exc_info:
                await upload_documents(
                    [_make_upload_file("report.pdf", b"%PDF-1.4 valid content here")],
                    session_id="test-session",
                )

            assert exc_info.value.status_code == 502, (
                f"Expected HTTP 502 for storage failure, got {exc_info.value.status_code}"
            )
            db_insert_mock.assert_not_called(), (
                "db.insert must NOT be called when storage upload fails"
            )
