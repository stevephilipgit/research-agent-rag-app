import hashlib
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Iterator, List

from fastapi import UploadFile, HTTPException
from langchain_core.messages import AIMessage, HumanMessage
import logging
from threading import Lock

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.agent import run_research_agent, run_research_agent_stream
from core.document_loader import ingest_documents
from core.telemetry import emit_log, get_logs as get_structured_logs
from services.security import validate_file, sanitize_filename
from services.self_healing import self_healing_flow, get_retrieval_params, get_model, reset_adaptive_state
from services.metrics_service import MetricsService
from infra.storage import upload_file, delete_file, get_file_url, file_exists
from infra.vector_db import get_collection_count, delete_vectors_by_doc_id, get_session_document_count, is_file_hash_indexed_in_qdrant
from utils.sanitize import clean_query
from utils.cache_db import invalidate_cache
from config.settings import PROCESSED_PATH, MAX_DOCS_PER_SESSION, MAX_FILE_SIZE_MB, ENABLE_SELF_HEALING

# In-memory session store
_session_histories: dict[str, list[dict]] = {}
_session_lock = Lock()


def log_event(step: str, status: str = "success", detail: str = "", scope: str = "system") -> dict:
    entry = emit_log(step=step, status=status, detail=detail, scope=scope)
    logger.info("[%s] %s [%s] %s", entry["time"], step, status, detail)
    return entry


def get_history(session_id: str = "default") -> List[dict]:
    with _session_lock:
        return list(_session_histories.get(session_id, []))


def get_logs() -> List[dict]:
    return get_structured_logs()


async def get_documents(session_id: str = "default") -> List[dict]:
    from infra.db import db
    return await db.query_documents(session_id)


def _ingestion_callback(message: str) -> None:
    lowered = message.lower()
    if "stored in vector db" in lowered or "vector db count" in lowered or "checking existing vectors" in lowered:
        status = "failure" if "failed" in lowered else "success"
        if "checking existing vectors" in lowered:
            status = "in_progress"
        log_event("Vector Store Insert", status, message, "pipeline")
        return
    if "chunk" in lowered:
        log_event("Chunking", "in_progress" if "started" in lowered else "success", message, "pipeline")
        return
    elif "embedding" in lowered:
        status = "failure" if "failed" in lowered else "in_progress"
        if "stored" in lowered or "using local hash embeddings" in lowered:
            status = "success"
        log_event("Embedding", status, message, "pipeline")
        return
    elif "loading document" in lowered or "extracting text" in lowered or "documents loaded" in lowered:
        status = "in_progress" if "loading" in lowered or "extracting" in lowered else "success"
        log_event("Document Load", status, message, "pipeline")
        return
    elif "failed" in lowered:
        log_event("Pipeline", "failure", message, "pipeline")
        return
    else:
        log_event("Pipeline", "success", message, "pipeline")


async def upload_documents(files: List[UploadFile], session_id: str = "default") -> dict:
    from infra.db import db
    # Check file count for this session from Qdrant
    existing_count = get_session_document_count(session_id)
    if existing_count + len(files) > MAX_DOCS_PER_SESSION:
        log_event("File Upload", "failure", f"Max documents ({MAX_DOCS_PER_SESSION}) exceeded for session", "pipeline")
        raise HTTPException(
            status_code=429,
            detail=f"Maximum {MAX_DOCS_PER_SESSION} documents per session allowed. You already have {existing_count}.",
        )

    saved_paths: List[str] = []
    saved_names: List[str] = []
    total_vectors = 0
    reused_count = 0

    log_event("File Upload", "in_progress", f"Receiving {len(files)} file(s)", "pipeline")

    for file in files:
        raw_name = file.filename or "unnamed"
        file_name = sanitize_filename(raw_name)
        logger.info(f"AUDIT: Received file upload | Raw: {raw_name} | Sanitized: {file_name}")
        content = await file.read()
        file_size = len(content)

        # File size and security validation
        if file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
            log_event("File Upload", "failure", f"File too large: {file_name}", "pipeline")
            raise HTTPException(
                status_code=413,
                detail=f"File {file_name} is too large. Maximum size is {MAX_FILE_SIZE_MB}MB.",
            )

        if not validate_file(file_name, file_size, content_bytes=content):
            log_event("File Upload", "failure", f"Security block: {file_name} failed validation", "pipeline")
            continue

        f_hash = hashlib.sha256(content).hexdigest()

        # ── Duplicate check ──────────────────────────────────────────────────
        existing_doc_id = None
        existing = await db.get_by_hash(f_hash)
        if existing:
            status = existing.get("status")
            s_path = existing.get("storage_path")
            exists_in_storage = file_exists(s_path)
            exists_in_vectors = is_file_hash_indexed_in_qdrant(f_hash, session_id)
            existing_doc_id = existing.get("id")

            if status == "indexed" and exists_in_storage and exists_in_vectors:
                log_event("File Upload", "success", f"Duplicate file detected: {file_name} reused", "pipeline")
                reused_count += 1
                saved_names.append(file_name)
                saved_paths.append(s_path)
                continue
            else:
                log_event(
                    "File Upload", "in_progress",
                    f"Duplicate found but inconsistent (status={status}, storage={exists_in_storage}, "
                    f"vectors={exists_in_vectors}). Retrying upload for {file_name}",
                    "pipeline",
                )
                try:
                    delete_vectors_by_doc_id(existing_doc_id)
                    logger.info(f"[Upload] Deleted old vectors for doc_id={existing_doc_id} before re-ingestion")
                except Exception as exc:
                    logger.warning(f"[Upload] Failed to delete old vectors: {exc}")

                try:
                    await db.update("documents", existing_doc_id, {"status": "re-indexing"})
                except Exception as exc:
                    logger.warning(f"[Upload] Failed to update registry status: {exc}")

        # ── FIX 1: Correct write order ───────────────────────────────────────
        # Step 1 — upload to storage first; raises on any failure so DB is never touched
        log_event("File Upload", "in_progress", f"Step 1/4 — uploading to storage: {file_name}", "pipeline")
        try:
            storage_path = upload_file(content, file_name, session_id=session_id)
        except RuntimeError as storage_err:
            log_event("File Upload", "failure", f"Upload failed to storage: {file_name} — {storage_err}", "pipeline")
            raise HTTPException(
                status_code=502,
                detail=f"Storage upload failed for '{file_name}': {storage_err}. No DB record was created.",
            )
        if not storage_path:
            # Defence-in-depth: shouldn't be reachable after the raise above, but guard anyway
            log_event("File Upload", "failure", f"Upload failed to storage (empty path): {file_name}", "pipeline")
            raise HTTPException(
                status_code=502,
                detail=f"Storage upload returned empty path for '{file_name}'. No DB record was created.",
            )

        # Steps 2-4 — extract text, chunk, and write vectors; roll back storage on failure
        try:
            doc_id = existing_doc_id or str(uuid.uuid4())
            log_event("File Upload", "in_progress", f"Step 2/4 — extracting text and chunking: {file_name}", "pipeline")
            # ingest_documents handles: load → chunk → embed → upsert_vectors
            # It raises RuntimeError if 0 chunks or vectors are produced.
            result = ingest_documents(
                [storage_path],
                session_id=session_id,
                doc_id=doc_id,
                callback=_ingestion_callback,
            )

            if result.get("status") != "success":
                chunks_created = result.get("chunks_created", 0)
                if chunks_created == 0:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"Document could not extract any text. "
                            f"If this is a scanned PDF, OCR is required. File: {file_name}"
                        ),
                    )
                raise RuntimeError(f"Ingestion failed for {file_name}: {result.get('message', 'unknown error')}")  

            v_count = result.get("upserted_count", 0)
            if v_count == 0:
                raise RuntimeError(f"No vectors were written for {file_name}; ingestion aborted.")

            log_event("File Upload", "in_progress", f"Step 3/4 — {v_count} vectors confirmed in ChromaDB/Qdrant", "pipeline")

            # Step 5 — ALL preconditions met; write DB record LAST
            log_event("File Upload", "in_progress", f"Step 4/4 — updating registry entry: {file_name}", "pipeline")
            doc_data = {
                "file_hash": f_hash,
                "filename": file_name,
                "storage_path": storage_path,
                "storage_url": get_file_url(storage_path),
                "user_id": session_id,
                "status": "indexed",
                "vector_count": v_count,
                "document_type": "general",
                "topic": "general",
            }
            if existing_doc_id:
                await db.update("documents", existing_doc_id, doc_data)
            else:
                doc_data["id"] = doc_id
                await db.insert("documents", doc_data)

            total_vectors += v_count
            saved_paths.append(storage_path)
            saved_names.append(file_name)
            log_event("File Upload", "success", f"Ingestion complete: {file_name} ({v_count} vectors, doc_id={doc_id})", "pipeline")

        except HTTPException:
            # Roll back storage — DB was never written
            logger.warning(f"AUDIT: Upload pipeline failed for '{file_name}' — rolling back storage path: {storage_path}")
            try:
                delete_file(storage_path)
            except Exception as del_exc:
                logger.warning(f"AUDIT: Storage rollback failed for {storage_path}: {del_exc}")
            raise

        except Exception as e:
            # Roll back storage — DB was never written
            logger.exception(f"AUDIT: Ingestion failed for '{file_name}' — rolling back storage path: {storage_path}")
            log_event("File Upload", "failure", f"Ingestion exception: {e}", "pipeline")
            try:
                delete_file(storage_path)
            except Exception as del_exc:
                logger.warning(f"AUDIT: Storage rollback failed for {storage_path}: {del_exc}")
            # Surface as a 500 so the caller knows exactly what happened
            raise HTTPException(
                status_code=500,
                detail=f"Ingestion failed for '{file_name}': {e}",
            )

    # Cleanup processed temp files
    if os.path.exists(PROCESSED_PATH):
        for fname in os.listdir(PROCESSED_PATH):
            fpath = os.path.join(PROCESSED_PATH, fname)
            try:
                if os.path.isfile(fpath):
                    os.remove(fpath)
            except Exception:
                pass

    if not saved_names and files:
        raise HTTPException(
            status_code=422,
            detail="Ingestion failed for all uploaded files. Check logs for details."
        )

    return {
        "status": "success",
        "uploaded_files": saved_names,
        "vector_count": total_vectors,
        "reused_count": reused_count,
        "message": "Upload process finished",
        "documents": await db.query_documents(session_id),
        "logs": get_logs(),
    }


async def delete_registered_document(doc_id: str, session_id: str = "default") -> dict:
    from infra.db import db
    try:
        log_event("Delete Document", "in_progress", f"Deleting document {doc_id}...", "pipeline")

        doc = await db.get_document(doc_id, session_id)
        if not doc:
            raise Exception("Document not found or access denied")

        # Delete from Supabase Storage
        storage_path = doc.get("storage_path")
        if storage_path:
            try:
                delete_file(storage_path)
            except Exception as e:
                logger.warning(f"Failed to delete storage file: {e}")

        # Delete from Qdrant Vector DB
        try:
            delete_vectors_by_doc_id(doc_id)
        except Exception as e:
            logger.warning(f"Failed to delete vectors: {e}")

        # Remove from database
        await db.delete("documents", doc_id)

        # Invalidate cache if needed
        invalidate_cache("*", session_id=session_id)

        log_event("Delete Document", "success", "Deleted document and all related assets", "pipeline")
        return {
            "status": "success",
            "message": "Document deleted successfully.",
            "documents": await db.query_documents(session_id),
            "logs": get_logs(),
        }
    except Exception as e:
        logger.exception(f"Failed to delete document | doc_id={doc_id}")
        log_event("Delete Document", "failure", f"Deletion failed: {e}", "pipeline")
        return {
            "status": "error",
            "message": f"Failed to delete document: {e}",
            "documents": await db.query_documents(session_id),
            "logs": get_logs(),
        }


def query_agent(query: str, session_id: str = "default", enable_self_healing: bool = False) -> dict:
    query = clean_query(query)
    history_messages = []
    with _session_lock:
        session_history = _session_histories.setdefault(session_id, [])

    for message in session_history:
        if message["role"] == "user":
            history_messages.append(HumanMessage(content=message["content"]))
        elif message["role"] == "assistant":
            history_messages.append(AIMessage(content=message["content"]))

    vector_count = get_collection_count()
    # Strip newlines from query before logging to prevent log injection
    safe_query_log = query.replace("\n", " ").replace("\r", " ")[:200]
    log_event("Query received", "success", safe_query_log, "query")
    log_event("Retrieval", "in_progress", f"Number of documents in vector DB: {vector_count}", "query")
    log_event("Agent Execution", "in_progress", "Agent started", "query")

    # Self-Healing Integration
    eval_score = None
    retry_count = 0
    use_self_healing = ENABLE_SELF_HEALING or enable_self_healing

    if use_self_healing:
        last_result = {"answer": "", "steps": [], "citations": []}

        def generate_answer(modified_query: str) -> str:
            """Generate answer for modified query, storing full result."""
            nonlocal last_result
            result = run_research_agent(modified_query, history_messages, session_id=session_id)
            last_result = {
                "answer": result.get("answer", "") if isinstance(result, dict) else str(result),
                "steps": result.get("steps", []) if isinstance(result, dict) else [],
                "citations": result.get("citations", []) if isinstance(result, dict) else [],
            }
            return last_result["answer"]

        start_time = time.time()
        answer, eval_score, retry_count = self_healing_flow(
            query=query,
            generate_fn=generate_answer,
            context="",
        )
        elapsed_time = time.time() - start_time

        retrieval_params = get_retrieval_params()
        model_used = get_model()

        MetricsService.log_self_healing_complete(
            total_retries=retry_count,
            final_score=eval_score,
            elapsed_time=elapsed_time,
            accepted=eval_score >= 0.75,
            best_score=eval_score,
            model_used=model_used,
            top_k=retrieval_params.get("top_k", 5),
        )

        reset_adaptive_state()

        steps = last_result.get("steps", [])
        citations = last_result.get("citations", [])
    else:
        result = run_research_agent(query, history_messages, session_id=session_id)
        answer = result.get("answer", "") if isinstance(result, dict) else str(result)
        steps = result.get("steps", []) if isinstance(result, dict) else []
        citations = result.get("citations", []) if isinstance(result, dict) else []

    with _session_lock:
        session_history.append({"role": "user", "content": query})
        session_history.append(
            {
                "role": "assistant",
                "content": answer,
                "steps": steps,
                "citations": citations,
            }
        )

    log_event("Agent Execution", "success", f"Agent completed with {len(steps)} step(s)", "query")
    log_event("Final answer", "success", f"Generated {len(answer)} characters", "query")

    return {
        "answer": answer,
        "steps": steps,
        "citations": citations,
        "messages": get_history(session_id),
        "logs": get_logs(),
        "debug": {"vector_count": vector_count},
        "eval_score": eval_score,
        "retry_count": retry_count,
        "self_healing_enabled": use_self_healing,
    }


def stream_query_events(query: str, session_id: str = "default", enable_self_healing: bool = False) -> Iterator[dict]:
    query = clean_query(query)
    history_messages = []
    with _session_lock:
        session_history = _session_histories.setdefault(session_id, [])

    for message in session_history:
        if message["role"] == "user":
            history_messages.append(HumanMessage(content=message["content"]))
        elif message["role"] == "assistant":
            history_messages.append(AIMessage(content=message["content"]))

    vector_count = get_collection_count()
    # Strip newlines from query before logging to prevent log injection
    safe_query_log = query.replace("\n", " ").replace("\r", " ")[:200]
    log_event("Query received", "success", safe_query_log, "query")
    log_event("Retrieval", "in_progress", f"Number of documents in vector DB: {vector_count}", "query")
    log_event("Agent Execution", "in_progress", "Streaming response started", "query")

    yield {"type": "meta", "data": {"logs": get_logs(), "debug": {"vector_count": vector_count}}}

    streamed_answer = ""
    steps = []
    citations = []
    eval_score = None
    retry_count = 0
    use_self_healing = ENABLE_SELF_HEALING or enable_self_healing

    for event in run_research_agent_stream(query, history_messages, session_id=session_id):
        if event["type"] == "token":
            streamed_answer += event["data"]
            yield event
            continue

        if event["type"] == "error":
            yield event
            return

        if event["type"] == "done":
            streamed_answer = event["data"].get("answer", streamed_answer)
            steps = event["data"].get("steps", [])
            citations = event["data"].get("citations", [])

    # If self-healing enabled, evaluate the response quality
    if use_self_healing and streamed_answer:
        try:
            from services.eval_engine import EvaluationEngine
            evaluator = EvaluationEngine()
            eval_scores = evaluator.evaluate(query, streamed_answer, context="")
            eval_score = evaluator.final_score(eval_scores)
            retry_count = 0
        except Exception:
            logger.warning("Self-healing evaluation failed")
            eval_score = None

    with _session_lock:
        session_history.append({"role": "user", "content": query})
        session_history.append(
            {
                "role": "assistant",
                "content": streamed_answer,
                "steps": steps,
                "citations": citations,
                "eval_score": eval_score,
                "retry_count": retry_count,
                "self_healing_enabled": use_self_healing,
            }
        )

    log_event("Agent Execution", "success", f"Agent completed with {len(steps)} step(s)", "query")
    log_event("Final answer", "success", f"Generated {len(streamed_answer)} characters", "query")

    yield {
        "type": "done",
        "data": {
            "answer": streamed_answer,
            "steps": steps,
            "citations": citations,
            "messages": get_history(session_id),
            "logs": get_logs(),
            "debug": {"vector_count": vector_count},
            "eval_score": eval_score,
            "retry_count": retry_count,
            "self_healing_enabled": use_self_healing,
        },
    }
