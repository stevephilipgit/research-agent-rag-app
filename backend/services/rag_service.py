import json
import os
import sys
import time
import uuid
import traceback
from pathlib import Path
from typing import Iterator, List

from fastapi import UploadFile
from langchain_core.messages import AIMessage, HumanMessage
import logging
from threading import Lock

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.agent import run_research_agent, run_research_agent_stream
from config import DOCUMENTS_PATH, ROOT_DIR
from core.document_loader import delete_document, ingest_documents
from infra.db import load_registry, save_doc_to_registry
from core.telemetry import emit_log, get_logs as get_structured_logs
from services.security import validate_file
from infra.storage import upload_file, delete_file, get_file_url
from infra.vector_db import get_collection_count, delete_vectors_by_doc_id, get_session_document_count
from utils.sanitize import clean_query
from utils.cache_db import invalidate_cache
from config.settings import PROCESSED_PATH, MAX_DOCS_PER_SESSION, MAX_FILE_SIZE_MB

# In-memory session store
_session_histories: dict[str, list[dict]] = {}
_session_lock = Lock()


def log_event(step: str, status: str = "success", detail: str = "", scope: str = "system") -> dict:
    entry = emit_log(step=step, status=status, detail=detail, scope=scope)
    print(f"[{entry['time']}] {step} [{status}] {detail}".strip())
    return entry


def get_history(session_id: str = "default") -> List[dict]:
    with _session_lock:
        return list(_session_histories.get(session_id, []))


def get_logs() -> List[dict]:
    return get_structured_logs()


def get_documents(session_id: str = "default") -> List[dict]:
    # Filter the registry to only show documents for this session
    registry = load_registry()
    return [doc for doc in registry if f"uploads/{session_id}/" in (doc.get("storage_path") or "")]


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
    # Addition 3: Check file count for this session from Qdrant
    existing_count = get_session_document_count(session_id)
    if existing_count + len(files) > MAX_DOCS_PER_SESSION:
        log_event("File Upload", "failure", f"Max documents ({MAX_DOCS_PER_SESSION}) exceeded for session", "pipeline")
        raise HTTPException(
            status_code=429,
            detail=f"Maximum {MAX_DOCS_PER_SESSION} documents per session allowed. You already have {existing_count}."
        )

    docs = load_registry()
    processed_hashes = {doc.get("file_hash") for doc in docs if doc.get("file_hash")}
    existing_names = {doc.get("file_name") for doc in docs}
    
    saved_paths: List[str] = []
    saved_names: List[str] = []
    debug_files: List[dict] = []

    log_event("File Upload", "in_progress", f"Receiving {len(files)} file(s)", "pipeline")

    for file in files:
        file_name = file.filename or "unnamed"
        content = await file.read()
        
        # Phase 5: Duplicate Prevention via Hash Check
        import hashlib
        f_hash = hashlib.md5(content).hexdigest()
        
        if f_hash in processed_hashes:
            log_event("File Upload", "failure", f"Duplicate content: {file_name} already exists", "pipeline")
            continue
            
        if file_name in existing_names:
            log_event("File Upload", "failure", f"Duplicate name: {file_name} already exists", "pipeline")
            continue

        # Security: File Validation & Size Check
        file_size = len(content)
        if file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
             log_event("File Upload", "failure", f"File too large: {file_name}", "pipeline")
             raise HTTPException(
                status_code=413,
                detail=f"File {file_name} too large. Maximum size is {MAX_FILE_SIZE_MB}MB."
            )

        if not validate_file(file_name, file_size):
            log_event("File Upload", "failure", f"Security block: {file_name} failed validation", "pipeline")
            continue
            
        doc_id = str(uuid.uuid4())

        try:
            log_event("File Upload", "in_progress", f"Uploading {file_name} to cloud storage (session: {session_id})...", "pipeline")
            storage_path = upload_file(content, file_name, session_id=session_id)
            public_url = get_file_url(storage_path)
            file.file.close()

            log_event("File Upload", "success", f"File saved to cloud: {storage_path}", "pipeline")
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            log_event("File Upload", "failure", "Upload failed", "pipeline")
            return {"status": "error", "message": "Upload failed", "logs": get_logs()}

        save_doc_to_registry(
            {
                "doc_id": doc_id,
                "file_name": file_name,
                "file_hash": f_hash,
                "storage_path": storage_path,
                "public_url": public_url,
                "upload_time": time.time(),
            }
        )

        saved_paths.append(storage_path)
        saved_names.append(file_name)
        debug_files.append(
            {
                "doc_id": doc_id,
                "file_name": file_name,
                "size_bytes": file_size,
                "storage_path": storage_path,
                "public_url": public_url,
            }
        )

    if not saved_paths:
        return {
            "status": "success",
            "uploaded_files": [],
            "message": "No new unique files uploaded.",
            "documents": get_documents(),
            "logs": get_logs(),
        }

    try:
        logger.info(f"Starting ingestion process for {len(saved_paths)} new file(s) for session {session_id}")
        result = ingest_documents(saved_paths, session_id=session_id, callback=_ingestion_callback)
        logger.info(f"Ingestion completed with status: {result.get('status')}. Vector count: {result.get('vector_count')}")
        if result.get("status") == "success":
            log_event("File Upload", "success", "Upload & Ingestion completed", "pipeline")
            invalidate_cache()
        else:
            msg = result.get("message", "Ingestion failed")
            log_event("File Upload", "failure", msg, "pipeline")
            return {"status": "error", "message": "Ingestion failed", "logs": get_logs()}
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        return {"status": "error", "message": "Ingestion failed", "logs": get_logs()}
    finally:
        # Cleanup
        if os.path.exists(PROCESSED_PATH):
            for fname in os.listdir(PROCESSED_PATH):
                fpath = os.path.join(PROCESSED_PATH, fname)
                try:
                    if os.path.isfile(fpath):
                        os.remove(fpath)
                except Exception:
                    pass

    return {
        "status": "success",
        "uploaded_files": saved_names,
        "documents_loaded": result.get("documents_loaded", 0),
        "vector_count": result.get("vector_count", 0),
        "message": "Upload successful",
        "steps": result.get("steps", []),
        "documents": get_documents(),
        "logs": get_logs(),
    }


def delete_registered_document(doc_id: str) -> dict:
    try:
        log_event("Delete Document", "in_progress", f"Deleting document {doc_id} from cloud...", "pipeline")
        
        # 1. Get metadata from registry
        registry = load_registry()
        entry = next((e for e in registry if e.get("doc_id") == doc_id), None)
        
        if entry:
            # 2. Delete from Supabase Storage
            storage_path = entry.get("storage_path")
            if storage_path:
                delete_file(storage_path)
            
            # 3. Delete from Qdrant Vector DB
            delete_vectors_by_doc_id(doc_id)
            
            # 4. Remove from local JSON registry
            delete_document(doc_id) # This call actually removes from registry
            
        log_event("Delete Document", "success", f"Deleted all cloud assets for {doc_id}", "pipeline")
        return {
            "status": "success",
            "message": "Document deleted successfully.",
            "documents": get_documents(getattr(entry, 'session_id', 'default')),
            "logs": get_logs(),
        }
    except Exception as e:
        logger.error(f"Failed to delete document {doc_id}: {e}", exc_info=True)
        log_event("Delete Document", "failure", str(e), "pipeline")
        return {
            "status": "error",
            "message": f"Failed to delete: {str(e)}",
            "documents": get_documents(),
            "logs": get_logs(),
        }


def query_agent(query: str, session_id: str = "default") -> dict:
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
    log_event("Query received", "success", query, "query")
    log_event("Retrieval", "in_progress", f"Number of documents in vector DB: {vector_count}", "query")
    log_event("Agent Execution", "in_progress", "Agent started", "query")

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
    }


def stream_query_events(query: str, session_id: str = "default") -> Iterator[dict]:
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
    log_event("Query received", "success", query, "query")
    log_event("Retrieval", "in_progress", f"Number of documents in vector DB: {vector_count}", "query")
    log_event("Agent Execution", "in_progress", "Streaming response started", "query")

    yield {"type": "meta", "data": {"logs": get_logs(), "debug": {"vector_count": vector_count}}}

    streamed_answer = ""
    steps = []
    citations = []

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

    with _session_lock:
        session_history.append({"role": "user", "content": query})
        session_history.append(
            {
                "role": "assistant",
                "content": streamed_answer,
                "steps": steps,
                "citations": citations,
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
        },
    }
