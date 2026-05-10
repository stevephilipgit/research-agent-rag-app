"""Document loader for the RAG ingestion pipeline - focused on loading and chunking."""

import hashlib
import logging
import os
import re
import time
import uuid
from typing import Callable, List, Optional

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    CSVLoader,
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
)
from config.settings import (
    DOCUMENTS_PATH,
    SUPPORTED_EXTENSIONS,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    PROCESSED_PATH,
)
from infra.embeddings import get_embeddings
from infra.vector_db import (
    upsert_vectors,
    delete_vectors_by_doc_id,
    get_collection_count,
    is_indexed_in_qdrant,
    is_file_hash_indexed_in_qdrant,
)
from infra.vector_db import QDRANT_AVAILABLE
from infra.db import load_registry, remove_from_registry
from utils.file_handling import download_file, is_url
from infra.storage import get_file_url

logger = logging.getLogger(__name__)


def get_file_hash(file_path: str) -> str:
    """Generates SHA256 hash of file content for duplicate prevention."""
    digest = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "")
    return text.strip()


def _emit_step(callback: Optional[Callable[[str], None]], message: str) -> None:
    logger.info(message)
    if callback:
        callback(message)


def _pick_loader(file_path: str):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return PyPDFLoader(file_path)
    if ext == ".txt":
        return TextLoader(file_path, encoding="utf-8")
    if ext == ".csv":
        return CSVLoader(file_path)
    if ext == ".docx":
        try:
            return Docx2txtLoader(file_path)
        except Exception:
            return UnstructuredWordDocumentLoader(file_path)
    return None


def _normalize_source_name(file_name: str) -> str:
    return os.path.basename((file_name or "Unknown").strip())


def _detect_document_type(text: str, source: str) -> str:
    hay = f"{source} {text}".lower()
    checks = [
        ("research_paper", ["abstract", "methodology", "references", "doi", "experiment"]),
        ("textbook", ["chapter", "exercise", "figure", "lesson"]),
        ("notes", ["todo", "bullet", "summary", "meeting notes"]),
        ("code", ["def ", "class ", "import ", "function", "{", "};"]),
        ("cybersecurity", ["vulnerability", "cve", "threat", "attack", "malware", "zero-day"]),
        ("medical", ["patient", "diagnosis", "symptom", "treatment", "clinical"]),
        ("legal", ["plaintiff", "defendant", "statute", "court", "contract"]),
    ]
    scores = []
    for label, words in checks:
        score = sum(1 for w in words if w in hay)
        if score:
            scores.append((score, label))
    if not scores:
        return "general"
    scores.sort(reverse=True)
    best_score, best_label = scores[0]
    if len(scores) > 1 and scores[1][0] == best_score:
        return "mixed"
    return best_label


def _detect_topics(text: str, source: str) -> list[str]:
    hay = f"{source} {text}".lower()
    topic_map = {
        "machine_learning": ["machine learning", "neural", "model", "training", "regression"],
        "cybersecurity": ["cyber", "cve", "malware", "threat", "vulnerability", "attack"],
        "cloud": ["aws", "azure", "gcp", "cloud", "kubernetes", "serverless"],
        "networking": ["tcp", "udp", "router", "switch", "network", "bandwidth"],
        "mathematics": ["theorem", "proof", "equation", "algebra", "calculus", "matrix"],
    }
    scored = []
    for name, words in topic_map.items():
        score = sum(1 for w in words if w in hay)
        if score:
            scored.append((score, name))
    if not scored:
        return ["general"]
    scored.sort(reverse=True)
    ordered = [name for _, name in scored]
    return ordered


def load_documents(
    file_paths: List[str],
    session_id: str = "default",
    callback: Optional[Callable[[str], None]] = None,
) -> List[Document]:
    registry = load_registry()
    registry_by_name = {d.get("filename", d.get("file_name")): d.get("id", d.get("doc_id")) for d in registry}

    documents: List[Document] = []
    processed_paths = set()

    for path in file_paths:
        if path in processed_paths:
            logger.warning(f"Skipping duplicate path: {path}")
            continue
        processed_paths.add(path)

        file_path = path
        is_temp = False
        try:
            if path.startswith("uploads/"):
                _emit_step(callback, f"Fetching cloud document: {path}")
                public_url = get_file_url(path)
                file_path = download_file(public_url)
                is_temp = True
            elif is_url(path):
                _emit_step(callback, f"Downloading URL: {path}")
                file_path = download_file(path)
                is_temp = True

            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                continue

            ext = os.path.splitext(file_path)[1].lower()
            if ext not in SUPPORTED_EXTENSIONS:
                continue

            loader = _pick_loader(file_path)
            if not loader:
                continue
            docs = loader.load()

            doc_id = registry_by_name.get(os.path.basename(file_path))
            original_name = os.path.basename(path)
            for doc in docs:
                doc.metadata["file_name"] = original_name
                if doc_id:
                    doc.metadata["doc_id"] = doc_id
                doc.page_content = clean_text(doc.page_content)
            documents.extend(docs)
        except Exception as exc:
            logger.error(f"Failed to load {file_path}: {exc}")
            _emit_step(callback, f"Failed to process document: {os.path.basename(file_path)}")
        finally:
            if is_temp and os.path.exists(file_path):
                os.remove(file_path)
    return documents


def split_into_sections(text: str) -> List[str]:
    pattern = r"(?=\n\s*\d+(?:\.\d+)*\s+)"
    return [s.strip() for s in re.split(pattern, text or "") if s and s.strip()]


def create_chunks_with_overlap(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    chunks = []
    start = 0
    step = max(1, chunk_size - overlap)
    while start < len(text):
        chunks.append(text[start : start + chunk_size])
        start += step
    return chunks


def chunk_documents(documents: List[Document], callback: Optional[Callable[[str], None]] = None) -> List[Document]:
    _emit_step(callback, "Chunking started...")
    all_chunks = []
    for doc in documents:
        sections = split_into_sections(doc.page_content) or [doc.page_content]
        for section in sections:
            metadata = dict(doc.metadata or {})
            if len(section) <= CHUNK_SIZE:
                all_chunks.append(Document(page_content=section, metadata=metadata))
            else:
                for chunk in create_chunks_with_overlap(section):
                    all_chunks.append(Document(page_content=chunk, metadata=metadata))
    _emit_step(callback, f"Chunks created: {len(all_chunks)}")
    return all_chunks


def delete_document(doc_id: str):
    """Cleanup metadata and vectors for a document."""
    registry = load_registry()
    remove_from_registry(doc_id)
    logger.info(f"Removed doc_id {doc_id} from registry")


def ingest_documents(file_paths: Optional[List[str]] = None, file_names: Optional[List[str]] = None, session_id: str = "default", callback: Optional[Callable[[str], None]] = None) -> dict:
    steps = []

    def record(m):
        steps.append(m)
        _emit_step(callback, m)

    logger.info(f"ingest_documents called with {len(file_paths) if file_paths else 0} paths and {len(file_names) if file_names else 0} names.")
    if file_paths:
        logger.debug(f"Input paths: {file_paths}")

    try:
        if file_paths is None:
            file_paths = [os.path.join(DOCUMENTS_PATH, f) for f in os.listdir(DOCUMENTS_PATH) if os.path.isfile(os.path.join(DOCUMENTS_PATH, f))]

        if file_paths:
            file_paths = list(dict.fromkeys(file_paths))
            logger.info(f"After dedup: {len(file_paths)} unique path(s)")

        if file_names:
            file_names = list(dict.fromkeys(file_names))

        valid_paths = []
        for path in file_paths:
            file_name = os.path.basename(path)

            if not os.path.exists(path) and not path.startswith("uploads/"):
                record(f"Skipping {path} - file not found locally and not a cloud path")
                continue

            f_hash = None
            if not path.startswith("uploads/"):
                f_hash = get_file_hash(path)

            record(f"Queueing for ingestion: {file_name}")
            valid_paths.append((path, f_hash))

        if not valid_paths:
            return {"status": "success", "message": "No new unique documents to ingest.", "steps": steps, "chunks_created": 0, "file_hashes": {}}

        docs = load_documents([p[0] for p in valid_paths], session_id=session_id, callback=record)

        chunks = chunk_documents(docs, callback=record)
        now_ts = int(time.time())
        embeddings_model = get_embeddings(callback=record)
        embedding_model_name = type(embeddings_model).__name__
        per_source_chunk_idx: dict[str, int] = {}
        hash_by_source = {os.path.basename(p[0]): p[1] or "" for p in valid_paths}

        for chunk in chunks:
            original_name = _normalize_source_name(chunk.metadata.get("file_name") or "Unknown")
            per_source_chunk_idx[original_name] = per_source_chunk_idx.get(original_name, 0) + 1
            detected_topics = _detect_topics(chunk.page_content, original_name)

            chunk.metadata["source"] = original_name
            chunk.metadata["display_name"] = original_name
            chunk.metadata["session_id"] = session_id or "default"
            chunk.metadata["user_id"] = chunk.metadata.get("user_id", session_id or "default")
            chunk.metadata["page"] = chunk.metadata.get("page", chunk.metadata.get("page_number", "N/A"))
            chunk.metadata["chunk_index"] = chunk.metadata.get("chunk_index", per_source_chunk_idx[original_name])
            chunk.metadata["document_type"] = chunk.metadata.get("document_type") or _detect_document_type(chunk.page_content, original_name)
            chunk.metadata["topic"] = chunk.metadata.get("topic") or detected_topics[0]
            chunk.metadata["topics"] = chunk.metadata.get("topics") or detected_topics
            chunk.metadata["created_at"] = chunk.metadata.get("created_at", now_ts)
            chunk.metadata["file_hash"] = chunk.metadata.get("file_hash") or hash_by_source.get(original_name, "")
            chunk.metadata["embedding_model"] = chunk.metadata.get("embedding_model", embedding_model_name)

        record(f"Chunks created: {len(chunks)}")
        record(f"Generating embeddings for {len(chunks)} chunks...")

        points = []
        for c in chunks:
            cid = str(uuid.uuid4())
            c.metadata["id"] = cid
            points.append(
                {
                    "id": cid,
                    "vector": embeddings_model.embed_query(c.page_content),
                    "payload": {
                        "page_content": c.page_content,
                        **c.metadata,
                    },
                }
            )

        upserted_count = 0
        if not points:
            raise RuntimeError("No vectors generated from chunks; ingestion aborted.")

        if not QDRANT_AVAILABLE:
            raise RuntimeError("Qdrant is unavailable; ingestion cannot continue.")

        if points and QDRANT_AVAILABLE:
            upsert_vectors(points, session_id=session_id)
            upserted_count = len(points)
            record(f"Upserted {upserted_count} vectors to Qdrant")

        if QDRANT_AVAILABLE:
            count = get_collection_count()
            record(f"VECTOR DB COUNT: {count}")
            if count == 0:
                raise RuntimeError("Vector DB is empty after ingestion.")
        else:
            count = 0
            record("Qdrant unavailable: skipping vector DB validation.")

        record(f"Ingestion complete. Unique files processed: {len(valid_paths)}")
        return {
            "status": "success",
            "vector_count": count,
            "upserted_count": upserted_count,
            "chunks_created": len(chunks),
            "steps": steps,
            "file_hashes": {os.path.basename(p[0]): p[1] for p in valid_paths},
        }
    except Exception as e:
        record(f"Ingestion failed: {e}")
        return {"status": "error", "message": str(e), "steps": steps, "chunks_created": 0, "file_hashes": {}}
