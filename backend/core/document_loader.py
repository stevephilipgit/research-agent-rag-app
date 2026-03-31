"""Document loader for the RAG ingestion pipeline - focused on loading and chunking."""

import hashlib
import logging
import os
import re
from typing import Callable, List, Optional

from langchain.docstore.document import Document
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
from infra.vector_db import upsert_vectors, delete_vectors_by_doc_id, get_collection_count
from infra.db import load_registry, remove_from_registry
from utils.file_handling import download_file, is_url
from infra.storage import get_file_url

logger = logging.getLogger(__name__)

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

def load_documents(
    file_paths: List[str],
    callback: Optional[Callable[[str], None]] = None,
) -> List[Document]:
    registry = load_registry()
    registry_by_name = {d.get("file_name"): d.get("doc_id") for d in registry}

    documents: List[Document] = []
    for path in file_paths:
        file_path = path
        is_temp = False
        try:
            # If it's a cloud path (doesn't start with http but is in our 'uploads/' storage)
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
            if not loader: continue
            docs = loader.load()
            
            doc_id = registry_by_name.get(os.path.basename(file_path))
            for doc in docs:
                doc.metadata["file_name"] = os.path.basename(file_path)
                if doc_id: doc.metadata["doc_id"] = doc_id
                doc.page_content = clean_text(doc.page_content)
            documents.extend(docs)
        except Exception as exc:
            logger.error(f"Failed to load {file_path}: {exc}")
            _emit_step(callback, f"⚠️ Failed to process document: {os.path.basename(file_path)}")
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
    # Vectors are now deleted directly in rag_service via vector_db.delete_vectors_by_doc_id
    # This remains for registry cleanup
    registry = load_registry()
    remove_from_registry(doc_id)
    logger.info(f"Removed doc_id {doc_id} from registry")
    

def ingest_documents(file_paths: Optional[List[str]] = None, callback: Optional[Callable[[str], None]] = None) -> dict:
    steps = []
    def record(m): steps.append(m); _emit_step(callback, m)
    
    try:
        if file_paths is None:
            file_paths = [os.path.join(DOCUMENTS_PATH, f) for f in os.listdir(DOCUMENTS_PATH) if os.path.isfile(os.path.join(DOCUMENTS_PATH, f))]
        
        docs = load_documents(file_paths, callback=record)
        chunks = chunk_documents(docs, callback=record)
        
        embeddings_model = get_embeddings(callback=record)
        
        points = []
        for c in chunks:
            cid = hashlib.sha256(c.page_content.encode()).hexdigest()
            # We don't perform a 'contains' check here for performance on cloud, 
            # Qdrant upsert will handle duplicates/updates
            c.metadata["id"] = cid
            points.append({
                "id": cid,
                "vector": embeddings_model.embed_query(c.page_content),
                "payload": {
                    "page_content": c.page_content,
                    **c.metadata
                }
            })
        
        if points:
            upsert_vectors(points)
            
        record(f"Ingestion complete. Vector count: {get_collection_count()}")
        return {"status": "success", "vector_count": get_collection_count(), "steps": steps}
    except Exception as e:
        record(f"Error: {e}")
        return {"status": "error", "message": str(e), "steps": steps}
