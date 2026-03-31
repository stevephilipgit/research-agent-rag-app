import os
import logging
from typing import Optional, Callable
from chromadb.config import Settings
from langchain_chroma import Chroma
from config import VECTOR_STORE_PATH
from infra.embeddings import get_embeddings

logger = logging.getLogger(__name__)
_vector_store: Optional[Chroma] = None

def load_vector_store(callback: Optional[Callable[[str], None]] = None) -> Chroma:
    global _vector_store
    if _vector_store is None:
        os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
        embeddings = get_embeddings(callback)
        _vector_store = Chroma(
            persist_directory=VECTOR_STORE_PATH,
            embedding_function=embeddings,
            client_settings=Settings(anonymized_telemetry=False),
        )
    return _vector_store

def get_vector_store_count() -> int:
    try:
        vector_store = load_vector_store()
        data = vector_store.get()
        return len(data.get("ids", []) or [])
    except Exception as exc:
        logger.warning(f"Unable to read vector store count: {exc}")
        return 0
