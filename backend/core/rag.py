import logging
import math
import re
from collections import Counter, defaultdict

from langchain.docstore.document import Document

from services.query_rewriter import rewrite_query
from services.context_compressor import compress_context
from config import ENABLE_CACHE, ENABLE_HYBRID, ENABLE_REWRITE, ENABLE_COMPRESSION
from infra.embeddings import get_embeddings
from infra.vector_db import search_vectors, get_collection_count
from core.reranker import rerank
from core.telemetry import emit_log
from utils.cache import (
    get_embedding_cache,
    get_query_cache,
    set_embedding_cache,
    set_query_cache,
)

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+", (text or "").lower())


def _dense_retrieve(query: str, top_k: int) -> list[Document]:
    cached_embedding = get_embedding_cache(query) if ENABLE_CACHE else None
    if cached_embedding is not None:
        emit_log("Cache", "success", "Cache hit for embeddings", "query")
        query_embedding = cached_embedding
    else:
        emit_log("Cache", "in_progress", "Cache miss for embeddings", "query")
        query_embedding = get_embeddings().embed_query(query)
        if ENABLE_CACHE:
            set_embedding_cache(query, query_embedding)

    results = search_vectors(query_embedding, limit=top_k)
    
    docs = []
    for r in results:
        payload = r.get("payload", {})
        content = r.get("content", "")
        docs.append(Document(page_content=content, metadata=payload))
    
    return docs


def _bm25_retrieve(query: str, top_k: int) -> list[Document]:
    # Cloud Optimization: Standard BM25 requires scanning all documents.
    # For this refactor, we disable local BM25 to avoid massive cloud data transfer.
    # Qdrant's Full Text Search could be implemented here in a future task.
    logger.warning("Local BM25 retrieval disabled in cloud mode to prevent full collection scan.")
    return []


def _merge_results(dense_docs: list[Document], bm25_docs: list[Document], top_k: int) -> list[Document]:
    merged: list[Document] = []
    seen = set()

    for doc in [*dense_docs, *bm25_docs]:
        content = (doc.page_content or "").strip()
        if not content or content in seen:
            continue
        seen.add(content)
        merged.append(doc)

    return merged[: max(top_k * 2, top_k)]


def hybrid_retrieve(query: str, top_k: int = 5) -> list[Document]:
    if ENABLE_CACHE:
        cached_docs = get_query_cache(query)
        if cached_docs is not None:
            emit_log("Cache", "success", "Cache hit for retrieval", "query")
            return cached_docs
        emit_log("Cache", "in_progress", "Cache miss for retrieval", "query")

    dense_docs = _dense_retrieve(query, top_k=top_k)
    bm25_docs = _bm25_retrieve(query, top_k=top_k)
    merged_docs = _merge_results(dense_docs, bm25_docs, top_k=top_k)

    emit_log(
        "Retrieval",
        "success",
        f"Hybrid retrieval used | dense={len(dense_docs)} | bm25={len(bm25_docs)} | merged={len(merged_docs)}",
        "query",
    )

    if ENABLE_CACHE:
        set_query_cache(query, merged_docs)

    return merged_docs


def _format_context(docs: list[Document]) -> str:
    parts = []
    for doc in docs:
        source = doc.metadata.get("source") or doc.metadata.get("file_name") or "unknown"
        page = doc.metadata.get("page", "N/A")
        section = doc.metadata.get("section", "")
        parts.append(f"[Source: {source}, Page: {page}, Section: {section}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def retrieve_context_with_extensions(query: str, top_k: int = 5) -> str:
    if not query or not query.strip():
        return ""

    if not any([ENABLE_REWRITE, ENABLE_HYBRID, ENABLE_CACHE, ENABLE_COMPRESSION]):
        docs = _dense_retrieve(query, top_k)
        return _format_context(docs)

    final_query = query
    try:
        if ENABLE_REWRITE:
            rewritten_query = rewrite_query(query)
            if rewritten_query and rewritten_query.strip():
                final_query = rewritten_query.strip()
    except Exception as exc:
        logger.warning("Query rewrite wrapper failed: %s", exc)
        emit_log("Query Rewrite", "failure", f"Wrapper fallback to original query: {exc}", "query")
        final_query = query

    try:
        docs = hybrid_retrieve(final_query, top_k=top_k) if ENABLE_HYBRID else _dense_retrieve(final_query, top_k=top_k)
        if not docs:
            return ""

        # [NEW] Context Compression
        if ENABLE_COMPRESSION:
            try:
                docs = compress_context(docs)
            except Exception as exc:
                logger.warning(f"Context compression failed: {exc}")

        ranked_docs = rerank(query, docs, top_k=top_k)
        if not ranked_docs:
            return ""

        return _format_context(ranked_docs)
    except Exception as exc:
        logger.error("Hybrid retrieval wrapper failed: %s", exc, exc_info=True)
        emit_log("Retrieval", "failure", f"Hybrid wrapper fallback triggered: {exc}", "query")
        docs = _dense_retrieve(query, top_k)
        return _format_context(docs)
