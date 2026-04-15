# Alias for test compatibility
def retrieve_context(query: str, top_k: int = 5, session_id: Optional[str] = None) -> str:
    return retrieve_context_with_extensions(query, top_k=top_k, session_id=session_id)
import logging
import math
import os
import re
from typing import Optional
from collections import Counter, defaultdict

from langchain_core.documents import Document

from backend.services.query_rewriter import rewrite_query
from backend.services.context_compressor import compress_context
from backend.config.config import ENABLE_CACHE, ENABLE_HYBRID, ENABLE_REWRITE, ENABLE_COMPRESSION
from backend.infra.embeddings import get_embeddings
from backend.infra.vector_db import search_vectors, get_collection_count
from backend.core.reranker import rerank
from backend.core.telemetry import emit_log
from backend.utils.cache import get_embedding_cache, get_query_cache, set_embedding_cache, set_query_cache

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+", (text or "").lower())


def _dense_retrieve(query: str, top_k: int, session_id: Optional[str] = None) -> list[Document]:
    cached_embedding = get_embedding_cache(query) if ENABLE_CACHE else None
    if cached_embedding is not None:
        emit_log("Cache", "success", "Cache hit for embeddings", "query")
        query_embedding = cached_embedding
    else:
        emit_log("Cache", "in_progress", "Cache miss for embeddings", "query")
        query_embedding = get_embeddings().embed_query(query)
        if ENABLE_CACHE:
            set_embedding_cache(query, query_embedding)

    results = search_vectors(query_embedding, limit=top_k, session_id=session_id)
    
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


def hybrid_retrieve(query: str, top_k: int = 5, session_id: Optional[str] = None) -> list[Document]:
    if ENABLE_CACHE:
        cached_docs = get_query_cache(query, session_id=session_id)
        if cached_docs is not None:
            emit_log("Cache", "success", "Cache hit for retrieval", "query")
            return cached_docs
        emit_log("Cache", "in_progress", "Cache miss for retrieval", "query")

    dense_docs = _dense_retrieve(query, top_k=top_k, session_id=session_id)
    bm25_docs = _bm25_retrieve(query, top_k=top_k)
    merged_docs = _merge_results(dense_docs, bm25_docs, top_k=top_k)

    emit_log(
        "Retrieval",
        "success",
        f"Hybrid retrieval used | dense={len(dense_docs)} | bm25={len(bm25_docs)} | merged={len(merged_docs)}",
        "query",
    )

    if ENABLE_CACHE:
        set_query_cache(query, merged_docs, session_id=session_id)

    return merged_docs


def _format_context(docs: list[Document]) -> str:
    parts = []
    for doc in docs:
        source = doc.metadata.get("display_name") or doc.metadata.get("source") or doc.metadata.get("file_name") or "unknown"
        # Ensure only basename is displayed
        source = os.path.basename(source)
        page = doc.metadata.get("page", "N/A")
        section = doc.metadata.get("section", "")
        parts.append(f"[Source: {source}, Page: {page}, Section: {section}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def group_by_source(docs: list[Document]) -> list[Document]:
    """Groups chunks by source and picks the best source (one with most chunks)."""
    if not docs:
        return []
    
    grouped = defaultdict(list)
    for doc in docs:
        source = doc.metadata.get("display_name") or doc.metadata.get("source") or doc.metadata.get("file_name") or "unknown"
        source = os.path.basename(source)
        grouped[source].append(doc)
    
    if not grouped:
        return docs
        
    # pick best source (most chunks)
    best_source = max(grouped.items(), key=lambda x: len(x[1]))[0]
    logger.info(f"Source grouping prioritized source: {best_source}")
    return grouped[best_source]


def retrieve_context_with_extensions(query: str, top_k: int = 5, session_id: Optional[str] = None) -> str:
    if not query or not query.strip():
        return ""

    final_query = query
    is_rewritten = False
    
    try:
        if ENABLE_REWRITE:
            rewritten_query = rewrite_query(query)
            if rewritten_query and rewritten_query.strip():
                final_query = rewritten_query.strip()
                is_rewritten = True
    except Exception as exc:
        logger.warning("Query rewrite failed: %s", exc)
        final_query = query

    try:
        # Phase 2: Hybrid Query Fallback (normalized -> raw)
        docs = hybrid_retrieve(final_query, top_k=top_k, session_id=session_id) if ENABLE_HYBRID else _dense_retrieve(final_query, top_k=top_k, session_id=session_id)
        
        if not docs and is_rewritten:
            logger.info("Hybrid fallback: No results for normalized query, retrying with raw query.")
            docs = hybrid_retrieve(query, top_k=top_k, session_id=session_id) if ENABLE_HYBRID else _dense_retrieve(query, top_k=top_k, session_id=session_id)

        if not docs:
            # Fix 1 & 10: Specific "not found" return
            return ""

        # [NEW] Context Compression
        if ENABLE_COMPRESSION:
            try:
                docs = compress_context(docs)
            except Exception as exc:
                logger.warning(f"Context compression failed: {exc}")

        # Rerank
        ranked_docs = rerank(query, docs, top_k=top_k)
        if not ranked_docs:
            return ""

        # Fix 3: Source Grouping
        prioritized_docs = group_by_source(ranked_docs)
        
        # Fix 9: Enforce Top-K strictly after rerank/grouping
        final_docs = prioritized_docs[:3]

        return _format_context(final_docs)
    except Exception as exc:
        logger.error("Retrieval pipeline failed: %s", exc, exc_info=True)
        emit_log("Retrieval", "failure", f"Pipeline fallback triggered: {exc}", "query")
        docs = _dense_retrieve(query, top_k, session_id=session_id)
        return _format_context(docs[:3])
