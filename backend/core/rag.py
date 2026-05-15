# Alias for test compatibility
from typing import Optional, List


def retrieve_context(query: str, top_k: int = 5, session_id: Optional[str] = None) -> str:
    return retrieve_context_with_extensions(query, top_k=top_k, session_id=session_id)


import logging
import os
import re
import time
from collections import defaultdict

from langchain_core.documents import Document

from services.query_rewriter import rewrite_query
from services.context_compressor import compress_context
from config.config import ENABLE_CACHE, ENABLE_HYBRID, ENABLE_REWRITE, ENABLE_COMPRESSION
from infra.embeddings import get_embeddings
from infra.vector_db import search_vectors, get_collection_count
from core.reranker import rerank
from core.telemetry import emit_log
from utils.cache import get_embedding_cache, get_query_cache, set_embedding_cache, set_query_cache

logger = logging.getLogger(__name__)


def _parse_retrieval_filters(query: str) -> dict:
    """Extract optional source/topic/document_type filters from query text."""
    source_filters: list[str] = []

    # Only search <file> / source:<file>
    m = re.findall(r"(?:only\s+search|source\s*:)\s*([A-Za-z0-9_.\- ]+\.(?:pdf|txt|docx|csv))", query or "", flags=re.IGNORECASE)
    if m:
        source_filters = [os.path.basename(x.strip()) for x in m]

    topic_filters: list[str] = []
    for topic in ["machine_learning", "cybersecurity", "cloud", "networking", "mathematics"]:
        probe = topic.replace("_", " ")
        if probe in (query or "").lower() or topic in (query or "").lower():
            topic_filters.append(topic)

    doc_type = None
    for dt in ["textbook", "research_paper", "notes", "code", "cybersecurity", "medical", "legal", "mixed", "general"]:
        if dt.replace("_", " ") in (query or "").lower() or dt in (query or "").lower():
            doc_type = dt
            break

    return {
        "source_filters": source_filters or None,
        "topic_filters": topic_filters or None,
        "document_type": doc_type,
    }


def _dense_retrieve(
    query: str,
    top_k: int,
    session_id: Optional[str] = None,
    source_filters: Optional[List[str]] = None,
    topic_filters: Optional[List[str]] = None,
    document_type: Optional[str] = None,
) -> list[Document]:
    cached_embedding = get_embedding_cache(query) if ENABLE_CACHE else None
    if cached_embedding is not None:
        emit_log("Cache", "success", "Cache hit for embeddings", "query")
        query_embedding = cached_embedding
    else:
        emit_log("Cache", "in_progress", "Cache miss for embeddings", "query")
        query_embedding = get_embeddings().embed_query(query)
        if ENABLE_CACHE:
            set_embedding_cache(query, query_embedding)

    # metadata filtering occurs before vector search inside vector_db.search_vectors
    results = search_vectors(
        query_embedding,
        limit=max(top_k * 4, 12),
        session_id=session_id or "default",
        source_filters=source_filters,
        topic_filters=topic_filters,
        document_type=document_type,
        user_id=session_id or "default",
    )

    if not results and session_id:
        logger.warning(f"[Retrieval] Session-scoped search returned 0 results for session {session_id}, retrying globally")
        results = search_vectors(
            query_embedding,
            limit=max(top_k * 4, 12),
            session_id=session_id,
            source_filters=source_filters,
            topic_filters=topic_filters,
            document_type=document_type,
            user_id=None,
        )

    docs = []
    for r in results:
        payload = r.get("payload", {})
        content = r.get("content", "")
        docs.append(Document(page_content=content, metadata=payload))
    return docs


def _bm25_retrieve(query: str, top_k: int) -> list[Document]:
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

    return merged[: max(top_k * 3, top_k)]


def _diversify_by_source(docs: list[Document], top_k: int) -> list[Document]:
    """Simple source diversification: round-robin across sources after relevance ordering."""
    if not docs:
        return []

    groups = defaultdict(list)
    for d in docs:
        source = os.path.basename(d.metadata.get("source") or d.metadata.get("display_name") or d.metadata.get("file_name") or "unknown")
        groups[source].append(d)

    buckets = list(groups.values())
    out = []
    idx = 0
    while len(out) < top_k:
        progressed = False
        for bucket in buckets:
            if idx < len(bucket):
                out.append(bucket[idx])
                progressed = True
                if len(out) >= top_k:
                    break
        if not progressed:
            break
        idx += 1
    return out


def hybrid_retrieve(
    query: str,
    top_k: int = 5,
    session_id: Optional[str] = None,
    source_filters: Optional[List[str]] = None,
    topic_filters: Optional[List[str]] = None,
    document_type: Optional[str] = None,
) -> list[Document]:
    cache_key = f"{query}|s={session_id}|src={source_filters}|topic={topic_filters}|dt={document_type}"
    if ENABLE_CACHE:
        cached_docs = get_query_cache(cache_key, session_id=session_id)
        if cached_docs is not None:
            emit_log("Cache", "success", "Cache hit for retrieval", "query")
            return cached_docs
        emit_log("Cache", "in_progress", "Cache miss for retrieval", "query")

    dense_docs = _dense_retrieve(
        query,
        top_k=top_k,
        session_id=session_id,
        source_filters=source_filters,
        topic_filters=topic_filters,
        document_type=document_type,
    )
    bm25_docs = _bm25_retrieve(query, top_k=top_k)
    merged_docs = _merge_results(dense_docs, bm25_docs, top_k=top_k)

    if len(merged_docs) == 0 and len(dense_docs) > 0:
        logger.warning("[Retrieval] Hybrid returned 0 results, falling back to dense-only")
        merged_docs = dense_docs

    emit_log(
        "Retrieval",
        "success",
        (
            f"Hybrid retrieval used | session={session_id or 'default'} | "
            f"sources={source_filters} | topics={topic_filters} | doc_type={document_type} | "
            f"dense={len(dense_docs)} | bm25={len(bm25_docs)} | merged={len(merged_docs)}"
        ),
        "query",
    )

    if ENABLE_CACHE:
        set_query_cache(cache_key, merged_docs, session_id=session_id)

    return merged_docs


def _format_context(docs: list[Document]) -> str:
    parts = []
    for doc in docs:
        source = doc.metadata.get("display_name") or doc.metadata.get("source") or doc.metadata.get("file_name") or "unknown"
        source = os.path.basename(source)
        page = doc.metadata.get("page", "N/A")
        section = doc.metadata.get("section", "")
        chunk_index = doc.metadata.get("chunk_index", "N/A")
        parts.append(f"[Source: {source}, Page: {page}, Chunk: {chunk_index}, Section: {section}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def retrieve_context_with_extensions(query: str, top_k: int = 5, session_id: Optional[str] = None) -> str:
    if not query or not query.strip():
        return ""

    started = time.perf_counter()
    filters = _parse_retrieval_filters(query)
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
        docs = (
            hybrid_retrieve(
                final_query,
                top_k=top_k,
                session_id=session_id,
                source_filters=filters["source_filters"],
                topic_filters=filters["topic_filters"],
                document_type=filters["document_type"],
            )
            if ENABLE_HYBRID
            else _dense_retrieve(
                final_query,
                top_k=top_k,
                session_id=session_id,
                source_filters=filters["source_filters"],
                topic_filters=filters["topic_filters"],
                document_type=filters["document_type"],
            )
        )

        if not docs and is_rewritten:
            logger.info("Hybrid fallback: No results for normalized query, retrying with raw query.")
            docs = (
                hybrid_retrieve(
                    query,
                    top_k=top_k,
                    session_id=session_id,
                    source_filters=filters["source_filters"],
                    topic_filters=filters["topic_filters"],
                    document_type=filters["document_type"],
                )
                if ENABLE_HYBRID
                else _dense_retrieve(
                    query,
                    top_k=top_k,
                    session_id=session_id,
                    source_filters=filters["source_filters"],
                    topic_filters=filters["topic_filters"],
                    document_type=filters["document_type"],
                )
            )

        if not docs:
            return ""

        if ENABLE_COMPRESSION:
            try:
                docs = compress_context(docs)
            except Exception as exc:
                logger.warning(f"Context compression failed: {exc}")

        rerank_start = time.perf_counter()
        ranked_docs = rerank(query, docs, top_k=max(top_k * 2, top_k))
        rerank_ms = (time.perf_counter() - rerank_start) * 1000.0
        if not ranked_docs:
            return ""

        final_docs = _diversify_by_source(ranked_docs, top_k=min(max(top_k, 3), 8))
        unique_sources = len({os.path.basename(d.metadata.get("source", "unknown")) for d in final_docs})

        total_ms = (time.perf_counter() - started) * 1000.0
        logger.info(
            "Retrieval summary | session=%s | filters=%s | chunks=%d | unique_sources=%d | reranker_latency_ms=%.2f | total_latency_ms=%.2f",
            session_id or "default",
            filters,
            len(final_docs),
            unique_sources,
            rerank_ms,
            total_ms,
        )
        emit_log("Retrieval", "success", f"retrieved {len(final_docs)} chunks from {unique_sources} sources", "query")

        return _format_context(final_docs)
    except Exception as exc:
        logger.error("Retrieval pipeline failed: %s", exc, exc_info=True)
        emit_log("Retrieval", "failure", f"Pipeline fallback triggered: {exc}", "query")
        docs = _dense_retrieve(query, top_k, session_id=session_id)
        return _format_context(docs[:3])
