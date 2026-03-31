"""RAG retrieval pipeline - uses the shared ChromaDB client from document_loader."""

import logging

from backend.utils.cache_db import get_cached_retrieval, set_cached_retrieval
from backend.core.document_loader import load_vector_store
from backend.core.query_rewriter_legacy import rewrite_query
from backend.core.reranker import rerank
from backend.core.telemetry import emit_log

logger = logging.getLogger(__name__)


def get_retriever():
    vector_store = load_vector_store()
    return vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 8},
    )


def retrieve_context(query: str) -> str:
    cached = get_cached_retrieval(query)
    if cached:
        emit_log("Retrieval", "success", "Cache hit for retrieval", "query")
        return cached

    try:
        retriever = get_retriever()
    except Exception as e:
        logger.error(f"Failed to initialize retriever: {e}", exc_info=True)
        emit_log("Retrieval", "failure", str(e), "query")
        return "Retrieval failed due to an internal error. Please try again."

    rewritten = rewrite_query(query, history=[])
    search_queries = list(set([query, rewritten]))
    emit_log("Retrieval", "in_progress", f"Running retriever for {len(search_queries)} query variant(s)", "query")

    docs = []
    for q in search_queries:
        if not q:
            continue
        try:
            docs.extend(retriever.invoke(q))
        except Exception as e:
            logger.warning(f"Retriever failed for query '{q}': {e}")
            emit_log("Retrieval", "failure", f"Retriever failed for '{q}': {e}", "query")

    if not docs:
        emit_log("Retrieval", "success", "Retrieved 0 documents", "query")
        return "No relevant content found."

    # deduplicate
    seen = set()
    unique = []
    for d in docs:
        txt = d.page_content.strip()
        if txt not in seen:
            seen.add(txt)
            unique.append(d)

    ranked_docs = rerank(query, unique, top_k=5)
    emit_log("Retrieval", "success", f"Retrieved {len(unique)} documents before reranking", "query")

    # light filter
    def valid(t):
        return len(t) > 50 and t.count(",") < 15

    filtered = [d for d in ranked_docs if valid(d.page_content)]

    if not filtered:
        filtered = ranked_docs[:4]

    final_docs = filtered

    parts = []
    for d in final_docs:
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", "N/A")
        section = d.metadata.get("section", "")
        parts.append(f"[Source: {src}, Page: {page}, Section: {section}]\n{d.page_content}")

    context = "\n\n---\n\n".join(parts)
    set_cached_retrieval(query, context)
    return context
