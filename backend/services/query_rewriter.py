import logging

from langchain_core.messages import HumanMessage, SystemMessage
from config import ENABLE_REWRITE
from core.query_rewriter_legacy import rewrite_query as legacy_rewrite_query
from core.telemetry import emit_log

logger = logging.getLogger(__name__)

_rewriter_llm = None

_SYSTEM_PROMPT = """Rewrite the user's question for document retrieval.

Rules:
- Preserve meaning.
- Keep it short and search-oriented.
- Include key entities and technical terms.
- Return only the rewritten query text.
"""


def _get_rewriter_llm():
    global _rewriter_llm
    if _rewriter_llm is None:
        from config.llm import get_llm
        _rewriter_llm = get_llm()
    return _rewriter_llm


def rewrite_query(query: str) -> str:
    if not ENABLE_REWRITE or not query or not query.strip():
        return query

    try:
        llm = _get_rewriter_llm()
        if llm is not None:
            response = llm.invoke(
                [
                    SystemMessage(content=_SYSTEM_PROMPT),
                    HumanMessage(content=query.strip()),
                ]
            )
            improved_query = str(getattr(response, "content", "") or "").strip()
            if improved_query:
                emit_log("Query Rewrite", "success", "Query rewritten", "query")
                return improved_query
    except Exception as exc:
        logger.warning("LLM query rewrite failed: %s", exc)
        emit_log("Query Rewrite", "failure", f"Rewrite failed, using fallback: {exc}", "query")

    try:
        improved_query = legacy_rewrite_query(query, history=[])
        if improved_query:
            emit_log("Query Rewrite", "success", "Query rewritten", "query")
            return improved_query
    except Exception as exc:
        logger.warning("Legacy query rewrite failed: %s", exc)

    return query
