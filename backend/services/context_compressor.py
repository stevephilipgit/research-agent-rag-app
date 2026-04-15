import logging
from typing import List
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from backend.config.config import ENABLE_COMPRESSION
from backend.core.telemetry import emit_log

logger = logging.getLogger(__name__)

_compressor_llm = None

def _get_compressor_llm():
    global _compressor_llm
    if _compressor_llm is None:
        from backend.config.llm import get_llm
        _compressor_llm = get_llm()
    return _compressor_llm

def summarize(text: str) -> str:
    """Simple LLM summarization for context compression."""
    if not text or len(text) < 200:
        return text

    try:
        llm = _get_compressor_llm()
        if llm:
            response = llm.invoke([
                SystemMessage(content="Summarize the following text concisely, preserving key facts and entities."),
                HumanMessage(content=text)
            ])
            return str(getattr(response, "content", text)).strip()
    except Exception as exc:
        logger.warning(f"Summarization failed: {exc}")
    
    return text[:500] + "..." # Fallback truncation

def compress_context(docs: List[Document]) -> List[Document]:
    if not ENABLE_COMPRESSION or not docs:
        return docs

    emit_log("Context Compression", "in_progress", f"Compressing {len(docs)} documents", "query")
    compressed_docs = []
    
    try:
        for doc in docs:
            # Hard limit truncation first to save tokens for summary if needed
            text = doc.page_content[:2000]
            
            # Optional: LLM summarization
            summary = summarize(text)
            
            new_doc = Document(
                page_content=summary,
                metadata=doc.metadata
            )
            compressed_docs.append(new_doc)
            
        emit_log("Context Compression", "success", f"Compressed to {len(compressed_docs)} summaries", "query")
        return compressed_docs
    except Exception as exc:
        logger.error(f"Compression failed: {exc}")
        emit_log("Context Compression", "failure", f"Fallback to original docs: {exc}", "query")
        return docs
