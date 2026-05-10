import logging
from typing import List, Union, Optional
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage
from config import ENABLE_VALIDATION
from core.telemetry import emit_log
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

_validator_llm = None
GROUNDING_COSINE_THRESHOLD = 0.25

_VALIDATION_SYSTEM_PROMPT = """You are a RAG answer validator.
Check if the Answer is supported by the Context.

Context:
{context}

Answer:
{answer}

Rules:
Return ONLY "VALID" if the answer is mostly supported.
Return "INVALID" if the answer is completely unrelated or contains major hallucinations.
"""

def _get_validator_llm():
    global _validator_llm
    if _validator_llm is None:
        from config.llm import get_llm
        _validator_llm = get_llm()
    return _validator_llm

def validate_answer(answer: str, docs: List[Document]) -> Union[bool, dict]:
    """
    Validates if the answer is grounded in the retrieved documents.
    Uses cosine similarity between context and answer embeddings for better validation.
    Returns True/False or a dictionary (truthy) with warning info.
    """
    if not ENABLE_VALIDATION or not answer or not docs:
        return True

    context_text = " ".join([doc.page_content for doc in docs])
    answer_text = answer

    try:
        from infra.embeddings import get_embeddings
        embedder = get_embeddings()

        if not context_text or not answer_text:
            return False

        if hasattr(embedder, "encode"):
            context_emb = embedder.encode([context_text])
            answer_emb = embedder.encode([answer_text])
        else:
            context_emb = np.array([embedder.embed_query(context_text)])
            answer_emb = np.array([embedder.embed_query(answer_text)])

        score = cosine_similarity(context_emb, answer_emb)[0][0]
        logger.info(f"Grounding cosine score: {score:.4f} (threshold: {GROUNDING_COSINE_THRESHOLD})")
        emit_log("Validation", "success" if float(score) >= GROUNDING_COSINE_THRESHOLD else "failure", f"Grounding cosine score={score:.4f}", "query")
        return float(score) >= GROUNDING_COSINE_THRESHOLD
    except Exception as exc:
        logger.warning(f"Semantic validation (cosine sim) failed: {exc}")
        emit_log("Validation", "failure", "Grounding cosine validation failed", "query")
        return False
