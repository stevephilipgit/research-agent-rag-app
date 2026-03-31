import logging
import re
from typing import List, Union
from langchain.docstore.document import Document
from langchain_core.messages import SystemMessage
from backend.config import ENABLE_VALIDATION
from backend.core.telemetry import emit_log

logger = logging.getLogger(__name__)

_validator_llm = None

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
        from backend.config.llm import get_llm
        _validator_llm = get_llm()
    return _validator_llm

def validate_answer(answer: str, docs: List[Document]) -> Union[bool, dict]:
    """
    Validates if the answer is grounded in the retrieved documents.
    Returns True/False or a dictionary (truthy) with warning info.
    """
    if not ENABLE_VALIDATION or not answer or not docs:
        return True

    # Preparation
    context_text = " ".join([doc.page_content for doc in docs]).lower()
    answer_clean = answer.lower()
    
    # 1. Keyword overlap check (STEP 2)
    def get_words(text):
        return set(re.findall(r"\w+", text))
    
    context_words = get_words(context_text)
    answer_words = get_words(answer_clean)
    
    match_count = sum(1 for word in answer_words if word in context_words)
    
    # 2. Rejection criteria (STEP 4)
    if match_count == 0:
        emit_log("Validation", "failure", "Strict rejection: No keyword overlap found", "query")
        return False

    # 3. Exact match bypass
    if answer_clean in context_text:
        emit_log("Validation", "success", "Validation passed (exact match)", "query")
        return True

    # 4. Semantic / LLM Check (STEP 6)
    try:
        llm = _get_validator_llm()
        if llm:
            prompt = _VALIDATION_SYSTEM_PROMPT.format(context=context_text[:3000], answer=answer)
            response = llm.invoke([SystemMessage(content=prompt)])
            result = str(getattr(response, "content", "VALID")).strip().upper()
            
            if "VALID" in result:
                emit_log("Validation", "success", "Validation passed (LLM checked)", "query")
                return True
            
            # If LLM says INVALID but we have decent keyword overlap, issue warning instead of rejection (STEP 3)
            if match_count > 5:
                emit_log("Validation", "warning", "Low grounding confidence (LLM mismatch but overlap > 5)", "query")
                return {
                    "answer": answer,
                    "warning": "Low confidence: answer may not be fully grounded in documents"
                }

            emit_log("Validation", "failure", f"Answer rejected by grounding validator: {result}", "query")
            return False
    except Exception as exc:
        logger.warning(f"Semantic validation failed: {exc}")
        # Fail safe to True if overlap is decent
        if match_count > 3:
            return True

    return match_count > 5 # Fallback to keyword count check
