import time
import logging

logger = logging.getLogger(__name__)

def safe_stream(llm, prompt, retries=3, delay=1):
    """
    Safely stream from an LLM with retries and a fallback to invoke.
    """
    for attempt in range(retries):
        try:
            for chunk in llm.stream(prompt):
                yield chunk
            return
        except Exception as e:
            logger.error(f"[STREAM ERROR] Attempt {attempt+1}: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                # Final attempt failed, try one last time with invoke (fallback)
                try:
                    logger.info("Retries exhausted. Falling back to llm.invoke()")
                    response = llm.invoke(prompt)
                    yield response
                    return
                except Exception as final_e:
                    logger.error(f"[CRITICAL ERROR] Fallback failed: {final_e}")
                    # Yielding a dummy message object as expected by the loop
                    from langchain_core.messages import AIMessage
                    yield AIMessage(content="⚠️ Connection issue. Please try again.")
