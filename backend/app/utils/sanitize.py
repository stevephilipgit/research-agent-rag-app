import re
import logging
from typing import Callable, Any
from langchain_core.messages import AIMessage

logger = logging.getLogger(__name__)

def clean_query(query: str, max_length: int = 500) -> str:
    """
    Sanitizes user input by trimming whitespace, normalizing spacing,
    removing strictly unsafe characters, and limiting length.
    """
    if not query:
        return ""
    
    # Trim and normalize spaces
    cleaned = re.sub(r'\s+', ' ', query.strip())
    
    # Remove null bytes or other control characters using printable regex subset
    cleaned = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', cleaned)
    
    # Limit length precisely
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length]
        
    return cleaned

def safe_tool_call(func: Callable, *args, **kwargs) -> Any:
    """
    Wraps tool executions to prevent crashes in the agent loop.
    Returns a safe string fallback.
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.error(f"Tool {getattr(func, '__name__', 'unknown')} failed: {e}")
        return "[Error: Internal tool execution failed. The information might not be available.]"

def safe_llm_call(llm, messages, retries: int = 2) -> AIMessage:
    """
    Invokes the LLM safely with internal retries to ensure pipeline stability.
    If all retries fail, returns a fallback AIMessage.
    """
    last_err = None
    for attempt in range(retries):
        try:
            return llm.invoke(messages)
        except Exception as e:
            last_err = e
            logger.warning(f"LLM call failed (attempt {attempt + 1}/{retries}): {e}")
            
    logger.error(f"LLM strictly failed after {retries} attempts: {last_err}")
    # Return a fallback AIMessage containing standard failure message
    return AIMessage(content="The information is not available in the provided documents.")
