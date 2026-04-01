import logging
import re

logger = logging.getLogger(__name__)

def normalize_query(query: str) -> str:
    """Fast query normalization by lowercasing and removing noise words."""
    if not query:
        return ""
    
    query = query.lower().strip()
    
    # Noise words removal
    noise_words = ["please", "can you", "could you", "tell me", "explain", "give me", "find", "search for", "about", "the"]
    for word in noise_words:
        # Use regex to match whole words only
        query = re.sub(rf"\b{word}\b", "", query)
    
    # Remove extra spaces
    query = re.sub(r"\s+", " ", query).strip()
    return query

def rewrite_query(query: str, history=None) -> str:
    """Fast normalization first."""
    normalized = normalize_query(query)
    logger.info(f"Query normalized: '{query}' -> '{normalized}'")
    return normalized
