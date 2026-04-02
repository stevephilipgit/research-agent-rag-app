import hashlib
import os
import json
import logging
from typing import Optional, Any
from config.settings import ENABLE_CACHE

logger = logging.getLogger(__name__)

# ===== CACHE CONFIG =====
# Redis is optional — if not installed or REDIS_URL not set, falls back to in-memory cache.
REDIS_URL = os.getenv("REDIS_URL")
redis_client = None

if REDIS_URL:
    try:
        import redis as _redis
        redis_client = _redis.Redis.from_url(REDIS_URL, decode_responses=True)
    except Exception as e:
        logger.warning(f"Redis unavailable, using in-memory cache: {e}")

# In-memory implementation for fallback
_cache = {}

def _get_key(prefix: str, query: str, session_id: Optional[str] = None) -> str:
    combined = f"{session_id}:{query}" if session_id else query
    h = hashlib.md5(combined.encode()).hexdigest()
    return f"{prefix}:{h}"

def get_cache_raw(key: str) -> Optional[Any]:
    if not ENABLE_CACHE:
        return None
    if redis_client:
        try:
            val = redis_client.get(key)
            if val:
                return json.loads(val)
        except Exception as e:
            logger.warning(f"Redis get failed for {key}: {e}")
    return _cache.get(key)

def set_cache_raw(key: str, value: Any, ttl: int = 300):
    if not ENABLE_CACHE:
        return
    if redis_client:
        try:
            redis_client.set(key, json.dumps(value), ex=ttl)
            return
        except Exception as e:
            logger.warning(f"Redis set failed for {key}: {e}")
    _cache[key] = value

def get_cache(query: str, session_id: Optional[str] = None) -> Optional[Any]:
    return get_cache_raw(_get_key("query", query, session_id))

def set_cache(query: str, value: Any, session_id: Optional[str] = None, ttl: int = 300):
    set_cache_raw(_get_key("query", query, session_id), value, ttl)

def get_query_cache(query: str, session_id: Optional[str] = None) -> Optional[Any]:
    return get_cache_raw(_get_key("query", query, session_id))

def set_query_cache(query: str, docs: Any, session_id: Optional[str] = None):
    set_cache_raw(_get_key("query", query, session_id), docs)

def get_embedding_cache(query: str) -> Optional[Any]:
    # Embeddings are global for the same query text
    return get_cache_raw(_get_key("emb", query))

def set_embedding_cache(query: str, embedding: Any):
    set_cache_raw(_get_key("emb", query), embedding)

def get_cached_response(query: str, session_id: Optional[str] = None) -> Optional[Any]:
    return get_cache_raw(_get_key("resp", query, session_id))

def set_cached_response(query: str, response: Any, session_id: Optional[str] = None):
    set_cache_raw(_get_key("resp", query, session_id), response)
