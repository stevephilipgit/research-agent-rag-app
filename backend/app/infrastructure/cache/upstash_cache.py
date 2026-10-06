"""
cache_db.py — Upstash Redis response cache.

Uses the Upstash HTTP client (upstash-redis), NOT redis-py.
Upstash uses a REST API, so it works on Render's free tier without TCP ports.

All calls are wrapped in try/except — if Redis is down the app continues
without caching (graceful degradation, never crashes).
"""
import hashlib
import json
import logging
import os
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

# BUG 1 FIX: Import once at module load — never per-request.
# If the package is missing we degrade gracefully without spamming logs.
try:
    from upstash_redis import Redis as _UpstashRedis
    _UPSTASH_AVAILABLE = True
except ImportError:
    _UpstashRedis = None  # type: ignore[assignment,misc]
    _UPSTASH_AVAILABLE = False
    logger.warning(
        "upstash_redis package not installed — Redis cache disabled. "
        "Run: pip install upstash-redis>=0.15.0"
    )

_redis_client = None


def _get_redis():
    """Lazily initialise the Upstash Redis client once and reuse it."""
    global _redis_client
    if not _UPSTASH_AVAILABLE:
        return None
    if _redis_client is not None:
        return _redis_client

    url = os.getenv("UPSTASH_REDIS_REST_URL")
    token = os.getenv("UPSTASH_REDIS_REST_TOKEN")
    if not url or not token:
        logger.warning(
            "UPSTASH_REDIS_REST_URL or UPSTASH_REDIS_REST_TOKEN not set — "
            "response cache disabled, falling back to in-memory."
        )
        return None

    try:
        _redis_client = _UpstashRedis(url=url, token=token)
        logger.info("Upstash Redis client initialised.")
        return _redis_client
    except Exception as exc:
        logger.warning(f"Upstash Redis client init failed, cache disabled: {exc}")
        return None


# ── BUG 2 FIX: Document serialisation helpers ────────────────────────────────
# LangChain Document objects are not JSON-serialisable; convert to plain dicts.

def _serialize_docs(docs: List[Any]) -> List[dict]:
    """Convert a list of Document objects (or plain dicts) to JSON-safe dicts."""
    out = []
    for doc in docs:
        if hasattr(doc, "page_content") and hasattr(doc, "metadata"):
            out.append({"page_content": doc.page_content, "metadata": doc.metadata})
        elif isinstance(doc, dict):
            out.append(doc)
        else:
            out.append({"page_content": str(doc), "metadata": {}})
    return out


def _deserialize_docs(raw: List[dict]) -> List[Any]:
    """Reconstruct Document objects from serialised dicts."""
    try:
        from langchain_core.documents import Document
        return [Document(page_content=d["page_content"], metadata=d.get("metadata", {})) for d in raw]
    except Exception:
        return raw  # graceful: return plain dicts if langchain unavailable


# In-memory fallback when Redis is unavailable
_memory_cache: dict = {}


def _make_key(query: str) -> str:
    """MD5 hash of normalised query with a 'resp:' namespace prefix."""
    digest = hashlib.md5(query.strip().lower().encode()).hexdigest()
    return f"resp:{digest}"


def get_cached_response(query: str, session_id: str = None) -> Optional[str]:
    """
    Return cached answer string for *query*, or None on a MISS.
    Tries Upstash Redis first, falls back to in-memory dict.
    Uses session_id to isolate cache per user.
    """
    # Use session_id as part of cache key to isolate per user
    cache_key = f"{session_id}:{query}" if session_id else query
    key = _make_key(cache_key)
    try:
        redis = _get_redis()
        if redis is not None:
            value = redis.get(key)
            if value:
                logger.info(f"Cache HIT  (Redis) for query: {query[:50]}")
                return value
            logger.info(f"Cache MISS (Redis) for query: {query[:50]}")
            return None
    except Exception as exc:
        logger.warning(f"Redis get failed, checking memory: {exc}")

    # In-memory fallback
    value = _memory_cache.get(key)
    if value:
        logger.info(f"Cache HIT  (memory) for query: {query[:50]}")
        return value
    logger.info(f"Cache MISS (memory) for query: {query[:50]}")
    return None


def set_cached_response(
    query: str, response: str, session_id: str = None, ttl_seconds: int = 3600
) -> None:
    """
    Store *response* in Redis (TTL 1h) and in-memory fallback.
    Uses session_id for key isolation.
    """
    cache_key = f"{session_id}:{query}" if session_id else query
    key = _make_key(cache_key)
    try:
        redis = _get_redis()
        if redis is not None:
            redis.set(key, response, ex=ttl_seconds)
            logger.info(f"Cached response (Redis) for query: {query[:50]}")
            return
    except Exception as exc:
        logger.warning(f"Redis set failed, storing in memory: {exc}")

    # In-memory fallback
    _memory_cache[key] = response
    logger.info(f"Cached response (memory) for query: {query[:50]}")


def invalidate_cache(pattern: str = "*", session_id: str = None) -> None:
    """
    Clear cache entries. 
    Currently flushes ALL keys from Upstash Redis (global) and clears in-memory fallback.
    """
    global _memory_cache
    _memory_cache = {}
    try:
        redis = _get_redis()
        if redis is None:
            logger.info("Cache invalidated — memory cache cleared (Redis not available).")
            return
        redis.flushdb()
        logger.info("Cache invalidated — all Redis keys flushed + memory cleared.")
    except Exception as exc:
        logger.warning(f"Redis flush skipped (permission denied): {exc}")


# ── Retrieval cache ──────────────────────────────────────────────────────────

def get_cached_retrieval(query: str) -> Optional[List[Any]]:
    """Retrieval-level cache — deserialises Document objects on read."""
    key = _make_key(f"retrieval:{query}")
    try:
        redis = _get_redis()
        if redis is not None:
            raw = redis.get(key)
            if raw:
                return _deserialize_docs(json.loads(raw))
    except Exception as exc:
        logger.warning(f"Redis get failed, checking memory: {exc}")
    mem = _memory_cache.get(key)
    if mem and isinstance(mem, list) and mem and isinstance(mem[0], dict):
        return _deserialize_docs(mem)
    return mem


def set_cached_retrieval(query: str, docs: Any) -> None:
    """Store retrieval docs in cache — serialises Document objects before storing.

    BUG 2 FIX: json.dumps(docs) previously crashed when docs contained
    LangChain Document objects. We now convert them to plain dicts first.
    """
    key = _make_key(f"retrieval:{query}")
    serialized = _serialize_docs(docs) if isinstance(docs, list) else docs
    try:
        redis = _get_redis()
        if redis is not None:
            payload = json.dumps(serialized) if not isinstance(serialized, str) else serialized
            redis.set(key, payload, ex=3600)
            return
    except Exception as exc:
        logger.warning(f"Redis set failed, storing in memory: {exc}")
    _memory_cache[key] = serialized
