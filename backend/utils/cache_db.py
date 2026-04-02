"""
cache_db.py — Upstash Redis response cache.

Uses the Upstash HTTP client (upstash-redis), NOT redis-py.
Upstash uses a REST API, so it works on Render's free tier without TCP ports.

All calls are wrapped in try/except — if Redis is down the app continues
without caching (graceful degradation, never crashes).
"""
import hashlib
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

_redis_client = None


def _get_redis():
    """Lazily initialise the Upstash Redis client once and reuse it."""
    global _redis_client
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
        from upstash_redis import Redis
        _redis_client = Redis(url=url, token=token)
        logger.info("Upstash Redis client initialised.")
        return _redis_client
    except Exception as exc:
        logger.warning(f"Upstash Redis unavailable, cache disabled: {exc}")
        return None


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


def invalidate_cache() -> None:
    """
    Flush ALL keys from Upstash Redis and clear the in-memory fallback.
    Call this after new documents are ingested so stale answers are cleared.
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


# ── Retrieval cache (kept for backward compatibility with any future callers) ──

def get_cached_retrieval(query: str) -> Optional[object]:
    """Retrieval-level cache — uses same Redis/memory backend."""
    key = _make_key(f"retrieval:{query}")
    try:
        redis = _get_redis()
        if redis is not None:
            return redis.get(key)
    except Exception as exc:
        logger.warning(f"Redis get failed, checking memory: {exc}")
    return _memory_cache.get(key)


def set_cached_retrieval(query: str, docs: object) -> None:
    """Store retrieval docs in cache."""
    key = _make_key(f"retrieval:{query}")
    try:
        redis = _get_redis()
        if redis is not None:
            import json
            redis.set(key, json.dumps(docs) if not isinstance(docs, str) else docs, ex=3600)
            return
    except Exception as exc:
        logger.warning(f"Redis set failed, storing in memory: {exc}")
    _memory_cache[key] = docs
