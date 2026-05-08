import logging
import os
from typing import Dict, Tuple

from backend.config.settings import (
    EMBEDDING_MODEL,
    ENVIRONMENT,
    GROQ_API_KEY,
    QDRANT_URL,
    QDRANT_API_KEY,
    SUPABASE_URL,
    SUPABASE_KEY,
)
from backend.infra.vector_db import get_client
from backend.infra.storage import _get_client as get_supabase_client
from backend.utils.cache import REDIS_URL, redis_client

logger = logging.getLogger(__name__)


def _is_production() -> bool:
    return (ENVIRONMENT or "development").strip().lower() == "production"


def validate_startup_config() -> None:
    env = (ENVIRONMENT or "").strip().lower()
    logger.info("Startup validation started | environment=%s", env or "(missing)")
    if env not in {"development", "production"}:
        raise RuntimeError("ENVIRONMENT must be either 'development' or 'production'")

    if _is_production():
        required_missing = []
        if not QDRANT_URL:
            required_missing.append("QDRANT_URL")
        if not QDRANT_API_KEY:
            required_missing.append("QDRANT_API_KEY")
        if not GROQ_API_KEY:
            required_missing.append("GROQ_API_KEY")
        if not SUPABASE_URL:
            required_missing.append("SUPABASE_URL")
        if not SUPABASE_KEY:
            required_missing.append("SUPABASE_KEY")
        if not EMBEDDING_MODEL:
            required_missing.append("EMBEDDING_MODEL")

        if required_missing:
            raise RuntimeError(f"Missing required production configuration: {', '.join(required_missing)}")

    logger.info("Startup validation completed | environment=%s | embedding_model=%s", env, EMBEDDING_MODEL)


def check_qdrant_health() -> bool:
    try:
        client = get_client()
        client.get_collections()
        return True
    except Exception:
        logger.exception("Qdrant health check failed")
        return False


def check_llm_health() -> bool:
    """Check GROQ LLM availability by verifying the API key is set."""
    if not GROQ_API_KEY:
        logger.warning("LLM health check failed: GROQ_API_KEY missing")
        return False
    return True


def check_storage_health() -> bool:
    try:
        client = get_supabase_client()
        if client is None:
            return False
        client.storage.list_buckets()
        return True
    except Exception:
        logger.exception("Supabase health check failed")
        return False


def check_cache_health() -> bool:
    if not REDIS_URL:
        return True
    if redis_client is None:
        return False
    try:
        redis_client.ping()
        return True
    except Exception:
        logger.exception("Redis health check failed")
        return False


def full_health_check() -> Tuple[str, Dict[str, bool]]:
    checks = {
        "qdrant": check_qdrant_health(),
        "llm": check_llm_health(),
        "storage": check_storage_health(),
        "cache": check_cache_health(),
    }
    status = "healthy" if all(checks.values()) else "degraded"
    return status, checks
