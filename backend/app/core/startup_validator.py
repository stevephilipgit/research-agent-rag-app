import logging
import threading
import time
from typing import Any, Dict, Tuple

from app.core.config import (
    EMBEDDING_MODEL,
    ENVIRONMENT,
    GROQ_API_KEY,
    QDRANT_URL,
    QDRANT_API_KEY,
    SUPABASE_URL,
    SUPABASE_KEY,
    SUPABASE_SERVICE_ROLE_KEY,
    BUCKET_NAME,
)
from app.infrastructure.vector_store.qdrant import probe_vector_store
from app.infrastructure.storage.supabase import _get_client as get_supabase_client
from app.infrastructure.database.registry import _get_client as get_db_client
from app.infrastructure.cache.redis_cache import REDIS_URL, redis_client

logger = logging.getLogger(__name__)

_HEALTH_TTL_SECONDS = 30.0
_DEGRADED_STATUSES = {"unavailable", "misconfigured", "timeout", "authentication_failure"}

_cache_locks: Dict[str, threading.Lock] = {}
_cache_locks_guard = threading.Lock()
_health_cache: Dict[str, Dict[str, Any]] = {}


def _is_production() -> bool:
    return (ENVIRONMENT or "development").strip().lower() == "production"


def _lock_for(name: str) -> threading.Lock:
    with _cache_locks_guard:
        lock = _cache_locks.get(name)
        if lock is None:
            lock = threading.Lock()
            _cache_locks[name] = lock
        return lock


def _cached_check(name: str, fn, ttl: float = _HEALTH_TTL_SECONDS, refresh: bool = False) -> Dict[str, str]:
    """Run a dependency check with per-check TTL caching (single-flight)."""
    with _lock_for(name):
        now = time.monotonic()
        entry = _health_cache.get(name)
        if not refresh and entry and (now - float(entry["checked_at"])) < ttl:
            return {"status": entry["status"], "detail": entry.get("detail", "")}

        status, detail = fn()
        _health_cache[name] = {"status": status, "detail": detail, "checked_at": time.monotonic()}
        return {"status": status, "detail": detail}


def _check_qdrant() -> Tuple[str, str]:
    result = probe_vector_store()
    return result["status"], result["detail"]


def _check_llm() -> Tuple[str, str]:
    if not GROQ_API_KEY:
        return "unavailable", "GROQ_API_KEY missing"
    return "healthy", "GROQ_API_KEY configured"


def _check_storage() -> Tuple[str, str]:
    """Validate that the 'documents' bucket exists and is writable."""
    try:
        client = get_supabase_client()
        if client is None:
            return "unavailable", "Supabase client could not be initialized"

        buckets = client.storage.list_buckets()
        bucket_names = [b.name for b in buckets]

        if BUCKET_NAME not in bucket_names:
            return "unavailable", f"bucket '{BUCKET_NAME}' not found in storage"

        test_path = ".healthcheck/ping.txt"
        try:
            upload_res = client.storage.from_(BUCKET_NAME).upload(
                path=test_path,
                file=b"ping",
                file_options={"upsert": "true", "content-type": "text/plain"},
            )

            status_code = getattr(upload_res, "status_code", None)
            if status_code is not None and not (200 <= status_code < 300):
                return "unavailable", f"upload permission check failed (HTTP {status_code})"

            if isinstance(upload_res, dict) and "error" in upload_res:
                return "unavailable", f"upload permission check failed: {upload_res}"

            client.storage.from_(BUCKET_NAME).remove([test_path])
            return "healthy", f"bucket '{BUCKET_NAME}' verified (read/write)"
        except Exception as perm_exc:
            logger.exception(
                "AUDIT: Upload permissions check failed. The current key may lack storage admin privileges."
            )
            return "unavailable", f"upload permission check failed: {perm_exc}"

    except Exception as exc:
        logger.exception("Supabase storage health check failed")
        return "unavailable", f"storage check failed: {exc}"


def _check_database() -> Tuple[str, str]:
    """Validate that the required 'documents' table exists with expected columns."""
    try:
        client = get_db_client()
        if client is None:
            return "unavailable", "database client could not be initialized"

        required_columns = [
            "id", "file_hash", "filename", "storage_path", "storage_url",
            "user_id", "status", "document_type", "topic", "vector_count",
            "created_at", "schema_version",
        ]
        client.table("documents").select(",".join(required_columns)).limit(1).execute()
        return "healthy", "table 'documents' schema verified"
    except Exception as e:
        error_str = str(e)
        if "relation \"documents\" does not exist" in error_str:
            logger.error("AUDIT: CRITICAL - Table 'documents' NOT FOUND in database.")
            return "misconfigured", "table 'documents' not found; run schema migrations"
        if "Could not find the" in error_str and "column" in error_str:
            logger.error(f"AUDIT: CRITICAL - Schema mismatch. {error_str}. Hint: Run the schema migration script.")
            return "misconfigured", "schema mismatch on table 'documents'; run schema migrations"
        logger.exception("Supabase database schema validation failed")
        return "unavailable", f"database check failed: {error_str}"


def _check_cache() -> Tuple[str, str]:
    if not REDIS_URL:
        return "not_configured", "REDIS_URL not set; in-memory cache in use"
    if redis_client is None:
        import app.infrastructure.cache.redis_cache as cache_module

        init_error = getattr(cache_module, "_redis_init_error", None)
        return "unavailable", init_error or "redis client not initialized"
    try:
        redis_client.ping()
        return "healthy", "redis reachable"
    except Exception as exc:
        logger.exception("Redis health check failed")
        return "unavailable", f"redis ping failed: {exc}"


def _dependency_report(refresh: bool = False) -> Dict[str, Dict[str, str]]:
    return {
        "qdrant": _cached_check("qdrant", _check_qdrant, refresh=refresh),
        "llm": _cached_check("llm", _check_llm, refresh=refresh),
        "storage": _cached_check("storage", _check_storage, refresh=refresh),
        "database": _cached_check("database", _check_database, refresh=refresh),
        "cache": _cached_check("cache", _check_cache, refresh=refresh),
    }


def _overall_status(report: Dict[str, Dict[str, str]]) -> str:
    for state in report.values():
        if state["status"] in _DEGRADED_STATUSES:
            return "degraded"
    return "healthy"


def validate_startup_config() -> Dict[str, Any]:
    """Validate configuration at startup.

    Configuration errors still raise (they are deterministic and must be fixed).
    Dependency *availability* failures are non-fatal: they are logged, recorded,
    and reported as degraded so the application can boot.
    """
    env = (ENVIRONMENT or "").strip().lower()
    logger.info("Startup validation started | environment=%s", env or "(missing)")
    if env not in {"development", "production"}:
        raise RuntimeError("ENVIRONMENT must be either 'development' or 'production'")

    required_missing = []
    if _is_production():
        if not QDRANT_URL:
            required_missing.append("QDRANT_URL")
        if not QDRANT_API_KEY:
            required_missing.append("QDRANT_API_KEY")
        if not GROQ_API_KEY:
            required_missing.append("GROQ_API_KEY")
        if not SUPABASE_URL:
            required_missing.append("SUPABASE_URL")
        if not SUPABASE_KEY and not SUPABASE_SERVICE_ROLE_KEY:
            required_missing.append("SUPABASE_SERVICE_ROLE_KEY")
        if not EMBEDDING_MODEL:
            required_missing.append("EMBEDDING_MODEL")

    if required_missing:
        raise RuntimeError(f"Missing required configuration: {', '.join(required_missing)}")

    logger.info("Performing resource validation (non-fatal for availability)...")
    report = _dependency_report(refresh=True)
    status = _overall_status(report)

    for name, state in report.items():
        if state["status"] in ("healthy", "not_configured"):
            logger.info("Startup dependency OK | dep=%s | status=%s", name, state["status"])
        else:
            logger.critical(
                "CRITICAL: dependency degraded | dep=%s | status=%s | detail=%s",
                name, state["status"], state["detail"],
            )

    if status == "healthy":
        logger.info(
            "Startup validation completed | environment=%s | embedding_model=%s",
            env, EMBEDDING_MODEL,
        )
    else:
        logger.warning(
            "Starting in DEGRADED mode | status=%s | environment=%s | dependencies=%s",
            status, env,
            {name: state["status"] for name, state in report.items()},
        )

    return {"status": status, "environment": env, "dependencies": report}


def check_qdrant_health() -> bool:
    return _cached_check("qdrant", _check_qdrant)["status"] == "healthy"


def check_llm_health() -> bool:
    """Check GROQ LLM availability by verifying the API key is set."""
    if not GROQ_API_KEY:
        logger.warning("LLM health check failed: GROQ_API_KEY missing")
        return False
    return True


def check_storage_health() -> bool:
    """Validate that the 'documents' bucket exists and is accessible."""
    return _cached_check("storage", _check_storage)["status"] == "healthy"


def check_database_schema() -> bool:
    """Validate that the required tables exist in Supabase and match the expected schema."""
    return _cached_check("database", _check_database)["status"] == "healthy"


def check_cache_health() -> bool:
    return _cached_check("cache", _check_cache)["status"] in ("healthy", "not_configured")


def full_health_check() -> Tuple[str, Dict[str, Dict[str, str]]]:
    """Return overall status and per-dependency state (TTL-cached)."""
    report = _dependency_report()
    return _overall_status(report), report
