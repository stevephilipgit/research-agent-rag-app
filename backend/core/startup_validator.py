import logging
import os
import time
from typing import Dict, Tuple

from config.settings import (
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
from infra.vector_db import get_client
from infra.storage import _get_client as get_supabase_client
from infra.db import _get_client as get_db_client
from utils.cache import REDIS_URL, redis_client

logger = logging.getLogger(__name__)


def _is_production() -> bool:
    return (ENVIRONMENT or "development").strip().lower() == "production"


def validate_startup_config() -> None:
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

    # Deep validation
    logger.info("Performing resource validation...")
    
    if not check_qdrant_health():
        logger.error("CRITICAL: Qdrant unreachable.")
        if _is_production(): raise RuntimeError("Qdrant unreachable")

    if not check_storage_health():
        logger.error(f"CRITICAL: Supabase bucket '{BUCKET_NAME}' missing or unreachable.")
        if _is_production(): raise RuntimeError(f"Storage bucket '{BUCKET_NAME}' missing")

    if not check_database_schema():
        logger.error("CRITICAL: Database schema validation failed. Table 'documents' missing.")
        if _is_production(): raise RuntimeError("Database schema invalid")

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
    """Validate that the 'documents' bucket exists and is accessible."""
    try:
        client = get_supabase_client()
        if client is None:
            logger.error("AUDIT: Supabase client could not be initialized.")
            return False
            
        logger.info("AUDIT: Verifying Supabase project reachability and listing buckets...")
        buckets = client.storage.list_buckets()
        bucket_names = [b.name for b in buckets]
        logger.info(f"AUDIT: Connected to Supabase. Available buckets: {bucket_names}")
        
        if BUCKET_NAME not in bucket_names:
            logger.error(f"AUDIT: CRITICAL - Bucket '{BUCKET_NAME}' NOT FOUND in storage. Auto-creation is disabled.")
            return False
            
        logger.info(f"AUDIT: Verifying storage access and upload permissions for bucket '{BUCKET_NAME}'...")
        test_path = ".healthcheck/ping.txt"
        try:
            upload_res = client.storage.from_(BUCKET_NAME).upload(
                path=test_path,
                file=b"ping",
                file_options={"upsert": "true", "content-type": "text/plain"}
            )
            
            status_code = getattr(upload_res, "status_code", None)
            if status_code is not None and not (200 <= status_code < 300):
                 logger.error(f"AUDIT: Failed upload permissions check: HTTP {status_code} - {getattr(upload_res, 'text', getattr(upload_res, 'content', 'No content'))}")
                 return False
                 
            if isinstance(upload_res, dict) and "error" in upload_res:
                 logger.error(f"AUDIT: Failed upload permissions check: {upload_res}")
                 return False

            client.storage.from_(BUCKET_NAME).remove([test_path])
            logger.info("AUDIT: Storage access and upload permissions verified successfully.")
            return True
        except Exception as perm_exc:
            logger.exception("AUDIT: CRITICAL - Upload permissions check failed. The current key may lack storage admin privileges.")
            return False
            
    except Exception:
        logger.exception("Supabase storage health check failed")
        return False


def check_database_schema() -> bool:
    """Validate that the required tables exist in Supabase and match the expected schema."""
    try:
        client = get_db_client()
        if client is None:
            return False
            
        required_columns = [
            "id", "file_hash", "filename", "storage_path", "storage_url", 
            "user_id", "status", "document_type", "topic", "vector_count", 
            "created_at", "schema_version"
        ]
        
        # Try a simple select to see if table exists and has all required columns
        client.table("documents").select(",".join(required_columns)).limit(1).execute()
        return True
    except Exception as e:
        error_str = str(e)
        if "relation \"documents\" does not exist" in error_str:
            logger.error("AUDIT: CRITICAL - Table 'documents' NOT FOUND in database.")
        elif "Could not find the" in error_str and "column" in error_str:
            logger.error(f"AUDIT: CRITICAL - Schema mismatch. {error_str}. Hint: Run the schema migration script.")
        else:
            logger.exception("Supabase database schema validation failed")
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
