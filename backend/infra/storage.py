import logging
import os
from typing import List
from config.settings import SUPABASE_URL, SUPABASE_KEY, SUPABASE_SERVICE_ROLE_KEY, BUCKET_NAME

logger = logging.getLogger(__name__)

_supabase_client = None

def _get_client():
    """Lazily initialise the Supabase client once and reuse it. Logs and tests connection."""
    global _supabase_client
    if _supabase_client is not None:
        return _supabase_client

    url = SUPABASE_URL
    # Prioritize service_role key for backend infrastructure
    key = SUPABASE_SERVICE_ROLE_KEY
    if not key:
        logger.warning("AUDIT: SUPABASE_SERVICE_ROLE_KEY missing. Falling back to SUPABASE_KEY. This is NOT recommended for backend services.")
        key = SUPABASE_KEY
    
    masked_key = key[:4] + "..." + key[-4:] if key and len(key) > 8 else "(missing)"
    
    logger.info(f"AUDIT: Supabase Config | URL: {url} | Key: {masked_key}")

    if not url or not key:
        logger.error("AUDIT: SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set in the environment.")
        return None

    try:
        from supabase import create_client, Client
        client: Client = create_client(
            supabase_url=url,
            supabase_key=key
        )
        # Connection test and bucket audit
        try:
            buckets = client.storage.list_buckets()
            bucket_names = [b.name for b in buckets]
            logger.info(f"AUDIT: Connected to Supabase project. Available buckets: {bucket_names}")
            
            if BUCKET_NAME not in bucket_names:
                logger.error(f"AUDIT: CRITICAL - Bucket '{BUCKET_NAME}' NOT FOUND. Please ensure the bucket exists in Supabase. Auto-creation is disabled for production safety.")
        except Exception as test_exc:
            logger.warning(f"AUDIT: Supabase connection test/bucket check failed: {test_exc}")
            
        _supabase_client = client
        return _supabase_client
    except Exception as exc:
        logger.exception(f"AUDIT: Supabase client initialization failed for {url}")
        return None


def upload_file(file_bytes: bytes, file_name: str, session_id: str = "default") -> str:
    """
    Upload file_bytes to Supabase Storage bucket under a session-scoped path.
    Returns the storage path (e.g. 'uploads/uuid-123/report.pdf') on success.

    FIX 2 + FIX 4: Raises RuntimeError on every failure path so callers
    can never silently swallow a storage error and proceed to write the DB.
    The 'upsert' flag avoids 409 conflicts on retry.
    """
    storage_path = f"uploads/{session_id}/{file_name}"
    logger.info(f"AUDIT: Starting upload | Bucket: {BUCKET_NAME} | Path: {storage_path} | Filename: {file_name}")

    client = _get_client()
    if client is None:
        raise RuntimeError(f"Supabase client unavailable — cannot upload '{file_name}'")

    try:
        logger.debug(f"AUDIT: Calling supabase.storage.upload(path={storage_path}, upsert=true)")

        # FIX 4: upsert=true avoids 409 on retry when the path already exists in storage
        response = client.storage.from_(BUCKET_NAME).upload(
            path=storage_path,
            file=file_bytes,
            file_options={"upsert": "true", "content-type": "application/octet-stream"},
        )

        # supabase-py v1 may return a dict with an 'error' key instead of raising
        if isinstance(response, dict) and response.get("error"):
            error_msg = response["error"]
            raise RuntimeError(f"Supabase storage upload failed (v1 error dict): {error_msg}")

        # supabase-py v2 objects may expose .error on some code paths
        if hasattr(response, "error") and response.error:
            raise RuntimeError(f"Supabase storage upload failed: {response.error.message}")

        # Any non-2xx HTTP status is a hard failure
        status_code = getattr(response, "status_code", None)
        if status_code is not None and not (200 <= status_code < 300):
            error_body = getattr(response, "text", getattr(response, "content", "No content"))
            raise RuntimeError(
                f"Supabase upload failed | HTTP {status_code} | Bucket: {BUCKET_NAME} | Response: {error_body}"
            )

        logger.info(f"AUDIT: Upload SUCCESS | Path: {storage_path} | Response type: {type(response).__name__}")
        return storage_path

    except RuntimeError:
        raise  # already formatted — re-raise without wrapping
    except Exception as exc:
        # Log full traceback for unexpected exceptions, then re-raise as RuntimeError
        logger.exception(
            f"AUDIT: FULL EXCEPTION TRACE - Supabase upload failed for file '{file_name}' "
            f"to path '{storage_path}' in bucket '{BUCKET_NAME}'"
        )
        if hasattr(exc, "response"):
            resp = exc.response  # type: ignore[attr-defined]
            status = getattr(resp, "status_code", "N/A")
            body = getattr(resp, "text", getattr(resp, "content", "N/A"))
            logger.error(
                f"AUDIT: Storage operation failed | HTTP Status: {status} | "
                f"Bucket: {BUCKET_NAME} | Operation: upload | Response: {body}"
            )
        if any(kw in str(exc) for kw in ("Unauthorized", "RLS", "row-level security")):
            logger.error(
                "AUDIT: Potential RLS Policy violation detected. "
                "Verify SUPABASE_SERVICE_ROLE_KEY is used and has required privileges."
            )
        raise RuntimeError(f"Supabase upload exception for '{file_name}': {exc}") from exc



def delete_file(path: str) -> None:
    """Delete a file from Supabase Storage by its storage path."""
    try:
        client = _get_client()
        if client is None:
            logger.error(f"Supabase delete skipped (client unavailable) for {path}")
            return
        client.storage.from_(BUCKET_NAME).remove([path])
        logger.info(f"Deleted {path} from Supabase Storage")
    except Exception as exc:
        logger.exception(f"Supabase delete failed for {path}")


def get_file_url(path: str) -> str:
    """Return the public URL for a file stored in Supabase Storage."""
    try:
        client = _get_client()
        if client is None:
            logger.error(f"Supabase get_file_url skipped (client unavailable) for {path}")
            return path
        url = client.storage.from_(BUCKET_NAME).get_public_url(path)
        return url
    except Exception as exc:
        logger.exception(f"Failed to get public URL for {path}")
        return path


def get_signed_file_url(path: str) -> str:
    """Return a signed URL for a file stored in Supabase Storage."""
    try:
        client = _get_client()
        if client is None:
            logger.error(f"Supabase get_signed_file_url skipped (client unavailable) for {path}")
            raise ValueError("Supabase client unavailable")
            
        try:
            res = client.storage.from_(BUCKET_NAME).create_signed_url(path, 300)
            if hasattr(res, "get") and res.get("signedURL"):
                return res["signedURL"]
            elif hasattr(res, "signedURL"):
                return res.signedURL
            elif hasattr(res, "get") and res.get("signedUrl"):
                return res["signedUrl"]
            elif hasattr(res, "signedUrl"):
                return res.signedUrl
            else:
                return str(res)
        except Exception as inner_exc:
            logger.warning(f"Failed to create signed URL via direct method: {inner_exc}. Fallback to path.")
            raise ValueError(f"Failed to create signed URL: {inner_exc}")
    except Exception as exc:
        logger.exception(f"Failed to create signed URL for {path}")
        raise ValueError(f"Failed to fetch signed URL: {exc}")


def file_exists(path: str) -> bool:
    """Check if a file exists in the Supabase Storage bucket."""
    try:
        client = _get_client()
        if client is None:
            return False
        
        # list() returns a list of objects in the path
        folder = os.path.dirname(path)
        filename = os.path.basename(path)
        
        files = client.storage.from_(BUCKET_NAME).list(folder)
        return any(f['name'] == filename for f in files)
    except Exception:
        logger.debug(f"File existence check failed for {path}")
        return False


def list_files_in_bucket(prefix: str = "") -> List[str]:
    """List all files in the bucket with a given prefix."""
    try:
        client = _get_client()
        if client is None:
            return []
        
        # Note: This is a simplified recursive list. 
        # Supabase list() is shallow, so for deep paths we'd need more logic.
        # But for 'uploads/{session_id}/' it usually works if we know the sessions.
        res = client.storage.from_(BUCKET_NAME).list(prefix)
        return [f['name'] for f in res]
    except Exception:
        logger.exception(f"Failed to list files in bucket with prefix '{prefix}'")
        return []
