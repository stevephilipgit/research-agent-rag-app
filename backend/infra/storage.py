import logging
import os

logger = logging.getLogger(__name__)

BUCKET_NAME = "documents"
_supabase_client = None



def _get_client():
    """Lazily initialise the Supabase client once and reuse it. Logs and tests connection."""
    global _supabase_client
    if _supabase_client is not None:
        return _supabase_client

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    masked_key = key[:4] + "..." + key[-4:] if key and len(key) > 8 else "(missing)"
    logger.info(f"SUPABASE_URL: {url}")
    logger.info(f"SUPABASE_KEY: {masked_key}")

    if not url or not key:
        logger.error("SUPABASE_URL and SUPABASE_KEY must be set in the environment.")
        return None

    try:
        from supabase import create_client
        client = create_client(url, key)
        # Connection test
        try:
            client.storage.list_buckets()
            logger.info(f"Supabase connection test succeeded: {url}")
        except Exception as test_exc:
            logger.error(f"Supabase connection test failed for {url}: {test_exc}")
            return None
        _supabase_client = client
        return _supabase_client
    except Exception as exc:
        logger.error(f"Supabase client initialization failed for {url}: {exc}")
        return None


def upload_file(file_bytes: bytes, file_name: str, session_id: str = "default") -> str:
    """
    Upload file_bytes to Supabase Storage bucket 'documents' under session-scoped path.
    Returns the storage path (e.g. 'uploads/uuid-123/report.pdf').
    Overwrites an existing file with the same name.
    """
    storage_path = f"uploads/{session_id}/{file_name}"
    try:
        client = _get_client()
        if client is None:
            logger.error(f"Supabase upload skipped (client unavailable) for {file_name}")
            return ""
        client.storage.from_(BUCKET_NAME).upload(
            path=storage_path,
            file=file_bytes,
            file_options={"upsert": "true", "content-type": "application/octet-stream"},
        )
        logger.info(f"Uploaded {file_name} to Supabase Storage → {storage_path}")
        return storage_path
    except Exception as exc:
        logger.error(f"Supabase upload failed for {file_name}: {exc}")
        return ""


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
        logger.error(f"Supabase delete failed for {path}: {exc}")


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
        logger.error(f"Failed to get public URL for {path}: {exc}")
        return path
