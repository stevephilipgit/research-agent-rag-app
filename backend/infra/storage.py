import logging
import os

logger = logging.getLogger(__name__)

BUCKET_NAME = "documents"
_supabase_client = None


def _get_client():
    """Lazily initialise the Supabase client once and reuse it."""
    global _supabase_client
    if _supabase_client is not None:
        return _supabase_client

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        raise RuntimeError(
            "SUPABASE_URL and SUPABASE_KEY must be set in the environment."
        )

    from supabase import create_client
    _supabase_client = create_client(url, key)
    return _supabase_client


def upload_file(file_bytes: bytes, file_name: str) -> str:
    """
    Upload file_bytes to Supabase Storage bucket 'documents'.
    Returns the storage path (e.g. 'uploads/report.pdf').
    Overwrites an existing file with the same name.
    """
    storage_path = f"uploads/{file_name}"
    try:
        client = _get_client()
        client.storage.from_(BUCKET_NAME).upload(
            path=storage_path,
            file=file_bytes,
            file_options={"upsert": "true", "content-type": "application/octet-stream"},
        )
        logger.info(f"Uploaded {file_name} to Supabase Storage → {storage_path}")
        return storage_path
    except Exception as exc:
        logger.error(f"Supabase upload failed for {file_name}: {exc}")
        raise


def delete_file(path: str) -> None:
    """Delete a file from Supabase Storage by its storage path."""
    try:
        client = _get_client()
        client.storage.from_(BUCKET_NAME).remove([path])
        logger.info(f"Deleted {path} from Supabase Storage")
    except Exception as exc:
        logger.error(f"Supabase delete failed for {path}: {exc}")
        raise


def get_file_url(path: str) -> str:
    """Return the public URL for a file stored in Supabase Storage."""
    try:
        client = _get_client()
        url = client.storage.from_(BUCKET_NAME).get_public_url(path)
        return url
    except Exception as exc:
        logger.error(f"Failed to get public URL for {path}: {exc}")
        # Fall back to path so callers don't hard-crash
        return path
