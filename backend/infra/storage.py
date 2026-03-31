import logging
from supabase import create_client, Client
from config.settings import SUPABASE_URL, SUPABASE_KEY

logger = logging.getLogger(__name__)

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

BUCKET_NAME = "documents"

def upload_file(file_bytes: bytes, file_name: str) -> str:
    """Uploads a file to Supabase Storage and returns the storage path."""
    try:
        path = f"uploads/{file_name}"
        # Using upsert=True to allow overwriting if the user re-uploads the same filename
        res = supabase.storage.from_(BUCKET_NAME).upload(
            path=path,
            file=file_bytes,
            file_options={"content-type": "application/octet-stream", "x-upsert": "true"}
        )
        logger.info(f"Successfully uploaded {file_name} to Supabase at {path}")
        return path
    except Exception as exc:
        logger.error(f"Supabase upload failed for {file_name}: {exc}")
        raise exc

def delete_file(path: str):
    """Removes a file from Supabase Storage."""
    try:
        supabase.storage.from_(BUCKET_NAME).remove([path])
        logger.info(f"Successfully deleted {path} from Supabase")
    except Exception as exc:
        logger.error(f"Supabase deletion failed for {path}: {exc}")
        raise exc

def get_file_url(path: str) -> str:
    """Returns the public URL for a file in Supabase Storage."""
    try:
        res = supabase.storage.from_(BUCKET_NAME).get_public_url(path)
        return res
    except Exception as exc:
        logger.error(f"Failed to get public URL for {path}: {exc}")
        return ""
