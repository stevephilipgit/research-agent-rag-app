import requests
import os
import uuid
import logging
import hashlib
from config.settings import PROCESSED_PATH, MAX_FILE_SIZE

logger = logging.getLogger(__name__)

def download_file(url: str) -> str:
    """Download a file from a URL to a local temporary location within the processed path."""
    os.makedirs(PROCESSED_PATH, exist_ok=True)
    
    # Try to guess extension from URL or use .pdf as fallback
    ext = os.path.splitext(url.split('?')[0])[1].lower()
    if not ext or ext not in {".pdf", ".txt", ".csv", ".docx"}:
        ext = ".pdf"
        
    temp_name = f"url_{uuid.uuid4().hex[:8]}{ext}"
    temp_path = os.path.join(PROCESSED_PATH, temp_name)
    
    try:
        response = requests.get(url, timeout=10, stream=True)
        response.raise_for_status()
        
        # Check size before downloading to satisfy MAX_FILE_SIZE limit
        content_length = response.headers.get('content-length')
        content_type = response.headers.get('content-type', 'unknown')
        if content_length and int(content_length) > MAX_FILE_SIZE:
             raise ValueError(f"File too large: {content_length} bytes exceeds limit.")
             
        downloaded = 0
        digest = hashlib.sha256()
        with open(temp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    downloaded += len(chunk)
                    if downloaded > MAX_FILE_SIZE:
                        raise ValueError("File too large during download.")
                    f.write(chunk)
                    digest.update(chunk)
                
        if downloaded == 0:
            raise ValueError(f"Downloaded file is empty (0 bytes). Content-Type: {content_type}")
            
        checksum = digest.hexdigest()
        logger.info(f"AUDIT: Cloud Download | Path: {temp_path} | Bytes: {downloaded} | Content-Type: {content_type} | SHA256: {checksum}")
                
        return temp_path
    except Exception as exc:
        logger.error(f"Failed to download URL {url}: {exc}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise exc

def is_url(path: str) -> bool:
    """Simple check if path is a URL."""
    return path.startswith(("http://", "https://"))
