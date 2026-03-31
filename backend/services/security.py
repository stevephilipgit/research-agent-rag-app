import logging
import re
from typing import List

from config import ENABLE_SECURITY
from core.telemetry import emit_log

logger = logging.getLogger(__name__)

# Basic sanitization
CLEAN_CHARS = re.compile(r"[^\x00-\x7F]+") # remove non-ascii chars if needed

# Prompt Injection Patterns (simplified)
BLOCK_PATTERNS = [
    "ignore previous instructions",
    "reveal system prompt",
    "act as system",
    "forget what you were told",
    "you are now a",
]

def validate_input(query: str):
    return bool(query and len(query.strip()) > 0 and len(query) < 2000)

def detect_injection(query: str):
    q = query.lower()
    return any(p in q for p in BLOCK_PATTERNS)

# File Validation
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024 # 10MB
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".csv", ".docx"}

def sanitize_input(query: str) -> str:
    """Sanitize and validate user input."""
    if not ENABLE_SECURITY or not query:
        return query

    # Basic cleaning
    clean_query = query.strip()
    
    # Check for prompt injection
    if detect_injection(clean_query):
        emit_log("Security Layer", "failure", "Blocked injection pattern detected", "query")
        return "Security alert: Suspicious input pattern detected. Query blocked."

    # Success: sanitize or return clean
    emit_log("Security Layer", "success", f"Input sanitized (len={len(clean_query)})", "query")
    return clean_query

def validate_file(file_name: str, file_size: int) -> bool:
    """Validate file upload."""
    if not ENABLE_SECURITY:
        return True

    ext = "." + file_name.split(".")[-1].lower() if "." in file_name else ""
    
    if ext not in ALLOWED_EXTENSIONS:
        emit_log("Security Layer", "failure", f"Blocked file type: {ext}", "pipeline")
        return False
    
    if file_size > MAX_FILE_SIZE_BYTES:
        emit_log("Security Layer", "failure", f"Blocked oversized file: {file_size} bytes", "pipeline")
        return False
        
    emit_log("Security Layer", "success", f"File validated: {file_name}", "pipeline")
    return True
