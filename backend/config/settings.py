import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Limit native threads to prevent OMP/MKL related crashes in high-concurrency environments
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Safe fallback
load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[1]

# ===== API KEYS =====
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# ===== CLOUD SERVICES =====
ENVIRONMENT = os.getenv("ENVIRONMENT", "development").strip().lower()
if ENVIRONMENT not in {"development", "production"}:
    raise RuntimeError("ENVIRONMENT must be either 'development' or 'production'")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
BUCKET_NAME = os.getenv("SUPABASE_BUCKET", "documents")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "local-hash")

if not GROQ_API_KEY:
    if ENVIRONMENT == "production":
        raise RuntimeError(
            "GROQ_API_KEY is required in production. Set it in the Render environment tab."
        )
    logger.warning("Missing GROQ_API_KEY. Set it in .env or Render environment tab.")
if not TAVILY_API_KEY:
    logger.warning("Missing TAVILY_API_KEY. Set it in .env or Render environment tab.")
if ENVIRONMENT == "production" and (not QDRANT_URL or not QDRANT_API_KEY):
    raise RuntimeError(
        "QDRANT_URL and QDRANT_API_KEY are required in production. Set them in the Render environment tab."
    )

# ===== LLM CONFIG =====
DEFAULT_MODEL = "llama-3.1-8b-instant"
GROQ_MODEL = DEFAULT_MODEL

LLM_TIMEOUT = 30
LLM_TEMPERATURE = 0

# ===== EMBEDDING CONFIG =====
EMBEDDING_DIMENSION = 256

# ===== RETRIEVAL =====
TOP_K = 5
MAX_CONTEXT_CHARS = 4000
VECTOR_STORE_PATH = str(ROOT_DIR / "vector_store")
DOCUMENTS_PATH = str(ROOT_DIR / "data" / "uploads")
PROCESSED_PATH = str(ROOT_DIR / "data" / "processed")

# ===== FILE CONFIG =====
MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE = MAX_FILE_SIZE_MB * 1024 * 1024  # 10MB
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".csv", ".docx"}
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100

# ===== RESOURCE LIMITS =====
MAX_DOCS_PER_SESSION = 5

# ===== FEATURES =====
def _env_flag(name: str, default: str = "true") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}

ENABLE_CACHE = _env_flag("ENABLE_CACHE", "true")
ENABLE_RERANKER = _env_flag("ENABLE_RERANKER", "false")  # disabled by default on free tier
ENABLE_SECURITY = _env_flag("ENABLE_SECURITY", "true")
ENABLE_VALIDATION = _env_flag("ENABLE_VALIDATION", "true")
ENABLE_MEMORY = _env_flag("ENABLE_MEMORY", "true")
ENABLE_REWRITE = _env_flag("ENABLE_REWRITE", "true")
ENABLE_HYBRID = _env_flag("ENABLE_HYBRID", "true")
ENABLE_COMPRESSION = _env_flag("ENABLE_COMPRESSION", "true")
ENABLE_RETRY = _env_flag("ENABLE_RETRY", "true")
ENABLE_TOOL_GUARD = _env_flag("ENABLE_TOOL_GUARD", "true")
ENABLE_TOOLS_ADVANCED = _env_flag("ENABLE_TOOLS_ADVANCED", "true")
ENABLE_SELF_HEALING = _env_flag("ENABLE_SELF_HEALING", "false")  # disabled by default - opt-in feature
USE_LLM_EVAL = _env_flag("USE_LLM_EVAL", "false")  # disabled by default - use LLM-based response evaluation (higher cost)

# ===== RATE LIMIT =====
REQUESTS_PER_MINUTE = "10/minute"
STREAM_REQUESTS_PER_MINUTE = "5/minute"
UPLOAD_REQUESTS_PER_MINUTE = "3/minute"

# ===== LOGGING =====
LOG_PATH = str(ROOT_DIR / "logs" / "app.log")

# ===== RUNTIME CHECK =====
if sys.version_info >= (3, 14):
    logger.warning("Python 3.14+ detected. System stability is only guaranteed for Python 3.10-3.12.")
    logger.warning("Consider using a compatible Python version if you experience native runtime crashes.")
