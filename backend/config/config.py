import os
import logging
import sys
from dotenv import load_dotenv
from typing import Final
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

# Load environment variables
load_dotenv()
os.environ.setdefault("USER_AGENT", "research-assistant/1.0")


def _env_flag(name: str, default: str = "true") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}

# Constants
GROQ_API_KEY: Final[str] = os.getenv("GROQ_API_KEY", "")
TAVILY_API_KEY: Final[str] = os.getenv("TAVILY_API_KEY", "")
VECTOR_STORE_PATH: Final[str] = str(ROOT_DIR / "vector_store")
DOCUMENTS_PATH: Final[str] = str(ROOT_DIR / "data" / "uploads")
PROCESSED_PATH: Final[str] = str(ROOT_DIR / "data" / "processed")
LOG_PATH: Final[str] = str(ROOT_DIR / "logs" / "app.log")
GROQ_MODEL: Final[str] = "llama-3.1-8b-instant"
ENABLE_REWRITE: Final[bool] = _env_flag("ENABLE_REWRITE", "true")
ENABLE_HYBRID: Final[bool] = _env_flag("ENABLE_HYBRID", "true")
ENABLE_CACHE: Final[bool] = _env_flag("ENABLE_CACHE", "true")
ENABLE_COMPRESSION: Final[bool] = _env_flag("ENABLE_COMPRESSION", "true")
ENABLE_VALIDATION: Final[bool] = _env_flag("ENABLE_VALIDATION", "true")
ENABLE_MEMORY: Final[bool] = _env_flag("ENABLE_MEMORY", "true")
ENABLE_TOOLS_ADVANCED: Final[bool] = _env_flag("ENABLE_TOOLS_ADVANCED", "true")
ENABLE_SECURITY: Final[bool] = _env_flag("ENABLE_SECURITY", "true")
ENABLE_TOOL_GUARD: Final[bool] = _env_flag("ENABLE_TOOL_GUARD", "true")
ENABLE_RETRY: Final[bool] = _env_flag("ENABLE_RETRY", "true")
PYTHON_VERSION_WARNING: Final[str] = (
    "Python 3.14+ may be unstable with the current LangChain stack. "
    "Prefer Python 3.10-3.13 for local development."
)

# Validate required API keys
if not GROQ_API_KEY:
    raise RuntimeError("Missing required API key: GROQ_API_KEY")
if not TAVILY_API_KEY:
    raise RuntimeError("Missing required API key: TAVILY_API_KEY")

# Logging configuration
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler()
    ]
)

# Keep third-party model download chatter out of normal app logs.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

if sys.version_info >= (3, 14):
    logging.getLogger(__name__).warning(PYTHON_VERSION_WARNING)

# Test command
# python -c "from backend.config import GROQ_API_KEY, GROQ_MODEL; print(GROQ_MODEL, GROQ_API_KEY[:8])"
