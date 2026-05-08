
import sys
import os
import psutil
import logging
from pathlib import Path

# Ensure backend directory is in sys.path for imports
BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from config.settings import REQUESTS_PER_MINUTE, LOG_PATH

# Initial environment loading is handled by settings.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi.middleware import SlowAPIMiddleware
from slowapi.errors import RateLimitExceeded

from utils.rate_limiter import limiter
from routes.query import router as query_router

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.reranker import warmup_reranker
from infra.vector_db import delete_vectors_older_than, ensure_collection_exists
from core.startup_validator import validate_startup_config, full_health_check
from config.settings import ENVIRONMENT
import time
from apscheduler.schedulers.asyncio import AsyncIOScheduler

scheduler = AsyncIOScheduler()

@scheduler.scheduled_job("interval", hours=1)
def cleanup_old_sessions():
    """Delete vectors older than 2 hours."""
    logger = logging.getLogger(__name__)
    cutoff = time.time() - (2 * 60 * 60)
    try:
        delete_vectors_older_than(cutoff)
    except Exception:
        logger.exception("Periodic cleanup failed")

@scheduler.scheduled_job("interval", hours=2)
async def cleanup_orphans():
    from infra.vector_db import cleanup_orphan_vectors
    await cleanup_orphan_vectors()

app = FastAPI(title="RAG Agent Assistant API", version="1.0.0")

app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)

@app.exception_handler(RateLimitExceeded)
def rate_limit_handler(request, exc):
    return JSONResponse(
        status_code=429,
        content={"message": f"Too many requests. Limit is {REQUESTS_PER_MINUTE}."},
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://research-rag-agent.netlify.app",
        "https://research-assistant-frontend.onrender.com",
        "http://localhost:5173",
        "http://localhost:3000",
    ],
    allow_credentials=True,

    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query_router)

def init_db():
    logger = logging.getLogger(__name__)
    sql = """
    CREATE TABLE IF NOT EXISTS documents (
        id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        file_hash     TEXT NOT NULL,
        filename      TEXT NOT NULL,
        storage_path  TEXT NOT NULL,
        user_id       TEXT NOT NULL,
        vector_ids    TEXT[] DEFAULT '{}',
        vector_count  INTEGER DEFAULT 0,
        status        TEXT DEFAULT 'pending',
        document_type TEXT,
        topic         TEXT,
        created_at    TIMESTAMPTZ DEFAULT NOW()
    );
    CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_hash_user
        ON documents (file_hash, user_id);
    CREATE INDEX IF NOT EXISTS idx_documents_user_id
        ON documents (user_id);
    """
    try:
        from infra.storage import _get_client
        client = _get_client()
        if client:
            client.rpc('exec_sql', {'query': sql}).execute()
            logger.info("Executed init_db SQL migration successfully.")
    except Exception as e:
        logger.warning(f"init_db SQL migration could not be run automatically via RPC: {e}. Please ensure the tables are created manually in your Supabase SQL editor.")

@app.on_event("startup")
async def startup():
    logger = logging.getLogger(__name__)
    validate_startup_config()

    try:
        process = psutil.Process(os.getpid())
        mem = process.memory_info().rss / 1024 / 1024
        logger.info("Memory usage at startup: %.1fMB", mem)
    except Exception as e:
        logger.warning("Memory check missing/failed: %s", e)

    if not scheduler.running:
        scheduler.start()
        logger.info("APScheduler started: Session cleanup job registered.")
    
    ensure_collection_exists()
    logger.info("Qdrant collection and indexes verified on startup")
    
    init_db()

    # Supabase connection log and test
    try:
        from infra.storage import _get_client
        _get_client()
    except Exception as e:
        logger.exception("Supabase startup check failed: %s", e)


@app.get("/")
def health():
    return {"status": "ok"}


# Warmup logic removed from startup to prevent blocking Render port detection.
# Models will now load lazily on the first request.


@app.get("/health")
def healthcheck():
    status, checks = full_health_check()
    return {
        "status": status,
        "environment": ENVIRONMENT,
        "qdrant": checks["qdrant"],
        "llm": checks["llm"],
        "storage": checks["storage"],
        "cache": checks["cache"],
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
