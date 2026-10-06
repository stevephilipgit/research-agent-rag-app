import sys
import os
import asyncio
import psutil
import logging
from pathlib import Path

from app.core.config import REQUESTS_PER_MINUTE

# Initial environment loading is handled by app.core.config
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi.middleware import SlowAPIMiddleware
from slowapi.errors import RateLimitExceeded

from app.core.rate_limiter import limiter
from app.api.routes import health_router, router as api_router

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.core.exceptions import VECTOR_UNAVAILABLE_DETAIL, VectorServiceUnavailable
from app.infrastructure.vector_store.qdrant import (
    delete_vectors_older_than,
    ensure_collection_exists,
)
from app.core.startup_validator import validate_startup_config
from app.services.maintenance_service import full_consistency_audit
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
    from app.infrastructure.vector_store.qdrant import cleanup_orphan_vectors
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


@app.exception_handler(VectorServiceUnavailable)
def vector_service_unavailable_handler(request, exc):
    logging.getLogger(__name__).warning(
        "Vector store unavailable | path=%s | reason=%s", request.url.path, exc
    )
    return JSONResponse(status_code=503, content={"detail": VECTOR_UNAVAILABLE_DETAIL})

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

app.include_router(api_router)
app.include_router(health_router)



@app.on_event("startup")
async def startup():
    logger = logging.getLogger(__name__)
    # Configuration errors raise here; dependency unavailability does not —
    # the app boots degraded and /health reports the failing dependencies.
    report = validate_startup_config()
    app.state.startup_report = report

    # Run consistency audit in background so it doesn't block startup (Task 7 & 8)
    asyncio.create_task(full_consistency_audit())

    try:
        process = psutil.Process(os.getpid())
        mem = process.memory_info().rss / 1024 / 1024
        logger.info("Memory usage at startup: %.1fMB", mem)
    except Exception as e:
        logger.warning("Memory check missing/failed: %s", e)

    if not scheduler.running:
        scheduler.start()
        logger.info("APScheduler started: Session cleanup job registered.")

    try:
        ensure_collection_exists()
        logger.info("Qdrant collection and indexes verified on startup")
    except Exception:
        logger.exception("Collection verification failed; continuing in degraded mode")
