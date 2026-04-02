import sys
import os
import logging
from pathlib import Path

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
import time
from apscheduler.schedulers.asyncio import AsyncIOScheduler

scheduler = AsyncIOScheduler()

@scheduler.scheduled_job("interval", hours=1)
def cleanup_old_sessions():
    """Delete vectors older than 2 hours."""
    cutoff = time.time() - (2 * 60 * 60)
    try:
        delete_vectors_older_than(cutoff)
    except Exception as e:
        print(f"Periodic cleanup failed: {e}")

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

@app.on_event("startup")
async def startup():
    if not scheduler.running:
        scheduler.start()
        print("APScheduler started: Session cleanup job registered.")
    
    try:
        ensure_collection_exists()
        print("Qdrant collection and indexes verified on startup")
    except Exception as e:
        print(f"Startup Qdrant check failed: {e}")


@app.get("/")
def health():
    return {"status": "ok"}


# Warmup logic removed from startup to prevent blocking Render port detection.
# Models will now load lazily on the first request.


@app.get("/health")
def healthcheck():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
