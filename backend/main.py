import sys
import os
import logging
from pathlib import Path
from backend.config.settings import REQUESTS_PER_MINUTE, LOG_PATH

# Initial environment loading is handled by settings.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi.middleware import SlowAPIMiddleware
from slowapi.errors import RateLimitExceeded
from backend.utils.rate_limiter import limiter

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.routes.query import router as query_router
from backend.core.reranker import warmup_reranker

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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query_router)


@app.on_event("startup")
def warmup_models() -> None:
    warmup_reranker()


@app.get("/health")
def healthcheck():
    return {"status": "ok"}
