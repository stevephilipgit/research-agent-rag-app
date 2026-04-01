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

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from routes.query import router as query_router

from core.reranker import warmup_reranker

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
