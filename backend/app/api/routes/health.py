"""Root and liveness/health endpoints (mounted at the application root)."""
from fastapi import APIRouter

from app.core.config import ENVIRONMENT
from app.core.startup_validator import full_health_check

router = APIRouter(tags=["api"])


@router.get("/")
def health():
    return {"status": "ok"}


# Warmup logic removed from startup to prevent blocking Render port detection.
# Models will now load lazily on the first request.


@router.get("/health")
def healthcheck():
    status, checks = full_health_check()

    def _ok(name: str) -> bool:
        return checks[name]["status"] in ("healthy", "not_configured")

    return {
        "status": status,
        "environment": ENVIRONMENT,
        "qdrant": _ok("qdrant"),
        "llm": _ok("llm"),
        "storage": _ok("storage"),
        "cache": _ok("cache"),
        "dependencies": {
            name: {"status": state["status"], "detail": state["detail"]}
            for name, state in checks.items()
        },
    }
