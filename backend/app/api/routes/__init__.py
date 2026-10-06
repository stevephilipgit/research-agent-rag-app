"""HTTP route registry.

Each module owns one vertical slice of the API and declares its own
`/api` prefix. `router` aggregates every `/api` route; `health_router`
carries the unprefixed root/liveness routes.
"""
from fastapi import APIRouter

from . import admin, documents, health, logs, query, sessions

router = APIRouter()
for _module in (query, documents, sessions, logs, admin):
    router.include_router(_module.router)

health_router = health.router

__all__ = ["router", "health_router"]
