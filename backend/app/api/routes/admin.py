"""Maintenance/audit endpoints (currently unauthenticated — see hardening phase)."""
import logging

from fastapi import APIRouter, HTTPException, Query

from app.services.maintenance_service import cleanup_orphan_documents

router = APIRouter(prefix="/api", tags=["api"])
logger = logging.getLogger(__name__)


@router.get("/admin/audit")
async def admin_audit_endpoint(
    dry_run: bool = Query(True, description="When true (default), report only — no deletions. Set false to run live cleanup."),
):
    """
    Run the orphan-document audit.

    - **dry_run=true** (default): Returns orphan counts and full record details
      without modifying any data. Safe to call any time.
    - **dry_run=false**: Executes live cleanup — deletes unrecoverable orphans
      from the registry and marks storage-present / vector-missing records as
      'corrupted'. Use with caution.

    Records created within the last 5 minutes are always skipped regardless of
    mode (grace period protects in-flight ingestions).
    """
    try:
        result = await cleanup_orphan_documents(dry_run=dry_run)
        return {
            "status": "ok",
            "mode": "dry_run" if dry_run else "live",
            **result,
        }
    except Exception:
        logger.exception("Admin audit endpoint error")
        raise HTTPException(status_code=500, detail="Audit job failed. Check server logs.")
