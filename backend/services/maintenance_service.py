import logging
import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List
from infra.db import db
from infra.storage import file_exists, delete_file
from infra.vector_db import delete_vectors_by_doc_id, is_doc_id_indexed_in_qdrant

logger = logging.getLogger(__name__)

# Documents created within this window may still be mid-ingestion; skip them.
_GRACE_PERIOD_SECONDS = 300

async def cleanup_orphan_documents(dry_run: bool = False) -> Dict[str, Any]:
    """
    Identify documents in the registry that are missing from storage or vectors.

    FIX 3 changes:
    - Logs the full document record before deletion so failures are debuggable.
    - Skips records created within the last 5 minutes (grace period) — they may
      still be mid-ingestion under the new write-order pipeline.
    - dry_run=True reports counts/reasons without modifying any data (used by
      the /api/admin/audit endpoint).
    """
    mode_label = "DRY-RUN" if dry_run else "LIVE"
    logger.info(f"AUDIT [{mode_label}]: Starting orphan document detection...")

    # Fetch the full registry
    try:
        from infra.db import _get_client
        client = _get_client()
        if not client:
            logger.error("AUDIT: Supabase client unavailable — skipping audit")
            return {"error": "DB unavailable", "orphans_detected": 0, "cleaned_up": 0, "corrupted_detected": 0, "skipped_grace": 0}
        res = client.table("documents").select("*").execute()
        documents = res.data
    except Exception as e:
        logger.error(f"AUDIT: Failed to fetch documents for audit: {e}")
        return {"error": str(e), "orphans_detected": 0, "cleaned_up": 0, "corrupted_detected": 0, "skipped_grace": 0}

    orphans_detected = 0
    corrupted_detected = 0
    cleaned_up = 0
    skipped_grace = 0
    orphan_details: List[Dict[str, Any]] = []
    now_utc = datetime.now(timezone.utc)

    for doc in documents:
        doc_id = doc.get("id")
        storage_path = doc.get("storage_path")
        file_hash = doc.get("file_hash")
        session_id = doc.get("user_id")
        filename = doc.get("filename")
        created_at_raw = doc.get("created_at")
        status = doc.get("status")

        # ── FIX 3: Grace period ──────────────────────────────────────────────
        # Skip very recent records — they may still be mid-ingestion.
        if created_at_raw:
            try:
                if isinstance(created_at_raw, str):
                    created_at = datetime.fromisoformat(created_at_raw.replace("Z", "+00:00"))
                else:
                    created_at = created_at_raw
                age_seconds = (now_utc - created_at).total_seconds()
                if age_seconds < _GRACE_PERIOD_SECONDS:
                    logger.info(
                        f"AUDIT: Skipping recent record {doc_id} ('{filename}') "
                        f"— age {int(age_seconds)}s < grace {_GRACE_PERIOD_SECONDS}s, may still be ingesting"
                    )
                    skipped_grace += 1
                    continue
            except Exception as ts_err:
                logger.warning(f"AUDIT: Could not parse created_at for {doc_id}: {ts_err}")

        # ── Storage + vector consistency checks ──────────────────────────────
        exists_in_storage = file_exists(storage_path)
        exists_in_vectors = is_doc_id_indexed_in_qdrant(doc_id)

        if not exists_in_storage or not exists_in_vectors:
            reasons: List[str] = []
            if not exists_in_storage:
                reasons.append("missing_storage")
            if not exists_in_vectors:
                reasons.append("missing_vectors")

            orphans_detected += 1
            orphan_details.append({
                "id": doc_id,
                "filename": filename,
                "status": status,
                "storage_path": storage_path,
                "created_at": str(created_at_raw),
                "reasons": reasons,
            })

            # ── FIX 3: Verbose pre-delete log ────────────────────────────────
            logger.warning(
                f"AUDIT: Orphan/Corrupted document detected — "
                f"id={doc_id} | filename={filename} | status={status} | "
                f"storage_path={storage_path} | created_at={created_at_raw} | "
                f"Reasons: {reasons}"
            )

            if dry_run:
                logger.info(f"AUDIT [DRY-RUN]: Would process orphan {doc_id} (reasons={reasons}) — no action taken")
                continue

            # ── Live cleanup ─────────────────────────────────────────────────
            if not exists_in_storage:
                # File is gone from storage — unrecoverable; delete registry + vectors
                logger.error(
                    f"AUDIT: Deleting orphan document | id={doc_id} | filename={filename} "
                    f"| reasons={reasons} | created_at={created_at_raw} | storage_path={storage_path}"
                )
                try:
                    await db.delete("documents", doc_id)
                    delete_vectors_by_doc_id(doc_id)
                    cleaned_up += 1
                    logger.info(f"AUDIT: Orphan {doc_id} successfully deleted from registry and vector DB")
                except Exception as e:
                    logger.error(f"AUDIT: Failed to cleanup orphan doc {doc_id}: {e}")
            else:
                # File exists but vectors are missing — mark corrupted for manual re-ingestion
                logger.info(
                    f"AUDIT: Vectors missing but file exists for '{filename}' ({doc_id}). "
                    "Marking as corrupted for re-ingestion."
                )
                try:
                    await db.update("documents", doc_id, {"status": "corrupted"})
                    corrupted_detected += 1
                except Exception as e:
                    logger.error(f"AUDIT: Failed to mark {doc_id} as corrupted: {e}")

    summary = {
        "orphans_detected": orphans_detected,
        "cleaned_up": cleaned_up,
        "corrupted_detected": corrupted_detected,
        "skipped_grace": skipped_grace,
        "dry_run": dry_run,
        "orphan_details": orphan_details if dry_run else [],
    }
    logger.info(
        f"AUDIT [{mode_label}]: Finished. "
        f"Detected={orphans_detected} | Cleaned={cleaned_up} | "
        f"Corrupted={corrupted_detected} | SkippedGrace={skipped_grace}"
    )
    return summary


async def full_consistency_audit():
    """Global consistency validation — runs at startup and on a schedule."""
    logger.info("AUDIT: Running startup consistency validation...")
    await cleanup_orphan_documents(dry_run=False)

    from infra.vector_db import cleanup_orphan_vectors
    await cleanup_orphan_vectors()

    logger.info("AUDIT: Startup consistency validation finished.")


def rebuild_document_registry():
    """(Reserved) Logic to rebuild registry from storage files if needed."""
    # This would involve iterating through storage, calculating hashes, and re-inserting to DB.
    # Implementation depends on how much metadata we can recover.
    logger.warning("AUDIT: rebuild_document_registry not fully implemented — requires manual mapping.")
    pass
