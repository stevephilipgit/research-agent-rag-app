"""Regression: a Qdrant outage must never be interpreted as missing vectors.

Previously `cleanup_orphan_documents(dry_run=False)` asked
`is_doc_id_indexed_in_qdrant()` which returned False on ANY failure — so a
Qdrant outage looked identical to genuinely missing vectors and legitimate
documents were marked `corrupted` (or deleted) during the startup audit.
"""

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

import app.infrastructure.database.registry as db_module
import app.services.maintenance_service as maintenance


def _old_timestamp():
    return datetime.now(timezone.utc) - timedelta(hours=2)


def _registry_rows():
    return [{
        "id": "doc-1",
        "storage_path": "uploads/sess1/report.pdf",
        "file_hash": "abc123",
        "user_id": "sess1",
        "filename": "report.pdf",
        "created_at": _old_timestamp(),
        "status": "indexed",
    }]


class _FakeQuery:
    def __init__(self, rows):
        self._rows = rows

    def select(self, *args, **kwargs):
        return self

    def limit(self, *args, **kwargs):
        return self

    def execute(self):
        result = type("Result", (), {})()
        result.data = self._rows
        return result


class _FakeClient:
    def __init__(self, rows):
        self._rows = rows

    def table(self, name):
        return _FakeQuery(self._rows)


class _RecordingDB:
    def __init__(self):
        self.deleted = []
        self.updated = []

    async def delete(self, table, doc_id):
        self.deleted.append((table, doc_id))

    async def update(self, table, doc_id, data):
        self.updated.append((table, doc_id, data))


@pytest.fixture
def audit_env(monkeypatch):
    rows = _registry_rows()
    rec_db = _RecordingDB()
    deleted_vectors = []
    monkeypatch.setattr(db_module, "_get_client", lambda: _FakeClient(rows))
    monkeypatch.setattr(maintenance, "db", rec_db)
    monkeypatch.setattr(maintenance, "delete_vectors_by_doc_id", lambda doc_id: deleted_vectors.append(doc_id))
    monkeypatch.setattr(maintenance, "file_exists", lambda path: True)
    return {
        "db": rec_db,
        "deleted_vectors": deleted_vectors,
        "rows": rows,
    }


def _set_probe(monkeypatch, status, detail="probe detail"):
    monkeypatch.setattr(
        maintenance,
        "probe_vector_store",
        lambda *args, **kwargs: {"status": status, "detail": detail},
    )


def _run_audit(dry_run=False):
    return asyncio.run(maintenance.cleanup_orphan_documents(dry_run=dry_run))


# ─────────────────────────────────────────────────────────────
# Qdrant outage: non-destructive
# ─────────────────────────────────────────────────────────────

def test_qdrant_outage_never_marks_document_corrupted(audit_env, monkeypatch):
    _set_probe(monkeypatch, "unavailable", "connection reset by peer")
    # The per-doc vector check must not even run when the probe says outage.
    monkeypatch.setattr(
        maintenance,
        "get_doc_vector_state",
        lambda doc_id: pytest.fail("vector check must be skipped while Qdrant is unavailable"),
    )

    summary = _run_audit(dry_run=False)

    assert summary["skipped_qdrant"] == 1
    assert summary["corrupted_detected"] == 0
    assert summary["cleaned_up"] == 0
    assert summary["qdrant_status"] == "unavailable"
    # No destructive database/vector state changes at all:
    assert audit_env["db"].updated == []
    assert audit_env["db"].deleted == []
    assert audit_env["deleted_vectors"] == []
    # Document status untouched.
    assert audit_env["rows"][0]["status"] == "indexed"


def test_qdrant_outage_dry_run_reports_without_mutations(audit_env, monkeypatch):
    _set_probe(monkeypatch, "timeout", "probe timed out")
    monkeypatch.setattr(
        maintenance,
        "get_doc_vector_state",
        lambda doc_id: pytest.fail("vector check must be skipped while Qdrant is unavailable"),
    )

    summary = _run_audit(dry_run=True)

    assert summary["skipped_qdrant"] == 1
    assert summary["qdrant_status"] == "timeout"
    assert audit_env["db"].updated == []
    assert audit_env["db"].deleted == []
    assert audit_env["deleted_vectors"] == []


def test_mid_run_vector_check_outage_is_non_destructive(audit_env, monkeypatch):
    """Probe healthy, but the per-document check itself cannot reach Qdrant."""
    _set_probe(monkeypatch, "healthy", "ok")
    monkeypatch.setattr(maintenance, "get_doc_vector_state", lambda doc_id: "unavailable")

    summary = _run_audit(dry_run=False)

    assert summary["skipped_qdrant"] == 1
    assert summary["corrupted_detected"] == 0
    assert summary["cleaned_up"] == 0
    assert audit_env["db"].updated == []
    assert audit_env["db"].deleted == []
    assert audit_env["deleted_vectors"] == []
    assert audit_env["rows"][0]["status"] == "indexed"


# ─────────────────────────────────────────────────────────────
# Qdrant healthy: existing reconciliation behaviour preserved
# ─────────────────────────────────────────────────────────────

def test_healthy_qdrant_still_marks_genuinely_missing_vectors_corrupted(audit_env, monkeypatch):
    _set_probe(monkeypatch, "healthy", "ok")
    monkeypatch.setattr(maintenance, "get_doc_vector_state", lambda doc_id: "missing")

    summary = _run_audit(dry_run=False)

    assert summary["skipped_qdrant"] == 0
    assert summary["corrupted_detected"] == 1
    assert summary["cleaned_up"] == 0
    assert ("documents", "doc-1") in [
        (t, d) for t, d, _ in audit_env["db"].updated
    ]
    assert audit_env["db"].updated[0][2] == {"status": "corrupted"}
    assert audit_env["db"].deleted == []
    assert audit_env["deleted_vectors"] == []


def test_healthy_qdrant_still_deletes_storage_orphans(audit_env, monkeypatch):
    _set_probe(monkeypatch, "healthy", "ok")
    monkeypatch.setattr(maintenance, "get_doc_vector_state", lambda doc_id: "indexed")
    monkeypatch.setattr(maintenance, "file_exists", lambda path: False)

    summary = _run_audit(dry_run=False)

    assert summary["skipped_qdrant"] == 0
    assert summary["cleaned_up"] == 1
    assert summary["corrupted_detected"] == 0
    assert audit_env["db"].deleted == [("documents", "doc-1")]
    assert audit_env["deleted_vectors"] == ["doc-1"]
    assert audit_env["db"].updated == []
