"""Degraded dependency coverage.

Unavailable dependencies must NOT crash startup and must NOT be reported as
empty successful retrievals — they surface as a degraded /health body and
HTTP 503 on retrieval-dependent endpoints.

conftest.py adds ROOT and BACKEND to sys.path, so both `from main import app`
and package-style imports work.
"""

import httpx
import pytest
from fastapi.testclient import TestClient
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

import app.core.startup_validator as validator
import app.infrastructure.vector_store.qdrant as vector_db
import app.main as main_module
from app.core.exceptions import VECTOR_UNAVAILABLE_DETAIL, VectorServiceUnavailable
from main import app


@pytest.fixture(autouse=True)
def _reset_dependency_caches(monkeypatch):
    monkeypatch.setattr(vector_db, "_probe_cache", {"status": None, "detail": "", "checked_at": 0.0})
    monkeypatch.setattr(validator, "_health_cache", {})
    yield


class _RaisingClientFactory:
    """QdrantClient stand-in that always raises the given exception."""

    def __init__(self, exc):
        self._exc = exc

    def __call__(self, *args, **kwargs):
        raise self._exc


def _force_cloud_mode(monkeypatch):
    monkeypatch.setattr(vector_db, "_IS_DEVELOPMENT", False)
    monkeypatch.setattr(vector_db, "QDRANT_URL", "https://example.qdrant.io")
    monkeypatch.setattr(vector_db, "QDRANT_API_KEY", "test-key")


# ─────────────────────────────────────────────────────────────
# Probe classification
# ─────────────────────────────────────────────────────────────

def test_probe_classifies_connection_reset(monkeypatch):
    _force_cloud_mode(monkeypatch)
    exc = ResponseHandlingException(ConnectionResetError("[Errno 104] Connection reset by peer"))
    monkeypatch.setattr(vector_db, "QdrantClient", _RaisingClientFactory(exc))
    status, detail = vector_db._run_probe(timeout=0.1)
    assert status == "unavailable"
    assert "reset" in detail.lower()


def test_probe_classifies_authentication_failure(monkeypatch):
    _force_cloud_mode(monkeypatch)
    exc = UnexpectedResponse(401, "Unauthorized", b"{}", httpx.Headers())
    monkeypatch.setattr(vector_db, "QdrantClient", _RaisingClientFactory(exc))
    status, _ = vector_db._run_probe(timeout=0.1)
    assert status == "authentication_failure"


def test_probe_classifies_timeout(monkeypatch):
    _force_cloud_mode(monkeypatch)
    monkeypatch.setattr(vector_db, "QdrantClient", _RaisingClientFactory(httpx.ReadTimeout("timed out")))
    status, _ = vector_db._run_probe(timeout=0.1)
    assert status == "timeout"


def test_probe_reports_misconfiguration_in_production(monkeypatch):
    monkeypatch.setattr(vector_db, "_IS_DEVELOPMENT", False)
    monkeypatch.setattr(vector_db, "QDRANT_URL", "")
    monkeypatch.setattr(vector_db, "QDRANT_API_KEY", "")
    status, _ = vector_db._run_probe(timeout=0.1)
    assert status == "misconfigured"


def test_probe_recovers_without_restart(monkeypatch):
    monkeypatch.setattr(vector_db, "_run_probe", lambda timeout: ("unavailable", "down"))
    assert vector_db.probe_vector_store(force=True)["status"] == "unavailable"
    with pytest.raises(VectorServiceUnavailable):
        vector_db.ensure_vector_service_available()

    # Dependency comes back: cached state until TTL/force refresh, then healthy.
    monkeypatch.setattr(vector_db, "_run_probe", lambda timeout: ("healthy", "up"))
    assert vector_db.probe_vector_store()["status"] == "unavailable"
    assert vector_db.probe_vector_store(force=True)["status"] == "healthy"
    vector_db.ensure_vector_service_available()  # no exception


# ─────────────────────────────────────────────────────────────
# Retrieval must not fabricate an empty, successful result
# ─────────────────────────────────────────────────────────────

def test_search_vectors_raises_instead_of_returning_empty(monkeypatch):
    monkeypatch.setattr(vector_db, "is_qdrant_available", lambda: False)
    with pytest.raises(VectorServiceUnavailable):
        vector_db.search_vectors([0.0] * 4)


# ─────────────────────────────────────────────────────────────
# Startup validation
# ─────────────────────────────────────────────────────────────

def test_startup_does_not_raise_when_dependencies_down(monkeypatch):
    monkeypatch.setattr(validator, "ENVIRONMENT", "development")
    monkeypatch.setattr(validator, "_check_qdrant", lambda: ("unavailable", "connection reset"))
    monkeypatch.setattr(validator, "_check_storage", lambda: ("unavailable", "bucket missing"))
    monkeypatch.setattr(validator, "_check_database", lambda: ("misconfigured", "schema mismatch"))
    monkeypatch.setattr(validator, "_check_cache", lambda: ("not_configured", "no redis"))
    monkeypatch.setattr(validator, "_check_llm", lambda: ("healthy", "key set"))

    report = validator.validate_startup_config()

    assert report["status"] == "degraded"
    assert report["dependencies"]["qdrant"]["status"] == "unavailable"
    assert report["dependencies"]["database"]["status"] == "misconfigured"
    assert report["dependencies"]["cache"]["status"] == "not_configured"


def test_startup_still_raises_on_missing_config(monkeypatch):
    monkeypatch.setattr(validator, "ENVIRONMENT", "production")
    monkeypatch.setattr(validator, "QDRANT_URL", "")
    monkeypatch.setattr(validator, "QDRANT_API_KEY", "")
    with pytest.raises(RuntimeError, match="Missing required configuration"):
        validator.validate_startup_config()


# ─────────────────────────────────────────────────────────────
# Health endpoint
# ─────────────────────────────────────────────────────────────

def _set_all_checks(monkeypatch, status="healthy", detail="ok"):
    for name in ("qdrant", "llm", "storage", "database", "cache"):
        monkeypatch.setattr(validator, f"_check_{name}", lambda s=status, d=detail: (s, d))


def test_health_all_healthy(monkeypatch):
    _set_all_checks(monkeypatch, "healthy")
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "healthy"
    assert body["environment"]
    assert body["qdrant"] is True
    assert body["dependencies"]["qdrant"]["status"] == "healthy"


def test_health_degraded_stays_http_200(monkeypatch):
    _set_all_checks(monkeypatch, "healthy")
    monkeypatch.setattr(validator, "_check_qdrant", lambda: ("unavailable", "connection reset by peer"))
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "degraded"
    assert body["qdrant"] is False
    assert body["dependencies"]["qdrant"]["status"] == "unavailable"
    assert "connection reset" in body["dependencies"]["qdrant"]["detail"]


def test_health_multiple_dependencies_unavailable(monkeypatch):
    monkeypatch.setattr(validator, "_check_qdrant", lambda: ("unavailable", "down"))
    monkeypatch.setattr(validator, "_check_storage", lambda: ("unavailable", "down"))
    monkeypatch.setattr(validator, "_check_llm", lambda: ("healthy", "ok"))
    monkeypatch.setattr(validator, "_check_database", lambda: ("healthy", "ok"))
    monkeypatch.setattr(validator, "_check_cache", lambda: ("not_configured", "no redis"))
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "degraded"


# ─────────────────────────────────────────────────────────────
# Redis is optional
# ─────────────────────────────────────────────────────────────

def test_cache_not_configured_without_redis_url(monkeypatch):
    monkeypatch.setattr(validator, "REDIS_URL", "")
    monkeypatch.setattr(validator, "redis_client", None)
    status, _ = validator._check_cache()
    assert status == "not_configured"
    # not_configured must not degrade the overall status
    assert validator._overall_status({"cache": {"status": "not_configured", "detail": ""}}) == "healthy"


def test_cache_unavailable_when_client_missing(monkeypatch):
    monkeypatch.setattr(validator, "REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setattr(validator, "redis_client", None)
    status, _ = validator._check_cache()
    assert status == "unavailable"


# ─────────────────────────────────────────────────────────────
# API behaviour: 503 instead of fabricated results
# ─────────────────────────────────────────────────────────────

def _deny_vector_store(monkeypatch):
    monkeypatch.setattr(vector_db, "_run_probe", lambda timeout: ("unavailable", "connection reset by peer"))


def test_query_endpoint_returns_503_when_vector_unavailable(monkeypatch):
    _deny_vector_store(monkeypatch)
    client = TestClient(app)
    resp = client.post(
        "/api/query",
        json={"query": "what is in my documents?"},
        headers={"X-Session-ID": "sess-503-query"},
    )
    assert resp.status_code == 503
    body = resp.json()
    assert body == {"detail": VECTOR_UNAVAILABLE_DETAIL}
    assert "answer" not in body


def test_stream_endpoint_returns_503_before_streaming(monkeypatch):
    _deny_vector_store(monkeypatch)
    client = TestClient(app)
    resp = client.post(
        "/api/query/stream",
        json={"query": "what is in my documents?"},
        headers={"X-Session-ID": "sess-503-stream"},
    )
    assert resp.status_code == 503
    assert "event-stream" not in resp.headers.get("content-type", "")
    assert resp.json() == {"detail": VECTOR_UNAVAILABLE_DETAIL}


def test_upload_endpoint_returns_503_when_vector_unavailable(monkeypatch):
    _deny_vector_store(monkeypatch)
    client = TestClient(app)
    resp = client.post(
        "/api/upload",
        files=[("files", ("x.pdf", b"%PDF-1.4 fake", "application/pdf"))],
        headers={"X-Session-ID": "sess-503-upload"},
    )
    assert resp.status_code == 503
    assert resp.json() == {"detail": VECTOR_UNAVAILABLE_DETAIL}


# ─────────────────────────────────────────────────────────────
# Startup integration: app boots with every dependency down
# ─────────────────────────────────────────────────────────────

async def _noop_audit():
    return None


def test_app_starts_and_reports_degraded_when_dependencies_down(monkeypatch):
    _set_all_checks(monkeypatch, "unavailable", "down")
    monkeypatch.setattr(main_module, "full_consistency_audit", _noop_audit)

    def _boom():
        raise RuntimeError("collection check failed")

    monkeypatch.setattr(main_module, "ensure_collection_exists", _boom)

    with TestClient(main_module.app) as client:
        root = client.get("/")
        assert root.status_code == 200
        assert root.json()["status"] == "ok"

        health = client.get("/health")
        assert health.status_code == 200
        body = health.json()
        assert body["status"] == "degraded"
        assert body["qdrant"] is False
