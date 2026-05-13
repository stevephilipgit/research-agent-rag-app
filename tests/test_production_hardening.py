from pathlib import Path

from fastapi.testclient import TestClient

from backend.core import agent as agent_module
from backend.core import document_loader as loader_module
from backend.main import app


def test_stream_cached_response_path_is_stable(monkeypatch):
    monkeypatch.setattr(agent_module, "ENABLE_SECURITY", False)
    monkeypatch.setattr(agent_module, "ENABLE_CACHE", False)
    monkeypatch.setattr(agent_module, "get_cached_response", lambda q, session_id=None: "cached-answer")

    events = list(agent_module.run_research_agent_stream("test", session_id="s1"))
    assert events[0]["type"] == "token"
    assert events[0]["data"] == "cached-answer"
    assert events[1]["type"] == "done"
    assert events[1]["data"]["answer"] == "cached-answer"


def test_ingestion_fails_when_qdrant_unavailable(tmp_path, monkeypatch):
    file_path = tmp_path / "doc.txt"
    file_path.write_text("machine learning chapter 3 basics", encoding="utf-8")

    monkeypatch.setattr(loader_module, "QDRANT_AVAILABLE", False)
    res = loader_module.ingest_documents([str(file_path)], session_id="test-session")
    assert res["status"] == "error"
    assert "Qdrant is unavailable" in res.get("message", "")


def test_upload_endpoint_wires_through_service(monkeypatch):
    async def fake_upload(files, session_id="default"):
        return {"status": "success", "uploaded_files": ["x.pdf"], "logs": []}

    import routes.query as runtime_query_routes
    from backend.routes import query as backend_query_routes

    monkeypatch.setattr(runtime_query_routes, "upload_documents", fake_upload)
    monkeypatch.setattr(backend_query_routes, "upload_documents", fake_upload)
    client = TestClient(app)
    resp = client.post(
        "/api/upload",
        files=[("files", ("x.pdf", b"%PDF-1.4 fake", "application/pdf"))],
        headers={"X-Session-ID": "sess-1"},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"
