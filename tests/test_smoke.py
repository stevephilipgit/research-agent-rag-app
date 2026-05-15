"""Smoke tests — verify the FastAPI app starts and key endpoints respond."""
from fastapi.testclient import TestClient

# conftest.py adds both ROOT and BACKEND to sys.path
from main import app


def test_health_endpoint_returns_200():
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert "status" in body
    assert "environment" in body


def test_root_endpoint_returns_ok():
    client = TestClient(app)
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
