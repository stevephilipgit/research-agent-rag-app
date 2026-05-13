from fastapi.testclient import TestClient

from backend.main import app


def test_health_endpoint_works():
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert "status" in body
    assert "environment" in body

