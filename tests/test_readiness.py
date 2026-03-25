import os
from fastapi.testclient import TestClient
from src.app.fastapi_app import app


def test_health_ok():
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json().get("status") == "healthy"


def test_ready_endpoint_depends_on_faiss_store():
    client = TestClient(app)
    has_store = os.path.isdir("./faiss_store")
    r = client.get("/ready")
    if has_store:
        assert r.status_code == 200
        assert r.json().get("ready") is True
    else:
        assert r.status_code == 503
        assert r.json().get("detail") == "Not ready"
