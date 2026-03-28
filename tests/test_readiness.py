import os
from fastapi.testclient import TestClient
from src.app.fastapi_app import app


def test_health_ok():
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json().get("status") == "healthy"


def test_ready_endpoint_depends_on_faiss_store():
    from src.config import Config
    client = TestClient(app)
    has_store = os.path.isdir("./faiss_store")
    r = client.get("/ready")
    
    # If in mock mode or not using faiss, it should be 200 regardless of store existence
    if Config.MOCK_MODE or Config.RETRIEVER_BACKEND != "faiss":
        assert r.status_code == 200
    else:
        if has_store:
            assert r.status_code == 200
            assert r.json().get("ready") is True
        else:
            assert r.status_code == 503
            assert r.json().get("detail") == "Not ready"

