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

    # The readiness probe now returns 503 if not ready (READY=False) or faiss_store missing
    # In MOCK_MODE, READY is set to True during startup, and faiss_store check is skipped
    # Since pytest runs with MOCK_MODE=False by default:
    if Config.MOCK_MODE or Config.RETRIEVER_BACKEND != "faiss":
        assert r.status_code == 200
    else:
        if has_store:
            assert r.status_code == 200
            assert r.json().get("ready") is True
        else:
            assert r.status_code == 503
            assert r.json().get("detail") == "Not ready"
