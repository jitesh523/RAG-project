from fastapi.testclient import TestClient
from tests.conftest import reload_app_with_env


def test_metrics_forbidden_with_invalid_hs256_token(monkeypatch):
    # Force metrics to require auth
    appmod = reload_app_with_env(
        METRICS_PUBLIC="false",
        JWT_ALG="HS256",
        JWT_SECRET="secret",
        API_KEY="SECRETKEY",
    )
    client = TestClient(appmod.app)

    # Invalid bearer token and wrong api key
    r = client.get(
        "/metrics",
        headers={
            "authorization": "Bearer this.is.not.jwt",
            "x-api-key": "WRONG",
        },
    )
    assert r.status_code == 403


def test_ask_unauthorized_with_invalid_bearer_when_api_key_required(monkeypatch):
    appmod = reload_app_with_env(API_KEY="SECRETKEY")
    client = TestClient(appmod.app)
    # Ensure service is marked ready and has a fake chain
    appmod.READY = True

    class FakeChain:
        def invoke(self, q):
            from types import SimpleNamespace

            doc = SimpleNamespace(metadata={"source": "t.pdf", "page": 1})
            return {"result": "ok", "source_documents": [doc]}

    appmod.qa_chain = FakeChain()

    r = client.post(
        "/ask",
        json={"query": "x"},
        headers={"authorization": "Bearer invalid.invalid.invalid"},
    )
    assert r.status_code == 401
