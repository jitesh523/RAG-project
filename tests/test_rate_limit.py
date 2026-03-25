from fastapi.testclient import TestClient
from tests.conftest import reload_app_with_env


def test_rate_limit_exceeded_returns_429(monkeypatch):
    # Set low rate limit to trigger quickly
    appmod = reload_app_with_env(RATE_LIMIT_PER_MIN="2")
    client = TestClient(appmod.app)

    # Mark ready and supply fake chain
    appmod.READY = True

    class FakeChain:
        def invoke(self, q):
            from types import SimpleNamespace

            doc = SimpleNamespace(metadata={"source": "t.pdf", "page": 1})
            return {"result": "ok", "source_documents": [doc]}

    appmod.qa_chain = FakeChain()

    # First two should pass
    r1 = client.post("/ask", json={"query": "q1"})
    assert r1.status_code == 200
    r2 = client.post("/ask", json={"query": "q2"})
    assert r2.status_code == 200

    # Third should be rate-limited
    r3 = client.post("/ask", json={"query": "q3"})
    assert r3.status_code == 429
