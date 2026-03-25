import time
from fastapi.testclient import TestClient
from tests.conftest import reload_app_with_env


def test_inmemory_cache_hit_then_expire(monkeypatch):
    # Enable in-memory cache by ensuring no REDIS_URL and enabling CACHE
    appmod = reload_app_with_env(CACHE_ENABLED="true", CACHE_TTL_SECONDS="1")
    client = TestClient(appmod.app)

    # Mark ready and use a deterministic fake chain
    appmod.READY = True

    class FakeChain:
        def __init__(self):
            self.calls = 0

        def invoke(self, q):
            self.calls += 1
            from types import SimpleNamespace

            doc = SimpleNamespace(metadata={"source": "cached.pdf", "page": 1})
            return {"result": f"answer-{self.calls}", "source_documents": [doc]}

    fc = FakeChain()
    appmod.qa_chain = fc

    r1 = client.post("/ask", json={"query": "cached"})
    assert r1.status_code == 200
    first = r1.json()

    # Second call should be served from cache (same answer), chain not invoked again
    r2 = client.post("/ask", json={"query": "cached"})
    assert r2.status_code == 200
    second = r2.json()
    assert first == second
    assert fc.calls == 1

    # After TTL, cache should expire and chain invoked again
    time.sleep(1.1)
    r3 = client.post("/ask", json={"query": "cached"})
    assert r3.status_code == 200
    third = r3.json()
    assert third != first
    assert fc.calls == 2
