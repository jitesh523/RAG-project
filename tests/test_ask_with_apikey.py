from fastapi.testclient import TestClient

from tests.conftest import reload_app_with_env


class FakeDoc:
    def __init__(self, source="doc.pdf", page=1):
        self.metadata = {"source": source, "page": page}


class FakeChain:
    def invoke(self, query):
        return {"result": f"answer to: {query}", "source_documents": [FakeDoc()]}


def test_ask_with_api_key_auth():
    # Reload app with API_KEY set
    appmod = reload_app_with_env(API_KEY="TESTKEY", METRICS_PUBLIC="true")
    client = TestClient(appmod.app)

    # Override chain and readiness post-startup
    appmod.qa_chain = FakeChain()
    appmod.READY = True

    r = client.post("/ask", json={"query": "hello"}, headers={"x-api-key": "TESTKEY"})
    assert r.status_code == 200
    data = r.json()
    assert data["answer"].startswith("answer to:")
