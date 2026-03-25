from fastapi.testclient import TestClient

from src.app import fastapi_app as appmod


class FakeDoc:
    def __init__(self, source="doc.pdf", page=1):
        self.metadata = {"source": source, "page": page}


class FakeChain:
    def invoke(self, query):
        return {"result": f"answer to: {query}", "source_documents": [FakeDoc()]}


def test_ask_endpoint_with_mock_chain():
    # Create client (triggers startup), then override chain/readiness
    client = TestClient(appmod.app)
    appmod.qa_chain = FakeChain()
    appmod.READY = True
    r = client.post("/ask", json={"query": "what is wing loading?"})
    assert r.status_code == 200
    body = r.json()
    assert "answer" in body and body["answer"].startswith("answer to:")
    assert isinstance(body.get("sources"), list)
