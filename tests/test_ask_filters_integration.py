from types import SimpleNamespace
from fastapi.testclient import TestClient

from src.app import fastapi_app as appmod


def test_ask_filters_sources(monkeypatch):
    # Prepare FakeChain returning mixed sources
    class FakeChain:
        def invoke(self, q):
            d1 = SimpleNamespace(
                page_content="a", metadata={"source": "aero_handbook.pdf", "page": 1}
            )
            d2 = SimpleNamespace(
                page_content="b", metadata={"source": "turbine.pdf", "page": 2}
            )
            d3 = SimpleNamespace(
                page_content="c", metadata={"source": "other.pdf", "page": 3}
            )
            return {"result": "ok", "source_documents": [d1, d2, d3]}

    appmod.qa_chain = FakeChain()
    appmod.READY = True

    client = TestClient(appmod.app)

    payload = {
        "query": "test",
        "filters": {"sources": ["aero_handbook.pdf", "turbine.pdf"]},
    }
    r = client.post("/ask", json=payload)
    assert r.status_code == 200, r.text
    data = r.json()
    got_sources = [s["source"] for s in data["sources"]]
    assert "other.pdf" not in got_sources
    assert set(got_sources).issubset({"aero_handbook.pdf", "turbine.pdf"})
