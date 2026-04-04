"""Tests for the ML reranker module."""

from types import SimpleNamespace


from src.retrieval.reranker import rerank_documents, _tf_score


def _make_doc(text: str, source: str = "test.pdf") -> SimpleNamespace:
    return SimpleNamespace(page_content=text, metadata={"source": source})


class TestTFScore:
    def test_empty_query(self):
        assert _tf_score("", "some text") == 0.0

    def test_basic_match(self):
        score = _tf_score("thrust weight", "thrust-to-weight ratio is important")
        assert score > 0.0

    def test_no_match(self):
        assert _tf_score("turbine blade", "aerodynamic lift coefficient") == 0.0


class TestRerankerFallback:
    """When no ML model is loaded, reranker falls back to TF scoring."""

    def test_empty_docs_returns_empty(self):
        result = rerank_documents("any query", [], model_name="", top_k=5)
        assert result == []

    def test_returns_sorted_by_score(self):
        docs = [
            _make_doc("unrelated content about weather patterns"),
            _make_doc("thrust-to-weight ratio thrust thrust analysis"),
            _make_doc("thrust mentioned once here"),
        ]
        # Use empty model_name to force TF fallback
        result = rerank_documents("thrust", docs, model_name="", top_k=3)
        assert len(result) == 3
        scores = [s for s, _ in result]
        assert scores == sorted(scores, reverse=True), "Results must be sorted desc"
        # The doc mentioning 'thrust' most should be first
        assert result[0][1].page_content.count("thrust") >= result[1][
            1
        ].page_content.count("thrust")

    def test_top_k_respected(self):
        docs = [_make_doc(f"doc number {i} about aerospace") for i in range(10)]
        result = rerank_documents("aerospace", docs, model_name="", top_k=3)
        assert len(result) == 3

    def test_top_k_none_returns_all(self):
        docs = [_make_doc(f"doc {i}") for i in range(5)]
        result = rerank_documents("doc", docs, model_name="", top_k=None)
        assert len(result) == 5

    def test_score_tuple_format(self):
        docs = [_make_doc("wing loading aerodynamics")]
        result = rerank_documents("wing", docs, model_name="", top_k=1)
        assert len(result) == 1
        score, doc = result[0]
        assert isinstance(score, float)
        assert hasattr(doc, "page_content")
