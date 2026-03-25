from types import SimpleNamespace

from src.app.deps import _milvus_expr_from_filters, _faiss_filter_callable_from_filters


def test_milvus_expr_sources_doc_type_dates():
    filters = SimpleNamespace(
        sources=["a.pdf", "b.pdf"],
        doc_type="handbook",
        date_from="2023-01-01",
        date_to="2024-12-31",
    )
    expr = _milvus_expr_from_filters(filters)
    assert expr is not None
    # order-insensitive checks
    assert "source in [" in expr and '"a.pdf"' in expr and '"b.pdf"' in expr
    assert 'doc_type == "handbook"' in expr
    assert 'date >= "2023-01-01"' in expr
    assert 'date <= "2024-12-31"' in expr


def test_faiss_predicate_accepts_and_rejects():
    filters = SimpleNamespace(
        sources=["ok.pdf"],
        doc_type="spec",
        date_from="2022-01-01",
        date_to="2022-12-31",
    )
    pred = _faiss_filter_callable_from_filters(filters)
    assert callable(pred)

    # Accept: matches all
    md_ok = {"source": "ok.pdf", "doc_type": "spec", "date": "2022-06-01"}
    assert pred(md_ok) is True

    # Reject wrong source
    md_src = {"source": "nope.pdf", "doc_type": "spec", "date": "2022-06-01"}
    assert pred(md_src) is False

    # Reject wrong doc_type
    md_type = {"source": "ok.pdf", "doc_type": "guide", "date": "2022-06-01"}
    assert pred(md_type) is False

    # Reject out of date range
    md_date = {"source": "ok.pdf", "doc_type": "spec", "date": "2023-01-01"}
    assert pred(md_date) is False
