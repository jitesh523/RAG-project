import pytest
from pydantic import ValidationError
from src.ingest.models import IngestData


def test_ingest_data_id_auto_gen():
    # Test that ID is automatically generated from text if not provided
    data = IngestData(text="hello world", source="test.txt")
    assert data.id != ""
    assert len(data.id) == 40  # sha1 hex digest


def test_ingest_data_validation_fails():
    # Test that empty text fails validation
    with pytest.raises(ValidationError):
        IngestData(text="", source="test.txt")


def test_ingest_data_dict():
    # Test that we can convert to dict for Redis xadd
    data = IngestData(text="aerospace engineering", source="manual.pdf", page=10)
    d = data.dict()
    assert d["text"] == "aerospace engineering"
    assert d["source"] == "manual.pdf"
    assert d["page"] == 10
    assert "id" in d
