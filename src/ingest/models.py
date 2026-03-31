import hashlib
from pydantic import BaseModel, Field, validator

class IngestData(BaseModel):
    id: str = Field(default_factory=lambda: "")
    text: str = Field(..., min_length=1)
    source: str = Field(default="")
    page: int = Field(default=-1)
    doc_type: str = Field(default="")
    date: str = Field(default="")

    @validator("id", pre=True, always=True)
    def set_id(cls, v, values):
        if not v:
            text = values.get("text", "")
            return hashlib.sha1(
                (text or "").encode("utf-8"), usedforsecurity=False
            ).hexdigest()
        return v

class VectorSchema:
    """Canonical schema for Milvus indexing, aligned with IngestData."""
    PRIMARY_KEY = "id"
    EMBEDDING = "embedding"
    TEXT = "text"
    SOURCE = "source"
    PAGE = "page"
    
    # Matching lengths from milvus_index.py for now
    MAX_ID_LEN = 64
    MAX_TEXT_LEN = 65535
    MAX_SOURCE_LEN = 1024
    
    @staticmethod
    def get_field_definitions(dim: int = 3072):
        from pymilvus import FieldSchema, DataType
        return [
            FieldSchema(name=VectorSchema.PRIMARY_KEY, dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=VectorSchema.MAX_ID_LEN),
            FieldSchema(name=VectorSchema.EMBEDDING, dtype=DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema(name=VectorSchema.TEXT, dtype=DataType.VARCHAR, max_length=VectorSchema.MAX_TEXT_LEN),
            FieldSchema(name=VectorSchema.SOURCE, dtype=DataType.VARCHAR, max_length=VectorSchema.MAX_SOURCE_LEN),
            FieldSchema(name=VectorSchema.PAGE, dtype=DataType.INT64),
        ]
