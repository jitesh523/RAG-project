from pydantic import BaseModel, Field, field_validator
import hashlib

class IngestData(BaseModel):
    id: str = Field(default="")
    text: str = Field(..., min_length=1)
    source: str = Field(default="")
    page: int = Field(default=-1)
    doc_type: str = Field(default="")
    date: str = Field(default="")

    @field_validator("id", mode="before")
    @classmethod
    def set_id(cls, v, info):
        if not v:
            # In V2 'before' mode, info.data might not be fully populated yet
            # However, for 'id' it will be called with the raw value.
            # Actually, id generation based on text Is better handled in model_validator(mode="after")
            return v
        return v
    
    # Actually, let's use a simpler approach for Pydantic V2: 
    # use model_validator(mode="after") to set the ID if it's missing.
    from pydantic import model_validator
    
    @model_validator(mode="after")
    def generate_id_if_missing(self) -> "IngestData":
        if not self.id:
            self.id = hashlib.sha1(
                (self.text or "").encode("utf-8"), usedforsecurity=False
            ).hexdigest()
        return self

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
