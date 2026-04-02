import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-large")
    EMBED_MODEL_SMALL = os.getenv("EMBED_MODEL_SMALL", "text-embedding-3-small")
    EMBED_MODEL_LARGE = os.getenv("EMBED_MODEL_LARGE", "text-embedding-3-large")
    EMBED_MODEL_MAP = os.getenv(
        "EMBED_MODEL_MAP", "default:large"
    )  # e.g. "pdf:small,html:small,docx:large,default:large"
    MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
    MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))
    MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "aero_docs_v1")
    MILVUS_PARTITIONED = os.getenv("MILVUS_PARTITIONED", "false").lower() == "true"
    PORT = int(os.getenv("PORT", "8000"))
    ENV = os.getenv("ENV", "local")
    # Retriever backend: "faiss" or "milvus"
    RETRIEVER_BACKEND = os.getenv("RETRIEVER_BACKEND", "faiss").lower()
    # Chunking params
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
    # Retrieval params
    RETRIEVER_K = int(os.getenv("RETRIEVER_K", "5"))
    RETRIEVER_FETCH_K = int(os.getenv("RETRIEVER_FETCH_K", "25"))
    # Auth & rate limiting
    API_KEY = os.getenv("API_KEY")  # if set, required for /ask
    METRICS_PUBLIC = os.getenv("METRICS_PUBLIC", "true").lower() == "true"
    _DEFAULT_RL = "60" if ENV == "local" else "30"
    RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", _DEFAULT_RL))
    REDIS_URL = os.getenv("REDIS_URL")
    # JWT (HMAC) support
    JWT_SECRET = os.getenv("JWT_SECRET")
    JWT_ISSUER = os.getenv("JWT_ISSUER")
    JWT_AUDIENCE = os.getenv("JWT_AUDIENCE")
    # JWT hardening (RS256 via JWKS)
    JWT_ALG = os.getenv("JWT_ALG", "HS256").upper()
    JWT_JWKS_URL = os.getenv("JWT_JWKS_URL")
    JWT_JWKS_CACHE_SECONDS = int(os.getenv("JWT_JWKS_CACHE_SECONDS", "3600"))
    # Pushgateway for ingestion metrics (optional)
    PUSHGATEWAY_URL = os.getenv("PUSHGATEWAY_URL")
    # Mock mode for e2e and CI (bypass LLM/retriever)
    MOCK_MODE = os.getenv("MOCK_MODE", "false").lower() == "true"
    # Response cache
    CACHE_ENABLED = os.getenv("CACHE_ENABLED", "false").lower() == "true"
    CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "300"))
    # OpenTelemetry tracing (optional)
    OTEL_ENABLED = os.getenv("OTEL_ENABLED", "false").lower() == "true"
    OTEL_EXPORTER_OTLP_ENDPOINT = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    OTEL_SERVICE_NAME = os.getenv("OTEL_SERVICE_NAME", "rag-aerospace")
    # Sentry error reporting (optional)
    SENTRY_DSN = os.getenv("SENTRY_DSN")
    # Security headers
    SECURITY_HSTS_ENABLED = os.getenv("SECURITY_HSTS_ENABLED", "true").lower() == "true"
    SECURITY_HSTS_MAX_AGE = int(os.getenv("SECURITY_HSTS_MAX_AGE", "31536000"))
    # Content Security Policy (optional)
    CONTENT_SECURITY_POLICY = os.getenv("CONTENT_SECURITY_POLICY", "")
    # Retry/backoff
    RETRY_MAX_ATTEMPTS = int(os.getenv("RETRY_MAX_ATTEMPTS", "3"))
    RETRY_BASE_DELAY_MS = int(os.getenv("RETRY_BASE_DELAY_MS", "100"))
    # CORS
    CORS_ALLOWED_ORIGINS = [
        s.strip() for s in os.getenv("CORS_ALLOWED_ORIGINS", "*").split(",")
    ]
    CORS_ALLOW_CREDENTIALS = (
        os.getenv("CORS_ALLOW_CREDENTIALS", "false").lower() == "true"
    )
    CORS_ALLOWED_METHODS = [
        s.strip()
        for s in os.getenv("CORS_ALLOWED_METHODS", "GET,POST,OPTIONS").split(",")
    ]
    CORS_ALLOWED_HEADERS = [
        s.strip() for s in os.getenv("CORS_ALLOWED_HEADERS", "*").split(",")
    ]
    # Reranking
    RERANK_ENABLED = os.getenv("RERANK_ENABLED", "false").lower() == "true"
    RERANK_MODEL = os.getenv("RERANK_MODEL", "")
    RERANK_MAX_DOCS = int(os.getenv("RERANK_MAX_DOCS", "10"))
    RERANK_MODEL_A = os.getenv("RERANK_MODEL_A", "")
    RERANK_MODEL_B = os.getenv("RERANK_MODEL_B", "")
    # Hybrid search (vector + term scoring)
    HYBRID_ENABLED = os.getenv("HYBRID_ENABLED", "false").lower() == "true"
    HYBRID_ALPHA = float(os.getenv("HYBRID_ALPHA", "0.7"))
    # Embedding batching controls
    EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "64"))
    # Streaming & compression & limits
    STREAMING_ENABLED = os.getenv("STREAMING_ENABLED", "false").lower() == "true"
    GZIP_ENABLED = os.getenv("GZIP_ENABLED", "true").lower() == "true"
    MAX_REQUEST_BYTES = int(os.getenv("MAX_REQUEST_BYTES", "1048576"))  # 1 MiB
    # Quotas
    QUOTA_ENABLED = os.getenv("QUOTA_ENABLED", "false").lower() == "true"
    QUOTA_DAILY_LIMIT = int(os.getenv("QUOTA_DAILY_LIMIT", "1000"))
    # LLM timeouts & circuit breaker
    LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", "20"))
    CB_FAIL_THRESHOLD = int(os.getenv("CB_FAIL_THRESHOLD", "3"))
    CB_RESET_SECONDS = int(os.getenv("CB_RESET_SECONDS", "60"))
    # Multi-tenancy
    MULTITENANT_ENABLED = os.getenv("MULTITENANT_ENABLED", "false").lower() == "true"
    TENANT_METADATA_FIELD = os.getenv("TENANT_METADATA_FIELD", "tenant")
    # Governance & compliance
    PII_REDACTION_ENABLED = (
        os.getenv("PII_REDACTION_ENABLED", "false").lower() == "true"
    )
    AUDIT_ENABLED = os.getenv("AUDIT_ENABLED", "true").lower() == "true"
    # Cost visibility (approximate)
    COST_ENABLED = os.getenv("COST_ENABLED", "true").lower() == "true"
    COST_PER_1K_PROMPT_TOKENS = float(os.getenv("COST_PER_1K_PROMPT_TOKENS", "0.005"))
    COST_PER_1K_COMPLETION_TOKENS = float(
        os.getenv("COST_PER_1K_COMPLETION_TOKENS", "0.015")
    )
    # Negative caching
    NEGATIVE_CACHE_ENABLED = (
        os.getenv("NEGATIVE_CACHE_ENABLED", "true").lower() == "true"
    )
    NEGATIVE_CACHE_TTL_SECONDS = int(os.getenv("NEGATIVE_CACHE_TTL_SECONDS", "15"))
    # Retention window (days). 0 disables retention sweeps.
    DOC_RETENTION_DAYS = int(os.getenv("DOC_RETENTION_DAYS", "0"))
    # Semantic cache
    SEMANTIC_CACHE_ENABLED = (
        os.getenv("SEMANTIC_CACHE_ENABLED", "false").lower() == "true"
    )
    SEMANTIC_CACHE_TTL_SECONDS = int(os.getenv("SEMANTIC_CACHE_TTL_SECONDS", "300"))
    SEMANTIC_CACHE_SIMHASH_BITS = int(os.getenv("SEMANTIC_CACHE_SIMHASH_BITS", "64"))
    SEMANTIC_CACHE_TTL_TENANT = os.getenv(
        "SEMANTIC_CACHE_TTL_TENANT", ""
    )  # e.g. "tenantA:600,tenantB:120"
    # Query expansion
    QUERY_EXPANSION_ENABLED = (
        os.getenv("QUERY_EXPANSION_ENABLED", "false").lower() == "true"
    )
    # Async ingestion (Redis Streams)
    INGEST_STREAM = os.getenv("INGEST_STREAM", "ingest:chunks")
    INGEST_GROUP = os.getenv("INGEST_GROUP", "workers")
    INGEST_CONCURRENCY = int(os.getenv("INGEST_CONCURRENCY", "1"))
    INGEST_MAX_RETRIES = int(os.getenv("INGEST_MAX_RETRIES", "5"))
    INGEST_DLQ_STREAM = os.getenv("INGEST_DLQ_STREAM", "ingest:dlq")
    INGEST_TENANT = os.getenv("INGEST_TENANT", "")
    # AB routing models
    LLM_MODEL_A = os.getenv("LLM_MODEL_A", "gpt-4o-mini")
    LLM_MODEL_B = os.getenv("LLM_MODEL_B", "gpt-4o-mini")
    # Phase 6: Cost controls
    BUDGET_ENABLED = os.getenv("BUDGET_ENABLED", "false").lower() == "true"
    BUDGET_DEFAULT_DAILY_USD = float(os.getenv("BUDGET_DEFAULT_DAILY_USD", "50.0"))
    BUDGET_WARN_FRACTION = float(os.getenv("BUDGET_WARN_FRACTION", "0.8"))
    LLM_MODEL_CHEAP = os.getenv("LLM_MODEL_CHEAP", "gpt-4o-mini")
    # Phase 7: Explainability
    EXPLAIN_ENABLED = os.getenv("EXPLAIN_ENABLED", "false").lower() == "true"
    EXPLAIN_PREVIEW_CHARS = int(os.getenv("EXPLAIN_PREVIEW_CHARS", "240"))
    # Phase 7: Export answers with citations
    EXPORT_ENABLED = os.getenv("EXPORT_ENABLED", "true").lower() == "true"
    EXPORT_FORMAT = os.getenv("EXPORT_FORMAT", "markdown").lower()
    # Phase 8: Offline eval & canary
    EVAL_GOLDEN_PATH = os.getenv("EVAL_GOLDEN_PATH", "./tests/eval/golden.jsonl")
    EVAL_PUSHGATEWAY_JOB = os.getenv("EVAL_PUSHGATEWAY_JOB", "offline-eval")
    CANARY_RERANK_WINDOW = int(os.getenv("CANARY_RERANK_WINDOW", "200"))
    CANARY_RERANK_MIN_HELPFUL = float(os.getenv("CANARY_RERANK_MIN_HELPFUL", "0.55"))
    # Feedback export
    EXPORT_S3_BUCKET = os.getenv("EXPORT_S3_BUCKET", "")
    EXPORT_S3_PREFIX = os.getenv("EXPORT_S3_PREFIX", "exports/")

    # Phase 9: Online eval & shadow traffic
    ONLINE_EVAL_ENABLED = os.getenv("ONLINE_EVAL_ENABLED", "false").lower() == "true"
    ONLINE_EVAL_SAMPLE_RATE = float(os.getenv("ONLINE_EVAL_SAMPLE_RATE", "0.1"))
    ONLINE_EVAL_DIFF_THRESHOLD = float(os.getenv("ONLINE_EVAL_DIFF_THRESHOLD", "0.15"))
    ONLINE_EVAL_WINDOW = int(os.getenv("ONLINE_EVAL_WINDOW", "200"))

    # Phase 10: Multi-Region DR
    DR_ENABLED = os.getenv("DR_ENABLED", "false").lower() == "true"
    REGION = os.getenv("REGION", "primary")
    # Primary Milvus (existing)
    # Secondary Milvus for DR
    MILVUS_HOST_SECONDARY = os.getenv("MILVUS_HOST_SECONDARY", "")
    MILVUS_PORT_SECONDARY = int(os.getenv("MILVUS_PORT_SECONDARY", "19530"))
    MILVUS_COLLECTION_SECONDARY = os.getenv(
        "MILVUS_COLLECTION_SECONDARY", MILVUS_COLLECTION
    )
    DR_DUAL_WRITE = os.getenv("DR_DUAL_WRITE", "true").lower() == "true"
    DR_READ_PREFERRED = os.getenv("DR_READ_PREFERRED", "primary")  # primary|secondary

    # Phase 12: Advanced Retrieval v2
    HYBRID_V2_ENABLED = os.getenv("HYBRID_V2_ENABLED", "true").lower() == "true"
    HYBRID_V2_DENSE_WEIGHT = float(os.getenv("HYBRID_V2_DENSE_WEIGHT", "0.6"))
    HYBRID_V2_BM25_WEIGHT = float(os.getenv("HYBRID_V2_BM25_WEIGHT", "0.4"))

    # Phase 13: Human-in-the-Loop (HITL)
    HITL_ENABLED = os.getenv("HITL_ENABLED", "true").lower() == "true"
    HITL_CONFIDENCE_THRESHOLD = float(os.getenv("HITL_CONFIDENCE_THRESHOLD", "0.35"))
    HITL_SAMPLE_RATE = float(os.getenv("HITL_SAMPLE_RATE", "0.1"))

    # Phase 18: Model Router & Vendor Neutrality
    ROUTER_ENABLED = os.getenv("ROUTER_ENABLED", "true").lower() == "true"
    ROUTER_DEFAULT_OBJECTIVE = os.getenv(
        "ROUTER_DEFAULT_OBJECTIVE", "balanced"
    )  # cost|latency|quality|balanced
    ROUTER_ALLOWED_PROVIDERS = [
        s.strip() for s in os.getenv("ROUTER_ALLOWED_PROVIDERS", "openai").split(",")
    ]

    # Bedrock (optional)
    BEDROCK_REGION = os.getenv("BEDROCK_REGION", "")
    BEDROCK_CHAT_MODEL = os.getenv("BEDROCK_CHAT_MODEL", "")
    BEDROCK_EMBEDDING_MODEL = os.getenv("BEDROCK_EMBEDDING_MODEL", "")
