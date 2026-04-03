import os
import time
import logging
import signal
import redis
from typing import Dict
from pydantic import ValidationError

from src.config import Config
from src.index.milvus_index import insert_rows
from src.ingest.models import IngestData
from langchain_openai import OpenAIEmbeddings
from prometheus_client import Counter, Histogram, REGISTRY, pushadd_to_gateway

from opentelemetry import trace as ot_trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
import sentry_sdk

_tracer = ot_trace.get_tracer("rag.worker")
if Config.OTEL_ENABLED and Config.OTEL_EXPORTER_OTLP_ENDPOINT:
    try:
        resource = Resource.create({"service.name": Config.OTEL_SERVICE_NAME or "rag-worker"})
        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(endpoint=Config.OTEL_EXPORTER_OTLP_ENDPOINT)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        ot_trace.set_tracer_provider(provider)
        _tracer = ot_trace.get_tracer("rag.worker")
    except Exception:
        pass

if Config.SENTRY_DSN:
    try:
        sentry_sdk.init(dsn=Config.SENTRY_DSN, traces_sample_rate=0.0)
    except Exception:
        pass

logger = logging.getLogger(__name__)

INGEST_WORKER_PROCESSED = Counter(
    "ingest_worker_processed_total", "Total messages processed", ["tenant", "doc_type"]
)
INGEST_WORKER_RETRIES = Counter(
    "ingest_worker_retries_total", "Total processing retries", ["tenant", "stage"]
)
INGEST_WORKER_FAILED = Counter(
    "ingest_worker_failed_total", "Total failed messages sent to DLQ", ["tenant"]
)
INGEST_WORKER_HANDLE = Histogram(
    "ingest_worker_handle_seconds", "Per-message handling time", ["tenant"]
)
INGEST_EMBED_DURATION = Histogram(
    "ingest_embed_seconds", "Time spent embedding text", ["tenant"]
)
INGEST_INSERT_DURATION = Histogram(
    "ingest_insert_seconds", "Time spent inserting into vector db", ["tenant"]
)


def _redis() -> redis.Redis:
    return redis.Redis.from_url(Config.REDIS_URL, decode_responses=True)


def _ensure_group(r: redis.Redis, stream: str, group: str):
    try:
        r.xgroup_create(stream, group, id="0-0", mkstream=True)
    except redis.ResponseError as e:
        if "BUSYGROUP" in str(e):
            return
        raise


def _parse_fields(fields: Dict[str, str]) -> IngestData:
    return IngestData(**fields)


def _parse_model_map(map_str: str):
    mp = {}
    try:
        for p in [x.strip() for x in (map_str or "").split(",") if x.strip()]:
            k, v = p.split(":", 1)
            mp[k.strip()] = v.strip().lower()
    except Exception as e:
        logger.debug("Failed to parse model map '%s': %s", map_str, e)
    return mp


def _select_embed_model(doc_type: str) -> str:
    mp = _parse_model_map(Config.EMBED_MODEL_MAP)
    key = (doc_type or "").lower() or "default"
    which = mp.get(key) or mp.get("default") or "large"
    return Config.EMBED_MODEL_SMALL if which == "small" else Config.EMBED_MODEL_LARGE


RUNNING = True


def handle_sigterm(sig, frame):
    global RUNNING
    logger.info("Signal received, stopping worker...")
    RUNNING = False


def run_worker():
    global RUNNING
    signal.signal(signal.SIGTERM, handle_sigterm)
    signal.signal(signal.SIGINT, handle_sigterm)
    r = _redis()
    stream = Config.INGEST_STREAM
    group = Config.INGEST_GROUP
    consumer = f"c-{os.getpid()}"
    _ensure_group(r, stream, group)
    embed_clients = {}
    seen_key = "ingest:seen"
    dlq = Config.INGEST_DLQ_STREAM
    tenant = Config.INGEST_TENANT or "__default__"

    logger.info("Worker started (PID: %s), tenant: %s", os.getpid(), tenant)

    def _process_batch(entries):
        rows_to_insert = []
        processed_metadata = []  # (msg_id, data, start_time)

        # 1. Parsing, Validation and Embedding
        for msg_id, fields in entries:
            start = time.time()
            try:
                data = _parse_fields(fields)
                if r.sismember(seen_key, data.id):
                    r.xack(stream, group, msg_id)
                    continue

                model_name = _select_embed_model(data.doc_type)
                if model_name not in embed_clients:
                    embed_clients[model_name] = OpenAIEmbeddings(
                        model=model_name, api_key=Config.OPENAI_API_KEY
                    )

                with _tracer.start_as_current_span("embed_query") as span:
                    span.set_attribute("tenant", tenant)
                    with INGEST_EMBED_DURATION.labels(tenant=tenant).time():
                        emb = embed_clients[model_name].embed_query(data.text)

                rows_to_insert.append(
                    (
                        data.id[:32],
                        emb,
                        data.text[:65000],
                        data.source,
                        data.page,
                    )
                )
                processed_metadata.append((msg_id, data, start))

            except ValidationError as e:
                logger.warning("Validation error on message %s: %s", msg_id, e)
                INGEST_WORKER_FAILED.labels(tenant=tenant).inc()
                r.xack(stream, group, msg_id)
            except Exception as e:
                _handle_single_failure(msg_id, fields, e)

        # 2. Batch Insertion with Fallback
        if rows_to_insert:
            part = (
                Config.INGEST_TENANT
                if Config.MILVUS_PARTITIONED and Config.INGEST_TENANT
                else None
            )
            try:
                with _tracer.start_as_current_span("insert_rows") as span:
                    span.set_attribute("tenant", tenant)
                    span.set_attribute("batch_size", len(rows_to_insert))
                    with INGEST_INSERT_DURATION.labels(tenant=tenant).time():
                        insert_rows(rows_to_insert, partition=part)

                # Success: update Redis metadata and ACK
                for msg_id, data, start in processed_metadata:
                    _finalize_success(msg_id, data, start)
            except Exception as e:
                logger.error(
                    "Batch insertion failed, falling back to individual processing: %s",
                    e,
                )
                # Fallback: try each row individually
                for i, (msg_id, data, start) in enumerate(processed_metadata):
                    try:
                        insert_rows([rows_to_insert[i]], partition=part)
                        _finalize_success(msg_id, data, start)
                    except Exception as ex:
                        _handle_single_failure(msg_id, entries[i][1], ex)

    def _handle_single_failure(msg_id, fields, e):
        attempts = int(fields.get("attempts", "0")) + 1
        logger.warning(
            "Error processing message %s (attempt %d): %s", msg_id, attempts, e
        )
        if attempts >= Config.INGEST_MAX_RETRIES:
            fields["attempts"] = str(attempts)
            r.xadd(dlq, fields)
            r.xack(stream, group, msg_id)
            INGEST_WORKER_FAILED.labels(tenant=tenant).inc()
        else:
            fields["attempts"] = str(attempts)
            r.xadd(stream, fields)  # Re-queue
            r.xack(stream, group, msg_id)
            INGEST_WORKER_RETRIES.labels(tenant=tenant, stage="requeue").inc()

    def _finalize_success(msg_id, data, start):
        tenant_val = Config.INGEST_TENANT or "__default__"
        r.sadd(f"filt:{tenant_val}:sources", str(data.source))
        if data.doc_type:
            r.sadd(f"filt:{tenant_val}:doc_types", str(data.doc_type))
        if data.date:
            d = str(data.date)[:10]
            kmin, kmax = f"filt:{tenant_val}:date_min", f"filt:{tenant_val}:date_max"
            r.set(kmin, d) if not r.get(kmin) or d < r.get(kmin) else None
            r.set(kmax, d) if not r.get(kmax) or d > r.get(kmax) else None

        r.sadd(seen_key, data.id)
        r.xack(stream, group, msg_id)
        INGEST_WORKER_PROCESSED.labels(tenant=tenant, doc_type=data.doc_type).inc()
        INGEST_WORKER_HANDLE.labels(tenant=tenant).observe(time.time() - start)

    try:
        while RUNNING:
            try:
                # 1. Processing Pending Messages (PEL recovery)
                pending = r.xreadgroup(
                    group,
                    consumer,
                    streams={stream: "0"},
                    count=Config.INGEST_CONCURRENCY,
                )
                if pending:
                    for _, entries in pending:
                        _process_batch(entries)
                    continue  # Check for more pending before moving to new ones

                # 2. Processing New Messages
                msgs = r.xreadgroup(
                    group,
                    consumer,
                    streams={stream: ">"},
                    count=Config.INGEST_CONCURRENCY,
                    block=2000,
                )
                if not msgs:
                    continue

                for _, entries in msgs:
                    _process_batch(entries)

                    # 3. Push metrics if Pushgateway is used
                    if Config.PUSHGATEWAY_URL:
                        try:
                            pushadd_to_gateway(
                                Config.PUSHGATEWAY_URL,
                                job="ingest-worker",
                                registry=REGISTRY,
                            )
                        except Exception as ex:
                            logger.debug("Failed to push metrics: %s", ex)

            except Exception as e:
                logger.error("Worker loop encountered an error: %s", e)
                time.sleep(1)
    finally:
        logger.info("Worker PID %s shutting down...", os.getpid())
        # Final cleanup if needed


if __name__ == "__main__":
    run_worker()
