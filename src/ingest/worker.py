import os
import time
import hashlib
import redis
import logging
from typing import Dict

from src.config import Config
from src.index.milvus_index import insert_rows
from langchain_openai import OpenAIEmbeddings
from prometheus_client import Counter, Histogram, REGISTRY, pushadd_to_gateway

logger = logging.getLogger(__name__)

INGEST_WORKER_PROCESSED = Counter(
    "ingest_worker_processed_total", "Total messages processed"
)
INGEST_WORKER_RETRIES = Counter(
    "ingest_worker_retries_total", "Total processing retries"
)
INGEST_WORKER_FAILED = Counter(
    "ingest_worker_failed_total", "Total failed messages sent to DLQ"
)
INGEST_WORKER_HANDLE = Histogram(
    "ingest_worker_handle_seconds", "Per-message handling time"
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


def _parse_fields(fields: Dict[str, str]):
    text = fields.get("text", "")
    source = fields.get("source", "")
    page = int(fields.get("page", "-1"))
    doc_type = fields.get("doc_type", "")
    date = fields.get("date", "")
    # derive id if not present
    mid = (
        fields.get("id")
        or hashlib.sha1((text or "").encode("utf-8"), usedforsecurity=False).hexdigest()
    )
    return {
        "id": mid,
        "text": text,
        "source": source,
        "page": page,
        "doc_type": doc_type,
        "date": date,
    }


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


def run_worker():
    r = _redis()
    stream = Config.INGEST_STREAM
    group = Config.INGEST_GROUP
    consumer = f"c-{os.getpid()}"
    _ensure_group(r, stream, group)
    embed_clients = {}
    seen_key = "ingest:seen"
    dlq = Config.INGEST_DLQ_STREAM
    while True:
        try:
            msgs = r.xreadgroup(
                group,
                consumer,
                streams={stream: ">"},
                count=Config.INGEST_CONCURRENCY,
                block=5000,
            )
            if not msgs:
                continue
            for sname, entries in msgs:
                for msg_id, fields in entries:
                    start = time.time()
                    try:
                        data = _parse_fields(fields)
                        # idempotency: skip if already processed
                        if r.sismember(seen_key, data["id"]):
                            r.xack(stream, group, msg_id)
                            continue
                        # embed with per-doc_type model
                        model_name = _select_embed_model(data.get("doc_type", ""))
                        if model_name not in embed_clients:
                            embed_clients[model_name] = OpenAIEmbeddings(
                                model=model_name, api_key=Config.OPENAI_API_KEY
                            )
                        emb = embed_clients[model_name].embed_query(data["text"])
                        part = (
                            Config.INGEST_TENANT
                            if Config.MILVUS_PARTITIONED and Config.INGEST_TENANT
                            else None
                        )
                        insert_rows(
                            [
                                (
                                    data["id"][:32],
                                    emb,
                                    data["text"][:65000],
                                    data["source"],
                                    data["page"],
                                )
                            ],
                            partition=part,
                        )
                        # update filter metadata per tenant (Redis-first)
                        try:
                            tenant = Config.INGEST_TENANT or "__default__"
                            r.sadd(
                                f"filt:{tenant}:sources", str(data.get("source", ""))
                            )
                            if data.get("doc_type"):
                                r.sadd(
                                    f"filt:{tenant}:doc_types",
                                    str(data.get("doc_type")),
                                )
                            if data.get("date"):
                                d = str(data.get("date"))[:10]
                                kmin = f"filt:{tenant}:date_min"
                                kmax = f"filt:{tenant}:date_max"
                                cur_min = r.get(kmin)
                                cur_max = r.get(kmax)
                                if not cur_min or d < cur_min:
                                    r.set(kmin, d)
                                if not cur_max or d > cur_max:
                                    r.set(kmax, d)
                        except Exception as e:
                            logger.debug(
                                "Failed to update filter metadata in Redis: %s", e
                            )
                        r.sadd(seen_key, data["id"])
                        r.xack(stream, group, msg_id)
                        INGEST_WORKER_PROCESSED.inc()
                    except Exception as e:
                        # retry logic using XCLAIM is more advanced; here we push to DLQ after max retries
                        logger.warning("Error processing message %s: %s", msg_id, e)
                        try:
                            attempts = int(fields.get("attempts", "0")) + 1
                        except (ValueError, TypeError):
                            attempts = 1

                        if attempts >= Config.INGEST_MAX_RETRIES:
                            try:
                                fields["attempts"] = str(attempts)
                                r.xadd(dlq, fields)
                            except Exception as ex:
                                logger.debug("Failed to add message to DLQ: %s", ex)
                            r.xack(stream, group, msg_id)
                            INGEST_WORKER_FAILED.inc()
                        else:
                            fields["attempts"] = str(attempts)
                            # requeue by adding back to stream
                            try:
                                r.xadd(stream, fields)
                                r.xack(stream, group, msg_id)
                                INGEST_WORKER_RETRIES.inc()
                            except Exception as ex:
                                logger.error("Failed to requeue message: %s", ex)
                    finally:
                        INGEST_WORKER_HANDLE.observe(time.time() - start)
                        if Config.PUSHGATEWAY_URL:
                            try:
                                pushadd_to_gateway(
                                    Config.PUSHGATEWAY_URL,
                                    job="ingest-worker",
                                    registry=REGISTRY,
                                )
                            except Exception as ex:
                                logger.debug(
                                    "Failed to push metrics to Pushgateway: %s", ex
                                )
        except Exception as e:
            logger.error("Worker loop encountered an error: %s", e)
            time.sleep(1)


if __name__ == "__main__":
    run_worker()
