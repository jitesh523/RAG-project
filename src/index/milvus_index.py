from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility,
)
from typing import List, Tuple, Optional
from src.config import Config
from prometheus_client import Counter
import time
import logging

logger = logging.getLogger(__name__)

try:
    import redis as _r
except Exception as e:
    logger.debug(
        "Optional redis import failed (this is okay if redis is not used): %s", e
    )
    _r = None

EMBED_DIM = 3072  # OpenAI text-embedding-3-large

DR_DUAL_WRITE_ERRORS_TOTAL = Counter(
    "dr_dual_write_errors_total",
    "Total errors when writing to secondary cluster",
)


_PRIMARY_CONNECTED = False
_SECONDARY_CONNECTED = False


def connect():
    global _PRIMARY_CONNECTED
    if _PRIMARY_CONNECTED:
        return
    try:
        connections.connect(
            alias="default", host=Config.MILVUS_HOST, port=str(Config.MILVUS_PORT)
        )
        _PRIMARY_CONNECTED = True
    except Exception as e:
        logger.warning("Primary Milvus connection failed: %s", e)
        _PRIMARY_CONNECTED = False


def connect_secondary():
    global _SECONDARY_CONNECTED
    if not Config.MILVUS_HOST_SECONDARY:
        return False
    if _SECONDARY_CONNECTED:
        return True
    try:
        connections.connect(
            alias="secondary",
            host=Config.MILVUS_HOST_SECONDARY,
            port=str(Config.MILVUS_PORT_SECONDARY),
        )
        _SECONDARY_CONNECTED = True
        return True
    except Exception as e:
        logger.debug("Secondary Milvus connection failed: %s", e)
        _SECONDARY_CONNECTED = False
        return False


def ensure_collection(name: str = Config.MILVUS_COLLECTION) -> Collection:
    connect()
    if utility.has_collection(name):
        return Collection(name)

    fields = [
        FieldSchema(
            name="id",
            dtype=DataType.VARCHAR,
            is_primary=True,
            auto_id=False,
            max_length=64,
        ),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBED_DIM),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=1024),
        FieldSchema(name="page", dtype=DataType.INT64),
    ]
    schema = CollectionSchema(fields, description="Aerospace chunks")
    col = Collection(name, schema)
    col.create_index(
        "embedding",
        {"index_type": "IVF_FLAT", "metric_type": "IP", "params": {"nlist": 1024}},
    )
    col.load()
    return col


def ensure_collection_secondary(
    name: str = Config.MILVUS_COLLECTION_SECONDARY,
) -> Collection | None:
    if not connect_secondary():
        return None
    if utility.has_collection(name, using="secondary"):
        return Collection(name, using="secondary")
    fields = [
        FieldSchema(
            name="id",
            dtype=DataType.VARCHAR,
            is_primary=True,
            auto_id=False,
            max_length=64,
        ),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBED_DIM),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=1024),
        FieldSchema(name="page", dtype=DataType.INT64),
    ]
    schema = CollectionSchema(fields, description="Aerospace chunks (secondary)")
    col = Collection(name, schema, using="secondary")
    col.create_index(
        "embedding",
        {"index_type": "IVF_FLAT", "metric_type": "IP", "params": {"nlist": 1024}},
    )
    col.load()
    return col


def _ensure_partition(col: Collection, partition: str):
    try:
        if partition and partition not in [p.name for p in col.partitions]:
            col.create_partition(partition)
    except Exception as e:
        logger.debug("Partition %s may already exist or error: %s", partition, e)


def insert_rows(
    rows: List[Tuple[str, list, str, str, int]],
    name: str = Config.MILVUS_COLLECTION,
    partition: str | None = None,
):
    # primary write
    col = ensure_collection(name)
    if partition:
        _ensure_partition(col, partition)
    ids, embeds, texts, sources, pages = zip(*rows)
    col.insert(
        [list(ids), list(embeds), list(texts), list(sources), list(pages)],
        partition_name=partition,
    )
    col.flush()
    # record primary write timestamp
    try:
        if _r is not None and Config.REDIS_URL:
            rc = _r.Redis.from_url(Config.REDIS_URL, decode_responses=True)
            rc.set("dr:last_write_ts:primary", str(int(time.time())))
    except Exception as e:
        logger.debug("Failed to record primary write timestamp: %s", e)
    # optional dual-write to secondary
    if Config.DR_ENABLED and Config.DR_DUAL_WRITE and Config.MILVUS_HOST_SECONDARY:
        try:
            scol = ensure_collection_secondary(Config.MILVUS_COLLECTION_SECONDARY)
            if scol is not None:
                if partition:
                    _ensure_partition(scol, partition)
                scol.insert(
                    [list(ids), list(embeds), list(texts), list(sources), list(pages)],
                    partition_name=partition,
                )
                scol.flush()
                try:
                    if _r is not None and Config.REDIS_URL:
                        rc = _r.Redis.from_url(Config.REDIS_URL, decode_responses=True)
                        rc.set("dr:last_write_ts:secondary", str(int(time.time())))
                except Exception as e:
                    logger.debug("Failed to record secondary write timestamp: %s", e)
        except Exception:
            try:
                DR_DUAL_WRITE_ERRORS_TOTAL.inc()
            except Exception as e:
                logger.debug("Failed to increment DR error counter: %s", e)


def check_milvus_readiness(name: str = Config.MILVUS_COLLECTION) -> dict:
    """Return readiness info for Milvus connection and collection.
    Example: {"connected": True, "has_collection": True, "loaded": True}
    """
    info = {"connected": False, "has_collection": False, "loaded": False}
    try:
        connect()
        info["connected"] = True
        info["has_collection"] = utility.has_collection(name)
        if info["has_collection"]:
            col = Collection(name)
            # Try to load; if already loaded this is no-op
            col.load()
            info["loaded"] = True
    except Exception as e:
        logger.warning("Milvus readiness check failed: %s", e)
    return info


# ---- Re-embedding backfill to canary ----
def reembed_active_to_canary(
    embeddings, provider: str, model: str, limit: int = 10000, batch_size: int = 256
) -> dict:
    """Reads up to 'limit' rows from active collection, re-embeds text, and writes to canary collection.
    Stores provider/model metadata in Redis for the canary index.
    """
    active = get_active_collection_name()
    canary = get_canary_collection_name()
    connect()
    ensure_collection(canary)
    col = Collection(active)
    # load to ensure query works
    try:
        col.load()
    except Exception as e:
        logger.debug("Silent exception (pass): %s", e)
    try:
        # Best-effort: pull up to 'limit' docs
        docs = col.query(
            expr="", output_fields=["id", "text", "source", "page"], limit=limit
        )
    except Exception as e:
        docs = []
        logger.warning("Failed to query active collection: %s", e)
    # Embed and write in batches
    total = 0
    buf_ids, buf_embs, buf_texts, buf_sources, buf_pages = [], [], [], [], []

    def _flush():
        nonlocal total, buf_ids, buf_embs, buf_texts, buf_sources, buf_pages
        if not buf_ids:
            return
        insert_rows(
            list(zip(buf_ids, buf_embs, buf_texts, buf_sources, buf_pages)), name=canary
        )
        total += len(buf_ids)
        buf_ids, buf_embs, buf_texts, buf_sources, buf_pages = [], [], [], [], []

    for d in docs:
        try:
            tid = str(d.get("id"))
            text = d.get("text") or ""
            src = d.get("source") or ""
            page = int(d.get("page") or 0)
            emb = embeddings.embed_documents([text])[0]
            buf_ids.append(tid)
            buf_embs.append(emb)
            buf_texts.append(text)
            buf_sources.append(src)
            buf_pages.append(page)
            if len(buf_ids) >= max(1, int(batch_size)):
                _flush()
        except Exception as e:
            logger.error("Error embedding document %s: %s", tid, e)
            continue
    _flush()
    # record metadata for canary
    r = _rc()
    if r is not None:
        try:
            r.hset(
                "index:canary:embed_meta",
                mapping={"provider": provider, "model": model, "rows": str(total)},
            )
        except Exception as e:
            logger.debug("Failed to record metadata for canary: %s", e)
    return {
        "active": active,
        "canary": canary,
        "rows": total,
        "provider": provider,
        "model": model,
    }


# ---- Phase 17: Canary index helpers ----
def _rc() -> Optional["_r.Redis"]:
    if _r is None or not Config.REDIS_URL:
        return None
    try:
        return _r.Redis.from_url(Config.REDIS_URL, decode_responses=True)
    except Exception as e:
        logger.debug("Failed to connect to Redis for connection helper: %s", e)
        return None


def get_active_collection_name() -> str:
    r = _rc()
    if r is None:
        return Config.MILVUS_COLLECTION
    try:
        v = r.get("index:active")
        return v or Config.MILVUS_COLLECTION
    except Exception as e:
        logger.debug("Failed to get active collection name from Redis: %s", e)
        return Config.MILVUS_COLLECTION


def get_canary_collection_name() -> str:
    r = _rc()
    base = get_active_collection_name()
    if r is None:
        return f"{base}_canary"
    try:
        v = r.get("index:canary")
        return v or f"{base}_canary"
    except Exception as e:
        logger.debug("Failed to get canary collection name from Redis: %s", e)
        return f"{base}_canary"


def begin_canary_build() -> dict:
    """Create canary collection (if missing) and mark build start time.
    Returns status dict with names and timestamps.
    """
    canary = get_canary_collection_name()
    ensure_collection(canary)
    r = _rc()
    now = int(time.time())
    if r is not None:
        try:
            r.set("index:canary_build_started", str(now))
            r.set("index:canary", canary)
        except Exception as e:
            logger.debug("Failed to mark canary build start in Redis: %s", e)
    return {"active": get_active_collection_name(), "canary": canary, "started": now}


def promote_canary() -> dict:
    """Promote canary collection by switching the active pointer.
    Does not delete old active collection.
    """
    r = _rc()
    active = get_active_collection_name()
    canary = get_canary_collection_name()
    now = int(time.time())
    if r is not None:
        try:
            r.set("index:active", canary)
            r.set("index:last_promotion_ts", str(now))
            r.set("index:previous_active", active)
        except Exception as e:
            logger.debug("Failed to promote canary in Redis: %s", e)
    return {"active": canary, "previous_active": active, "promoted_at": now}


def index_status() -> dict:
    r = _rc()
    st = {
        "active": get_active_collection_name(),
        "canary": get_canary_collection_name(),
        "build_started": None,
        "last_promotion_ts": None,
    }
    if r is not None:
        try:
            bs = r.get("index:canary_build_started")
            lp = r.get("index:last_promotion_ts")
            st["build_started"] = int(bs) if bs else None
            st["last_promotion_ts"] = int(lp) if lp else None
        except Exception as e:
            logger.debug("Failed to fetch index status from Redis: %s", e)
    # Include readiness of both
    st["active_ready"] = check_milvus_readiness(st["active"]).get("loaded", False)
    st["canary_ready"] = check_milvus_readiness(st["canary"]).get("loaded", False)
    return st


def check_milvus_readiness_secondary(
    name: str = Config.MILVUS_COLLECTION_SECONDARY,
) -> dict:
    info = {"connected": False, "has_collection": False, "loaded": False}
    if not Config.MILVUS_HOST_SECONDARY:
        return info
    try:
        if not connect_secondary():
            return info
        info["connected"] = True
        info["has_collection"] = utility.has_collection(name, using="secondary")
        if info["has_collection"]:
            col = Collection(name, using="secondary")
            col.load()
            info["loaded"] = True
    except Exception as e:
        logger.debug("Milvus secondary readiness check error: %s", e)
    return info
