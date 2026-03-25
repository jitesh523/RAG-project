import os
import json
import time
from typing import List, Dict, Any
from src.config import Config

try:
    import redis
except Exception:  # pragma: no cover
    redis = None


def _redis_client():
    if not redis:
        return None
    try:
        return redis.Redis.from_url(Config.REDIS_URL, decode_responses=True)
    except Exception:
        return None


def collect_feedback(
    limit: int = 1000, tenant: str | None = None
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    r = _redis_client()
    if r is not None:
        try:
            arr = r.lrange("feedback", -limit, -1) or []
            for s in arr:
                try:
                    obj = json.loads(s)
                    if tenant and obj.get("tenant") != tenant:
                        continue
                    out.append(obj)
                except Exception:
                    continue
            return out
        except Exception:
            pass
    # fallback: no memory fallback available here; return empty
    return out


def to_training_rows(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for it in items:
        rows.append(
            {
                "tenant": it.get("tenant"),
                "helpful": bool(it.get("helpful")),
                "reason": it.get("reason"),
                "ts": it.get("ts", int(time.time())),
                # reserved fields for future joining (query/answer ids)
            }
        )
    return rows


def write_jsonl(rows: List[Dict[str, Any]], path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def maybe_upload_s3(path: str) -> str | None:
    bucket = Config.EXPORT_S3_BUCKET
    prefix = Config.EXPORT_S3_PREFIX.rstrip("/") + "/"
    if not bucket:
        return None
    try:
        import boto3  # optional dependency
    except Exception:
        return None
    key = f"{prefix}{os.path.basename(path)}"
    try:
        s3 = boto3.client("s3")
        s3.upload_file(path, bucket, key)
        return f"s3://{bucket}/{key}"
    except Exception:
        return None


def export_feedback(
    limit: int = 1000,
    tenant: str | None = None,
    out_dir: str = "./exports",
    upload: bool = False,
) -> Dict[str, Any]:
    items = collect_feedback(limit=limit, tenant=tenant)
    rows = to_training_rows(items)
    ts = int(time.time())
    fname = f"feedback_{tenant or 'all'}_{ts}.jsonl"
    path = os.path.join(out_dir, fname)
    write_jsonl(rows, path)
    s3_uri = None
    if upload:
        s3_uri = maybe_upload_s3(path)
    return {"count": len(rows), "path": path, "s3": s3_uri}
