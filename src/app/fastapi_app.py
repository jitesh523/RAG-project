from langchain_openai import OpenAIEmbeddings
try:
    from langchain_openai import AzureOpenAIEmbeddings
except Exception:
    AzureOpenAIEmbeddings = None  # type: ignore
try:
    from langchain_aws import BedrockEmbeddings
except Exception:
    BedrockEmbeddings = None  # type: ignore
def _has_scope(request: 'Request', needle: str) -> bool:
    try:
        authz = request.headers.get("authorization", "")
        if not authz.lower().startswith("bearer "):
            return False
        token = authz.split(" ", 1)[1]
        claims = jwt.decode(token, options={"verify_signature": False}, algorithms=["RS256", "HS256"])
        scopes = claims.get("scope") or claims.get("scopes") or claims.get("scp") or []
        if isinstance(scopes, str):
            scopes = scopes.split()
        roles = claims.get("roles") or []
        if isinstance(roles, str):
            roles = [roles]
        return (needle in scopes) or (needle in roles)
    except Exception:
        return False

# ---- Phase 15: Tenant tier helpers ----
def _tenant_tier(tenant: str) -> str:
    t = tenant or "__default__"
    if _redis_usable():
        try:
            v = _redis.get(f"tier:{t}")
            if v:
                return str(v)
        except Exception:
            _record_redis_failure()
    return "pro"

def _tier_limits(tier: str) -> dict:
    defaults = {
        "free": {"rate_per_min": 60, "concurrent_max": 2, "k_min": 4, "k_max": 10, "fetch_min": 10, "fetch_max": 20, "daily_usd_cap": 5.0},
        "pro": {"rate_per_min": 300, "concurrent_max": 8, "k_min": 4, "k_max": 20, "fetch_min": 10, "fetch_max": 50, "daily_usd_cap": 50.0},
        "enterprise": {"rate_per_min": 1200, "concurrent_max": 32, "k_min": 4, "k_max": 32, "fetch_min": 10, "fetch_max": 80, "daily_usd_cap": 500.0},
    }
    base = defaults.get((tier or "pro").lower(), defaults["pro"]).copy()
    if _redis_usable():
        try:
            h = _redis.hgetall(f"tier:limits:{tier}") or {}
            for k, v in h.items():
                if k in ["rate_per_min", "concurrent_max", "k_min", "k_max", "fetch_min", "fetch_max"]:
                    base[k] = int(v)
                elif k == "daily_usd_cap":
                    base[k] = float(v)
        except Exception:
            _record_redis_failure()
    return base

def _rl_key(tenant: str) -> str:
    return f"rate:{tenant}:{int(time.time()//60)}"

def _conc_key(tenant: str) -> str:
    return f"conc:{tenant}:{int(time.time()//300)}"

def _incr_rate(tenant: str, limit_per_min: int) -> bool:
    if not _redis_usable():
        return True
    try:
        k = _rl_key(tenant)
        v = _redis.incr(k)
        if v == 1:
            _redis.expire(k, 90)
        return v <= max(1, int(limit_per_min))
    except Exception:
        _record_redis_failure()

def _output_guardrail_key(tenant: str) -> str:
    t = tenant or "__default__"
    return f"guard_out:{t}"

def _load_output_guardrail_cfg(tenant: str) -> dict:
    """Load per-tenant output guardrail config.

    Fields (all optional, with defaults):
      enable_pii_redaction: bool
      enable_secret_redaction: bool
      max_answer_chars: int | None
    """
    cfg: dict = {
        "enable_pii_redaction": True,
        "enable_secret_redaction": False,
        "max_answer_chars": None,
    }
    if not _redis_usable():
        return cfg
    try:
        raw = _redis.get(_output_guardrail_key(tenant))
        if raw:
            try:
                user_cfg = json.loads(raw)
                if isinstance(user_cfg, dict):
                    cfg.update(user_cfg)
            except Exception:
                pass
    except Exception:
        _record_redis_failure()
    return cfg

def _save_output_guardrail_cfg(tenant: str, cfg: dict) -> None:
    if not _redis_usable():
        return
    try:
        _redis.set(_output_guardrail_key(tenant), json.dumps(cfg or {}))
    except Exception:
        _record_redis_failure()

def _apply_output_guardrails(text: str, cfg: dict) -> tuple[str, list[str]]:
    """Apply output guardrails and return (text, flags).

    Flags are simple strings like ["pii_redacted", "secret_redacted", "truncated"] for observability.
    """
    flags: list[str] = []
    out = text or ""
    try:
        import re
        if cfg.get("enable_pii_redaction"):
            before = out
            # coarse email / phone / id patterns; reuse semantics of _redact_pii
            out = _redact_pii(out)
            if out != before:
                flags.append("pii_redacted")
        if cfg.get("enable_secret_redaction"):
            before = out
            # very rough secret-like patterns (tokens/keys)
            out = re.sub(r"(?i)(api[_-]?key|secret|token)\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{8,}['\"]?", r"\\1: [REDACTED_SECRET]", out)
            out = re.sub(r"(?i)(password)\s*[:=]\s*['\"]?[^\s'\"]{4,}['\"]?", r"\\1: [REDACTED_SECRET]", out)
            if out != before:
                flags.append("secret_redacted")
        max_chars = cfg.get("max_answer_chars")
        if isinstance(max_chars, int) and max_chars > 0 and len(out) > max_chars:
            out = out[:max_chars]
            flags.append("truncated")
    except Exception:
        return text or "", []
    return out, flags

def _acquire_concurrency(tenant: str, max_conc: int) -> bool:
    TENANT_INFLIGHT.labels(tenant=tenant).inc()
    if not _redis_usable():
        return True
    try:
        k = _conc_key(tenant)
        v = _redis.incr(k)
        if v == 1:
            _redis.expire(k, 600)
        ok = v <= max(1, int(max_conc))
        if not ok:
            _redis.decr(k)
        return ok
    except Exception:
        _record_redis_failure()
        return True

def _release_concurrency(tenant: str) -> None:
    try:
        TENANT_INFLIGHT.labels(tenant=tenant).dec()
    except Exception:
        pass
    if not _redis_usable():
        return
    try:
        k = _conc_key(tenant)
        _redis.decr(k)
    except Exception:
        _record_redis_failure()

def _estimate_tokens(texts: list[str]) -> int:
    # Simple heuristic without external calls
    total_chars = sum(len(t or "") for t in texts)
    return max(1, total_chars // 4)

def _tokens_per_dollar() -> float:
    try:
        return float(_cost_cfg().get("tokens_per_dollar", 25000))
    except Exception:
        return 25000.0

def _budget_key(tenant: str) -> str:
    day = time.strftime("%Y-%m-%d", time.gmtime())
    return f"budget:{tenant}:{day}"

def _budget_check_and_record(tenant: str, tokens: int, cap_usd: float) -> str:
    # returns action: "allow"|"downshift"|"reject"
    if not _redis_usable():
        return "allow"
    try:
        tpd = max(1.0, _tokens_per_dollar())
        cost_usd = float(tokens) / tpd
        k = _budget_key(tenant)
        new = _redis.incrbyfloat(k, cost_usd)
        if new == cost_usd:
            _redis.expire(k, 86400)
        if new > float(cap_usd) * 1.2:
            return "reject"
        if new > float(cap_usd):
            return "downshift"
        return "allow"
    except Exception:
        _record_redis_failure()
        return "allow"
import os
import time
import uuid
import logging
import json
import hashlib
from typing import Optional
from datetime import datetime, timedelta

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from starlette.responses import StreamingResponse, PlainTextResponse, JSONResponse
from starlette_exporter import PrometheusMiddleware, handle_metrics
from prometheus_client import Counter, Gauge, Histogram

import jwt
import redis
import requests
import concurrent.futures
import random

from opentelemetry import trace as ot_trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
import sentry_sdk
from sentry_sdk.integrations.asgi import SentryAsgiMiddleware

from src.app.deps import build_chain
from src.config import Config
from src.index.milvus_index import check_milvus_readiness, begin_canary_build, promote_canary, index_status, reembed_active_to_canary
from src.eval.offline_eval import run_offline_eval
from src.eval.feedback_export import export_feedback
from src.eval.online_eval import run_shadow_eval
from src.policy.engine import evaluate_pre as policy_eval_pre, evaluate_post as policy_eval_post, load_policy


def _get_cache_ns() -> int:
    if _redis is not None:
        try:
            v = _redis.get("cache:ns")
            if v:
                return int(v)
            _redis.set("cache:ns", 1)
            return 1
        except Exception:
            pass

# ---- Phase 16: Observability tracing init ----
_tracer = ot_trace.get_tracer("rag.app")
def _init_tracing():
    try:
        if Config.OTEL_ENABLED and Config.OTEL_EXPORTER_OTLP_ENDPOINT:
            provider = TracerProvider(resource=Resource.create({"service.name": Config.OTEL_SERVICE_NAME}))
            exporter = OTLPSpanExporter(endpoint=Config.OTEL_EXPORTER_OTLP_ENDPOINT)
            processor = BatchSpanProcessor(exporter)
            provider.add_span_processor(processor)
            ot_trace.set_tracer_provider(provider)
        global _tracer
        _tracer = ot_trace.get_tracer("rag.app")
    except Exception:
        # best-effort init
        pass

def _trace_event(name: str, attrs: dict | None = None) -> None:
    try:
        span = ot_trace.get_current_span()
        if span:
            span.add_event(name, attributes=attrs or {})
    except Exception:
        pass

def _trace_sampled() -> bool:
    # Allow dynamic sampling via Redis key trace:sampler_rate (0.0..1.0)
    rate = 0.1
    if _redis_usable():
        try:
            v = _redis.get("trace:sampler_rate")
            if v is not None:
                rate = max(0.0, min(1.0, float(v)))
        except Exception:
            _record_redis_failure()
    try:
        return random.random() < rate
    except Exception:
        return False

# ---- Phase 11: Policy admin endpoints ----
@app.post("/admin/policy/set", tags=["Admin"])
def admin_policy_set(request: Request, tenant: str, body: dict):
    require_admin(request)
    if not tenant:
        raise HTTPException(status_code=400, detail="tenant required")
    if _redis_usable():
        try:
            _redis.set(f"policy:{tenant}", json.dumps(body))
        except Exception:
            _record_redis_failure()
            raise HTTPException(status_code=500, detail="redis error")
    return {"ok": True}

@app.get("/admin/policy/get", tags=["Admin"])
def admin_policy_get(request: Request, tenant: str):
    require_admin(request)
    pol = load_policy(_redis if _redis_usable() else None, tenant)
    return {"ok": True, "policy": pol}

@app.post("/admin/policy/test", tags=["Admin"])
def admin_policy_test(request: Request, tenant: str, body: dict):
    require_admin(request)
    pol = load_policy(_redis if _redis_usable() else None, tenant)
    filt, decision = policy_eval_pre(body or {}, pol)
    ans, srcs, post_dec = policy_eval_post(body.get("answer", ""), body.get("sources", []) or [], pol, override=bool(body.get("override", False)))
    return {"ok": True, "pre": {"filters": filt, "decision": decision}, "post": {"answer": ans, "sources": srcs, "decision": post_dec}}

# ---- Phase 29: Guardrails & Safety Nets ----
def _guardrail_key(tenant: str) -> str:
    t = tenant or "__default__"
    return f"guard:{t}"

def _load_guardrail_cfg(tenant: str) -> dict:
    cfg: dict = {"block_patterns": [], "allow_untrusted": True}
    if not _redis_usable():
        return cfg
    try:
        raw = _redis.get(_guardrail_key(tenant))
        if raw:
            try:
                user_cfg = json.loads(raw)
                if isinstance(user_cfg, dict):
                    cfg.update(user_cfg)
            except Exception:
                pass
    except Exception:
        _record_redis_failure()
    return cfg

def _guardrails_check_input(text: str, cfg: dict) -> dict:
    """Simple pattern-based guardrail decision.

    cfg example:
      {
        "block_patterns": ["delete database", "drop table", "ignore previous instructions"],
        "allow_untrusted": true
      }
    Returns {"blocked": bool, "matched": str | None}.
    """
    t = (text or "").lower()
    blocked = False
    matched = None
    try:
        pats = cfg.get("block_patterns") or []
        for p in pats:
            try:
                s = str(p).lower()
            except Exception:
                continue
            if not s:
                continue
            if s in t:
                blocked = True
                matched = s
                break
    except Exception:
        blocked = False
        matched = None
    return {"blocked": bool(blocked), "matched": matched}

def _save_guardrail_cfg(tenant: str, cfg: dict) -> None:
    if not _redis_usable():
        return
    try:
        _redis.set(_guardrail_key(tenant), json.dumps(cfg or {}))
    except Exception:
        _record_redis_failure()

@app.get("/admin/guardrails/config", tags=["Admin"])
def admin_guardrails_get(request: Request, tenant: str):
    require_admin(request)
    cfg = _load_guardrail_cfg(tenant)
    return {"ok": True, "tenant": tenant, "config": cfg}

@app.post("/admin/guardrails/config", tags=["Admin"])
def admin_guardrails_set(request: Request, tenant: str, body: dict):
    require_admin(request)
    cfg = body or {}
    _save_guardrail_cfg(tenant, cfg)
    return admin_guardrails_get(request, tenant)

@app.post("/admin/guardrails/test", tags=["Admin"])
def admin_guardrails_test(request: Request, tenant: str, body: dict):
    require_admin(request)
    cfg = _load_guardrail_cfg(tenant)
    text = (body or {}).get("text") or ""
    decision = _guardrails_check_input(text, cfg)
    return {"ok": True, "tenant": tenant, "decision": decision}

@app.get("/admin/guardrails/output_config", tags=["Admin"])
def admin_guardrails_output_get(request: Request, tenant: str):
    require_admin(request)
    cfg = _load_output_guardrail_cfg(tenant)
    return {"ok": True, "tenant": tenant, "config": cfg}

@app.post("/admin/guardrails/output_config", tags=["Admin"])
def admin_guardrails_output_set(request: Request, tenant: str, body: dict):
    require_admin(request)
    cfg = body or {}
    _save_output_guardrail_cfg(tenant, cfg)
    return admin_guardrails_output_get(request, tenant)

@app.post("/admin/guardrails/output_test", tags=["Admin"])
def admin_guardrails_output_test(request: Request, tenant: str, body: dict):
    require_admin(request)
    cfg = _load_output_guardrail_cfg(tenant)
    text = (body or {}).get("text") or ""
    out, flags = _apply_output_guardrails(text, cfg)
    return {"ok": True, "tenant": tenant, "original": text, "output": out, "flags": flags, "config": cfg}

# ---- Phase 17: Data Lifecycle & SearchOps ----
def _freshness_touch(source: str, ts: int | None = None):
    if not source:
        return
    if _redis_usable():
        try:
            now = ts or int(time.time())
            _redis.hset("sources:last_ts", mapping={source: str(now)})
            _redis.sadd("sources:known", source)
        except Exception:
            _record_redis_failure()

def _freshness_update_metrics():
    if not _redis_usable():
        return
    try:
        now = int(time.time())
        known = _redis.smembers("sources:known") or []
        last = _redis.hgetall("sources:last_ts") or {}
        for s in known:
            try:
                ts = int(last.get(s, "0"))
            except Exception:
                ts = 0
            lag = max(0, now - ts) if ts else 0
            try:
                SOURCE_FRESHNESS_LAG_SECONDS.labels(source=s).set(lag)
            except Exception:
                pass
    except Exception:
        _record_redis_failure()

@app.post("/admin/index/canary/build", tags=["Admin"])
def admin_index_canary_build(request: Request):
    require_admin(request)
    try:
        st = begin_canary_build()
        INDEX_BUILDS_TOTAL.labels(status="started").inc()
        return {"ok": True, "status": st}
    except Exception as e:
        INDEX_BUILDS_TOTAL.labels(status="error").inc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/index/canary/promote", tags=["Admin"])
def admin_index_canary_promote(request: Request, force: bool = False):
    require_admin(request)
    try:
        # If not forced, require last evaluation pass
        if not force and _redis_usable():
            try:
                ev = _redis.get("index:last_canary_eval")
                if ev:
                    data = json.loads(ev)
                    if not bool(data.get("pass", False)):
                        raise HTTPException(status_code=412, detail="Canary gate not passed; use force=true to override")
            except HTTPException:
                raise
            except Exception:
                _record_redis_failure()
        st = promote_canary()
        INDEX_PROMOTIONS_TOTAL.inc()
        return {"ok": True, "status": st}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/admin/index/status", tags=["Admin"])
def admin_index_status(request: Request):
    require_admin(request)
    try:
        st = index_status()
        return {"ok": True, "status": st}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/source/freshness", tags=["Admin"])
def admin_source_freshness(request: Request, source: str, ts: int | None = None):
    require_admin(request)
    _freshness_touch(source, ts)
    _freshness_update_metrics()
    return {"ok": True}

def _index_gate_cfg() -> dict:
    cfg = {"max_latency_increase_pct": 20.0, "min_source_overlap": 0.8, "sample_count": 50}
    if _redis_usable():
        try:
            h = _redis.hgetall("index:gate") or {}
            if "max_latency_increase_pct" in h:
                cfg["max_latency_increase_pct"] = float(h.get("max_latency_increase_pct"))
            if "min_source_overlap" in h:
                cfg["min_source_overlap"] = float(h.get("min_source_overlap"))
            if "sample_count" in h:
                cfg["sample_count"] = int(h.get("sample_count"))
        except Exception:
            _record_redis_failure()
    return cfg

def _load_golden_queries(limit: int) -> list[str]:
    qs = []
    try:
        path = Config.EVAL_GOLDEN_PATH
        with open(path, "r") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    q = (obj.get("query") or obj.get("question") or "").strip()
                    if q:
                        qs.append(q)
                        if len(qs) >= limit:
                            break
                except Exception:
                    continue
    except Exception:
        pass
    return qs

def _measure_index(queries: list[str], collection_name: str) -> dict:
    latencies = []
    per_q_sources = []
    for q in queries:
        try:
            ch = build_chain(milvus_collection_override=collection_name)
            t0 = time.time()
            res = ch.invoke(q)
            latencies.append(time.time() - t0)
            docs = res.get("source_documents", [])
            srcs = set()
            for d in docs:
                try:
                    md = getattr(d, "metadata", {})
                    s = md.get("source", "")
                    if s:
                        srcs.add(s)
                except Exception:
                    continue
            per_q_sources.append(srcs)
        except Exception:
            latencies.append(9e9)
            per_q_sources.append(set())
    lat_sorted = sorted([x for x in latencies if x < 1e9])
    def pctl(p):
        if not lat_sorted:
            return None
        k = max(0, min(len(lat_sorted)-1, int(round(p * (len(lat_sorted)-1)))))
        return lat_sorted[k]
    return {"p50": pctl(0.5), "p95": pctl(0.95), "sources": per_q_sources}

@app.post("/admin/index/canary/evaluate", tags=["Admin"])
def admin_index_canary_evaluate(request: Request, sample_count: int | None = None):
    require_admin(request)
    cfg = _index_gate_cfg()
    n = int(sample_count or cfg.get("sample_count", 50))
    queries = _load_golden_queries(n)
    if not queries:
        raise HTTPException(status_code=400, detail="No golden queries available for evaluation")
    try:
        # Measure active
        from src.index.milvus_index import get_active_collection_name, get_canary_collection_name
        active = get_active_collection_name()
        canary = get_canary_collection_name()
        m_active = _measure_index(queries, active)
        m_canary = _measure_index(queries, canary)
        if not (m_active.get("p95") and m_canary.get("p95")):
            raise HTTPException(status_code=500, detail="Unable to compute latency percentiles")
        inc_pct = ((m_canary["p95"] - m_active["p95"]) / max(1e-6, m_active["p95"])) * 100.0
        # Compute average per-query source overlap Jaccard
        overlaps = []
        for i in range(min(len(m_active["sources"]), len(m_canary["sources"]))):
            a = m_active["sources"][i]
            b = m_canary["sources"][i]
            inter = len(a & b)
            uni = len(a | b) or 1
            overlaps.append(inter / uni)
        avg_overlap = sum(overlaps)/len(overlaps) if overlaps else 0.0
        gate_ok = (inc_pct <= float(cfg.get("max_latency_increase_pct", 20.0))) and (avg_overlap >= float(cfg.get("min_source_overlap", 0.8)))
        out = {"active": active, "canary": canary, "p95_active": m_active["p95"], "p95_canary": m_canary["p95"], "latency_increase_pct": inc_pct, "avg_source_overlap": avg_overlap, "pass": gate_ok}
        if _redis_usable():
            try:
                _redis.set("index:last_canary_eval", json.dumps(out))
            except Exception:
                _record_redis_failure()
        return {"ok": True, "result": out}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---- Tenant batch endpoints ----
@app.get("/admin/tenant/list", tags=["Admin"])
def admin_tenant_list(request: Request):
    require_admin(request)
    tenants = []
    if _redis_usable():
        try:
            # scan keys tier:* for tenant list
            # simple approach: assume small cardinality and use KEYS
            keys = _redis.keys("tier:*")
            for k in keys:
                if k.startswith("tier:") and ":limits:" not in k:
                    tenants.append(k.split(":",1)[1])
        except Exception:
            _record_redis_failure()
    return {"ok": True, "tenants": sorted(set(tenants))}

@app.post("/admin/tenant/batch_summary", tags=["Admin"])
def admin_tenant_batch_summary(request: Request, tenants: list[str]):
    require_admin(request)
    out = []
    for t in tenants or []:
        try:
            s = admin_tenant_summary(request, t)
            out.append(s.get("summary"))
        except Exception:
            continue
    return {"ok": True, "summaries": out}

@app.post("/admin/tenant/export_summaries", tags=["Admin"])
def admin_tenant_export_summaries(request: Request, tenants: list[str] | None = None):
    require_admin(request)
    if not tenants:
        tl = admin_tenant_list(request)
        tenants = tl.get("tenants", [])
    rows = []
    for t in tenants:
        try:
            s = admin_tenant_summary(request, t).get("summary", {})
            rows.append(s)
        except Exception:
            continue
    content = "\n".join([json.dumps(r) for r in rows])
    if _redis_usable():
        try:
            _redis.set("tenant:last_export", content)
        except Exception:
            _record_redis_failure()
    return PlainTextResponse(content=content, media_type="application/jsonl")

# ---- Embeddings provider/model map admin ----
@app.get("/admin/embeddings/map", tags=["Admin"])
def admin_embeddings_map_get(request: Request):
    require_admin(request)
    out = {"provider_map": {}, "model_map": {}}
    if _redis_usable():
        try:
            out["provider_map"] = _redis.hgetall("emb:provider_map") or {}
            out["model_map"] = _redis.hgetall("emb:model_map") or {}
        except Exception:
            _record_redis_failure()
    return {"ok": True, "map": out}

@app.post("/admin/embeddings/map", tags=["Admin"])
def admin_embeddings_map_set(request: Request, provider_map: dict | None = None, model_map: dict | None = None):
    require_admin(request)
    if _redis_usable():
        try:
            if provider_map:
                _redis.hset("emb:provider_map", mapping={k: str(v) for k, v in provider_map.items()})
            if model_map:
                _redis.hset("emb:model_map", mapping={k: str(v) for k, v in model_map.items()})
        except Exception:
            _record_redis_failure()
    return admin_embeddings_map_get(request)

# ---- Router providers overview ----
@app.get("/admin/router/providers", tags=["Admin"])
def admin_router_providers(request: Request):
    require_admin(request)
    providers = []
    # openai
    providers.append({
        "name": "openai",
        "imported": True,
        "configured": bool(Config.OPENAI_API_KEY),
    })
    # azure_openai
    try:
        imported = True if AzureChatOpenAI is not None else False  # type: ignore
    except Exception:
        imported = False
    providers.append({
        "name": "azure_openai",
        "imported": imported,
        "configured": bool(os.getenv("AZURE_OPENAI_ENDPOINT")) and bool(Config.OPENAI_API_KEY),
    })
    # bedrock
    try:
        from langchain_aws import ChatBedrock  # type: ignore
        bed_imported = True
    except Exception:
        bed_imported = False
    providers.append({
        "name": "bedrock",
        "imported": bed_imported,
        "configured": bool(Config.BEDROCK_REGION) and bool(Config.BEDROCK_CHAT_MODEL),
        "region": Config.BEDROCK_REGION,
    })
    return {"ok": True, "providers": providers, "allowed": Config.ROUTER_ALLOWED_PROVIDERS}

@app.get("/admin/eval/drift/status", tags=["Admin"])
def admin_eval_drift_status(request: Request, tenant: str | None = None, window: int = 3600):
    require_admin(request)
    if not _redis_usable():
        raise HTTPException(status_code=503, detail="Redis not available for eval status")
    now_ts = int(time.time())
    total = 0
    ctrl_wins = 0
    treat_wins = 0
    ties = 0
    avg_delta = 0.0
    try:
        rows = _redis.lrange("online:judge", 0, 9999) or []
    except Exception:
        _record_redis_failure()
        rows = []
    deltas = []
    for r in rows:
        try:
            obj = json.loads(r)
        except Exception:
            continue
        ts = int(obj.get("ts", 0))
        if window and ts and now_ts - ts > int(window):
            continue
        t = str(obj.get("tenant", ""))
        if tenant and t != tenant:
            continue
        total += 1
        c_sc = float(obj.get("control_score", 0.0))
        t_sc = float(obj.get("treatment_score", 0.0))
        deltas.append(t_sc - c_sc)
        w = str(obj.get("winner", "tie"))
        if w == "control":
            ctrl_wins += 1
        elif w == "treatment":
            treat_wins += 1
        else:
            ties += 1
    if deltas:
        avg_delta = float(sum(deltas) / len(deltas))
    status = "stable"
    if total >= 10:
        if avg_delta < -0.05:
            status = "regressed"
        elif avg_delta > 0.05:
            status = "improved"
    return {
        "ok": True,
        "tenant": tenant,
        "window_seconds": int(window),
        "total": total,
        "control_wins": ctrl_wins,
        "treatment_wins": treat_wins,
        "ties": ties,
        "avg_treatment_minus_control": avg_delta,
        "status": status,
    }

@app.get("/admin/playbook", tags=["Admin"])
def admin_playbook_get(request: Request, tenant: str):
    require_admin(request)
    cfg = _load_playbook(tenant)
    return {"ok": True, "tenant": tenant, "playbook": cfg}

@app.post("/admin/playbook", tags=["Admin"])
def admin_playbook_set(request: Request, tenant: str, body: dict):
    require_admin(request)
    cfg = body or {}
    _save_playbook(tenant, cfg)
    return admin_playbook_get(request, tenant)

@app.get("/admin/temporal/status", tags=["Admin"])
def admin_temporal_status(request: Request, tenant: str | None = None):
    require_admin(request)
    if not _redis_usable():
        raise HTTPException(status_code=503, detail="Redis not available for temporal status")
    def _read_range(prefix: str) -> dict:
        try:
            start = _redis.get(f"{prefix}:start")
            end = _redis.get(f"{prefix}:end")
        except Exception:
            _record_redis_failure()
            start = None
            end = None
        def _fmt(ts: str | None):
            try:
                if not ts:
                    return None
                iv = int(ts)
                from datetime import datetime
                return datetime.utcfromtimestamp(iv).isoformat() + "Z"
            except Exception:
                return None
        return {"start_ts": start, "end_ts": end, "start_iso": _fmt(start), "end_iso": _fmt(end)}
    global_range = _read_range("temporal:global")
    tenant_range = None
    if tenant:
        tenant_range = _read_range(f"temporal:{tenant}")
    return {"ok": True, "tenant": tenant, "global": global_range, "tenant_range": tenant_range}

@app.post("/admin/policy/simulate", tags=["Admin"])
def admin_policy_simulate(request: Request, tenant: str, body: dict):
    """Simulate policy pre/post effects for a tenant on a hypothetical Q&A."""
    require_admin(request)
    pol = load_policy(_redis if _redis_usable() else None, tenant)
    q = (body or {}).get("query") or ""
    doc_type = (body or {}).get("doc_type")
    region = (body or {}).get("region")
    qctx = {"tenant": tenant, "doc_type": doc_type, "region": region}
    pre_filters, pre_decision = policy_eval_pre(qctx, pol)
    post_answer = None
    post_sources = None
    post_decision = None
    ans = (body or {}).get("answer")
    sources = (body or {}).get("sources") or []
    override = bool((body or {}).get("override"))
    if ans is not None:
        # normalize sources into list of dicts
        norm_sources = []
        try:
            for s in sources:
                if isinstance(s, dict):
                    norm_sources.append(dict(s))
                else:
                    norm_sources.append({"source": str(s)})
        except Exception:
            norm_sources = []
        post_answer, post_sources, post_decision = policy_eval_post(ans, norm_sources, pol, override=override)
    return {
        "ok": True,
        "tenant": tenant,
        "pre": {"filters": pre_filters, "decision": pre_decision},
        "post": {
            "answer": post_answer,
            "sources": post_sources,
            "decision": post_decision,
        },
    }

def _track_query_volume(tenant: str, doc_type: str | None):
    if not _redis_usable():
        return
    try:
        day = time.strftime("%Y-%m-%d", time.gmtime())
        key = f"q:vol:{day}"
        dt = doc_type or "__none__"
        field = f"{tenant}|{dt}"
        _redis.hincrby(key, field, 1)
    except Exception:
        _record_redis_failure()

def _append_query_trail(tenant: str, q: str, intent: str | None, doc_type: str | None):
    if not _redis_usable():
        return
    try:
        key = f"q:trail:{tenant}"
        evt = {
            "ts": int(time.time()),
            "q": q,
            "intent": intent,
            "doc_type": doc_type,
        }
        _redis.lpush(key, json.dumps(evt))
        _redis.ltrim(key, 0, 199)
    except Exception:
        _record_redis_failure()

@app.get("/admin/query/hotspots", tags=["Admin"])
def admin_query_hotspots(request: Request, day: str | None = None, limit: int = 50):
    require_admin(request)
    if not _redis_usable():
        raise HTTPException(status_code=503, detail="Redis not available for query hotspots")
    try:
        d = day or time.strftime("%Y-%m-%d", time.gmtime())
        key = f"q:vol:{d}"
        h = _redis.hgetall(key) or {}
        rows = []
        for k, v in h.items():
            try:
                tenant, dt = k.split("|", 1)
            except ValueError:
                tenant, dt = k, "__none__"
            try:
                c = int(v)
            except Exception:
                c = 0
            rows.append({"tenant": tenant, "doc_type": None if dt == "__none__" else dt, "count": c})
        rows.sort(key=lambda x: x["count"], reverse=True)
        if limit and limit > 0:
            rows = rows[: int(limit)]
        return {"ok": True, "day": d, "hotspots": rows}
    except HTTPException:
        raise
    except Exception:
        _record_redis_failure()
        raise HTTPException(status_code=500, detail="redis error")

@app.get("/admin/query/trail", tags=["Admin"])
def admin_query_trail(request: Request, tenant: str, limit: int = 50):
    require_admin(request)
    if not _redis_usable():
        raise HTTPException(status_code=503, detail="Redis not available for query trail")
    try:
        key = f"q:trail:{tenant}"
        raw = _redis.lrange(key, 0, max(0, int(limit) - 1)) or []
        out = []
        for r in raw:
            try:
                out.append(json.loads(r))
            except Exception:
                continue
        return {"ok": True, "tenant": tenant, "events": out}
    except HTTPException:
        raise
    except Exception:
        _record_redis_failure()
        raise HTTPException(status_code=500, detail="redis error")

# ---- Phase 18: Router admin endpoints ----
def _router_policy(tenant: str) -> dict:
    pol = {"objective": Config.ROUTER_DEFAULT_OBJECTIVE, "allowed_providers": Config.ROUTER_ALLOWED_PROVIDERS}
    if _redis_usable():
        try:
            raw = _redis.get(f"router:policy:{tenant}")
            if raw:
                d = json.loads(raw)
                pol.update(d)
        except Exception:
            _record_redis_failure()
    return pol

@app.get("/admin/router/policy", tags=["Admin"])
def admin_router_get_policy(request: Request, tenant: str):
    require_admin(request)
    return {"ok": True, "tenant": tenant, "policy": _router_policy(tenant)}

@app.post("/admin/router/policy", tags=["Admin"])
def admin_router_set_policy(request: Request, tenant: str, body: dict):
    require_admin(request)
    if _redis_usable():
        try:
            _redis.set(f"router:policy:{tenant}", json.dumps(body or {}))
        except Exception:
            _record_redis_failure()
    return {"ok": True, "tenant": tenant, "policy": _router_policy(tenant)}

@app.get("/admin/router/status", tags=["Admin"])
def admin_router_status(request: Request, limit: int = 100):
    require_admin(request)
    items = []
    if _redis_usable():
        try:
            raw = _redis.lrange("router:hist", 0, max(0, limit - 1))
            for r in raw:
                try:
                    items.append(json.loads(r))
                except Exception:
                    continue
        except Exception:
            _record_redis_failure()
    # Provider health summary
    def _percentile(vals: list[float], pct: float) -> float:
        if not vals:
            return 0.0
        s = sorted(vals)
        k = int(max(0, min(len(s) - 1, round((pct/100.0) * (len(s)-1)))))
        return float(s[k])
    def _prov_stats(p: str) -> dict:
        stats = {"provider": p, "p50": None, "p95": None, "error_rate_5m": None, "qps_1m": None, "last_error": None}
        if not _redis_usable():
            return stats
        try:
            lats = [float(x) for x in (_redis.lrange(f"prov:lat:{p}", 0, 199) or [])]
            stats["p50"] = _percentile(lats, 50)
            stats["p95"] = _percentile(lats, 95)
            now = int(time.time())
            # errors
            err_ts = [int(x) for x in (_redis.lrange(f"prov:errts:{p}", 0, 499) or [])]
            stats["error_rate_5m"] = float(sum(1 for t in err_ts if now - t <= 300)) / max(1.0, 300.0)
            # qps
            req_ts = [int(x) for x in (_redis.lrange(f"prov:reqts:{p}", 0, 999) or [])]
            stats["qps_1m"] = float(sum(1 for t in req_ts if now - t <= 60)) / 60.0
            le = _redis.get(f"prov:errlast:{p}")
            if le:
                stats["last_error"] = str(le)
        except Exception:
            _record_redis_failure()
        return stats
    provs = list(set((Config.ROUTER_ALLOWED_PROVIDERS or ["openai"])) | set(["openai","azure_openai","bedrock"]))
    health = [_prov_stats(p) for p in provs]
    return {"ok": True, "items": items, "health": health}

@app.post("/admin/index/canary/reembed", tags=["Admin"])
def admin_index_canary_reembed(request: Request, body: dict):
    require_admin(request)
    provider = (body or {}).get("provider") or "openai"
    model = (body or {}).get("model") or Config.EMBED_MODEL
    limit = int((body or {}).get("limit") or 10000)
    batch_size = int((body or {}).get("batch_size") or 256)
    # ensure canary exists
    begin_canary_build()
    # construct embeddings per provider
    embeddings = None
    if provider == "azure_openai" and AzureOpenAIEmbeddings is not None:
        embeddings = AzureOpenAIEmbeddings(azure_deployment=model, api_key=Config.OPENAI_API_KEY, azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"), api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"))
    elif provider == "bedrock" and BedrockEmbeddings is not None and Config.BEDROCK_REGION:
        embeddings = BedrockEmbeddings(model_id=model or Config.BEDROCK_EMBEDDING_MODEL, region_name=Config.BEDROCK_REGION)
    else:
        embeddings = OpenAIEmbeddings(model=model, api_key=Config.OPENAI_API_KEY)
    res = reembed_active_to_canary(embeddings, provider=provider, model=model, limit=limit, batch_size=batch_size)
    return {"ok": True, "reembedded": res}

def _index_route_key(tenant: str, doc_type: str | None) -> str:
    t = (tenant or "__default__").lower()
    dt = (doc_type or "").lower()
    return f"{t}:{dt}" if dt else t

def _index_collection_for(tenant: str, doc_type: str | None) -> str | None:
    if not _redis_usable():
        return None
    try:
        key_exact = _index_route_key(tenant, doc_type)
        v = _redis.hget("index:route", key_exact)
        if v:
            return str(v)
        # fallback: tenant-only route
        v2 = _redis.hget("index:route", (tenant or "__default__").lower())
        if v2:
            return str(v2)
    except Exception:
        _record_redis_failure()
    return None

@app.get("/admin/index/routes", tags=["Admin"])
def admin_index_routes_get(request: Request):
    require_admin(request)
    routes = {}
    if _redis_usable():
        try:
            routes = _redis.hgetall("index:route") or {}
        except Exception:
            _record_redis_failure()
    return {"ok": True, "routes": routes}

@app.post("/admin/index/routes", tags=["Admin"])
def admin_index_routes_set(request: Request, routes: dict):
    require_admin(request)
    if _redis_usable() and routes:
        try:
            mapping = {str(k): str(v) for k, v in routes.items()}
            _redis.hset("index:route", mapping=mapping)
        except Exception:
            _record_redis_failure()
            raise HTTPException(status_code=500, detail="redis error")
    return admin_index_routes_get(request)

@app.get("/admin/router/costs", tags=["Admin"])
def admin_router_get_costs(request: Request, provider: str, model: str):
    require_admin(request)
    data = {"prompt": None, "completion": None}
    if _redis_usable():
        try:
            h = _redis.hgetall(f"router:cost:{provider}:{model}") or {}
            if "prompt" in h:
                data["prompt"] = float(h.get("prompt"))
            if "completion" in h:
                data["completion"] = float(h.get("completion"))
        except Exception:
            _record_redis_failure()
    return {"ok": True, "provider": provider, "model": model, "costs": data}

@app.post("/admin/router/costs", tags=["Admin"])
def admin_router_set_costs(request: Request, body: dict):
    require_admin(request)
    provider = (body or {}).get("provider")
    model = (body or {}).get("model")
    if not provider or not model:
        raise HTTPException(status_code=400, detail="provider and model are required")
    if _redis_usable():
        try:
            mapping = {}
            if "prompt" in (body or {}):
                mapping["prompt"] = str(float(body.get("prompt")))
            if "completion" in (body or {}):
                mapping["completion"] = str(float(body.get("completion")))
            if mapping:
                _redis.hset(f"router:cost:{provider}:{model}", mapping=mapping)
        except Exception:
            _record_redis_failure()
    return admin_router_get_costs(request, provider=provider, model=model)

def _router_decide(tenant: str, fallback_model: str, doc_type: str | None) -> tuple[str, str, str]:
    """Return (model, provider, reason)."""
    if not Config.ROUTER_ENABLED:
        return fallback_model, "openai", "disabled"
    pol = _router_policy(tenant)
    objective = (pol.get("objective") or "balanced").lower()
    allowed = pol.get("allowed_providers") or ["openai"]
    slo_max_p95_ms = pol.get("max_p95_ms")
    slo_max_err = pol.get("max_error_rate_5m")
    slo_max_cost = pol.get("max_prompt_cost_per_1k")
    dt_over = (pol.get("doc_type_overrides") or {})
    # Helper readers for health and cost
    def _p95(p: str) -> float:
        if not _redis_usable():
            return 999.0
        try:
            lats = [float(x) for x in (_redis.lrange(f"prov:lat:{p}", 0, 199) or [])]
            if not lats:
                return 999.0
            s = sorted(lats)
            idx = int(max(0, min(len(s)-1, round(0.95*(len(s)-1)))))
            return float(s[idx])
        except Exception:
            return 999.0
    def _err_rate_5m(p: str) -> float:
        if not _redis_usable():
            return 0.0
        try:
            now = int(time.time())
            err_ts = [int(x) for x in (_redis.lrange(f"prov:errts:{p}", 0, 499) or [])]
            req_ts = [int(x) for x in (_redis.lrange(f"prov:reqts:{p}", 0, 999) or [])]
            errs = sum(1 for t in err_ts if now - t <= 300)
            reqs = sum(1 for t in req_ts if now - t <= 300)
            if reqs <= 0:
                return 0.0
            return float(errs) / float(reqs)
        except Exception:
            return 0.0
    def _cost_prompt_per_1k(p: str, m: str) -> float:
        if not _redis_usable():
            return 999.0
        try:
            h = _redis.hgetall(f"router:cost:{p}:{m}") or {}
            if "prompt" in h:
                return float(h.get("prompt"))
        except Exception:
            pass
        return 999.0
    # Candidate list with SLO evaluation
    doc_type_key = (doc_type or "").lower() or None
    order = list(allowed)
    # doc_type override preferred provider
    if doc_type_key and isinstance(dt_over, dict):
        try:
            ov = dt_over.get(doc_type_key) or {}
            ovp = ov.get("provider")
            if ovp and ovp in allowed:
                order = [ovp] + [p for p in allowed if p != ovp]
        except Exception:
            pass
    cands = []
    for p in order:
        p95 = _p95(p)
        err = _err_rate_5m(p)
        cost = _cost_prompt_per_1k(p, fallback_model)
        ms = p95 * 1000.0
        slo_ok = True
        if slo_max_p95_ms is not None:
            try:
                slo_ok = slo_ok and (ms <= float(slo_max_p95_ms))
            except Exception:
                pass
        if slo_max_err is not None:
            try:
                slo_ok = slo_ok and (err <= float(slo_max_err))
            except Exception:
                pass
        if slo_max_cost is not None:
            try:
                slo_ok = slo_ok and (cost <= float(slo_max_cost))
            except Exception:
                pass
        cands.append({"provider": p, "p95": p95, "err": err, "cost": cost, "slo_ok": slo_ok})
    def _pick_best(key: str, prefer_ok: bool, reverse: bool = False) -> tuple[dict, bool]:
        chosen = None
        used_ok = False
        pool = cands
        if prefer_ok and any(c["slo_ok"] for c in cands):
            pool = [c for c in cands if c["slo_ok"]]
            used_ok = True
        best_val = None
        for c in pool:
            v = c[key]
            if best_val is None or ((v > best_val) if reverse else (v < best_val)):
                best_val = v
                chosen = c
        return chosen or cands[0], used_ok
    model = fallback_model
    provider = order[0] if order else "openai"
    reason = objective
    # choose based on objective with SLO-aware selection
    if objective == "latency":
        chosen, used_ok = _pick_best("p95", prefer_ok=True)
        provider = chosen["provider"]
        model = Config.LLM_MODEL_CHEAP
        reason = "latency" if used_ok else "latency_slo_violation"
    elif objective == "cost":
        chosen, used_ok = _pick_best("cost", prefer_ok=True)
        provider = chosen["provider"]
        model = Config.LLM_MODEL_CHEAP
        reason = "cost" if used_ok else "cost_slo_violation"
    elif objective == "quality":
        # prefer doc_type override / first allowed but fail over if SLOs badly violated and another provider is OK
        chosen, used_ok = _pick_best("err", prefer_ok=True)
        provider = chosen["provider"]
        model = Config.LLM_MODEL_A
        reason = "quality" if used_ok else "quality_slo_failover"
    else:
        # balanced: latency + cost combined; prefer SLO-ok
        for c in cands:
            c["score"] = c["p95"] + c["cost"]
        def _pick_bal() -> tuple[dict, bool]:
            pool = [c for c in cands if c["slo_ok"]]
            used_ok_local = True
            if not pool:
                pool = cands
                used_ok_local = False
            best = None
            best_val = None
            for c in pool:
                v = c["score"]
                if best_val is None or v < best_val:
                    best_val = v
                    best = c
            return best or cands[0], used_ok_local
        chosen, used_ok = _pick_bal()
        provider = chosen["provider"]
        model = fallback_model
        reason = "balanced" if used_ok else "balanced_slo_violation"
    try:
        _redis.lpush("router:hist", json.dumps({
            "ts": int(time.time()),
            "tenant": tenant,
            "provider": provider,
            "model": model,
            "reason": reason,
            "doc_type": doc_type_key,
        }))
        _redis.ltrim("router:hist", 0, 999)
    except Exception:
        _record_redis_failure()
    try:
        ROUTER_DECISIONS_TOTAL.labels(tenant=tenant, provider=provider, reason=reason).inc()
    except Exception:
        pass
    return model, provider, reason

# ---- Phase 14: Cache and Cost admin endpoints ----
def _cache_cfg() -> dict:
    cfg = {
        "enabled": bool(Config.SEMANTIC_CACHE_ENABLED),
        "ttl_seconds": int(Config.SEMANTIC_CACHE_TTL_SECONDS),
    }
    if _redis_usable():
        try:
            h = _redis.hgetall("cache:config") or {}
            if "ttl_seconds" in h:
                cfg["ttl_seconds"] = int(h.get("ttl_seconds"))
            if "enabled" in h:
                cfg["enabled"] = str(h.get("enabled")).lower() == "true"
        except Exception:
            _record_redis_failure()
    return cfg

@app.get("/admin/cache/config", tags=["Admin"])
def admin_cache_get(request: Request):
    require_admin(request)
    return {"ok": True, "config": _cache_cfg()}

@app.post("/admin/cache/config", tags=["Admin"])
def admin_cache_set(request: Request, body: dict):
    require_admin(request)
    if _redis_usable():
        try:
            mapping = {}
            if "enabled" in body:
                mapping["enabled"] = str(bool(body.get("enabled"))).lower()
            if "ttl_seconds" in body:
                mapping["ttl_seconds"] = str(int(body.get("ttl_seconds")))
            if mapping:
                _redis.hset("cache:config", mapping=mapping)
        except Exception:
            _record_redis_failure()
    return {"ok": True, "config": _cache_cfg()}

@app.post("/admin/cache/invalidate", tags=["Admin"])
def admin_cache_invalidate(request: Request, tenant: str = None):
    require_admin(request)
    # bump namespace globally or per-tenant by policy: we support global bump only here
    try:
        _bump_cache_ns()
        try:
            CACHE_EVICTIONS_TOTAL.inc()
        except Exception:
            pass
    except Exception:
        pass
    return {"ok": True, "ns": _get_cache_ns()}

def _cost_cfg() -> dict:
    cfg = {
        "k_min": 4,
        "k_max": 20,
        "fetch_k_min": 10,
        "fetch_k_max": 50,
        "downshift_model": Config.LLM_MODEL_CHEAP,
        "tokens_per_dollar": 25000,
    }
    if _redis_usable():
        try:
            h = _redis.hgetall("cost:config") or {}
            for k in list(cfg.keys()):
                if k in h:
                    v = h.get(k)
                    cfg[k] = int(v) if k.endswith("_min") or k.endswith("_max") else v
            if "tokens_per_dollar" in h:
                cfg["tokens_per_dollar"] = float(h.get("tokens_per_dollar"))
        except Exception:
            _record_redis_failure()
    return cfg

@app.get("/admin/cost/config", tags=["Admin"])
def admin_cost_get(request: Request):
    require_admin(request)
    return {"ok": True, "config": _cost_cfg()}

@app.post("/admin/cost/config", tags=["Admin"])
def admin_cost_set(request: Request, body: dict):
    require_admin(request)
    if _redis_usable():
        try:
            mapping = {}
            for k in ["k_min","k_max","fetch_k_min","fetch_k_max","downshift_model","tokens_per_dollar"]:
                if k in (body or {}):
                    mapping[k] = str(body.get(k))
            if mapping:
                _redis.hset("cost:config", mapping=mapping)
        except Exception:
            _record_redis_failure()
    return {"ok": True, "config": _cost_cfg()}

# ---- Phase 13: HITL admin endpoints ----
@app.get("/admin/hitl/queue", tags=["Admin"])
def admin_hitl_queue(request: Request, limit: int = 100):
    require_admin(request)
    items = []
    if _redis_usable():
        try:
            raw = _redis.lrange("hitl:queue", 0, max(0, limit - 1))
            for r in raw:
                try:
                    items.append(json.loads(r))
                except Exception:
                    continue
        except Exception:
            _record_redis_failure()
    return {"ok": True, "items": items}

@app.post("/admin/hitl/resolve", tags=["Admin"])
def admin_hitl_resolve(request: Request, resolution: str, item: dict):
    require_admin(request)
    resv = (resolution or "ack").lower()
    tenant = str(item.get("tenant", "")) or "__default__"
    try:
        HITL_REVIEWED_TOTAL.labels(tenant=tenant, resolution=resv).inc()
    except Exception:
        pass
    # Optional: keep a short history of resolutions
    if _redis_usable():
        try:
            _redis.lpush("hitl:resolved", json.dumps({"ts": int(time.time()), "resolution": resv, "item": item}))
            _redis.ltrim("hitl:resolved", 0, 999)
        except Exception:
            _record_redis_failure()
    return {"ok": True}

# ---- Phase 9: Online eval runtime config helpers ----
def _online_eval_cfg() -> dict:
    cfg = {
        "enabled": Config.ONLINE_EVAL_ENABLED,
        "sample_rate": Config.ONLINE_EVAL_SAMPLE_RATE,
        "diff_threshold": Config.ONLINE_EVAL_DIFF_THRESHOLD,
        "window": Config.ONLINE_EVAL_WINDOW,
    }
    if _redis_usable():
        try:
            raw = _redis.hgetall("online:eval:config") or {}
            if raw:
                if str(raw.get("enabled", "")).lower() in ("true","false"):
                    cfg["enabled"] = str(raw.get("enabled")).lower() == "true"
                if "sample_rate" in raw:
                    cfg["sample_rate"] = float(raw.get("sample_rate"))
                if "diff_threshold" in raw:
                    cfg["diff_threshold"] = float(raw.get("diff_threshold"))
                if "window" in raw:
                    cfg["window"] = int(raw.get("window"))
        except Exception:
            _record_redis_failure()
            pass
    return cfg

def _online_eval_enabled() -> bool:
    try:
        return bool(_online_eval_cfg().get("enabled", False))
    except Exception:
        return False

def _online_eval_sample_rate() -> float:
    try:
        r = float(_online_eval_cfg().get("sample_rate", 0.0))
        return max(0.0, min(1.0, r))
    except Exception:
        return 0.0

def _tenant_ttl(tenant: str) -> int:
    try:
        cfg = (Config.SEMANTIC_CACHE_TTL_TENANT or "").strip()
        if not cfg:
            return int(Config.SEMANTIC_CACHE_TTL_SECONDS)
        mapping = {}
        for pair in [p.strip() for p in cfg.split(",") if p.strip()]:
            k, v = pair.split(":", 1)
            mapping[k.strip()] = int(v.strip())
        return int(mapping.get(tenant, Config.SEMANTIC_CACHE_TTL_SECONDS))
    except Exception:
        return int(Config.SEMANTIC_CACHE_TTL_SECONDS)

def _simhash(q: str, filters) -> str:
    base = (q or "")
    try:
        f = filters.dict() if filters else {}
    except Exception:
        f = {}
    s = base + "|" + json.dumps(f, sort_keys=True)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]

# Budget helpers (Redis-first with in-memory fallback)
_budget_mem = {}

def _budget_key(tenant: str) -> str:
    return f"budget:{tenant}"

def _budget_get(tenant: str):
    try:
        if _redis_usable():
            try:
                h = _redis.hgetall(_budget_key(tenant)) or {}
                # decode if bytes
                if isinstance(h, dict):
                    h = {
                        (k.decode() if isinstance(k, (bytes, bytearray)) else k):
                        (v.decode() if isinstance(v, (bytes, bytearray)) else v)
                        for k, v in h.items()
                    }
                limit = float(h.get("limit", Config.BUDGET_DEFAULT_DAILY_USD))
                spent = float(h.get("spent", "0"))
                window = int(h.get("window", str(int(time.time() // 86400))))
                return (limit, spent, window)
            except Exception:
                _record_redis_failure()
        m = _budget_mem.get(tenant, {"limit": Config.BUDGET_DEFAULT_DAILY_USD, "spent": 0.0, "window": int(time.time() // 86400)})
        return (float(m["limit"]), float(m["spent"]), int(m["window"]))
    except Exception:
        return (Config.BUDGET_DEFAULT_DAILY_USD, 0.0, int(time.time() // 86400))

def _budget_set(tenant: str, limit: float, spent: float, window: int):
    try:
        if _redis_usable():
            try:
                _redis.hset(_budget_key(tenant), mapping={"limit": limit, "spent": spent, "window": window})
                return
            except Exception:
                _record_redis_failure()
        _budget_mem[tenant] = {"limit": limit, "spent": spent, "window": window}
    except Exception:
        pass

# ---- Phase 10: DR admin endpoints ----
def _dr_last_write_ts(role: str) -> int:
    if _redis_usable():
        try:
            v = _redis.get(f"dr:last_write_ts:{role}")
            return int(v) if v else 0
        except Exception:
            _record_redis_failure()
    return 0

def _dr_set_read_preferred(val: str):
    if _redis_usable():
        try:
            _redis.set("dr:read_preferred", str(val))
        except Exception:
            _record_redis_failure()

@app.post("/admin/dr/failover", tags=["Admin"])
def admin_dr_failover(request: Request, to: str, mode: str = "drain"):
    require_admin(request)
    to = (to or "primary").lower()
    if to not in ("primary", "secondary"):
        raise HTTPException(status_code=400, detail="to must be primary|secondary")
    _dr_set_read_preferred(to)
    # record action
    if _redis_usable():
        try:
            act = {"ts": int(time.time()), "action": "failover", "to": to, "mode": mode}
            _redis.lpush("dr:actions", json.dumps(act))
            _redis.ltrim("dr:actions", 0, 99)
        except Exception:
            _record_redis_failure()
    return {"ok": True, "read_preferred": to}

@app.get("/admin/dr/status", tags=["Admin"])
def admin_dr_status(request: Request):
    require_admin(request)
    prim = check_milvus_readiness()
    try:
        from src.index.milvus_index import check_milvus_readiness_secondary
        sec = check_milvus_readiness_secondary()
    except Exception:
        sec = {"connected": False, "has_collection": False, "loaded": False}
    t_primary = _dr_last_write_ts("primary")
    t_secondary = _dr_last_write_ts("secondary")
    lag = 0
    if t_primary and t_secondary:
        lag = max(0, int(t_primary - t_secondary))
    try:
        DR_REPLICATION_LAG_SECONDS.set(lag)
    except Exception:
        pass
    read_pref = "primary"
    if _redis_usable():
        try:
            v = _redis.get("dr:read_preferred")
            if v:
                read_pref = str(v)
        except Exception:
            _record_redis_failure()
    actions = []
    if _redis_usable():
        try:
            raw = _redis.lrange("dr:actions", 0, 20)
            for r in raw:
                try:
                    actions.append(json.loads(r))
                except Exception:
                    continue
        except Exception:
            _record_redis_failure()
    return {
        "ok": True,
        "read_preferred": read_pref,
        "primary": prim,
        "secondary": sec,
        "lag_seconds": lag,
        "actions": actions,
    }

def _budget_add_spend(tenant: str, usd: float):
    try:
        limit, spent, window = _budget_get(tenant)
        today = int(time.time() // 86400)
        if window != today:
            spent = 0.0
            window = today
        spent += max(0.0, float(usd))
        _budget_set(tenant, limit, spent, window)
    except Exception:
        pass

def _budget_should_throttle_and_model(tenant: str) -> tuple[bool, Optional[str]]:
    try:
        if not Config.BUDGET_ENABLED:
            return (False, None)
        limit, spent, _ = _budget_get(tenant)
        if spent >= limit:
            return (True, None)
        if spent >= Config.BUDGET_WARN_FRACTION * limit:
            return (False, Config.LLM_MODEL_CHEAP)
        return (False, None)
    except Exception:
        return (False, None)
    return _cache_ns_mem

def _bump_cache_ns() -> int:
    global _cache_ns_mem
    if _redis is not None:
        try:
            return int(_redis.incr("cache:ns"))
        except Exception:
            pass
    _cache_ns_mem += 1
    return _cache_ns_mem
def require_auth(request: Request) -> None:
    if not Config.API_KEY:
        return
    api_key = request.headers.get("x-api-key")
    if api_key == Config.API_KEY:
        return
    authz = request.headers.get("authorization", "")
    if authz.lower().startswith("bearer ") and _verify_jwt(authz.split(" ", 1)[1]):
        return
    raise HTTPException(status_code=401, detail="Invalid or missing API key/JWT")

CACHE_HITS_TOTAL = Counter(
    "cache_hits_total",
    "/ask cache hits",
    labelnames=["tenant"],
)
CACHE_MISSES_TOTAL = Counter(
    "cache_misses_total",
    "/ask cache misses",
    labelnames=["tenant"],
)
CACHE_EVICTIONS_TOTAL = Counter(
    "cache_evictions_total",
    "Cache evictions or invalidations",
)
DOCS_SOFT_DELETES_TOTAL = Counter(
    "docs_soft_deletes_total",
    "Total soft-deleted sources",
)
DOCS_SOFT_UNDELETES_TOTAL = Counter(
    "docs_soft_undeletes_total",
    "Total undeleted sources",
)

def require_admin(request: Request) -> None:
    """Require caller to be admin.

    - API key auth: treated as admin
    - JWT auth: must include 'admin' in scope/scopes/scp or in roles claim
    """
    require_auth(request)
    # If API key is used, allow
    api_key = request.headers.get("x-api-key")
    if Config.API_KEY and api_key == Config.API_KEY:
        return
    # If JWT used, verify admin scope
    authz = request.headers.get("authorization", "")
    if authz.lower().startswith("bearer "):
        token = authz.split(" ", 1)[1]
        try:
            # We only read claims; signature verification already enforced by require_auth
            claims = jwt.decode(token, options={"verify_signature": False}, algorithms=["RS256", "HS256"])
        except Exception:
            raise HTTPException(status_code=403, detail="Invalid token claims")
        scopes = claims.get("scope") or claims.get("scopes") or claims.get("scp") or []
        if isinstance(scopes, str):
            scopes = scopes.split()
        roles = claims.get("roles") or []
        if isinstance(roles, str):
            roles = [roles]
        if ("admin" in scopes) or ("admin" in roles):
            return
        raise HTTPException(status_code=403, detail="Admin scope required")
    # Fallback deny
    raise HTTPException(status_code=403, detail="Admin scope required")

def _tenant_from_key(api_key: str | None) -> str:
    if not api_key:
        return "anon"
    try:
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, api_key))[:8]
    except Exception:
        return "anon"

def _quota_inc_and_check(key_label: str) -> int:
    day = int(time.time()) // 86400
    if _redis is not None:
        qkey = f"quota:{key_label}:{day}"
        try:
            newv = _redis.incr(qkey)
            if newv == 1:
                _redis.expire(qkey, 90000)
            return int(newv)
        except Exception:
            pass
    st = _rate_state.get(f"q:{key_label}")
    if not st or st.get("day") != day:
        st = {"day": day, "count": 0}
    st["count"] += 1
    _rate_state[f"q:{key_label}"] = st
    return st["count"]
# Prometheus: retries for LLM/ask
ASK_RETRIES = Counter(
    "ask_retries_total",
    "Total retries performed for LLM invocations in /ask",
)
ASK_TIMEOUTS = Counter(
    "ask_timeouts_total",
    "Total LLM timeouts in /ask",
)
CIRCUIT_OPEN = Counter(
    "circuit_open_total",
    "Number of times a circuit was opened",
    labelnames=["component"],
)
CIRCUIT_STATE = Gauge(
    "circuit_state",
    "Current circuit state (0=closed,1=open)",
    labelnames=["component"],
)
ASK_USAGE_TOTAL = Counter(
    "ask_usage_total",
    "Total /ask requests counted towards quota",
    labelnames=["tenant"],
)
TOKENS_PROMPT_TOTAL = Counter(
    "tokens_prompt_total",
    "Estimated prompt tokens",
    labelnames=["tenant"],
)
TOKENS_COMPLETION_TOTAL = Counter(
    "tokens_completion_total",
    "Estimated completion tokens",
    labelnames=["tenant"],
)
COST_USD_TOTAL = Counter(
    "cost_usd_total",
    "Estimated USD cost",
    labelnames=["tenant"],
)
DENYLIST_SIZE = Gauge(
    "denylist_size",
    "Number of sources currently in denylist",
)
NEG_CACHE_HITS_TOTAL = Counter(
    "cache_negative_hits_total",
    "/ask negative cache hits",
    labelnames=["tenant"],
)
AB_DECISIONS_TOTAL = Counter(
    "ab_decisions_total",
    "AB routing decisions for /ask",
    labelnames=["tenant", "llm", "rerank"],
)
SEM_CACHE_HITS_TOTAL = Counter(
    "semantic_cache_hits_total",
    "/ask semantic cache hits",
    labelnames=["tenant"],
)
SEM_CACHE_MISSES_TOTAL = Counter(
    "semantic_cache_misses_total",
    "/ask semantic cache misses",
    labelnames=["tenant"],
)
REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "Request latency by route",
    labelnames=["method", "path"],
)
ASK_RETRIES_TOTAL = Counter(
    "ask_retries_total",
    "Total LLM retries for /ask",
)
ASK_LLM_DURATION = Histogram(
    "ask_llm_duration_seconds",
    "Latency of LLM invocation in /ask",
)
FEEDBACK_TOTAL = Counter(
    "feedback_total",
    "User feedback submissions",
    labelnames=["tenant", "helpful"],
)
OFFLINE_EVAL_RUNS_TOTAL = Counter(
    "offline_eval_runs_total",
    "Offline eval runs triggered",
)
OFFLINE_EVAL_RECALL_LAST = Gauge(
    "offline_eval_recall_last",
    "Last offline eval recall@k",
)
OFFLINE_EVAL_ANS_SCORE_LAST = Gauge(
    "offline_eval_answer_contains_last",
    "Last offline eval answer contains score",
)
DR_REPLICATION_LAG_SECONDS = Gauge(
    "dr_replication_lag_seconds",
    "Replication lag between primary and secondary (s)",
)
INDEX_BUILDS_TOTAL = Counter(
    "index_builds_total",
    "Index canary builds triggered",
    labelnames=["status"],
)
INDEX_PROMOTIONS_TOTAL = Counter(
    "index_promotions_total",
    "Index promotions from canary to active",
)
SOURCE_FRESHNESS_LAG_SECONDS = Gauge(
    "source_freshness_lag_seconds",
    "Source data freshness lag in seconds",
    labelnames=["source"],
)
ROUTER_DECISIONS_TOTAL = Counter(
    "router_decisions_total",
    "Router decisions by provider and reason",
    labelnames=["tenant", "provider", "reason"],
)
PROVIDER_ERRORS_TOTAL = Counter(
    "provider_errors_total",
    "Provider-level errors",
    labelnames=["provider"],
)
PROVIDER_LLM_LATENCY = Histogram(
    "provider_llm_latency_seconds",
    "LLM latency by provider",
    labelnames=["provider"],
)
HITL_ENQUEUED_TOTAL = Counter(
    "hitl_enqueued_total",
    "Total questions enqueued for human review",
    labelnames=["tenant", "reason"],
)
HITL_REVIEWED_TOTAL = Counter(
    "hitl_reviewed_total",
    "Total human reviews resolved",
    labelnames=["tenant", "resolution"],
)
HITL_CONFIDENCE = Histogram(
    "hitl_confidence",
    "Model confidence distribution",
)

app = FastAPI(
    title="Aerospace RAG API",
    version="1.0.0",
    description="API for question answering over aerospace documents using retrieval augmented generation.",
    contact={"name": "Aerospace RAG Team"},
    license_info={"name": "MIT"},
    swagger_ui_parameters={"displayOperationId": True},
)

# Metrics
app.add_middleware(PrometheusMiddleware)
"""Optional Sentry initialization"""
if Config.SENTRY_DSN:
    try:
        sentry_sdk.init(dsn=Config.SENTRY_DSN, traces_sample_rate=0.0)
        app.add_middleware(SentryAsgiMiddleware)
    except Exception:
        pass

"""Optional OpenTelemetry tracing initialization"""
_tracer = None
if Config.OTEL_ENABLED and Config.OTEL_EXPORTER_OTLP_ENDPOINT:
    try:
        resource = Resource.create({"service.name": Config.OTEL_SERVICE_NAME})
        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(endpoint=Config.OTEL_EXPORTER_OTLP_ENDPOINT)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        ot_trace.set_tracer_provider(provider)
        _tracer = ot_trace.get_tracer(__name__)
    except Exception:
        _tracer = None

# Basic security headers middleware
@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    response.headers.setdefault("X-XSS-Protection", "0")
    if Config.CONTENT_SECURITY_POLICY:
        response.headers.setdefault("Content-Security-Policy", Config.CONTENT_SECURITY_POLICY)
    if Config.SECURITY_HSTS_ENABLED and request.url.scheme == "https":
        response.headers.setdefault(
            "Strict-Transport-Security",
            f"max-age={Config.SECURITY_HSTS_MAX_AGE}; includeSubDomains; preload",
        )
    return response

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=Config.CORS_ALLOWED_ORIGINS,
    allow_credentials=Config.CORS_ALLOW_CREDENTIALS,
    allow_methods=Config.CORS_ALLOWED_METHODS,
    allow_headers=Config.CORS_ALLOWED_HEADERS,
)
# GZip compression (optional)
if Config.GZIP_ENABLED:
    app.add_middleware(GZipMiddleware, minimum_size=500)

# Request size limit middleware
@app.middleware("http")
async def limit_request_size(request: Request, call_next):
    cl = request.headers.get("content-length")
    if cl is not None:
        try:
            if int(cl) > Config.MAX_REQUEST_BYTES:
                return PlainTextResponse("Request entity too large", status_code=413)
        except Exception:
            pass
    return await call_next(request)
# Metrics exposure policy: in non-local envs, require auth regardless of METRICS_PUBLIC
if (Config.ENV == "local") and Config.METRICS_PUBLIC:
    app.add_route("/metrics", handle_metrics)
else:
    @app.get("/metrics")
    def metrics(request: Request):
        require_auth(request)
        return handle_metrics(request)

class AskFilters(BaseModel):
    sources: Optional[list[str]] = None
    doc_type: Optional[str] = None
    date_from: Optional[str] = None  # ISO date YYYY-MM-DD
    date_to: Optional[str] = None    # ISO date YYYY-MM-DD
    tenant: Optional[str] = None

class AskReq(BaseModel):
    query: str
    filters: Optional[AskFilters] = None

class SourceItem(BaseModel):
    source: str
    page: Optional[int] = None

class AskResp(BaseModel):
    answer: str
    sources: list[SourceItem]

class UsageResp(BaseModel):
    limit: int
    used_today: int

class HealthResp(BaseModel):
    status: str

class ReadyResp(BaseModel):
    ready: bool

class FeedbackReq(BaseModel):
    query: str
    answer: str
    helpful: bool
    reason: Optional[str] = None
    clicked_sources: Optional[list[str]] = None
    hallucinated: Optional[bool] = None
    style_score: Optional[int] = None
    tags: Optional[list[str]] = None

qa_chain = None
READY = False
# cache of QA chains by (model, rerank_enabled)
_qa_cache = {}
# AB routing maps (in-memory fallback)
_ab_llm_mem = {}   # tenant -> 'A'|'B'
_ab_rerank_mem = {}  # tenant -> 'true'|'false'
_rerank_model = None
_rerank_cache = {}
_cb_failures = 0
_cb_open_until = 0

# Structured logging setup
logger = logging.getLogger("api")
if not logger.handlers:
    handler = logging.StreamHandler()
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

# Rate limiter: Redis (if configured) with fallback to in-memory
_rate_state = {}
_deny_sources_mem = set()
_cache_ns_mem = 1
_cache_neg = {}
_source_index_mem = {}
_redis = None
_redis_cb_failures = 0
_redis_cb_open_until = 0
if Config.REDIS_URL:
    try:
        _redis = redis.Redis.from_url(Config.REDIS_URL, decode_responses=True)
        # ping to verify connectivity
        _redis.ping()
    except Exception:
        _redis = None
        # open redis circuit on init failure
        try:
            _redis_cb_failures += 1
            if _redis_cb_failures >= Config.CB_FAIL_THRESHOLD:
                _redis_cb_open_until = int(time.time()) + max(1, Config.CB_RESET_SECONDS)
                CIRCUIT_OPEN.labels(component="redis").inc()
                CIRCUIT_STATE.labels(component="redis").set(1)
        except Exception:
            pass
    
# Simple in-memory cache structure: key -> {v: response_json, t: epoch}
_cache = {}
# In-memory semantic cache: key -> {v: resp_json, t: epoch}
_sem_cache = {}
# In-memory feedback buffer fallback (tenant -> list of dict)
_feedback_mem = {}
_audit_mem = []
_budget_mem = {}
_eval_mem = []

# JWKS cache and verification helpers
_jwks_cache = {"keys": None, "fetched_at": 0}

def _fetch_jwks() -> Optional[dict]:
    if not Config.JWT_JWKS_URL:
        return None
    now = int(time.time())
    if _jwks_cache["keys"] and now - _jwks_cache["fetched_at"] < Config.JWT_JWKS_CACHE_SECONDS:
        return _jwks_cache["keys"]
    try:
        resp = requests.get(Config.JWT_JWKS_URL, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        _jwks_cache["keys"] = data
        _jwks_cache["fetched_at"] = now
        return data
    except Exception:
        return _jwks_cache["keys"]

def _verify_jwt(token: str) -> bool:
    try:
        headers = jwt.get_unverified_header(token)
    except Exception:
        headers = {}
    alg = (Config.JWT_ALG or "HS256").upper()
    try:
        if alg == "RS256" and Config.JWT_JWKS_URL:
            jwks = _fetch_jwks()
            if not jwks:
                return False
            kid = headers.get("kid")
            key = None
            for k in jwks.get("keys", []):
                if k.get("kid") == kid:
                    key = jwt.algorithms.RSAAlgorithm.from_jwk(json.dumps(k))
                    break
            if not key:
                return False
            jwt.decode(
                token,
                key=key,
                algorithms=["RS256"],
                issuer=Config.JWT_ISSUER,
                audience=Config.JWT_AUDIENCE,
            )
            return True
        elif Config.JWT_SECRET:
            jwt.decode(
                token,
                key=Config.JWT_SECRET,
                algorithms=["HS256"],
                issuer=Config.JWT_ISSUER,
                audience=Config.JWT_AUDIENCE,
            )
            return True
        else:
            return False
    except Exception:
        return False

def _expand_query(q: str) -> str:
    try:
        if not (Config.QUERY_EXPANSION_ENABLED and q):
            return q
        base = q
        low = q.lower()
        adds = []
        if "uav" in low or "drone" in low:
            adds.append("unmanned aerial vehicle")
        if "gnc" in low:
            adds.append("guidance navigation and control")
        if "aero" in low:
            adds.append("aerodynamics")
        if not adds:
            return q
        return base + " " + " ".join(adds)
    except Exception:
        return q

def _classify_intent(q: str) -> str:
    try:
        low = (q or "").lower()
        # Troubleshooting / incident style
        if any(t in low for t in ["error", "exception", "stack trace", "failed", "failure", "bug", "issue"]):
            return "troubleshoot"
        # Long, open-ended description
        tokens = low.split()
        if len(tokens) > 20 or any(t in low for t in ["overview", "explain", "guide", "walkthrough"]):
            return "explore"
        # Short fact lookup
        return "lookup"
    except Exception:
        return "lookup"

def _maybe_eval_math(q: str):
    """Best-effort arithmetic evaluator for simple calculator-style queries.
    Returns a float or None if not a pure math expression.
    """
    try:
        import re
        expr = (q or "").strip()
        if not expr:
            return None
        if not re.fullmatch(r"[0-9\s\+\-\*\/\(\)\.]+", expr):
            return None
        # Safe eval: no builtins, only math operators
        val = eval(expr, {"__builtins__": None}, {})  # type: ignore
        if isinstance(val, (int, float)):
            return float(val)
        return None
    except Exception:
        return None

def _episode_key(tenant: str, user: str) -> str:
    return f"ep:{tenant}:{user}"

def _append_episode(tenant: str, user: str, q: str, answer: str) -> None:
    if not _redis_usable():
        return
    try:
        rec = json.dumps({"q": q, "a": answer, "ts": int(time.time())})
        k = _episode_key(tenant, user)
        _redis.lpush(k, rec)
        _redis.ltrim(k, 0, 9)
        _redis.expire(k, 3600)
    except Exception:
        _record_redis_failure()

def _load_episode(tenant: str, user: str) -> list[dict]:
    out: list[dict] = []
    if not _redis_usable():
        return out
    try:
        k = _episode_key(tenant, user)
        rows = _redis.lrange(k, 0, 4) or []
        for r in rows:
            try:
                out.append(json.loads(r))
            except Exception:
                continue
    except Exception:
        _record_redis_failure()
    return out

def _playbook_key(tenant: str) -> str:
    return f"playbook:{tenant}"

def _load_playbook(tenant: str) -> dict:
    cfg: dict = {}
    if not _redis_usable():
        return cfg
    try:
        raw = _redis.get(_playbook_key(tenant))
        if raw:
            try:
                cfg = json.loads(raw)
            except Exception:
                cfg = {}
    except Exception:
        _record_redis_failure()
    return cfg

def _save_playbook(tenant: str, cfg: dict) -> None:
    if not _redis_usable():
        return
    try:
        _redis.set(_playbook_key(tenant), json.dumps(cfg or {}))
    except Exception:
        _record_redis_failure()

def _redact_pii(text: str) -> str:
    if not (Config.PII_REDACTION_ENABLED and text):
        return text
    try:
        import re
        t = text
        # emails
        t = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[REDACTED_EMAIL]", t)
        # phone-like
        t = re.sub(r"\b(?:\+?\d[\s-]?){7,15}\b", "[REDACTED_PHONE]", t)
        # passport/ids (very rough)
        t = re.sub(r"\b[0-9A-Z]{8,}\b", "[REDACTED_ID]", t)
        return t
    except Exception:
        return text

def _redis_usable() -> bool:
    try:
        return _redis is not None and (time.time() >= _redis_cb_open_until)
    except Exception:
        return False

def _record_redis_failure():
    global _redis, _redis_cb_failures, _redis_cb_open_until
    try:
        _redis_cb_failures += 1
        if _redis_cb_failures >= Config.CB_FAIL_THRESHOLD:
            _redis = None
            _redis_cb_open_until = int(time.time()) + max(1, Config.CB_RESET_SECONDS)
            CIRCUIT_OPEN.labels(component="redis").inc()
            CIRCUIT_STATE.labels(component="redis").set(1)
    except Exception:
        pass

@app.middleware("http")
async def add_request_id_and_logging(request: Request, call_next):
    req_id = str(uuid.uuid4())
    start = time.time()
    client_ip = request.client.host if request.client else ""
    request.state.request_id = req_id
    # Before
    logger.info(json.dumps({
        "request_id": req_id,
        "event": "request_start",
        "method": request.method,
        "path": request.url.path,
        "client_ip": client_ip,
    }))
    try:
        response = await call_next(request)
        return response
    finally:
        dur_ms = int((time.time() - start) * 1000)
        logger.info(json.dumps({
            "request_id": req_id,
            "event": "request_end",
            "status_code": getattr(locals().get('response', None), 'status_code', None),
            "duration_ms": dur_ms,
        }))
        try:
            REQUEST_DURATION.labels(method=request.method, path=request.url.path).observe((time.time() - start))
        except Exception:
            pass

@app.on_event("startup")
def _startup():
    global qa_chain, READY
    try:
        # Build a default chain for readiness checks
        qa_chain = build_chain()
        # Basic readiness check: FAISS store presence + chain built
        faiss_path_exists = os.path.isdir("./faiss_store")
        if Config.RETRIEVER_BACKEND == "milvus":
            # Retry Milvus readiness with backoff
            attempt = 0
            delay = max(0.001, Config.RETRY_BASE_DELAY_MS / 1000.0)
            while True:
                try:
                    milvus = check_milvus_readiness()
                    break
                except Exception:
                    attempt += 1
                    if attempt >= max(1, Config.RETRY_MAX_ATTEMPTS):
                        # mark milvus circuit open
                        try:
                            global _milvus_cb_failures, _milvus_cb_open_until
                            _milvus_cb_failures += 1
                            if _milvus_cb_failures >= Config.CB_FAIL_THRESHOLD:
                                _milvus_cb_open_until = int(time.time()) + max(1, Config.CB_RESET_SECONDS)
                                CIRCUIT_OPEN.labels(component="milvus").inc()
                                CIRCUIT_STATE.labels(component="milvus").set(1)
                        except Exception:
                            pass
                        raise
                    time.sleep(delay)
                    delay *= 2
            READY = qa_chain is not None and milvus.get("connected") and milvus.get("has_collection") and milvus.get("loaded")
        else:
            READY = qa_chain is not None and faiss_path_exists
    except Exception:
        # Do not crash on startup; mark as not ready
        READY = False
        qa_chain = None
    # Enforce auth configuration in non-local environments
    try:
        if Config.ENV != "local":
            has_auth = bool(Config.API_KEY) or bool(Config.JWT_SECRET) or ((Config.JWT_ALG or "HS256").upper() == "RS256" and bool(Config.JWT_JWKS_URL))
            if not has_auth:
                READY = False
                try:
                    logger.error(json.dumps({"event": "startup_auth_check_failed", "reason": "non_local_requires_auth"}))
                except Exception:
                    pass
    except Exception:
        pass

@app.post(
    "/ask",
    response_model=AskResp,
    tags=["Query"],
    summary="Ask a question",
    description="Returns an answer and source citations using the configured retriever and LLM.",
)
def ask(req: AskReq, request: Request):
    global _cb_failures, _cb_open_until
    span_ctx = None
    if _tracer is not None and _trace_sampled():
        span_ctx = _tracer.start_as_current_span("ask")
        span_ctx.__enter__()
    # Authorization (optional, enabled when API_KEY is set)
    require_auth(request)
    # Rate limiting
    api_key_hdr = request.headers.get("x-api-key")
    key = api_key_hdr or (request.client.host if request.client else "unknown")
    now = int(time.time())
    window = now // 60
    if _redis_usable():
        rl_key = f"rl:{key}:{window}"
        try:
            newv = _redis.incr(rl_key)
            if newv == 1:
                _redis.expire(rl_key, 65)
            if newv > Config.RATE_LIMIT_PER_MIN:
                raise HTTPException(status_code=429, detail="Rate limit exceeded")
        except Exception:
            # Fallback to in-memory if redis error
            _record_redis_failure()
            st = _rate_state.get(key)
            if not st or st["window"] != window:
                st = {"window": window, "count": 0}
            st["count"] += 1
            _rate_state[key] = st
            if st["count"] > Config.RATE_LIMIT_PER_MIN:
                raise HTTPException(status_code=429, detail="Rate limit exceeded")
    else:
        st = _rate_state.get(key)
        if not st or st["window"] != window:
            st = {"window": window, "count": 0}
        st["count"] += 1
        _rate_state[key] = st
        if st["count"] > Config.RATE_LIMIT_PER_MIN:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
    # Quota check
    if Config.QUOTA_ENABLED:
        tenant = _tenant_from_key(api_key_hdr)
        used = _quota_inc_and_check(tenant)
        ASK_USAGE_TOTAL.labels(tenant=tenant).inc()
        if used > Config.QUOTA_DAILY_LIMIT:
            raise HTTPException(status_code=429, detail="Daily quota exceeded")

    q = (req.query or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Query must not be empty")
    if len(q) > 4000:
        raise HTTPException(status_code=400, detail="Query too long (max 4000 chars)")
    # Phase 29: tenant guardrails on input text
    tenant_label = _tenant_from_key(api_key_hdr)
    try:
        gcfg = _load_guardrail_cfg(tenant_label)
        decision = _guardrails_check_input(q, gcfg)
        if decision.get("blocked"):
            try:
                logger.info(json.dumps({
                    "event": "guardrail_block",
                    "tenant": tenant_label,
                    "request_id": getattr(getattr(request, "state", None), "request_id", ""),
                    "matched": decision.get("matched"),
                }))
            except Exception:
                pass
            raise HTTPException(status_code=400, detail="Query blocked by tenant guardrails")
    except HTTPException:
        raise
    except Exception:
        pass
    # Phase 21: simple arithmetic tool path
    math_val = _maybe_eval_math(q)
    if math_val is not None:
        return {"answer": str(math_val), "sources": []}
    # Optional query expansion to improve recall (after episode context is read)
    if not READY or qa_chain is None:
        raise HTTPException(status_code=503, detail="Service not ready. Ingest documents to create ./faiss_store and restart.")
    # Determine AB routing for this tenant
    tenant_label = _tenant_from_key(api_key_hdr)
    # Phase 21: short episode context (per-tenant+user)
    user_id = key
    ep = _load_episode(tenant_label, user_id)
    if ep:
        try:
            history_txt = "\n".join([f"Q: {e.get('q','')}\nA: {e.get('a','')}" for e in ep])
            base_q = f"Previous context (most recent first):\n{history_txt}\n\nNew question: {q}"
        except Exception:
            base_q = q
    else:
        base_q = q
    # Phase 23: tenant playbook personalization
    pb = _load_playbook(tenant_label)
    style = ""
    try:
        if isinstance(pb, dict):
            style = str(pb.get("style") or pb.get("instructions") or "").strip()
    except Exception:
        style = ""
    if style:
        q_for_llm = f"Follow these tenant-specific instructions when answering:\n{style}\n\nQuestion: {base_q}"
    else:
        q_for_llm = base_q
    q_expanded = _expand_query(q_for_llm)
    # Phase 15: tier enforcement — rate limit and concurrency
    tier = _tenant_tier(tenant_label)
    limits = _tier_limits(tier)
    if not _incr_rate(tenant_label, limits.get("rate_per_min", 300)):
        try:
            TENANT_RATE_LIMITED_TOTAL.labels(tenant=tenant_label).inc()
        except Exception:
            pass
        raise HTTPException(status_code=429, detail="Rate limit exceeded for tenant")
    if not _acquire_concurrency(tenant_label, limits.get("concurrent_max", 8)):
        try:
            TENANT_CONCURRENCY_DROPPED_TOTAL.labels(tenant=tenant_label).inc()
        except Exception:
            pass
        raise HTTPException(status_code=429, detail="Too many concurrent requests for tenant")
    k_bounds = (limits.get("k_min", 4), limits.get("k_max", 20))
    fk_bounds = (limits.get("fetch_min", 10), limits.get("fetch_max", 50))
    # Phase 20: intent-aware dynamic retrieval depth
    intent = _classify_intent(q)
    k_min, k_max = k_bounds
    fk_min, fk_max = fk_bounds
    if intent == "lookup":
        k_eff = k_min
        fk_eff = max(fk_min, (fk_min + fk_max) // 2)
    elif intent == "troubleshoot":
        k_eff = min(k_max, k_min + 4)
        fk_eff = fk_max
    else:  # explore
        k_eff = k_max
        fk_eff = fk_max
    try:
        ASK_USAGE_TOTAL.labels(tenant_label).inc()
    except Exception:
        pass
    # Budget check and potential model downgrade
    throttle, model_override = _budget_should_throttle_and_model(tenant_label)
    if throttle:
        raise HTTPException(status_code=429, detail="Tenant budget exceeded. Try later or increase budget.")
    # Enforce server-side tenant isolation
    try:
        if Config.MULTITENANT_ENABLED:
            if req.filters is None:
                req.filters = AskFilters()
            req.filters.tenant = tenant_label
    except Exception:
        pass
    # Phase 11: Policy-aware pre-eval to derive additional filters/deny
    try:
        pol = load_policy(_redis if _redis_usable() else None, tenant_label)
        qctx = {
            "tenant": tenant_label,
            "doc_type": getattr(req.filters, "doc_type", None) if req.filters else None,
            "region": None,
        }
        add_filters, decision = policy_eval_pre(qctx, pol)
        if decision.get("deny"):
            try:
                logger.info(json.dumps({"event": "policy_deny", "tenant": tenant_label, "request_id": getattr(getattr(request, "state", None), "request_id", "")}))
            except Exception:
                pass
            raise HTTPException(status_code=403, detail="Policy denied the request")
        # merge filter hints
        if add_filters:
            if req.filters is None:
                req.filters = AskFilters()
            if add_filters.get("sources"):
                req.filters.sources = list(add_filters.get("sources"))
            if add_filters.get("doc_type"):
                req.filters.doc_type = str(add_filters.get("doc_type"))
    except HTTPException:
        raise
    except Exception:
        pass
    # Phase 28: append query trail event (tenant-level) with intent and doc_type
    try:
        dt_for_trail = getattr(req.filters, "doc_type", None) if req.filters else None
        _append_query_trail(tenant_label, q, intent, dt_for_trail)
    except Exception:
        pass
    # Phase 27: record query volume per tenant/doc_type for hotspot analytics
    try:
        dt_for_q = getattr(req.filters, "doc_type", None) if req.filters else None
        _track_query_volume(tenant_label, dt_for_q)
    except Exception:
        pass
    try:
        deny = set()
        srcs = []
        for d in result.get("source_documents", []):
            src = getattr(d, "metadata", {}).get("source")
            if src:
                srcs.append(src)
        post = policy_eval_post({"tenant": tenant_label, "sources": srcs}, pol)
        if post.get("deny"):
            raise HTTPException(status_code=403, detail="Policy denied the answer")
    except HTTPException:
        raise
    except Exception:
        pass
    # record episode context
    try:
        _append_episode(tenant_label, user_id, q, result.get("result") or result.get("answer") or "")
    except Exception:
        pass
    def _get_llm_variant(t: str) -> str:
        # redis hash ab:llm maps tenant->'A'|'B'
        if _redis is not None:
            try:
                v = _redis.hget("ab:llm", t)
                if v in ("A","B"):
                    return v
            except Exception:
                pass
        return _ab_llm_mem.get(t) or "A"
    def _get_rerank_enabled(t: str) -> bool:
        if _redis is not None:
            try:
                v = _redis.hget("ab:rerank", t)
                if v is not None:
                    return str(v).lower() == "true"
            except Exception:
                pass
        if t in _ab_rerank_mem:
            return str(_ab_rerank_mem.get(t)).lower() == "true"
        return bool(Config.HYBRID_ENABLED)
    llm_variant = _get_llm_variant(tenant_label)
    model_name = Config.LLM_MODEL_A if llm_variant == "A" else Config.LLM_MODEL_B
    if model_override:
        model_name = model_override
    # Phase 18+19: route provider/model decision with SLOs and doc_type overrides
    routed_provider = "openai"
    doc_type = None
    try:
        doc_type = getattr(req.filters, "doc_type", None) if req.filters else None
    except Exception:
        doc_type = None
    try:
        model_name, routed_provider, _rreason = _router_decide(tenant_label, model_name, doc_type)
    except Exception:
        pass
    rerank_enabled = _get_rerank_enabled(tenant_label)
    # Phase 26: multi-index routing – choose collection per tenant/doc_type
    routed_collection = None
    try:
        routed_collection = _index_collection_for(tenant_label, doc_type)
    except Exception:
        routed_collection = None
    # Phase 14: coarse prompt cache read (pre-LLM)
    coarse_cache_key = None
    if _redis_usable() and Config.SEMANTIC_CACHE_ENABLED:
        try:
            filters_key = {}
            if req.filters:
                try:
                    filters_key = req.filters.dict()
                except Exception:
                    filters_key = {}
            coarse_key_raw = {
                "ns": _get_cache_ns(),
                "tenant": tenant_label,
                "model": model_name,
                "rerank": bool(rerank_enabled),
                "q": q,
                "filters": filters_key,
                "intent": intent,
                "k": int(k_eff),
                "fetch_k": int(fk_eff),
            }
            coarse_cache_key = "ask:coarse:" + hashlib.sha1(json.dumps(coarse_key_raw, sort_keys=True).encode("utf-8")).hexdigest()
            v = _redis.get(coarse_cache_key)
            if v:
                try:
                    CACHE_HITS_TOTAL.labels(tenant=tenant_label).inc()
                except Exception:
                    pass
                _trace_event("cache.hit", {"kind": "coarse", "tenant": tenant_label})
                obj = json.loads(v)
                return obj
        except Exception:
            _record_redis_failure()
    # Phase 9: fire-and-forget shadow online eval with sampling
    try:
        if _online_eval_enabled() and random.random() < _online_eval_sample_rate():
            run_shadow_eval(tenant_label, q, req.filters, treatment_model=model_name, treatment_rerank=rerank_enabled)
    except Exception:
        pass
    # Build or reuse chain for this routing (include collection name in cache key)
    key_rt = (model_name, bool(rerank_enabled), int(k_eff), int(fk_eff), routed_collection or "__default__")
    chain = _qa_cache.get(key_rt)
    if chain is None:
        try:
            chain = build_chain(
                filters=req.filters,
                llm_model=model_name,
                rerank_enabled=rerank_enabled,
                k_override=int(k_eff),
                fetch_k_override=int(fk_eff),
                milvus_collection_override=routed_collection,
                llm_provider=routed_provider,
            )
            _qa_cache[key_rt] = chain
        except Exception:
            chain = None
    if chain is None:
        # Fallback to default
        chain = qa_chain
    try:
        AB_DECISIONS_TOTAL.labels(tenant=tenant_label, llm=llm_variant, rerank=str(bool(rerank_enabled)).lower()).inc()
    except Exception:
        pass
    # Cache get (if enabled) keyed by query+filters+tenant
    ns = _get_cache_ns()
    if Config.CACHE_ENABLED:
        try:
            filt = req.filters.dict() if req.filters else {}
        except Exception:
            filt = {}
        ckey = f"ask:{ns}:{tenant_label}:{q}:{json.dumps(filt, sort_keys=True)}:{int(k_eff)}:{int(fk_eff)}"
        nkey = f"{ckey}:neg"
        if _redis_usable():
            try:
                # negative first
                if Config.NEGATIVE_CACHE_ENABLED:
                    if _redis.get(nkey):
                        NEG_CACHE_HITS_TOTAL.labels(tenant=tenant_label).inc()
                        return {"answer": "", "sources": []}
                cached = _redis.get(ckey)
                if cached:
                    CACHE_HITS_TOTAL.labels(tenant=tenant_label).inc()
                    _trace_event("cache.hit", {"kind": "response", "tenant": tenant_label})
                    return json.loads(cached)
                else:
                    CACHE_MISSES_TOTAL.labels(tenant=tenant_label).inc()
            except Exception:
                _record_redis_failure()
                pass
        else:
            ent = _cache.get(ckey)
            if ent and (time.time() - ent["t"]) < Config.CACHE_TTL_SECONDS:
                CACHE_HITS_TOTAL.labels(tenant=tenant_label).inc()
                _trace_event("cache.hit", {"kind": "response_mem", "tenant": tenant_label})
                return ent["v"]
            else:
                CACHE_MISSES_TOTAL.labels(tenant=tenant_label).inc()
                # check in-memory negative cache
                if Config.NEGATIVE_CACHE_ENABLED:
                    nent = _cache_neg.get(nkey)
                    if nent and (time.time() - nent) < Config.NEGATIVE_CACHE_TTL_SECONDS:
                        NEG_CACHE_HITS_TOTAL.labels(tenant=tenant_label).inc()
                        return {"answer": "", "sources": []}

    # Semantic cache (feature-flagged)
    if Config.SEMANTIC_CACHE_ENABLED:
        simhash_key = _simhash(q, req.filters)
        skey = f"sem:{ns}:{tenant_label}:{simhash_key}"
        if _redis_usable():
            try:
                cached = _redis.get(skey)
                if cached:
                    SEM_CACHE_HITS_TOTAL.labels(tenant=tenant_label).inc()
                    _trace_event("cache.hit", {"kind": "semantic", "tenant": tenant_label})
                    return json.loads(cached)
                else:
                    SEM_CACHE_MISSES_TOTAL.labels(tenant=tenant_label).inc()
            except Exception:
                _record_redis_failure()
                pass
        else:
            ent = _sem_cache.get(skey)
            if ent and (time.time() - ent["t"]) < Config.SEMANTIC_CACHE_TTL_SECONDS:
                SEM_CACHE_HITS_TOTAL.labels(tenant=tenant_label).inc()
                _trace_event("cache.hit", {"kind": "semantic_mem", "tenant": tenant_label})
                return ent["v"]
            else:
                SEM_CACHE_MISSES_TOTAL.labels(tenant=tenant_label).inc()

    # Circuit breaker: short-circuit if open with graceful fallback
    now_ts = int(time.time())
    if _cb_open_until and now_ts < _cb_open_until:
        CIRCUIT_STATE.labels(component="llm").set(1)
        try:
            ns = _get_cache_ns()
            try:
                filt = req.filters.dict() if req.filters else {}
            except Exception:
                filt = {}
            ckey = f"ask:{ns}:{tenant_label}:{q}:{json.dumps(filt, sort_keys=True)}"
            skey = f"sem:{ns}:{tenant_label}:{_simhash(q, req.filters)}"
            # Prefer semantic cache, then response cache
            if _redis_usable():
                try:
                    cached = _redis.get(skey) or _redis.get(ckey)
                    if cached:
                        return json.loads(cached)
                except Exception:
                    _record_redis_failure()
                    pass
            ent = _sem_cache.get(skey)
            if ent and (time.time() - ent.get("t", 0)) < Config.SEMANTIC_CACHE_TTL_SECONDS:
                return ent.get("v", {"answer": "", "sources": []})
            ent2 = _cache.get(ckey)
            if ent2 and (time.time() - ent2.get("t", 0)) < Config.CACHE_TTL_SECONDS:
                return ent2.get("v", {"answer": "", "sources": []})
        except Exception:
            pass
        retry_after = max(1, int(_cb_open_until - now_ts)) if _cb_open_until else 10
        payload = {
            "answer": "Temporarily degraded: insufficient context. Please retry shortly.",
            "sources": [],
            "degraded": True,
            "retry_after_seconds": retry_after,
        }
        return JSONResponse(content=payload, headers={"Retry-After": str(retry_after)})

    # LLM invocation with retry/backoff + timeout
    attempt = 0
    delay = max(0.001, Config.RETRY_BASE_DELAY_MS / 1000.0)
    result = None
    # add tracing attributes
    rerank_model_name = None
    try:
        if _tracer is not None:
            cur = ot_trace.get_current_span()
            cur.set_attribute("tenant", tenant_label)
            cur.set_attribute("llm.variant", llm_variant)
            cur.set_attribute("llm.model", model_name)
            cur.set_attribute("rerank.enabled", bool(rerank_enabled))
            cur.set_attribute("rerank.model", "")
            cur.set_attribute("query.expanded", q_expanded != q)
            cur.set_attribute("request.id", getattr(getattr(request, "state", None), "request_id", ""))
    except Exception:
        pass
    while True:
        try:
            if _tracer is not None:
                cur = ot_trace.get_current_span()
                cur.set_attribute("query.length", len(q))
                cur.set_attribute("retriever.backend", os.getenv("RETRIEVER_BACKEND", Config.RETRIEVER_BACKEND))
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                llm_t0 = time.time()
                fut = ex.submit(chain.invoke, q_expanded)
                try:
                    result = fut.result(timeout=max(1, Config.LLM_TIMEOUT_SECONDS))
                    try:
                        dur = time.time() - llm_t0
                        ASK_LLM_DURATION.observe(dur)
                        try:
                            PROVIDER_LLM_LATENCY.labels(provider=routed_provider).observe(dur)
                        except Exception:
                            pass
                        # record provider success timestamp and latency window
                        if _redis_usable():
                            try:
                                nowts = int(time.time())
                                _redis.lpush(f"prov:reqts:{routed_provider}", nowts)
                                _redis.ltrim(f"prov:reqts:{routed_provider}", 0, 999)
                                _redis.lpush(f"prov:lat:{routed_provider}", dur)
                                _redis.ltrim(f"prov:lat:{routed_provider}", 0, 199)
                            except Exception:
                                _record_redis_failure()
                    except Exception:
                        pass
                except concurrent.futures.TimeoutError:
                    # timeout acts as a failure for CB
                    _cb_failures += 1
                    if _cb_failures >= Config.CB_FAIL_THRESHOLD:
                        _cb_open_until = time.time() + Config.CB_RESET_SECONDS
                    try:
                        PROVIDER_ERRORS_TOTAL.labels(provider=routed_provider).inc()
                    except Exception:
                        pass
                    # record provider error
                    if _redis_usable():
                        try:
                            nowts = int(time.time())
                            _redis.lpush(f"prov:errts:{routed_provider}", nowts)
                            _redis.ltrim(f"prov:errts:{routed_provider}", 0, 499)
                            _redis.set(f"prov:errlast:{routed_provider}", "timeout")
                        except Exception:
                            _record_redis_failure()
                    raise HTTPException(status_code=504, detail="LLM timeout")
        except HTTPException:
            # surfaced timeout
            attempt += 1
            try:
                if attempt < Config.RETRY_MAX_ATTEMPTS:
                    ASK_RETRIES_TOTAL.inc()
            except Exception:
                pass
            if attempt < Config.RETRY_MAX_ATTEMPTS:
                time.sleep(delay)
                delay *= 2
                continue
            else:
                raise
        except Exception:
            # other LLM failures
            attempt += 1
            try:
                if attempt < Config.RETRY_MAX_ATTEMPTS:
                    ASK_RETRIES_TOTAL.inc()
            except Exception:
                pass
            # record provider error
            if _redis_usable():
                try:
                    nowts = int(time.time())
                    _redis.lpush(f"prov:errts:{routed_provider}", nowts)
                    _redis.ltrim(f"prov:errts:{routed_provider}", 0, 499)
                    _redis.set(f"prov:errlast:{routed_provider}", "exception")
                except Exception:
                    _record_redis_failure()
            if attempt < Config.RETRY_MAX_ATTEMPTS:
                time.sleep(delay)
                delay *= 2
                continue
            else:
                raise
        else:
            break
    try:
        deny = set()
        if _redis_usable():
            try:
                members = _redis.smembers("deny:sources")
                deny = set([m.decode("utf-8") if isinstance(m, (bytes, bytearray)) else str(m) for m in members])
            except Exception:
                _record_redis_failure()
                deny = set(_deny_sources_mem)
        else:
            deny = set(_deny_sources_mem)
        if deny:
            docs = [d for d in docs if d.metadata.get("source") not in deny]
        try:
            DENYLIST_SIZE.set(len(deny))
        except Exception:
            pass
    except Exception:
        pass
    # Apply simple metadata filters
    if req.filters:
        try:
            if req.filters.sources:
                allowed = set(req.filters.sources)
                docs = [d for d in docs if d.metadata.get("source") in allowed]
            if req.filters.doc_type:
                dt = req.filters.doc_type
                docs = [d for d in docs if str(d.metadata.get("doc_type", "")) == dt]
            if req.filters.date_from or req.filters.date_to:
                from datetime import datetime
                df = datetime.fromisoformat(req.filters.date_from) if req.filters.date_from else None
                dt_ = datetime.fromisoformat(req.filters.date_to) if req.filters.date_to else None
                def _in_range(meta_date: str) -> bool:
                    try:
                        if not meta_date:
                            return False
                        d = datetime.fromisoformat(str(meta_date)[:10])
                        if df and d < df:
                            return False
                        if dt_ and d > dt_:
                            return False
                        return True
                    except Exception:
                        return False
                docs = [d for d in docs if _in_range(str(d.metadata.get("date", "")))]
        except Exception:
            pass
    # Reranking: ML model if configured, else TF-based if enabled
    # Determine rerank model variant (A/B) per-tenant if configured
    rerank_model_name = Config.RERANK_MODEL
    if Config.RERANK_MODEL_A or Config.RERANK_MODEL_B:
        def _get_rerank_variant(t: str) -> str:
            if _redis_usable():
                try:
                    v = _redis.hget("ab:rerank_model", t)
                    if v in ("A", "B"):
                        return v
                except Exception:
                    _record_redis_failure()
                    pass
            return (_ab_rerank_mem.get(f"model:{t}") or "A")
        v = _get_rerank_variant(tenant_label)
        if v == "B" and Config.RERANK_MODEL_B:
            rerank_model_name = Config.RERANK_MODEL_B
        elif Config.RERANK_MODEL_A:
            rerank_model_name = Config.RERANK_MODEL_A

    if rerank_model_name:
        global _rerank_model
        try:
            # cache reranker instances per model name
            model = _rerank_cache.get(rerank_model_name)
            if model is None:
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer(rerank_model_name)
                _rerank_cache[rerank_model_name] = model
            # encode query and docs, rank by cosine similarity
            # cost guardrail: cap reranked documents
            docs = docs[: max(1, Config.RERANK_MAX_DOCS)]
            texts = [getattr(d, "page_content", "") for d in docs]
            if texts:
                from numpy import dot
                qv = model.encode([q], normalize_embeddings=True)[0]
                dvs = model.encode(texts, normalize_embeddings=True)
                scores = [float(dot(qv, dv)) for dv in dvs]
                pairs = list(zip(scores, docs))
                pairs.sort(key=lambda x: x[0], reverse=True)
                docs = [d for _, d in pairs]
        except Exception:
            # fall back silently
            pass
    elif Config.RERANK_ENABLED:
        q_terms = [t for t in q.lower().split() if t]
        def _score(doc):
            text = getattr(doc, "page_content", "").lower()
            return sum(text.count(t) for t in q_terms)
        try:
            docs = sorted(docs[: max(1, Config.RERANK_MAX_DOCS)], key=_score, reverse=True)
        except Exception:
            pass
    sources = [
        {
            "source": d.metadata.get("source", ""),
            "page": d.metadata.get("page", None),
        }
        for d in docs
    ]
    # PII redaction and policy enforcement on answer
    safe_answer = _redact_pii(result.get("result", ""))
    override = _has_scope(request, "policy:override")
    try:
        pol = pol if 'pol' in locals() else load_policy(_redis if _redis_usable() else None, tenant_label)
        post_answer, post_sources, _pdec = policy_eval_post(safe_answer, sources, pol, override=override)
    except Exception:
        post_answer, post_sources = safe_answer, sources
    # Phase 33: tenant output guardrails (on top of policy post-processing)
    try:
        out_cfg = _load_output_guardrail_cfg(tenant_label)
        guarded_answer, gflags = _apply_output_guardrails(post_answer, out_cfg)
    except Exception:
        guarded_answer, gflags = post_answer, []
    resp = {"answer": guarded_answer, "sources": post_sources}
    if gflags:
        resp["guardrail_flags"] = gflags
    # Phase 15: budget enforcement (record and possibly signal downshift)
    try:
        approx_tokens = _estimate_tokens([q, post_answer])
        action = _budget_check_and_record(tenant_label, approx_tokens, limits.get("daily_usd_cap", 50.0))
        if action != "allow":
            TENANT_BUDGET_ENFORCED_TOTAL.labels(tenant=tenant_label, action=action).inc()
            _trace_event("budget.action", {"tenant": tenant_label, "action": action, "tokens": approx_tokens})
            if action == "reject":
                raise HTTPException(status_code=402, detail="Budget exceeded for tenant")
    except HTTPException:
        raise
    except Exception:
        pass
    # Phase 14: semantic cache write (fine key includes sources simhash)
    if _redis_usable() and Config.SEMANTIC_CACHE_ENABLED:
        try:
            fine_key_raw = {
                "ns": _get_cache_ns(),
                "tenant": tenant_label,
                "model": model_name,
                "rerank": bool(rerank_enabled),
                "q": q,
                "filters": getattr(req.filters, "__dict__", {}) or {},
                "sources": [s.get("source", "") for s in post_sources],
            }
            fine_key = "ask:fine:" + hashlib.sha1(json.dumps(fine_key_raw, sort_keys=True).encode("utf-8")).hexdigest()
            ttl = max(1, _tenant_ttl(tenant_label))
            _redis.setex(fine_key, ttl, json.dumps(resp))
            # also backfill coarse cache if we attempted
            if coarse_cache_key:
                _redis.setex(coarse_cache_key, ttl, json.dumps(resp))
        except Exception:
            _record_redis_failure()
    # Phase 13: HITL low-confidence routing
    def _compute_confidence() -> float:
        try:
            conf = 0.0
            # retrieval signal
            topk = len(docs)
            conf += min(topk, Config.RETRIEVER_K) / float(max(1, Config.RETRIEVER_K)) * 0.2
            # answer length signal
            al = len(post_answer or "")
            conf += min(al, 400) / 400.0 * 0.2
            # hybrid v2 blend margin
            margin = 0.0
            try:
                retr = getattr(chain, "_retriever_ref", None)
                if retr is not None and getattr(retr, "last_blend", None):
                    b = retr.last_blend
                    if len(b) >= 2:
                        margin = max(0.0, float(b[0][0] - b[1][0]))
                        conf += min(margin, 0.5) * 0.4
            except Exception:
                pass
            # rerank presence
            if rerank_enabled:
                conf += 0.2
            return max(0.0, min(1.0, conf))
        except Exception:
            return 0.0
    try:
        cval = _compute_confidence()
        HITL_CONFIDENCE.observe(cval)
        should_sample = (random.random() < max(0.0, min(1.0, float(Config.HITL_SAMPLE_RATE))))
        if Config.HITL_ENABLED and (cval < float(Config.HITL_CONFIDENCE_THRESHOLD) or should_sample):
            item = {
                "id": str(uuid.uuid4()),
                "ts": int(time.time()),
                "tenant": tenant_label,
                "query": q,
                "answer": post_answer,
                "sources": post_sources,
                "confidence": float(cval),
                "llm_model": model_name,
                "rerank_enabled": bool(rerank_enabled),
                "request_id": getattr(getattr(request, "state", None), "request_id", ""),
            }
            if _redis_usable():
                try:
                    _redis.lpush("hitl:queue", json.dumps(item))
                    _redis.ltrim("hitl:queue", 0, 4999)
                    HITL_ENQUEUED_TOTAL.labels(tenant=tenant_label, reason=("low_conf" if cval < float(Config.HITL_CONFIDENCE_THRESHOLD) else "sample")).inc()
                except Exception:
                    _record_redis_failure()
    except Exception:
        pass
    # Explainability: include previews and scores if enabled
    if Config.EXPLAIN_ENABLED:
        try:
            previews = []
            max_chars = max(0, int(Config.EXPLAIN_PREVIEW_CHARS))
            for d in docs:
                txt = getattr(d, "page_content", "") or ""
                previews.append(txt[:max_chars])
            scores = None
            try:
                retr = getattr(chain, "_retriever_ref", None)
                if retr is not None and hasattr(retr, "last_scores") and retr.last_scores:
                    scores = retr.last_scores
            except Exception:
                scores = None
            explain = []
            for i, s in enumerate(sources):
                item = {"source": s.get("source", ""), "page": s.get("page"), "preview": previews[i] if i < len(previews) else ""}
                if scores and i < len(scores):
                    sc = scores[i]
                    item.update({"score": sc.get("blended"), "v_sim": sc.get("v_sim"), "tf": sc.get("tf")})
                explain.append(item)
            resp["explain"] = explain
        except Exception:
            pass
    # Export payload (markdown or json) with citations
    if Config.EXPORT_ENABLED:
        try:
            fmt = (Config.EXPORT_FORMAT or "markdown").lower()
            if fmt == "markdown":
                lines = [safe_answer, "", "References:"]
                for i, s in enumerate(sources, start=1):
                    src = s.get("source", "")
                    pg = s.get("page")
                    ref = f"{i}. {src}"
                    if pg is not None:
                        ref += f" (p.{pg})"
                    lines.append(ref)
                resp["export"] = {"format": "markdown", "content": "\n".join(lines)}
            else:
                resp["export"] = {
                    "format": "json",
                    "content": {
                        "answer": safe_answer,
                        "citations": [{"source": s.get("source", ""), "page": s.get("page")} for s in sources],
                    },
                }
        except Exception:
            pass
    # Cost and token estimation metrics
    if Config.COST_ENABLED:
        try:
            tenant = _tenant_from_key(api_key_hdr)
            # very rough token estimate: ~4 chars per token
            p_tokens = max(1, int(len(q) / 4))
            c_tokens = max(1, int(len(resp.get("answer", "")) / 4))
            TOKENS_PROMPT_TOTAL.labels(tenant=tenant).inc(p_tokens)
            TOKENS_COMPLETION_TOTAL.labels(tenant=tenant).inc(c_tokens)
            cost = (p_tokens / 1000.0) * Config.COST_PER_1K_PROMPT_TOKENS + (c_tokens / 1000.0) * Config.COST_PER_1K_COMPLETION_TOKENS
            COST_USD_TOTAL.labels(tenant=tenant).inc(cost)
        except Exception:
            pass
    # Cache set
    if Config.CACHE_ENABLED:
        try:
            filt = req.filters.dict() if req.filters else {}
        except Exception:
            filt = {}
        ns = _get_cache_ns()
        ckey = f"ask:{ns}:{tenant_label}:{q}:{json.dumps(filt, sort_keys=True)}"
        nkey = f"{ckey}:neg"
        if _redis_usable():
            try:
                # negative caching for empty answers
                if Config.NEGATIVE_CACHE_ENABLED and ((not resp.get("answer")) or (not resp.get("sources"))):
                    _redis.setex(nkey, Config.NEGATIVE_CACHE_TTL_SECONDS, "1")
                else:
                    _redis.setex(ckey, Config.CACHE_TTL_SECONDS, json.dumps(resp))
            except Exception:
                _record_redis_failure()
                pass
        else:
            # negative caching in memory
            if Config.NEGATIVE_CACHE_ENABLED and ((not resp.get("answer")) or (not resp.get("sources"))):
                _cache_neg[nkey] = time.time()
            else:
                _cache[ckey] = {"v": resp, "t": time.time()}
    # Semantic cache set
    if Config.SEMANTIC_CACHE_ENABLED:
        simhash_key = _simhash(q, req.filters)
        skey = f"sem:{ns}:{tenant_label}:{simhash_key}"
        if _redis_usable():
            try:
                _redis.setex(skey, _tenant_ttl(tenant_label), json.dumps(resp))
            except Exception:
                _record_redis_failure()
                pass
        else:
            _sem_cache[skey] = {"v": resp, "t": time.time()}
    # Audit logging (Redis-first, fallback memory)
    try:
        if Config.AUDIT_ENABLED:
            audit = {
                "ts": int(time.time()),
                "tenant": tenant_label,
                "query_len": len(q),
                "answer_len": len(resp.get("answer", "")),
                "sources": [s.get("source", "") for s in sources],
                "request_id": getattr(getattr(request, "state", None), "request_id", ""),
            }
            if _redis_usable():
                try:
                    _redis.rpush("audit:ask", json.dumps(audit))
                except Exception:
                    _record_redis_failure()
                    _audit_mem.append(audit)
            else:
                _audit_mem.append(audit)
    except Exception:
        pass
    # Update budget with cost (if computed)
    try:
        if Config.BUDGET_ENABLED and Config.COST_ENABLED:
            # very rough token estimate is already computed above
            # cost variable is computed in cost block; recompute safely if absent
            if 'cost' not in locals():
                p_tokens = max(1, int(len(q) / 4))
                c_tokens = max(1, int(len(resp.get("answer", "")) / 4))
                cost = (p_tokens / 1000.0) * Config.COST_PER_1K_PROMPT_TOKENS + (c_tokens / 1000.0) * Config.COST_PER_1K_COMPLETION_TOKENS
            _budget_add_spend(tenant_label, float(cost))
    except Exception:
        pass
    try:
        return resp
    finally:
        if span_ctx is not None:
            try:
                span_ctx.__exit__(None, None, None)
            except Exception:
                pass

@app.post(
    "/feedback",
    tags=["Feedback"],
    summary="Submit user feedback",
)
def submit_feedback(req: FeedbackReq, request: Request):
    require_auth(request)
    tenant = _tenant_from_key(request.headers.get("x-api-key"))
    try:
        helpful = bool(req.helpful)
        FEEDBACK_TOTAL.labels(tenant=tenant, helpful=str(helpful).lower()).inc()
    except Exception:
        pass
    fb = {
        "tenant": tenant,
        "helpful": bool(req.helpful),
        "reason": req.reason or "",
        "query": req.query,
        "answer": req.answer,
        "clicked_sources": req.clicked_sources or [],
        "hallucinated": bool(req.hallucinated) if req.hallucinated is not None else None,
        "style_score": req.style_score,
        "tags": req.tags or [],
        "ts": int(time.time()),
    }
    if _redis_usable():
        try:
            _redis.rpush("feedback", json.dumps(fb))
        except Exception:
            _record_redis_failure()
            _feedback_mem.setdefault(tenant, []).append(fb)
    else:
        _feedback_mem.setdefault(tenant, []).append(fb)
    return {"ok": True}

@app.get(
    "/filters",
    tags=["Filters"],
    summary="List available filters for the authenticated tenant",
)
def list_filters(request: Request):
    require_auth(request)
    tenant = _tenant_from_key(request.headers.get("x-api-key"))
    res = {"tenant": tenant, "sources": [], "doc_types": [], "date_min": None, "date_max": None}
    if _redis_usable():
        try:
            s_key = f"filt:{tenant}:sources"
            d_key = f"filt:{tenant}:doc_types"
            res["sources"] = sorted([
                m.decode("utf-8") if isinstance(m, (bytes, bytearray)) else str(m)
                for m in (_redis.smembers(s_key) or set()) if str(m)
            ])
            res["doc_types"] = sorted([
                m.decode("utf-8") if isinstance(m, (bytes, bytearray)) else str(m)
                for m in (_redis.smembers(d_key) or set()) if str(m)
            ])
            res["date_min"] = _redis.get(f"filt:{tenant}:date_min")
            res["date_max"] = _redis.get(f"filt:{tenant}:date_max")
            return res
        except Exception:
            _record_redis_failure()
            return res
    return res

@app.post(
    "/admin/eval/run",
    tags=["Admin"],
    summary="Run offline evaluation over golden set",
)
def admin_eval_run(request: Request):
    require_admin(request)
    res = run_offline_eval()
    try:
        if res.get("ok"):
            OFFLINE_EVAL_RUNS_TOTAL.inc()
            OFFLINE_EVAL_RECALL_LAST.set(float(res.get("recall_at_k", 0.0)))
            OFFLINE_EVAL_ANS_SCORE_LAST.set(float(res.get("avg_answer_contains", 0.0)))
        payload = json.dumps(res)
        if _redis_usable():
            try:
                _redis.rpush("eval:history", payload)
            except Exception:
                _record_redis_failure()
                _eval_mem.append(res)
        else:
            _eval_mem.append(res)
    except Exception:
        pass
    return res

@app.get(
    "/admin/eval/history",
    tags=["Admin"],
    summary="List recent offline eval results",
)
def admin_eval_history(request: Request, limit: int = 20):
    require_admin(request)
    out = []
    if _redis_usable():
        try:
            arr = _redis.lrange("eval:history", -limit, -1) or []
            out = [json.loads(x) for x in arr]
        except Exception:
            _record_redis_failure()
            out = _eval_mem[-limit:]
    else:
        out = _eval_mem[-limit:]
    return {"items": out}

# Phase 8: Canary rerank application based on recent feedback helpful rate
@app.post(
    "/admin/canary/rerank/apply",
    tags=["Admin"],
    summary="Apply canary reranker enable/disable per tenant based on helpful rate",
)
def admin_canary_rerank(request: Request, tenant: str):
    require_admin(request)
    t = (tenant or "").strip()
    if not t:
        raise HTTPException(status_code=400, detail="tenant required")
    window = max(10, int(Config.CANARY_RERANK_WINDOW))
    helpful_rate = None
    try:
        items = []
        if _redis_usable():
            try:
                raw = _redis.lrange("feedback", -2000, -1) or []
                for r in raw[::-1]:
                    try:
                        obj = json.loads(r)
                        if obj.get("tenant") == t:
                            items.append(obj)
                            if len(items) >= window:
                                break
                    except Exception:
                        continue
            except Exception:
                _record_redis_failure()
        else:
            items = _feedback_mem.get(t, [])[-window:]
        if not items:
            raise HTTPException(status_code=400, detail="no feedback available for tenant")
        pos = sum(1 for x in items if bool(x.get("helpful")))
        helpful_rate = pos / float(len(items))
        enable = helpful_rate >= float(Config.CANARY_RERANK_MIN_HELPFUL)
        # write routing
        if _redis_usable():
            try:
                _redis.hset("ab:rerank", t, "true" if enable else "false")
            except Exception:
                _ab_rerank_mem[t] = "true" if enable else "false"
        else:
            _ab_rerank_mem[t] = "true" if enable else "false"
        # record history
        rec = {"tenant": t, "enable": enable, "helpful_rate": helpful_rate, "count": len(items), "ts": int(time.time())}
        if _redis_usable():
            try:
                _redis.rpush("canary:rerank:history", json.dumps(rec))
            except Exception:
                _record_redis_failure()
        return rec
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post(
    "/admin/feedback/export",
    tags=["Admin"],
    summary="Export recent feedback to JSONL and optionally upload to S3",
)
def admin_feedback_export(request: Request, limit: int = 1000, tenant: str | None = None, upload: bool = False):
    require_admin(request)
    try:
        res = export_feedback(limit=limit, tenant=tenant, upload=upload)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get(
    "/admin/feedback/summary",
    tags=["Admin"],
    summary="Summarize recent feedback for a tenant",
)
def admin_feedback_summary(request: Request, tenant: str, limit: int = 500):
    require_admin(request)
    t = (tenant or "").strip()
    if not t:
        raise HTTPException(status_code=400, detail="tenant required")
    items = []
    if _redis_usable():
        try:
            raw = _redis.lrange("feedback", -max(10, int(limit)), -1) or []
            for r in raw[::-1]:
                try:
                    obj = json.loads(r)
                    if obj.get("tenant") != t:
                        continue
                    items.append(obj)
                except Exception:
                    continue
        except Exception:
            _record_redis_failure()
    else:
        items = _feedback_mem.get(t, [])[-limit:]
    if not items:
        return {"ok": True, "tenant": t, "count": 0, "helpful_rate": None, "hallucinated_rate": None, "avg_style_score": None, "top_reasons": [], "top_tags": []}
    total = len(items)
    helpful_cnt = 0
    halluc_cnt = 0
    style_vals = []
    reasons: dict[str, int] = {}
    tags: dict[str, int] = {}
    for it in items:
        try:
            if bool(it.get("helpful")):
                helpful_cnt += 1
            if it.get("hallucinated") is True:
                halluc_cnt += 1
            sc = it.get("style_score")
            if isinstance(sc, (int, float)):
                style_vals.append(float(sc))
            r = (it.get("reason") or "").strip()
            if r:
                reasons[r] = reasons.get(r, 0) + 1
            for tg in it.get("tags") or []:
                try:
                    s = str(tg).strip()
                except Exception:
                    s = ""
                if not s:
                    continue
                tags[s] = tags.get(s, 0) + 1
        except Exception:
            continue
    helpful_rate = helpful_cnt / float(total) if total else None
    halluc_rate = halluc_cnt / float(total) if total else None
    avg_style = (sum(style_vals) / float(len(style_vals))) if style_vals else None
    top_reasons = sorted([{"reason": k, "count": v} for k, v in reasons.items()], key=lambda x: x["count"], reverse=True)[:10]
    top_tags = sorted([{"tag": k, "count": v} for k, v in tags.items()], key=lambda x: x["count"], reverse=True)[:10]
    return {
        "ok": True,
        "tenant": t,
        "count": total,
        "helpful_rate": helpful_rate,
        "hallucinated_rate": halluc_rate,
        "avg_style_score": avg_style,
        "top_reasons": top_reasons,
        "top_tags": top_tags,
    }

@app.get(
    "/admin/tuning/suggestions",
    tags=["Admin"],
    summary="Derive non-destructive tuning suggestions from recent signals",
)
def admin_tuning_suggestions(request: Request, tenant: str, day: str | None = None, limit: int = 500):
    """Suggest router/retrieval/index tweaks based on feedback, hotspots, and trails.

    This endpoint is read-only: it does not modify any config, only returns suggestions.
    """
    require_admin(request)
    t = (tenant or "").strip()
    if not t:
        raise HTTPException(status_code=400, detail="tenant required")
    # Feedback-based signals (reusing logic similar to admin_feedback_summary)
    items = []
    if _redis_usable():
        try:
            raw = _redis.lrange("feedback", -max(10, int(limit)), -1) or []
            for r in raw[::-1]:
                try:
                    obj = json.loads(r)
                    if obj.get("tenant") != t:
                        continue
                    items.append(obj)
                except Exception:
                    continue
        except Exception:
            _record_redis_failure()
    else:
        items = _feedback_mem.get(t, [])[-limit:]
    total_fb = len(items)
    helpful_rate = None
    halluc_rate = None
    if items:
        pos = sum(1 for it in items if bool(it.get("helpful")))
        halluc = sum(1 for it in items if it.get("hallucinated") is True)
        helpful_rate = pos / float(total_fb)
        halluc_rate = halluc / float(total_fb)
    # Hotspots: per-tenant doc_type concentration
    doc_type_counts: dict[str | None, int] = {}
    if _redis_usable():
        try:
            d = day or time.strftime("%Y-%m-%d", time.gmtime())
            key = f"q:vol:{d}"
            h = _redis.hgetall(key) or {}
            for k, v in h.items():
                try:
                    ten, dt = k.split("|", 1)
                except ValueError:
                    ten, dt = k, "__none__"
                if ten != t:
                    continue
                try:
                    c = int(v)
                except Exception:
                    c = 0
                dt_norm: str | None = None if dt == "__none__" else dt
                doc_type_counts[dt_norm] = doc_type_counts.get(dt_norm, 0) + c
        except Exception:
            _record_redis_failure()
    # Trails: intent mix for this tenant
    intent_counts: dict[str, int] = {}
    if _redis_usable():
        try:
            key_trail = f"q:trail:{t}"
            raw_trail = _redis.lrange(key_trail, 0, 199) or []
            for r in raw_trail:
                try:
                    obj = json.loads(r)
                except Exception:
                    continue
                inten = (obj.get("intent") or "").strip().lower()
                if not inten:
                    continue
                intent_counts[inten] = intent_counts.get(inten, 0) + 1
        except Exception:
            _record_redis_failure()
    # Build suggestions
    suggestions = []
    signals = {
        "helpful_rate": helpful_rate,
        "hallucinated_rate": halluc_rate,
        "doc_type_counts": doc_type_counts,
        "intent_counts": intent_counts,
        "feedback_count": total_fb,
    }
    # Suggest router objective tweaks based on helpful/hallucination
    if helpful_rate is not None:
        if helpful_rate < 0.6:
            suggestions.append({
                "kind": "router_policy",
                "action": "consider_quality_objective",
                "reason": "low_helpful_rate",
                "target_objective": "quality",
                "metrics": {"helpful_rate": helpful_rate, "hallucinated_rate": halluc_rate},
            })
        elif helpful_rate > 0.85 and (halluc_rate is None or halluc_rate < 0.05):
            suggestions.append({
                "kind": "router_policy",
                "action": "consider_cost_or_latency_objective",
                "reason": "high_helpful_low_hallucinations",
                "target_objective": "cost",
                "metrics": {"helpful_rate": helpful_rate, "hallucinated_rate": halluc_rate},
            })
    # Suggest retrieval depth tuning based on intent mix
    total_int = sum(intent_counts.values()) or 0
    if total_int > 0:
        frac_trouble = intent_counts.get("troubleshoot", 0) / float(total_int)
        frac_lookup = intent_counts.get("lookup", 0) / float(total_int)
        if frac_trouble > 0.3:
            suggestions.append({
                "kind": "retrieval_depth",
                "action": "increase_k_fetch_k",
                "reason": "high_troubleshoot_intent_share",
                "metrics": {"troubleshoot_share": frac_trouble, "lookup_share": frac_lookup},
            })
        elif frac_lookup > 0.6:
            suggestions.append({
                "kind": "retrieval_depth",
                "action": "consider_reducing_k_for_lookup",
                "reason": "dominant_lookup_intent",
                "metrics": {"troubleshoot_share": frac_trouble, "lookup_share": frac_lookup},
            })
    # Suggest index routing if one doc_type dominates volume
    total_dt = sum(doc_type_counts.values()) or 0
    if total_dt > 0:
        # find dominant doc_type (excluding None)
        best_dt = None
        best_cnt = 0
        for dt, c in doc_type_counts.items():
            if dt is None:
                continue
            if c > best_cnt:
                best_cnt = c
                best_dt = dt
        if best_dt is not None and best_cnt / float(total_dt) >= 0.5:
            suggestions.append({
                "kind": "index_routing",
                "action": "consider_dedicated_collection_for_doc_type",
                "reason": "dominant_doc_type_volume",
                "doc_type": best_dt,
                "metrics": {"doc_type_share": best_cnt / float(total_dt)},
                "hint_index_route_key": _index_route_key(t, best_dt),
            })
    return {"ok": True, "tenant": t, "signals": signals, "suggestions": suggestions}

@app.post(
    "/admin/tuning/apply",
    tags=["Admin"],
    summary="Apply conservative tuning changes based on recent signals",
)
def admin_tuning_apply(request: Request, tenant: str, day: str | None = None, limit: int = 500):
    """Mutate router policy and index routing based on current signals.

    This is an explicit, admin-triggered operation; no automatic background tuning.
    """
    require_admin(request)
    t = (tenant or "").strip()
    if not t:
        raise HTTPException(status_code=400, detail="tenant required")
    # Reuse suggestion logic
    sugg_resp = admin_tuning_suggestions(request, tenant=t, day=day, limit=limit)
    signals = sugg_resp.get("signals", {})
    suggestions = sugg_resp.get("suggestions", [])
    applied = []
    skipped = []
    # Mutate router policy objective if warranted
    for s in suggestions:
        try:
            kind = s.get("kind")
            if kind == "router_policy":
                action = s.get("action")
                target_obj = s.get("target_objective")
                if not target_obj:
                    skipped.append({"suggestion": s, "reason": "no_target_objective"})
                    continue
                # load current router policy and update objective field
                pol = _router_policy(t)
                old_obj = pol.get("objective")
                if str(old_obj or "").lower() == str(target_obj).lower():
                    skipped.append({"suggestion": s, "reason": "already_set"})
                    continue
                pol["objective"] = target_obj
                if _redis_usable():
                    try:
                        _redis.set(f"router:policy:{t}", json.dumps(pol))
                    except Exception:
                        _record_redis_failure()
                        skipped.append({"suggestion": s, "reason": "redis_error"})
                        continue
                applied.append({"kind": kind, "change": {"objective": target_obj}, "from": old_obj})
            elif kind == "index_routing":
                dt = s.get("doc_type")
                hint_key = s.get("hint_index_route_key") or _index_route_key(t, dt)
                if not dt or not hint_key:
                    skipped.append({"suggestion": s, "reason": "missing_doc_type"})
                    continue
                if not _redis_usable():
                    skipped.append({"suggestion": s, "reason": "redis_unusable"})
                    continue
                # only set mapping if not already present to avoid overriding manual config
                try:
                    existing = _redis.hget("index:route", hint_key)
                    if existing:
                        skipped.append({"suggestion": s, "reason": "route_already_set"})
                        continue
                    # suggest collection name convention: <tenant>_<doc_type>
                    coll_name = f"{t}_{dt}".replace(":", "_")
                    _redis.hset("index:route", mapping={hint_key: coll_name})
                    applied.append({"kind": kind, "change": {"route_key": hint_key, "collection": coll_name}})
                except Exception:
                    _record_redis_failure()
                    skipped.append({"suggestion": s, "reason": "redis_error"})
            else:
                skipped.append({"suggestion": s, "reason": "not_applicable"})
        except Exception:
            skipped.append({"suggestion": s, "reason": "exception"})
    return {"ok": True, "tenant": t, "signals": signals, "applied": applied, "skipped": skipped}

# SSE streaming endpoint (flag-gated)
@app.get("/ask/stream")
def ask_stream(query: str, request: Request):
    if not Config.STREAMING_ENABLED:
        raise HTTPException(status_code=404, detail="Streaming disabled")
    # Authorization (optional)
    require_auth(request)
    # Basic rate limiting (reuse same as /ask)
    key = request.headers.get("x-api-key") or (request.client.host if request.client else "unknown")
    now = int(time.time())
    window = now // 60
    if _redis_usable():
        rl_key = f"rl:{key}:{window}"
        try:
            newv = _redis.incr(rl_key)
            if newv == 1:
                _redis.expire(rl_key, 65)
            if newv > Config.RATE_LIMIT_PER_MIN:
                raise HTTPException(status_code=429, detail="Rate limit exceeded")
        except Exception:
            _record_redis_failure()
            st = _rate_state.get(key)
            if not st or st["window"] != window:
                st = {"window": window, "count": 0}
            st["count"] += 1
            _rate_state[key] = st
            if st["count"] > Config.RATE_LIMIT_PER_MIN:
                raise HTTPException(status_code=429, detail="Rate limit exceeded")
    else:
        st = _rate_state.get(key)
        if not st or st["window"] != window:
            st = {"window": window, "count": 0}
        st["count"] += 1
        _rate_state[key] = st
        if st["count"] > Config.RATE_LIMIT_PER_MIN:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")

    q = (query or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Query must not be empty")
    if len(q) > 4000:
        raise HTTPException(status_code=400, detail="Query too long (max 4000 chars)")

    def _gen():
        # compute once, then stream in chunks
        try:
            # respect circuit breaker and timeout logic by reusing same flow as /ask
            # (no retries for stream; we apply single timeout)
            if _cb_open_until and int(time.time()) < _cb_open_until:
                yield f"event: error\ndata: {json.dumps({'error': 'LLM unavailable (circuit open)'})}\n\n"
                return
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(qa_chain.invoke, q)
                try:
                    result = fut.result(timeout=max(1, Config.LLM_TIMEOUT_SECONDS))
                except concurrent.futures.TimeoutError:
                    ASK_TIMEOUTS.inc()
                    yield f"event: error\ndata: {json.dumps({'error': 'LLM timeout'})}\n\n"
                    return
            full = result.get("result", "")
            docs = result.get("source_documents", [])
            # Exclude soft-deleted sources (denylist)
            try:
                deny = set()
                if _redis is not None:
                    try:
                        members = _redis.smembers("deny:sources")
                        deny = set([m.decode("utf-8") if isinstance(m, (bytes, bytearray)) else str(m) for m in members])
                    except Exception:
                        deny = set(_deny_sources_mem)
                else:
                    deny = set(_deny_sources_mem)
                if deny:
                    docs = [d for d in docs if d.metadata.get("source") not in deny]
                try:
                    DENYLIST_SIZE.set(len(deny))
                except Exception:
                    pass
            except Exception:
                pass
            # first send sources metadata
            srcs = []
            for d in docs:
                srcs.append({"source": d.metadata.get("source", ""), "page": d.metadata.get("page", None)})
            yield f"event: sources\ndata: {json.dumps(srcs)}\n\n"
            # optionally send explain previews
            if Config.EXPLAIN_ENABLED:
                try:
                    max_chars = max(0, int(Config.EXPLAIN_PREVIEW_CHARS))
                    previews = []
                    for d in docs:
                        txt = getattr(d, "page_content", "") or ""
                        previews.append({
                            "source": d.metadata.get("source", ""),
                            "page": d.metadata.get("page", None),
                            "preview": txt[:max_chars],
                        })
                    yield f"event: explain\ndata: {json.dumps(previews)}\n\n"
                except Exception:
                    pass
            # Update source index (for retention) with any observed date
            try:
                for d in docs:
                    src = str(d.metadata.get("source", "")).strip()
                    ds = str(d.metadata.get("date", ""))[:10]
                    if not src or not ds:
                        continue
                    try:
                        dt = datetime.fromisoformat(ds)
                        score = int(dt.timestamp())
                        if _redis is not None:
                            try:
                                _redis.zadd("sources:index", {src: score})
                            except Exception:
                                _source_index_mem[src] = score
                        else:
                            _source_index_mem[src] = score
                    except Exception:
                        continue
            except Exception:
                pass
            # stream answer in small chunks (with PII redaction if enabled)
            safe_full = _redact_pii(full)
            chunk = []
            count = 0
            for ch in safe_full.split():
                chunk.append(ch)
                count += len(ch) + 1
                if count >= 128:
                    yield f"data: {' '.join(chunk)}\n\n"
                    chunk = []
                    count = 0
            if chunk:
                yield f"data: {' '.join(chunk)}\n\n"
            yield "event: done\ndata: [DONE]\n\n"
            # record estimated tokens/cost at end
            if Config.COST_ENABLED:
                try:
                    tenant = _tenant_from_key(request.headers.get("x-api-key"))
                    p_tokens = max(1, int(len(q) / 4))
                    c_tokens = max(1, int(len(full) / 4))
                    TOKENS_PROMPT_TOTAL.labels(tenant=tenant).inc(p_tokens)
                    TOKENS_COMPLETION_TOTAL.labels(tenant=tenant).inc(c_tokens)
                    cost = (p_tokens / 1000.0) * Config.COST_PER_1K_PROMPT_TOKENS + (c_tokens / 1000.0) * Config.COST_PER_1K_COMPLETION_TOKENS
                    COST_USD_TOTAL.labels(tenant=tenant).inc(cost)
                except Exception:
                    pass
        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(_gen(), media_type="text/event-stream")

class AdminSoftDeleteReq(BaseModel):
    source: str

class AdminABLLMReq(BaseModel):
    tenant: str
    variant: str  # 'A' or 'B'

class AdminABRerankReq(BaseModel):
    tenant: str
    enabled: bool

@app.post("/admin/docs/delete", tags=["Admin"], summary="Soft-delete a source (denylist)")
def admin_delete(req: AdminSoftDeleteReq, request: Request):
    require_admin(request)
    src = (req.source or "").strip()
    if not src:
        raise HTTPException(status_code=400, detail="source is required")
    ok = True
    if _redis is not None:
        try:
            _redis.sadd("deny:sources", src)
        except Exception:
            ok = False
    if not ok or _redis is None:
        _deny_sources_mem.add(src)
    DOCS_SOFT_DELETES_TOTAL.inc()
    _bump_cache_ns()
    try:
        # update gauge after mutation
        cur = set()
        if _redis is not None:
            try:
                cur = set(_redis.smembers("deny:sources"))
            except Exception:
                cur = set(_deny_sources_mem)
        else:
            cur = set(_deny_sources_mem)
        DENYLIST_SIZE.set(len(cur))
    except Exception:
        pass
    return {"status": "ok", "source": src}

@app.post("/admin/docs/undelete", tags=["Admin"], summary="Remove a source from denylist")
def admin_undelete(req: AdminSoftDeleteReq, request: Request):
    require_admin(request)
    src = (req.source or "").strip()
    if not src:
        raise HTTPException(status_code=400, detail="source is required")
    ok = True
    if _redis is not None:
        try:
            _redis.srem("deny:sources", src)
        except Exception:
            ok = False
    if not ok or _redis is None:
        try:
            _deny_sources_mem.discard(src)
        except Exception:
            pass
    DOCS_SOFT_UNDELETES_TOTAL.inc()
    _bump_cache_ns()
    try:
        cur = set()
        if _redis is not None:
            try:
                cur = set(_redis.smembers("deny:sources"))
            except Exception:
                cur = set(_deny_sources_mem)
        else:
            cur = set(_deny_sources_mem)
        DENYLIST_SIZE.set(len(cur))
    except Exception:
        pass
    return {"status": "ok", "source": src}

@app.post("/admin/ab/llm", tags=["Admin"], summary="Set LLM variant (A/B) for a tenant")
def admin_set_llm(req: AdminABLLMReq, request: Request):
    require_admin(request)
    t = (req.tenant or "").strip()
    v = (req.variant or "").strip().upper()
    if v not in ("A","B") or not t:
        raise HTTPException(status_code=400, detail="variant must be 'A' or 'B' and tenant required")
    try:
        if _redis is not None:
            try:
                _redis.hset("ab:llm", t, v)
            except Exception:
                _ab_llm_mem[t] = v
        else:
            _ab_llm_mem[t] = v
        # Clear cached chains for this tenant's variants is not tenant-specific; clear all caches to be safe
        _qa_cache.clear()
        return {"status": "ok", "tenant": t, "variant": v}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/ab/rerank", tags=["Admin"], summary="Enable/disable reranker for a tenant")
def admin_set_rerank(req: AdminABRerankReq, request: Request):
    require_admin(request)
    t = (req.tenant or "").strip()
    val = bool(req.enabled)
    if not t:
        raise HTTPException(status_code=400, detail="tenant required")
    try:
        if _redis is not None:
            try:
                _redis.hset("ab:rerank", t, "true" if val else "false")
            except Exception:
                _ab_rerank_mem[t] = "true" if val else "false"
        else:
            _ab_rerank_mem[t] = "true" if val else "false"
        _qa_cache.clear()
        return {"status": "ok", "tenant": t, "enabled": val}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Retention sweep metrics
DOCS_RETENTION_SWEEPS_TOTAL = Counter(
    "docs_retention_sweeps_total",
    "Total retention sweeps executed",
)
DOCS_RETENTION_SOFT_DELETES_TOTAL = Counter(
    "docs_retention_soft_deletes_total",
    "Total sources soft-deleted by retention sweeps",
)

@app.post("/admin/retention/sweep", tags=["Admin"], summary="Apply retention window to soft-delete old sources")
def retention_sweep(request: Request):
    require_admin(request)
    days = max(0, int(Config.DOC_RETENTION_DAYS))
    if days <= 0:
        return {"status": "skipped", "reason": "DOC_RETENTION_DAYS=0"}
    cutoff_ts = int((datetime.utcnow() - timedelta(days=days)).timestamp())
    deleted = 0
    try:
        if _redis is not None:
            try:
                olds = _redis.zrangebyscore("sources:index", "-inf", cutoff_ts)
            except Exception:
                olds = []
        else:
            olds = [s for s, sc in _source_index_mem.items() if sc <= cutoff_ts]
    except Exception:
        olds = []
    for src in olds:
        try:
            if _redis is not None:
                try:
                    _redis.sadd("deny:sources", src)
                except Exception:
                    _deny_sources_mem.add(src)
            else:
                _deny_sources_mem.add(src)
            deleted += 1
        except Exception:
            continue
    DOCS_RETENTION_SWEEPS_TOTAL.inc()
    if deleted:
        DOCS_RETENTION_SOFT_DELETES_TOTAL.inc(deleted)
        _bump_cache_ns()
    try:
        cur = set()
        if _redis is not None:
            try:
                cur = set(_redis.smembers("deny:sources"))
            except Exception:
                cur = set(_deny_sources_mem)
        else:
            cur = set(_deny_sources_mem)
        DENYLIST_SIZE.set(len(cur))
    except Exception:
        pass
    return {"status": "ok", "deleted": deleted, "cutoff_ts": cutoff_ts}

@app.get("/system", tags=["System"], summary="System state")
def system_state():
    now_ts = int(time.time())
    open_state = 1 if (_cb_open_until and now_ts < _cb_open_until) else 0
    try:
        CIRCUIT_STATE.labels(component="llm").set(open_state)
    except Exception:
        pass
    # also set denylist gauge on system read
    try:
        cur = set()
        if _redis is not None:
            try:
                cur = set(_redis.smembers("deny:sources"))
            except Exception:
                cur = set(_deny_sources_mem)
        else:
            cur = set(_deny_sources_mem)
        DENYLIST_SIZE.set(len(cur))
    except Exception:
        pass
    return {
        "circuit": {
            "component": "llm",
            "open": bool(open_state),
            "open_until": _cb_open_until,
            "failures": _cb_failures,
            "threshold": Config.CB_FAIL_THRESHOLD,
            "reset_seconds": Config.CB_RESET_SECONDS,
        },
        "ready": READY,
    }

@app.get("/health", response_model=HealthResp, tags=["System"], summary="Liveness probe")
def health():
    return {"status": "healthy"}

@app.get("/ready", response_model=ReadyResp, tags=["System"], summary="Readiness probe")
def ready():
    if READY:
        return {"ready": True}
    # Not ready yet
    return {"ready": False}

@app.get("/usage", response_model=UsageResp, tags=["System"], summary="Usage and quota for caller")
def usage(request: Request):
    require_auth(request)
    api_key_hdr = request.headers.get("x-api-key")
    tenant = _tenant_from_key(api_key_hdr)
    if not Config.QUOTA_ENABLED:
        return {"limit": 0, "used_today": 0}
    # read current without increment
    day = int(time.time()) // 86400
    used = 0
    if _redis is not None:
        try:
            val = _redis.get(f"quota:{tenant}:{day}")
            used = int(val) if val else 0
        except Exception:
            used = 0
    else:
        st = _rate_state.get(f"q:{tenant}") or {}
        used = int(st.get("count", 0)) if st.get("day") == day else 0
    return {"limit": Config.QUOTA_DAILY_LIMIT, "used_today": used}
