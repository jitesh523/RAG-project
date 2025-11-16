import time
import json
import threading
from typing import Optional, Dict, Any

from prometheus_client import Counter, Histogram
from src.config import Config
from src.app.deps import build_chain
from langchain_openai import ChatOpenAI

ONLINE_EVAL_EVENTS = Counter(
    "online_eval_events_total",
    "Online eval events (shadow requests)",
    labelnames=["tenant", "bucket"],
)
ONLINE_EVAL_LATENCY = Histogram(
    "online_eval_latency_seconds",
    "Latency of shadow path",
    labelnames=["bucket"],
)

ONLINE_JUDGE_EVENTS = Counter(
    "online_judge_events_total",
    "LLM-as-judge comparisons recorded",
    labelnames=["tenant", "winner"],
)

# Redis is optional; imported lazily to avoid import cycles in app startup
_redis = None

def _get_redis():
    global _redis
    if _redis is not None:
        return _redis
    try:
        import redis as _r
        if Config.REDIS_URL:
            _redis = _r.Redis.from_url(Config.REDIS_URL, decode_responses=True)
            _redis.ping()
            return _redis
    except Exception:
        _redis = None
    return None

_judge_llm: Optional[ChatOpenAI] = None

def _get_judge_llm() -> Optional[ChatOpenAI]:
    global _judge_llm
    if _judge_llm is not None:
        return _judge_llm
    try:
        if not Config.OPENAI_API_KEY:
            return None
        _judge_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=Config.OPENAI_API_KEY)
        return _judge_llm
    except Exception:
        _judge_llm = None
        return None


def _record_event(obj: Dict[str, Any]):
    try:
        r = _get_redis()
        if r is not None:
            r.lpush("online:eval", json.dumps(obj))
            r.ltrim("online:eval", 0, 9999)
    except Exception:
        pass


def _record_judge(obj: Dict[str, Any]):
    try:
        r = _get_redis()
        if r is not None:
            r.lpush("online:judge", json.dumps(obj))
            r.ltrim("online:judge", 0, 9999)
    except Exception:
        pass


def _run_judge(tenant: str, q: str, control_res: Dict[str, Any], treatment_res: Dict[str, Any]):
    llm = _get_judge_llm()
    if llm is None:
        return
    try:
        c_text = (control_res.get("result") or "") if isinstance(control_res, dict) else ""
        t_text = (treatment_res.get("result") or "") if isinstance(treatment_res, dict) else ""
        if not c_text and not t_text:
            return
        prompt = (
            "You are a strict evaluator for Q&A. Given a question and two answers, "
            "return STRICT JSON with keys 'control_score', 'treatment_score', 'winner'. "
            "Scores are floats from 0 to 1 reflecting helpfulness and correctness. "
            "Winner is 'control', 'treatment', or 'tie'.\n\n" 
            f"Question: {q}\n\nControl answer: {c_text}\n\nTreatment answer: {t_text}\n\n"
        )
        msg = [{"role": "user", "content": prompt}]
        out = llm.invoke(msg)
        txt = getattr(out, "content", "") or ""
        data = {}
        try:
            # best-effort JSON extraction
            start = txt.find("{")
            end = txt.rfind("}")
            if start != -1 and end != -1 and end > start:
                data = json.loads(txt[start : end + 1])
        except Exception:
            data = {}
        c_score = float(data.get("control_score", 0.0))
        t_score = float(data.get("treatment_score", 0.0))
        winner = str(data.get("winner", "tie")).lower()
        if winner not in ("control", "treatment", "tie"):
            if abs(t_score - c_score) < 0.05:
                winner = "tie"
            elif t_score > c_score:
                winner = "treatment"
            else:
                winner = "control"
        _record_judge({
            "ts": int(time.time()),
            "tenant": tenant,
            "q": q,
            "control_score": c_score,
            "treatment_score": t_score,
            "winner": winner,
        })
        try:
            ONLINE_JUDGE_EVENTS.labels(tenant, winner).inc()
        except Exception:
            pass
    except Exception:
        pass


def _shadow_worker(tenant: str, q: str, filters, control_model: str, treatment_model: str, treatment_rerank: bool):
    try:
        # CONTROL: model A, no rerank
        t0 = time.time()
        try:
            c_chain = build_chain(filters=filters, llm_model=control_model, rerank_enabled=False)
            c_res = c_chain.invoke(q)
            ONLINE_EVAL_LATENCY.labels("control").observe(time.time() - t0)
            ONLINE_EVAL_EVENTS.labels(tenant, "control").inc()
        except Exception as e:
            c_res = {"error": str(e)}
        # TREATMENT: configured model, rerank flag
        t1 = time.time()
        try:
            t_chain = build_chain(filters=filters, llm_model=treatment_model, rerank_enabled=treatment_rerank)
            t_res = t_chain.invoke(q)
            ONLINE_EVAL_LATENCY.labels("treatment").observe(time.time() - t1)
            ONLINE_EVAL_EVENTS.labels(tenant, "treatment").inc()
        except Exception as e:
            t_res = {"error": str(e)}
        evt = {
            "ts": int(time.time()),
            "tenant": tenant,
            "query": q,
            "filters": getattr(filters, "__dict__", {}) or {},
            "control": {
                "model": control_model,
                "rerank": False,
                "result": c_res.get("result") if isinstance(c_res, dict) else None,
                "sources": [getattr(d, "metadata", {}).get("source", "") for d in (c_res.get("source_documents") or [])] if isinstance(c_res, dict) else [],
                "error": c_res.get("error") if isinstance(c_res, dict) else None,
            },
            "treatment": {
                "model": treatment_model,
                "rerank": bool(treatment_rerank),
                "result": t_res.get("result") if isinstance(t_res, dict) else None,
                "sources": [getattr(d, "metadata", {}).get("source", "") for d in (t_res.get("source_documents") or [])] if isinstance(t_res, dict) else [],
                "error": t_res.get("error") if isinstance(t_res, dict) else None,
            },
        }
        _record_event(evt)
        # LLM-as-judge scoring
        _run_judge(tenant, q, evt["control"], evt["treatment"])
    except Exception:
        pass


def run_shadow_eval(tenant: str, q: str, filters, treatment_model: str, treatment_rerank: bool, control_model: Optional[str] = None):
    """
    Fire-and-forget shadow evaluation comparing control vs treatment.
    control_model defaults to Config.LLM_MODEL_A; treatment is the live routing.
    """
    try:
        if not Config.ONLINE_EVAL_ENABLED:
            return
        c_model = control_model or Config.LLM_MODEL_A
        t = threading.Thread(
            target=_shadow_worker,
            args=(tenant, q, filters, c_model, treatment_model, bool(treatment_rerank)),
            daemon=True,
        )
        t.start()
    except Exception:
        pass
