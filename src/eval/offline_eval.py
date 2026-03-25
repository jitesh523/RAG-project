import json
import time
from types import SimpleNamespace
from typing import Dict, Any, List
from prometheus_client import Gauge, Counter, pushadd_to_gateway, REGISTRY
import hashlib
from src.config import Config
from src.app.deps import build_chain

# Offline evaluation over a golden set
# Golden JSONL schema per line: {"tenant": "t1", "query": "...", "expected_sources": ["foo.pdf"], "expected_answer_contains": ["term1", "term2"]}


def _load_golden(path: str) -> List[Dict[str, Any]]:
    items = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    items.append(obj)
                except Exception:
                    continue
    except FileNotFoundError:
        return []
    return items


def _make_filters(tenant: str):
    # Minimal filters object compatible with deps._milvus_expr_from_filters/_faiss_filter
    return SimpleNamespace(tenant=tenant)


def _derive_doc_type_from_source(src: str) -> str:
    try:
        s = (src or "").lower()
        if s.endswith(".pdf"):
            return "pdf"
        if s.endswith(".html") or s.endswith(".htm"):
            return "html"
        if s.endswith(".docx"):
            return "docx"
        return "other"
    except Exception:
        return "other"


def _golden_hash(path: str) -> str:
    try:
        h = hashlib.sha1()
        with open(path, "rb") as f:
            while True:
                b = f.read(8192)
                if not b:
                    break
                h.update(b)
        return h.hexdigest()
    except Exception:
        return ""


def run_offline_eval(golden_path: str | None = None) -> Dict[str, Any]:
    path = golden_path or Config.EVAL_GOLDEN_PATH
    data = _load_golden(path)
    if not data:
        return {"ok": False, "reason": f"no golden data at {path}"}
    total = len(data)
    hits = 0
    answer_scores = 0.0
    per = []
    # slicing accumulators
    by_tenant = {}
    by_dtype = {}
    for item in data:
        t = str(item.get("tenant", "")).strip() or "__default__"
        q = str(item.get("query", ""))
        exp_sources = list(item.get("expected_sources", []) or [])
        exp_ans_parts = list(item.get("expected_answer_contains", []) or [])
        exp_dtype = str(item.get("expected_doc_type", "")).strip()
        try:
            chain = build_chain(filters=_make_filters(t))
            res = chain.invoke(q)
            ans = res.get("result", "") or ""
            docs = res.get("source_documents", []) or []
            got_sources = [getattr(d, "metadata", {}).get("source", "") for d in docs]
            # attempt a doc_type using the first expected source
            dtype = exp_dtype or _derive_doc_type_from_source(
                exp_sources[0] if exp_sources else ""
            )
            # retrieval hit
            hit = 1 if (set(exp_sources) & set(got_sources)) else 0
            hits += hit
            # answer contain score
            score = 0.0
            if exp_ans_parts:
                score = sum(
                    1 for p in exp_ans_parts if p and (p.lower() in ans.lower())
                ) / float(len(exp_ans_parts))
            answer_scores += score
            per.append(
                {
                    "tenant": t,
                    "query": q,
                    "hit": hit,
                    "answer_score": score,
                    "got_sources": got_sources,
                    "doc_type": dtype,
                }
            )
            # accumulate slices
            bt = by_tenant.setdefault(t, {"n": 0, "hits": 0, "ans": 0.0})
            bt["n"] += 1
            bt["hits"] += hit
            bt["ans"] += score
            bd = by_dtype.setdefault(dtype, {"n": 0, "hits": 0, "ans": 0.0})
            bd["n"] += 1
            bd["hits"] += hit
            bd["ans"] += score
        except Exception as e:
            per.append({"tenant": t, "query": q, "error": str(e)})
            continue
    recall = hits / float(total) if total else 0.0
    avg_answer = answer_scores / float(total) if total else 0.0
    # compute slice metrics
    slices = {"tenant": {}, "doc_type": {}}
    for k, v in by_tenant.items():
        n = max(1, v["n"])
        slices["tenant"][k] = {
            "recall_at_k": v["hits"] / float(n),
            "avg_answer_contains": v["ans"] / float(n),
            "n": v["n"],
        }
    for k, v in by_dtype.items():
        n = max(1, v["n"])
        slices["doc_type"][k] = {
            "recall_at_k": v["hits"] / float(n),
            "avg_answer_contains": v["ans"] / float(n),
            "n": v["n"],
        }
    ghash = _golden_hash(path)
    out = {
        "ok": True,
        "total": total,
        "recall_at_k": recall,
        "avg_answer_contains": avg_answer,
        "ts": int(time.time()),
        "golden_path": path,
        "golden_hash": ghash,
        "slices": slices,
        "details": per[:50],  # cap preview
    }
    # Optionally push summary to Pushgateway
    if Config.PUSHGATEWAY_URL:
        try:
            g_recall = Gauge(
                "offline_eval_recall", "Offline eval recall@k", registry=REGISTRY
            )
            g_ans = Gauge(
                "offline_eval_answer_contains",
                "Offline eval answer contains score",
                registry=REGISTRY,
            )
            c_runs = Counter(
                "offline_eval_runs_total", "Offline eval runs", registry=REGISTRY
            )
            g_recall.set(recall)
            g_ans.set(avg_answer)
            c_runs.inc()
            # slice metrics with labels
            g_slice_recall = Gauge(
                "offline_eval_slice_recall",
                "Recall@k per slice",
                ["slice_type", "slice_key", "golden_hash"],
                registry=REGISTRY,
            )
            g_slice_ans = Gauge(
                "offline_eval_slice_answer_contains",
                "Answer-contains per slice",
                ["slice_type", "slice_key", "golden_hash"],
                registry=REGISTRY,
            )
            for k, v in slices.get("tenant", {}).items():
                g_slice_recall.labels("tenant", k, ghash).set(v["recall_at_k"])
                g_slice_ans.labels("tenant", k, ghash).set(v["avg_answer_contains"])
            for k, v in slices.get("doc_type", {}).items():
                g_slice_recall.labels("doc_type", k, ghash).set(v["recall_at_k"])
                g_slice_ans.labels("doc_type", k, ghash).set(v["avg_answer_contains"])
            pushadd_to_gateway(
                Config.PUSHGATEWAY_URL,
                job=Config.EVAL_PUSHGATEWAY_JOB,
                registry=REGISTRY,
            )
        except Exception:
            pass
    return out


if __name__ == "__main__":
    res = run_offline_eval()
    print(json.dumps(res, indent=2))
