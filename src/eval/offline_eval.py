import concurrent.futures

def _process_item(item: Dict[str, Any]) -> Dict[str, Any]:
    t = str(item.get("tenant", "")).strip() or "__default__"
    q = str(item.get("query", ""))
    exp_sources = list(item.get("expected_sources", []) or [])
    exp_ans_parts = list(item.get("expected_answer_contains", []) or [])
    exp_dtype = str(item.get("expected_doc_type", "")).strip()
    try:
        chain = build_chain(filters=_make_filters(t))
        # This is a network/LLM bound operation, so threading is effective
        res = chain.invoke(q)
        ans = res.get("result", "") or ""
        docs = res.get("source_documents", []) or []
        got_sources = [getattr(d, "metadata", {}).get("source", "") for d in docs]
        dtype = exp_dtype or _derive_doc_type_from_source(
            exp_sources[0] if exp_sources else ""
        )
        hit = 1 if (set(exp_sources) & set(got_sources)) else 0
        score = 0.0
        if exp_ans_parts:
            score = sum(
                1 for p in exp_ans_parts if p and (p.lower() in ans.lower())
            ) / float(len(exp_ans_parts))
        return {
            "tenant": t,
            "query": q,
            "hit": hit,
            "answer_score": score,
            "got_sources": got_sources,
            "doc_type": dtype,
            "ok": True,
        }
    except Exception as e:
        return {"tenant": t, "query": q, "error": str(e), "ok": False}


def run_offline_eval(golden_path: str | None = None) -> Dict[str, Any]:
    path = golden_path or Config.EVAL_GOLDEN_PATH
    data = _load_golden(path)
    if not data:
        return {"ok": False, "reason": f"no golden data at {path}"}
    total = len(data)
    hits = 0
    answer_scores = 0.0
    per = []
    by_tenant = {}
    by_dtype = {}

    max_workers = min(10, total) if total > 0 else 1
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(_process_item, data))

    for res in results:
        t = res["tenant"]
        q = res["query"]
        if not res.get("ok"):
            per.append({"tenant": t, "query": q, "error": res.get("error")})
            continue

        hit = res["hit"]
        score = res["answer_score"]
        dtype = res["doc_type"]
        hits += hit
        answer_scores += score
        per.append(res)

        # accumulate slices
        bt = by_tenant.setdefault(t, {"n": 0, "hits": 0, "ans": 0.0})
        bt["n"] += 1
        bt["hits"] += hit
        bt["ans"] += score
        bd = by_dtype.setdefault(dtype, {"n": 0, "hits": 0, "ans": 0.0})
        bd["n"] += 1
        bd["hits"] += hit
        bd["ans"] += score

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
        except Exception as e:
            logger.debug("Silent exception (pass): %s", e)
    return out


if __name__ == "__main__":
    res = run_offline_eval()
    print(json.dumps(res, indent=2))
