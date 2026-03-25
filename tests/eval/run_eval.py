import os
import json
import sys
from typing import List, Dict

import requests


def ask_http(base_url: str, api_key: str | None, q: str) -> Dict:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["x-api-key"] = api_key
    r = requests.post(
        f"{base_url.rstrip('/')}/ask", json={"query": q}, headers=headers, timeout=60
    )
    r.raise_for_status()
    return r.json()


def ask_local(q: str) -> Dict:
    # Local in-process using TestClient
    from fastapi.testclient import TestClient
    from src.app import fastapi_app as appmod

    appmod.READY = True if getattr(appmod, "qa_chain", None) else appmod.READY
    client = TestClient(appmod.app)
    r = client.post("/ask", json={"query": q})
    r.raise_for_status()
    return r.json()


def load_golden(path: str) -> List[Dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def recall_at_k(
    expected_sources: List[str], got_sources: List[str], k: int = 3
) -> float:
    got_k = got_sources[:k]
    for src in expected_sources:
        if src in got_k:
            return 1.0
    return 0.0


def keyword_hit_rate(keywords: List[str], answer: str) -> float:
    if not keywords:
        return 1.0
    words = answer.lower()
    hits = sum(1 for kw in keywords if kw.lower() in words)
    return hits / len(keywords)


def main():
    base_url = os.getenv("EVAL_BASE_URL")  # if set, call HTTP; else local TestClient
    api_key = os.getenv("API_KEY")
    golden_path = os.getenv("GOLDEN_PATH", "tests/eval/golden.jsonl")
    k = int(os.getenv("EVAL_K", "3"))

    golden = load_golden(golden_path)
    results = []
    rec_sum = 0.0
    kw_sum = 0.0

    for row in golden:
        q = row["query"]
        exp = row.get("expected_sources", [])
        kws = row.get("keywords", [])
        try:
            if base_url:
                resp = ask_http(base_url, api_key, q)
            else:
                resp = ask_local(q)
        except Exception as e:
            results.append({"query": q, "error": str(e)})
            continue
        sources = [s.get("source", "") for s in resp.get("sources", [])]
        answer = resp.get("answer", "")
        r = recall_at_k(exp, sources, k)
        kh = keyword_hit_rate(kws, answer)
        rec_sum += r
        kw_sum += kh
        results.append(
            {
                "query": q,
                "recall@%d" % k: r,
                "keyword_hit": kh,
                "sources": sources,
            }
        )

    n = max(1, len(golden))
    summary = {
        "n": len(golden),
        "avg_recall@%d" % k: rec_sum / n,
        "avg_keyword_hit": kw_sum / n,
        "details": results,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    sys.exit(main())
