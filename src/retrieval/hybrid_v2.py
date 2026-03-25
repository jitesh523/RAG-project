from typing import List, Tuple, Dict, Any

from rank_bm25 import BM25Okapi
from prometheus_client import Histogram, Counter

RETR_STAGE_LATENCY = Histogram(
    "retrieval_stage_duration_seconds",
    "Latency by retrieval stage",
    labelnames=["stage"],
)
RETR_STAGE_HITS = Counter(
    "retrieval_stage_hits_total",
    "Candidates produced by stage",
    labelnames=["stage"],
)


class BM25Adapter:
    def __init__(self, corpus_docs: List[str], meta: List[Dict[str, Any]]):
        # Simple in-memory tokenization
        self._meta = meta
        self._tokenized = [self._tokenize(t) for t in corpus_docs]
        self._bm25 = BM25Okapi(self._tokenized)

    def _tokenize(self, text: str) -> List[str]:
        return [t for t in (text or "").lower().split() if t]

    def search(self, query: str, k: int = 20) -> List[Tuple[float, Dict[str, Any]]]:
        import time

        t0 = time.time()
        scores = self._bm25.get_scores(self._tokenize(query or ""))
        pairs = list(enumerate(scores))
        pairs.sort(key=lambda x: x[1], reverse=True)
        out = []
        for idx, sc in pairs[:k]:
            out.append((float(sc), self._meta[idx]))
        RETR_STAGE_LATENCY.labels("bm25").observe(time.time() - t0)
        RETR_STAGE_HITS.labels("bm25").inc(len(out))
        return out


def blend_candidates(
    dense: List[Tuple[float, Dict[str, Any]]],
    bm25: List[Tuple[float, Dict[str, Any]]],
    weights: Dict[str, float] = None,
    k: int = 10,
) -> List[Tuple[float, Dict[str, Any]]]:
    import time

    t0 = time.time()
    weights = weights or {"dense": 0.6, "bm25": 0.4}
    by_key: Dict[str, Dict[str, Any]] = {}
    by_score: Dict[str, float] = {}

    # normalize scores
    def _norm(
        arr: List[Tuple[float, Dict[str, Any]]],
    ) -> List[Tuple[float, Dict[str, Any]]]:
        if not arr:
            return []
        xs = [s for s, _ in arr]
        lo, hi = min(xs), max(xs)
        if hi == lo:
            return [(1.0, m) for (_, m) in arr]
        return [((s - lo) / (hi - lo), m) for (s, m) in arr]

    dense_n = _norm(dense)
    bm25_n = _norm(bm25)
    for s, m in dense_n:
        key = f"{m.get('source', '')}#{m.get('page', '')}"
        by_key.setdefault(key, m)
        by_score[key] = by_score.get(key, 0.0) + weights.get("dense", 0.6) * float(s)
    for s, m in bm25_n:
        key = f"{m.get('source', '')}#{m.get('page', '')}"
        by_key.setdefault(key, m)
        by_score[key] = by_score.get(key, 0.0) + weights.get("bm25", 0.4) * float(s)
    items = [(sc, by_key[k]) for k, sc in by_score.items()]
    items.sort(key=lambda x: x[0], reverse=True)
    out = items[:k]
    from time import time as _now

    RETR_STAGE_LATENCY.labels("blend").observe(_now() - t0)
    RETR_STAGE_HITS.labels("blend").inc(len(out))
    return out
