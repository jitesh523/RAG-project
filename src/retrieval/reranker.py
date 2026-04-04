"""ML-based reranker using SentenceTransformers cross-encoder.

Supports:
  - BAAI/bge-reranker-base (default)
  - Any cross-encoder compatible model via RERANK_MODEL config
  - Graceful fallback to TF-based scoring when ST is unavailable
"""

import logging
import time
from typing import List, Tuple, Any, Dict

from prometheus_client import Histogram

logger = logging.getLogger(__name__)

RERANK_LATENCY = Histogram(
    "reranker_duration_seconds",
    "Time spent reranking documents",
    labelnames=["model"],
)

_cross_encoder_cache: Dict[str, Any] = {}


def _load_cross_encoder(model_name: str):
    """Lazy-load and cache a CrossEncoder model."""
    if model_name in _cross_encoder_cache:
        return _cross_encoder_cache[model_name]
    try:
        from sentence_transformers import CrossEncoder  # type: ignore

        ce = CrossEncoder(model_name)
        _cross_encoder_cache[model_name] = ce
        logger.info("Loaded reranker model: %s", model_name)
        return ce
    except Exception as e:
        logger.warning("Failed to load reranker model %s: %s", model_name, e)
        return None


def _tf_score(query: str, text: str) -> float:
    """Fallback TF scoring when ML reranker unavailable."""
    terms = [t for t in (query or "").lower().split() if t]
    if not terms:
        return 0.0
    body = (text or "").lower()
    return float(sum(body.count(t) for t in terms))


def rerank_documents(
    query: str,
    docs: List[Any],
    model_name: str = "BAAI/bge-reranker-base",
    top_k: int | None = None,
) -> List[Tuple[float, Any]]:
    """Rerank a list of LangChain Document objects by relevance to the query.

    Args:
        query: The user query string.
        docs: List of Document objects (with .page_content attribute).
        model_name: HuggingFace model id for the cross-encoder.
        top_k: If set, return only the top-k documents after reranking.

    Returns:
        List of (score, doc) tuples sorted descending by score.
    """
    if not docs:
        return []

    t0 = time.time()
    label = model_name.split("/")[-1] if model_name else "tf"
    texts = [getattr(d, "page_content", "") for d in docs]

    ce = _load_cross_encoder(model_name) if model_name else None

    if ce is not None:
        try:
            pairs = [(query, t) for t in texts]
            scores = ce.predict(pairs)
            scored = list(zip([float(s) for s in scores], docs))
        except Exception as e:
            logger.warning("CrossEncoder prediction failed, falling back to TF: %s", e)
            scored = [(_tf_score(query, t), d) for t, d in zip(texts, docs)]
    else:
        scored = [(_tf_score(query, t), d) for t, d in zip(texts, docs)]

    scored.sort(key=lambda x: x[0], reverse=True)
    if top_k is not None:
        scored = scored[:top_k]

    RERANK_LATENCY.labels(label).observe(time.time() - t0)
    return scored
