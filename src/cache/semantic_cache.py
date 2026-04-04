"""Persistent Semantic Cache backed by Redis.

Features:
  - Exact-match caching via SHA256 key (fast path)
  - Embedding-based similarity matching via cosine similarity (optional, slower path)
  - Per-tenant TTL configuration
  - Cache namespace versioning for instant invalidation
  - Prometheus metrics for hit/miss rates
  - Graceful fallback to in-memory dict when Redis is unavailable
"""

import hashlib
import json
import logging
import math
import time
from typing import Any, Dict, List, Optional, Tuple

from prometheus_client import Counter

logger = logging.getLogger(__name__)

CACHE_HITS = Counter(
    "semantic_cache_hits_total",
    "Number of semantic cache hits",
    labelnames=["tenant", "kind"],
)
CACHE_MISSES = Counter(
    "semantic_cache_misses_total",
    "Number of semantic cache misses",
    labelnames=["tenant"],
)
CACHE_WRITES = Counter(
    "semantic_cache_writes_total",
    "Number of writes to semantic cache",
    labelnames=["tenant"],
)
CACHE_INVALIDATIONS = Counter(
    "semantic_cache_invalidations_total",
    "Number of cache namespace invalidations",
)


def _cosine_sim(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two equal-length vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _exact_key(query: str, filters: dict, namespace: int, tenant: str) -> str:
    raw = json.dumps(
        {"ns": namespace, "tenant": tenant, "q": query, "f": filters}, sort_keys=True
    )
    return "sem:exact:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _embed_index_key(namespace: int, tenant: str) -> str:
    return f"sem:embedidx:{namespace}:{tenant}"


class SemanticCache:
    """Two-tier Redis-backed semantic cache.

    Tier 1: Exact key match (O(1) Redis GET).
    Tier 2: Embedding similarity scan over stored query vectors (optional).

    Args:
        redis_client: A configured redis.Redis instance (or None for in-memory fallback).
        default_ttl: Default TTL in seconds for cached responses.
        similarity_threshold: Cosine similarity threshold for semantic hits (0.0-1.0).
        tenant_ttls: Optional dict of {tenant: ttl_seconds} overrides.
    """

    def __init__(
        self,
        redis_client=None,
        default_ttl: int = 300,
        similarity_threshold: float = 0.92,
        tenant_ttls: Optional[Dict[str, int]] = None,
    ):
        self._r = redis_client
        self._default_ttl = default_ttl
        self._similarity_threshold = similarity_threshold
        self._tenant_ttls = tenant_ttls or {}
        self._mem: Dict[str, Tuple[Any, float]] = {}  # in-memory fallback

    def _ttl(self, tenant: str) -> int:
        return self._tenant_ttls.get(tenant, self._default_ttl)

    def _get_namespace(self) -> int:
        if self._r is not None:
            try:
                v = self._r.get("sem:ns")
                return int(v) if v else 1
            except Exception as e:
                logger.debug("Redis namespace read failed: %s", e)
        return 1

    def invalidate(self) -> None:
        """Invalidate all cache entries by incrementing the namespace."""
        if self._r is not None:
            try:
                self._r.incr("sem:ns")
                CACHE_INVALIDATIONS.inc()
                logger.info("Semantic cache invalidated (namespace bumped)")
                return
            except Exception as e:
                logger.warning("Cache invalidation failed: %s", e)
        # In-memory fallback: clear dict
        self._mem.clear()
        CACHE_INVALIDATIONS.inc()

    def get(
        self,
        query: str,
        filters: dict,
        tenant: str = "__default__",
        query_embedding: Optional[List[float]] = None,
    ) -> Optional[Any]:
        """Look up a cached response.

        First tries exact match, then optional embedding-based similarity.

        Returns:
            Cached response dict, or None on miss.
        """
        ns = self._get_namespace()
        ekey = _exact_key(query, filters, ns, tenant)

        # --- Tier 1: exact match ---
        if self._r is not None:
            try:
                cached = self._r.get(ekey)
                if cached:
                    CACHE_HITS.labels(tenant=tenant, kind="exact").inc()
                    return json.loads(cached)
            except Exception as e:
                logger.debug("Redis GET failed: %s", e)
        else:
            entry = self._mem.get(ekey)
            if entry:
                val, ts = entry
                if time.time() - ts < self._ttl(tenant):
                    CACHE_HITS.labels(tenant=tenant, kind="exact").inc()
                    return val

        # --- Tier 2: embedding similarity (optional) ---
        if query_embedding and self._r is not None:
            try:
                idx_key = _embed_index_key(ns, tenant)
                raw_index = self._r.get(idx_key)
                if raw_index:
                    index: List[Dict] = json.loads(raw_index)
                    best_score = 0.0
                    best_response = None
                    for entry in index:
                        emb = entry.get("embedding", [])
                        score = _cosine_sim(query_embedding, emb)
                        if score > best_score:
                            best_score = score
                            best_response = entry.get("response")
                    if best_score >= self._similarity_threshold and best_response:
                        CACHE_HITS.labels(tenant=tenant, kind="semantic").inc()
                        logger.debug(
                            "Semantic cache hit (cosine=%.3f, tenant=%s)",
                            best_score,
                            tenant,
                        )
                        return best_response
            except Exception as e:
                logger.debug("Semantic similarity lookup failed: %s", e)

        CACHE_MISSES.labels(tenant=tenant).inc()
        return None

    def set(
        self,
        query: str,
        filters: dict,
        response: Any,
        tenant: str = "__default__",
        query_embedding: Optional[List[float]] = None,
    ) -> None:
        """Store a response in cache with exact key, and optionally index the embedding."""
        ns = self._get_namespace()
        ekey = _exact_key(query, filters, ns, tenant)
        ttl = self._ttl(tenant)
        serialized = json.dumps(response)

        if self._r is not None:
            try:
                self._r.setex(ekey, ttl, serialized)
                CACHE_WRITES.labels(tenant=tenant).inc()
            except Exception as e:
                logger.debug("Redis SET failed: %s", e)
                return
            # Store embedding in per-tenant index for Tier 2 lookups
            if query_embedding:
                try:
                    idx_key = _embed_index_key(ns, tenant)
                    raw_index = self._r.get(idx_key)
                    index: List[Dict] = json.loads(raw_index) if raw_index else []
                    # Deduplicate by exact key
                    index = [e for e in index if e.get("key") != ekey]
                    index.append(
                        {
                            "key": ekey,
                            "query": query,
                            "embedding": query_embedding,
                            "response": response,
                        }
                    )
                    # Keep index bounded (latest 200 entries)
                    if len(index) > 200:
                        index = index[-200:]
                    self._r.setex(idx_key, ttl * 2, json.dumps(index))
                except Exception as e:
                    logger.debug("Failed to update embedding index: %s", e)
        else:
            # In-memory fallback
            self._mem[ekey] = (response, time.time())
            CACHE_WRITES.labels(tenant=tenant).inc()
