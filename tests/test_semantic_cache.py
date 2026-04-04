"""Tests for the SemanticCache module."""

import time

import pytest

from src.cache.semantic_cache import SemanticCache, _cosine_sim, _exact_key


# ── helpers ─────────────────────────────────────────────────────────────────


class FakeRedis:
    """Minimal in-memory Redis stub for testing."""

    def __init__(self):
        self._store: dict = {}
        self._expiry: dict = {}

    def get(self, key):
        if key in self._expiry and time.time() > self._expiry[key]:
            del self._store[key]
            del self._expiry[key]
        return self._store.get(key)

    def set(self, key, value):
        self._store[key] = value

    def setex(self, key, ttl, value):
        self._store[key] = value
        self._expiry[key] = time.time() + ttl

    def incr(self, key):
        current = int(self._store.get(key, 0))
        self._store[key] = str(current + 1)
        return current + 1


# ── _cosine_sim ──────────────────────────────────────────────────────────────


class TestCosineSim:
    def test_identical_vectors(self):
        v = [1.0, 0.0, 0.0]
        assert _cosine_sim(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert _cosine_sim([1, 0], [0, 1]) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        result = _cosine_sim([1, 0], [-1, 0])
        assert result == pytest.approx(-1.0)

    def test_empty_vectors(self):
        assert _cosine_sim([], []) == 0.0

    def test_mismatched_lengths(self):
        assert _cosine_sim([1, 2], [1]) == 0.0


# ── exact key ───────────────────────────────────────────────────────────────


class TestExactKey:
    def test_deterministic(self):
        k1 = _exact_key("hello", {}, 1, "t1")
        k2 = _exact_key("hello", {}, 1, "t1")
        assert k1 == k2

    def test_different_query_different_key(self):
        k1 = _exact_key("hello", {}, 1, "t1")
        k2 = _exact_key("world", {}, 1, "t1")
        assert k1 != k2

    def test_different_namespace_different_key(self):
        k1 = _exact_key("hello", {}, 1, "t1")
        k2 = _exact_key("hello", {}, 2, "t1")
        assert k1 != k2

    def test_different_tenant_different_key(self):
        k1 = _exact_key("hello", {}, 1, "t1")
        k2 = _exact_key("hello", {}, 1, "t2")
        assert k1 != k2


# ── SemanticCache – in-memory fallback ──────────────────────────────────────


class TestSemanticCacheInMemory:
    """All tests without a Redis client use the in-memory fallback."""

    def setup_method(self):
        self.cache = SemanticCache(redis_client=None, default_ttl=5)

    def test_miss_on_empty(self):
        result = self.cache.get("q", {}, tenant="t1")
        assert result is None

    def test_set_then_get(self):
        self.cache.set("q", {}, {"answer": "42"}, tenant="t1")
        result = self.cache.get("q", {}, tenant="t1")
        assert result == {"answer": "42"}

    def test_different_queries_isolated(self):
        self.cache.set("q1", {}, {"answer": "A"}, tenant="t1")
        self.cache.set("q2", {}, {"answer": "B"}, tenant="t1")
        assert self.cache.get("q1", {}, tenant="t1") == {"answer": "A"}
        assert self.cache.get("q2", {}, tenant="t1") == {"answer": "B"}

    def test_ttl_expiry(self):
        cache = SemanticCache(redis_client=None, default_ttl=1)
        cache.set("q", {}, {"answer": "ephemeral"}, tenant="t1")
        assert cache.get("q", {}, tenant="t1") is not None
        time.sleep(1.1)
        assert cache.get("q", {}, tenant="t1") is None

    def test_invalidate_clears_all(self):
        self.cache.set("q1", {}, {"answer": "A"}, tenant="t1")
        self.cache.set("q2", {}, {"answer": "B"}, tenant="t1")
        self.cache.invalidate()
        assert self.cache.get("q1", {}, tenant="t1") is None
        assert self.cache.get("q2", {}, tenant="t1") is None

    def test_per_tenant_ttl(self):
        cache = SemanticCache(
            redis_client=None, default_ttl=60, tenant_ttls={"fast_tenant": 1}
        )
        cache.set("q", {}, {"answer": "X"}, tenant="fast_tenant")
        cache.set("q", {}, {"answer": "Y"}, tenant="slow_tenant")
        time.sleep(1.1)
        # fast_tenant entry expired
        assert cache.get("q", {}, tenant="fast_tenant") is None
        # slow_tenant still alive
        assert cache.get("q", {}, tenant="slow_tenant") == {"answer": "Y"}

    def test_filters_affect_key(self):
        self.cache.set("q", {"src": "a.pdf"}, {"answer": "with filter"}, tenant="t1")
        self.cache.set("q", {}, {"answer": "no filter"}, tenant="t1")
        assert self.cache.get("q", {"src": "a.pdf"}, "t1") == {"answer": "with filter"}
        assert self.cache.get("q", {}, "t1") == {"answer": "no filter"}


# ── SemanticCache – Redis-backed ─────────────────────────────────────────────


class TestSemanticCacheRedis:
    def setup_method(self):
        self.redis = FakeRedis()
        self.cache = SemanticCache(
            redis_client=self.redis, default_ttl=60, similarity_threshold=0.9
        )

    def test_exact_hit_via_redis(self):
        self.cache.set("q", {}, {"answer": "redis-cached"}, tenant="t1")
        result = self.cache.get("q", {}, tenant="t1")
        assert result == {"answer": "redis-cached"}

    def test_miss_returns_none(self):
        assert self.cache.get("unknown query", {}, tenant="t1") is None

    def test_invalidate_bumps_namespace(self):
        """Pre-seed ns=1 so invalidate takes it to 2, making ns=1 keys stale."""
        self.redis.set("sem:ns", "1")
        self.cache.set("q", {}, {"answer": "stale"}, tenant="t1")
        self.cache.invalidate()
        assert self.cache.get("q", {}, tenant="t1") is None

    def test_semantic_hit_above_threshold(self):
        """Embedding almost identical to stored → semantic hit."""
        emb = [1.0, 0.0, 0.0]
        self.cache.set(
            "aerospace lift", {}, {"answer": "lift"}, tenant="t1", query_embedding=emb
        )
        near_emb = [0.999, 0.001, 0.0]  # cosine ≈ 0.9999 → above 0.9 threshold
        result = self.cache.get("lift force", {}, tenant="t1", query_embedding=near_emb)
        assert result == {"answer": "lift"}

    def test_semantic_miss_below_threshold(self):
        """Unrelated embedding → below threshold → miss."""
        emb_a = [1.0, 0.0, 0.0]
        emb_b = [0.0, 1.0, 0.0]  # orthogonal → cosine = 0
        self.cache.set(
            "aerospace lift", {}, {"answer": "lift"}, tenant="t1", query_embedding=emb_a
        )
        result = self.cache.get("random query", {}, tenant="t1", query_embedding=emb_b)
        assert result is None
