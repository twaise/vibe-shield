"""Tests for the Circuit Breaker module."""

import pytest
from vibe_shield.circuit_breaker import CircuitBreaker, TripReason


class TestCircuitBreaker:
    """Test suite for CircuitBreaker."""

    def setup_method(self):
        self.breaker = CircuitBreaker(
            budget_cap=100.0,
            max_duplicates=3,
            similarity_threshold=0.90,
            max_calls_per_minute=10,
        )

    def test_allows_normal_calls(self):
        """Normal, varied calls should pass."""
        assert self.breaker.check("/generate", {"prompt": "sunset"}, cost=10.0)
        self.breaker.record_success("/generate", {"prompt": "sunset"}, cost=10.0)

        assert self.breaker.check("/generate", {"prompt": "mountains"}, cost=10.0)
        self.breaker.record_success("/generate", {"prompt": "mountains"}, cost=10.0)

        assert not self.breaker.is_tripped

    def test_detects_duplicate_calls(self):
        """Identical calls should trip the breaker after max_duplicates."""
        payload = {"prompt": "same thing", "model": "z-image"}

        for i in range(3):
            ok = self.breaker.check("/generate", payload, cost=5.0)
            if ok:
                self.breaker.record_success("/generate", payload, cost=5.0)

        # 4th identical call should be blocked
        assert not self.breaker.check("/generate", payload, cost=5.0)
        assert self.breaker.is_tripped
        assert self.breaker.state.trip_reason == TripReason.DUPLICATE_CALLS

    def test_detects_budget_exceeded(self):
        """Exceeding budget cap should trip the breaker."""
        # Budget is 100₽, try to spend 110₽
        assert not self.breaker.check("/generate", {"prompt": "expensive"}, cost=110.0)
        assert self.breaker.state.trip_reason == TripReason.BUDGET_EXCEEDED

    def test_detects_semantic_loop(self):
        """Similar prompts should be detected as a semantic loop."""
        # First two calls are fine
        p1 = {"prompt": "beautiful sunset over the calm ocean with golden light"}
        self.breaker.check("/generate", p1, cost=5.0)
        self.breaker.record_success("/generate", p1, cost=5.0)

        p2 = {"prompt": "another thing entirely different topic"}
        self.breaker.check("/generate", p2, cost=5.0)
        self.breaker.record_success("/generate", p2, cost=5.0)

        # This prompt is very similar to p1
        p3 = {"prompt": "beautiful sunset over the calm ocean with golden lights"}
        ok = self.breaker.check("/generate", p3, cost=5.0)

        assert not ok
        assert self.breaker.state.trip_reason == TripReason.SEMANTIC_LOOP

    def test_reset(self):
        """Reset should clear all state."""
        self.breaker.check("/generate", {"prompt": "x"}, cost=150.0)
        assert self.breaker.is_tripped

        self.breaker.reset()
        assert not self.breaker.is_tripped
        assert self.breaker.total_cost == 0.0

    def test_corrective_message(self):
        """Tripped breaker should provide a corrective message."""
        self.breaker.check("/generate", {"prompt": "x"}, cost=150.0)
        msg = self.breaker.get_corrective_message()

        assert "CIRCUIT BREAKER" in msg
        assert "budget" in msg.lower()

    def test_stats(self):
        """Stats should reflect session state."""
        self.breaker.check("/generate", {"prompt": "test"}, cost=10.0)
        self.breaker.record_success("/generate", {"prompt": "test"}, cost=10.0)

        stats = self.breaker.get_stats()
        assert stats["total_cost_rub"] == 10.0
        assert stats["call_count"] == 1
        assert stats["is_tripped"] is False


class TestSemanticCache:
    """Test suite for SemanticCache."""

    def setup_method(self):
        from vibe_shield.semantic_cache import SemanticCache
        self.cache = SemanticCache(similarity_threshold=0.90, ttl_seconds=3600)

    def test_exact_match(self):
        """Exact same prompt should return cached result."""
        self.cache.put("z-image", "sunset over ocean", {"display_url": "http://test"}, cost=15.0)
        result = self.cache.get("z-image", "sunset over ocean")
        assert result is not None
        assert result["display_url"] == "http://test"

    def test_fuzzy_match(self):
        """Similar prompts should match."""
        self.cache.put("z-image", "beautiful sunset over the calm ocean",
                       {"display_url": "http://cached"}, cost=15.0)

        result = self.cache.get("z-image", "beautiful sunset over the calm oceans")
        assert result is not None

    def test_different_models_isolated(self):
        """Cache entries for different models should not match."""
        self.cache.put("z-image", "sunset", {"display_url": "img"}, cost=10.0)
        result = self.cache.get("grok-ttv-10", "sunset")
        assert result is None

    def test_miss_on_different_prompt(self):
        """Very different prompts should NOT match."""
        self.cache.put("z-image", "sunset over ocean", {"display_url": "http://test"}, cost=15.0)
        result = self.cache.get("z-image", "a cat sitting on a windowsill")
        assert result is None

    def test_stats_tracking(self):
        """Cache should track hits, misses, and savings."""
        self.cache.put("z-image", "test", {"url": "x"}, cost=20.0)

        self.cache.get("z-image", "test")  # hit
        self.cache.get("z-image", "test")  # hit
        self.cache.get("z-image", "totally different")  # miss

        stats = self.cache.stats
        assert stats.cache_hits == 2
        assert stats.cache_misses == 1
        assert stats.total_saved_rub == 40.0  # 2 hits × 20₽

    def test_lru_eviction(self):
        """Cache should evict oldest entries when full."""
        small_cache = type(self.cache)(max_entries=3)

        for i in range(5):
            small_cache.put("z-image", f"prompt {i}", {"id": i}, cost=1.0)

        # First two should be evicted
        assert small_cache.get("z-image", "prompt 0") is None
        assert small_cache.get("z-image", "prompt 1") is None
        # Last three should remain
        assert small_cache.get("z-image", "prompt 4") is not None
