"""
VibeShield Semantic Cache — Prevents duplicate generations and saves money.

Problem:
    In marketing workflows, 20-40% of generation requests are near-duplicates.
    User prompts like "sunset over the ocean" and "sunset on the ocean, high quality"
    produce nearly identical results but both get charged.

Solution:
    Uses text similarity (SequenceMatcher for zero-dependency baseline,
    upgradable to sentence-transformers or CLIP for production) to detect
    near-duplicate prompts and return cached results.

Architecture:
    Request → SemanticCache.get() → cache hit? → return cached result
                                  → cache miss? → forward to API → cache result

Usage:
    cache = SemanticCache(similarity_threshold=0.90, ttl_seconds=86400)

    # Before calling API:
    cached = cache.get(model="z-image", prompt="sunset over ocean", params={...})
    if cached:
        print("Cache hit! Saved money:", cached.cost)
        return cached

    # After successful generation:
    cache.put(model="z-image", prompt="sunset over ocean", params={...}, result=result)
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Optional


@dataclass
class CacheEntry:
    """A cached generation result."""
    model: str
    prompt: str
    params_hash: str
    result: dict
    cost: float
    created_at: float
    hit_count: int = 0


@dataclass
class CacheStats:
    """Cache performance statistics."""
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_saved_rub: float = 0.0
    entries_count: int = 0

    @property
    def hit_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.cache_hits / self.total_requests

    def to_dict(self) -> dict:
        return {
            "total_requests": self.total_requests,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": f"{self.hit_rate:.1%}",
            "total_saved_rub": round(self.total_saved_rub, 2),
            "entries_count": self.entries_count,
        }


class SemanticCache:
    """
    Intelligent generation cache with semantic prompt matching.

    Prevents duplicate charges by detecting when a user's new prompt
    is semantically equivalent to a recently generated one.

    Features:
        - Exact match by (model + prompt hash) — O(1) lookup
        - Fuzzy match by text similarity — catches rephrased prompts
        - Configurable TTL (default: 24 hours, matching VibeMarketolog's cache)
        - LRU eviction to bound memory usage
        - Per-model isolation (image cache doesn't match video cache)
        - Statistics tracking for cost savings reporting
    """

    def __init__(
        self,
        similarity_threshold: float = 0.90,
        ttl_seconds: int = 86400,
        max_entries: int = 1000,
    ):
        """
        Args:
            similarity_threshold: Minimum text similarity (0-1) to count as duplicate.
            ttl_seconds: Time-to-live for cache entries (default: 24h).
            max_entries: Maximum cache size (LRU eviction).
        """
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries

        # Exact match index: hash → CacheEntry
        self._exact: OrderedDict[str, CacheEntry] = OrderedDict()

        # Per-model prompt lists for fuzzy search
        self._by_model: dict[str, list[CacheEntry]] = {}

        self._stats = CacheStats()

    @property
    def stats(self) -> CacheStats:
        return self._stats

    def get(self, model: str, prompt: str,
            params: Optional[dict] = None) -> Optional[dict]:
        """
        Look up a cached result for the given generation request.

        Tries exact match first, then fuzzy semantic match.

        Args:
            model: Model name (e.g., "z-image", "grok-ttv-10")
            prompt: Generation prompt
            params: Additional params (aspect_ratio, duration, etc.)

        Returns:
            Cached result dict if found, None if cache miss.
        """
        self._stats.total_requests += 1
        now = time.time()

        # Strategy 1: Exact hash match (fast O(1))
        key = self._make_key(model, prompt, params)
        if key in self._exact:
            entry = self._exact[key]
            if (now - entry.created_at) < self.ttl_seconds:
                entry.hit_count += 1
                self._stats.cache_hits += 1
                self._stats.total_saved_rub += entry.cost
                # Move to end (LRU)
                self._exact.move_to_end(key)
                return entry.result
            else:
                # Expired
                self._remove_entry(key, entry)

        # Strategy 2: Fuzzy semantic match (slower, per-model scan)
        if model in self._by_model:
            for entry in self._by_model[model]:
                if (now - entry.created_at) >= self.ttl_seconds:
                    continue  # skip expired (cleaned lazily)

                similarity = self._text_similarity(prompt, entry.prompt)
                if similarity >= self.similarity_threshold:
                    entry.hit_count += 1
                    self._stats.cache_hits += 1
                    self._stats.total_saved_rub += entry.cost
                    return entry.result

        self._stats.cache_misses += 1
        return None

    def put(self, model: str, prompt: str, result: dict,
            cost: float = 0.0, params: Optional[dict] = None) -> None:
        """
        Store a generation result in the cache.

        Args:
            model: Model name
            prompt: Generation prompt
            result: Full API response to cache
            cost: Cost in RUB of this generation
            params: Additional params for exact matching
        """
        key = self._make_key(model, prompt, params)

        entry = CacheEntry(
            model=model,
            prompt=prompt,
            params_hash=key,
            result=result,
            cost=cost,
            created_at=time.time(),
        )

        # Store in exact index
        self._exact[key] = entry
        self._exact.move_to_end(key)

        # Store in per-model list for fuzzy search
        if model not in self._by_model:
            self._by_model[model] = []
        self._by_model[model].append(entry)

        self._stats.entries_count = len(self._exact)

        # LRU eviction
        while len(self._exact) > self.max_entries:
            oldest_key, oldest_entry = self._exact.popitem(last=False)
            self._remove_from_model_index(oldest_entry)
            self._stats.entries_count = len(self._exact)

    def clear(self) -> None:
        """Clear all cache entries."""
        self._exact.clear()
        self._by_model.clear()
        self._stats.entries_count = 0

    def cleanup_expired(self) -> int:
        """Remove expired entries. Returns number of entries removed."""
        now = time.time()
        expired_keys = [
            key for key, entry in self._exact.items()
            if (now - entry.created_at) >= self.ttl_seconds
        ]
        for key in expired_keys:
            entry = self._exact.pop(key)
            self._remove_from_model_index(entry)

        self._stats.entries_count = len(self._exact)
        return len(expired_keys)

    # ── Private Helpers ─────────────────────────────────────────────────

    @staticmethod
    def _make_key(model: str, prompt: str, params: Optional[dict] = None) -> str:
        """Create a deterministic cache key from model + prompt + params."""
        parts = {"model": model, "prompt": prompt.strip().lower()}
        if params:
            # Only include generation-relevant params (not callbacks, etc.)
            relevant = {
                k: v for k, v in sorted(params.items())
                if k in (
                    "aspect_ratio", "resolution", "duration", "quality",
                    "output_format", "voice_id", "voice_name", "style",
                    "negative_prompt", "seed", "image_input",
                )
            }
            parts["params"] = relevant

        raw = json.dumps(parts, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode()).hexdigest()

    @staticmethod
    def _text_similarity(a: str, b: str) -> float:
        """
        Text similarity via SequenceMatcher (zero-dependency baseline).

        For production, upgrade to:
            - sentence-transformers (all-MiniLM-L6-v2) for multilingual semantic similarity
            - CLIP text encoder for cross-modal matching (text ↔ image prompts)
        """
        if not a or not b:
            return 0.0
        return SequenceMatcher(None, a.strip().lower(), b.strip().lower()).ratio()

    def _remove_entry(self, key: str, entry: CacheEntry) -> None:
        """Remove an entry from all indices."""
        self._exact.pop(key, None)
        self._remove_from_model_index(entry)
        self._stats.entries_count = len(self._exact)

    def _remove_from_model_index(self, entry: CacheEntry) -> None:
        """Remove an entry from the per-model fuzzy search list."""
        if entry.model in self._by_model:
            model_entries = self._by_model[entry.model]
            try:
                model_entries.remove(entry)
            except ValueError:
                pass
            if not model_entries:
                del self._by_model[entry.model]
