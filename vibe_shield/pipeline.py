"""
VibeShield Pipeline — Protected generation pipeline combining all modules.

Integrates VibeClient + CircuitBreaker + SemanticCache into a single
high-level interface for safe, cost-efficient AI content generation.

Usage:
    from vibe_shield.pipeline import VibePipeline

    async with VibePipeline(api_token="your_token", budget_cap=300.0) as pipe:
        result = await pipe.generate_safe({
            "type": "image",
            "model": "z-image",
            "prompt": "cyberpunk cityscape at night"
        })
        print(f"Result: {result.display_url}")
        print(f"Stats: {pipe.stats()}")
"""

from __future__ import annotations

from typing import Any, Optional

from vibe_shield.circuit_breaker import CircuitBreaker
from vibe_shield.client import GenerationResult, VibeAPIError, VibeClient
from vibe_shield.semantic_cache import SemanticCache


class VibePipeline:
    """
    All-in-one protected pipeline for VibeMarketolog API.

    Combines:
        - VibeClient: HTTP client with smart retries
        - CircuitBreaker: Agent loop and budget protection
        - SemanticCache: Duplicate generation prevention

    Flow:
        1. Check CircuitBreaker → is the agent stuck?
        2. Check SemanticCache → already generated this?
        3. Estimate cost via /generate/estimate → pre-flight validation
        4. Execute generation → VibeClient with retries
        5. Cache result → for future deduplication
        6. Record in CircuitBreaker → track session state
    """

    def __init__(
        self,
        api_token: str,
        *,
        budget_cap: float = 500.0,
        max_duplicates: int = 3,
        similarity_threshold: float = 0.90,
        cache_ttl: int = 86400,
        enable_cache: bool = True,
        enable_breaker: bool = True,
        enable_estimate: bool = True,
    ):
        self.client = VibeClient(api_token)
        self.breaker = CircuitBreaker(
            budget_cap=budget_cap,
            max_duplicates=max_duplicates,
            similarity_threshold=similarity_threshold,
        )
        self.cache = SemanticCache(
            similarity_threshold=similarity_threshold,
            ttl_seconds=cache_ttl,
        )
        self._enable_cache = enable_cache
        self._enable_breaker = enable_breaker
        self._enable_estimate = enable_estimate

    async def __aenter__(self) -> VibePipeline:
        await self.client.__aenter__()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.client.__aexit__(*exc)

    async def generate_safe(
        self,
        payload: dict,
        *,
        poll_interval: float = 10.0,
        timeout: float = 600.0,
    ) -> GenerationResult:
        """
        Generate content with full protection stack.

        Args:
            payload: Standard /generate payload (type, model, prompt, etc.)
            poll_interval: Status polling interval in seconds
            timeout: Max wait time for generation

        Returns:
            GenerationResult with display_url

        Raises:
            CircuitBreakerTripped: If agent loop detected
            VibeAPIError: If API returns an error
            TimeoutError: If generation times out
        """
        model = payload.get("model", "unknown")
        prompt = payload.get("prompt", "")
        endpoint = "/generate"

        # Step 1: Check Circuit Breaker
        if self._enable_breaker:
            estimated_cost = 0.0

            # Pre-estimate cost for budget check
            if self._enable_estimate:
                try:
                    estimate = await self.client.estimate(payload)
                    estimated_cost = float(estimate.get("estimated_cost_rub", 0))
                except VibeAPIError:
                    pass  # estimate failed — proceed without cost data

            if not self.breaker.check(endpoint, payload, estimated_cost):
                raise CircuitBreakerTripped(
                    self.breaker.trip_reason,
                    corrective_message=self.breaker.get_corrective_message(),
                    stats=self.breaker.get_stats(),
                )

        # Step 2: Check Semantic Cache
        if self._enable_cache:
            cached = self.cache.get(model, prompt, payload)
            if cached:
                print(f"💾 Cache hit! Returning cached result for '{prompt[:50]}...'")
                return GenerationResult(
                    generation_id=cached.get("generation_id", 0),
                    status="complete",
                    model=model,
                    type=payload.get("type", ""),
                    display_url=cached.get("display_url"),
                    result_url=cached.get("result_url"),
                    result_urls=cached.get("result_urls", []),
                    cost=0.0,  # Cache hit = FREE!
                    raw=cached,
                )

        # Step 3: Execute Generation
        gen_id = await self.client.generate(payload, strict=True)

        # Step 4: Wait for Result
        result = await self.client.wait_for_result(
            gen_id, poll_interval=poll_interval, timeout=timeout
        )

        # Step 5: Cache the Result
        if self._enable_cache and result.status == "complete":
            self.cache.put(model, prompt, result.raw, cost=result.cost, params=payload)

        # Step 6: Record in Circuit Breaker
        if self._enable_breaker:
            self.breaker.record_success(endpoint, payload, cost=result.cost)

        return result

    async def generate_batch(
        self,
        payloads: list[dict],
        *,
        poll_interval: float = 10.0,
        timeout: float = 600.0,
        stop_on_breaker: bool = True,
    ) -> list[GenerationResult | Exception]:
        """
        Generate multiple items with protection. Stops if circuit breaker trips.

        Returns a list of GenerationResult or Exception for each payload.
        """
        results: list[GenerationResult | Exception] = []

        for i, payload in enumerate(payloads):
            try:
                result = await self.generate_safe(
                    payload, poll_interval=poll_interval, timeout=timeout
                )
                results.append(result)
                print(f"✅ [{i + 1}/{len(payloads)}] Generated: {result.display_url}")
            except CircuitBreakerTripped as e:
                results.append(e)
                if stop_on_breaker:
                    print(f"🛑 Circuit breaker tripped at item {i + 1}. Stopping batch.")
                    # Fill remaining with the error
                    remaining = len(payloads) - len(results)
                    results.extend([e] * remaining)
                    break
            except Exception as e:
                results.append(e)
                print(f"❌ [{i + 1}/{len(payloads)}] Error: {e}")

        return results

    def stats(self) -> dict:
        """Combined statistics from all modules."""
        return {
            "circuit_breaker": self.breaker.get_stats(),
            "cache": self.cache.stats.to_dict(),
        }

    def reset(self) -> None:
        """Reset circuit breaker and cache."""
        self.breaker.reset()
        self.cache.clear()


class CircuitBreakerTripped(Exception):
    """Raised when the circuit breaker prevents an API call."""

    def __init__(self, reason: str, corrective_message: str = "",
                 stats: Optional[dict] = None):
        self.reason = reason
        self.corrective_message = corrective_message
        self.stats = stats or {}
        super().__init__(f"Circuit breaker tripped: {reason}")
