"""
Demo: VibeShield in action — protected content generation pipeline.

This example shows how VibeShield protects against:
    1. Duplicate generations (semantic cache saves money)
    2. Agent loops (circuit breaker stops infinite retries)
    3. Budget overruns (configurable spending caps)

Usage:
    export VIBE_API_TOKEN="your_token_here"
    python -m examples.demo
"""

import asyncio
import os

from vibe_shield.pipeline import VibePipeline, CircuitBreakerTripped


async def demo_basic_generation():
    """Generate an image with full protection."""
    token = os.environ.get("VIBE_API_TOKEN")
    if not token:
        print("❌ Set VIBE_API_TOKEN environment variable first!")
        print("   Get your key at: https://lk.vibemarketolog.ru/#agent")
        return

    async with VibePipeline(token, budget_cap=300.0) as pipe:
        # --- Generation 1: Normal image ---
        print("\n🎨 Generating image #1...")
        result1 = await pipe.generate_safe({
            "type": "image",
            "model": "z-image",
            "prompt": "futuristic neon cityscape at night, cyberpunk style",
            "aspect_ratio": "16:9",
        })
        print(f"   ✅ Done! Cost: {result1.cost}₽")
        print(f"   🔗 URL: {result1.display_url}")

        # --- Generation 2: Same prompt (cache hit!) ---
        print("\n🎨 Generating image #2 (same prompt — should hit cache)...")
        result2 = await pipe.generate_safe({
            "type": "image",
            "model": "z-image",
            "prompt": "futuristic neon cityscape at night, cyberpunk style",
            "aspect_ratio": "16:9",
        })
        print(f"   💾 Cache hit! Cost: {result2.cost}₽ (saved!)")

        # --- Generation 3: Similar prompt (semantic cache hit!) ---
        print("\n🎨 Generating image #3 (similar prompt — should detect duplicate)...")
        result3 = await pipe.generate_safe({
            "type": "image",
            "model": "z-image",
            "prompt": "futuristic neon city at night, cyberpunk aesthetic, high quality",
            "aspect_ratio": "16:9",
        })
        print(f"   💾 Semantic cache hit! Cost: {result3.cost}₽")

        # --- Print savings ---
        stats = pipe.stats()
        print("\n📊 Session Stats:")
        print(f"   Cache hits: {stats['cache']['cache_hits']}")
        print(f"   Total saved: {stats['cache']['total_saved_rub']}₽")
        print(f"   Total spent: {stats['circuit_breaker']['total_cost_rub']}₽")


async def demo_circuit_breaker():
    """Simulate an agent loop and watch the circuit breaker activate."""
    token = os.environ.get("VIBE_API_TOKEN")
    if not token:
        print("❌ Set VIBE_API_TOKEN environment variable first!")
        return

    async with VibePipeline(token, budget_cap=100.0, max_duplicates=3) as pipe:
        print("\n🔁 Simulating agent loop (identical requests)...")

        for i in range(5):
            try:
                result = await pipe.generate_safe({
                    "type": "image",
                    "model": "z-image",
                    "prompt": "a red apple on a white table",
                })
                print(f"   [{i+1}] Generated: {result.display_url}")
            except CircuitBreakerTripped as e:
                print(f"   [{i+1}] 🛑 CIRCUIT BREAKER: {e.reason}")
                print(f"   Corrective message for agent:")
                print(f"   {e.corrective_message[:200]}...")
                break

        print(f"\n📊 Final stats: {pipe.stats()}")


async def demo_cost_estimate():
    """Show dry-run cost estimation before spending."""
    token = os.environ.get("VIBE_API_TOKEN")
    if not token:
        print("❌ Set VIBE_API_TOKEN environment variable first!")
        return

    async with VibePipeline(token) as pipe:
        print("\n💰 Estimating costs (no charge)...")

        models_to_test = [
            {"type": "image", "model": "z-image", "prompt": "test"},
            {"type": "voice", "model": "gemini-flash-tts", "prompt": "Привет, мир!"},
            {"type": "video", "model": "grok-ttv-10", "prompt": "test", "duration": 5},
        ]

        for payload in models_to_test:
            try:
                estimate = await pipe.client.estimate(payload)
                cost = estimate.get("estimated_cost_rub", "?")
                valid = estimate.get("valid", False)
                print(f"   {payload['model']:25s} → {cost}₽ (valid: {valid})")
            except Exception as e:
                print(f"   {payload['model']:25s} → Error: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("  VibeShield Demo — Protected Content Generation")
    print("=" * 60)

    asyncio.run(demo_cost_estimate())
    asyncio.run(demo_basic_generation())
    # Uncomment to test circuit breaker:
    # asyncio.run(demo_circuit_breaker())
