"""
VibeShield — Intelligent middleware for VibeMarketolog Agent API.

Modules:
    - client: Async API client with smart retries and error handling
    - circuit_breaker: Agent loop detection and balance protection
    - semantic_cache: CLIP-based duplicate generation detection
    - openai_compat: OpenAI-compatible translation proxy
"""

__version__ = "0.1.0"
__author__ = "Maxim Li"

from vibe_shield.client import VibeClient
from vibe_shield.circuit_breaker import CircuitBreaker
from vibe_shield.semantic_cache import SemanticCache

__all__ = ["VibeClient", "CircuitBreaker", "SemanticCache"]
