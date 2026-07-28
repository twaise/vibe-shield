"""
VibeShield Circuit Breaker — Agent loop detection and balance protection.

Detects and prevents:
    1. Duplicate tool calls (same endpoint + same params N times in a row)
    2. Semantic thought loops (agent repeating similar prompts with cosine > 0.95)
    3. Budget overruns (cumulative session spend exceeding a configurable cap)
    4. Rapid-fire abuse (too many requests in a short time window)

Usage:
    breaker = CircuitBreaker(budget_cap=200.0, max_duplicates=3)

    # Wrap every API call:
    if breaker.check(endpoint="/generate", payload={...}, cost=15.0):
        # safe to proceed
        result = await client.generate(payload)
        breaker.record_success(cost=15.0)
    else:
        # breaker tripped — stop the agent
        print(breaker.trip_reason)
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import deque
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from enum import Enum
from typing import Optional


class TripReason(Enum):
    """Why the circuit breaker tripped."""
    DUPLICATE_CALLS = "duplicate_calls"
    SEMANTIC_LOOP = "semantic_loop"
    BUDGET_EXCEEDED = "budget_exceeded"
    RAPID_FIRE = "rapid_fire"


@dataclass
class CallRecord:
    """Record of a single API call."""
    endpoint: str
    payload_hash: str
    prompt: str
    cost: float
    timestamp: float


@dataclass
class CircuitBreakerState:
    """Current state of the circuit breaker."""
    is_tripped: bool = False
    trip_reason: Optional[TripReason] = None
    trip_message: str = ""
    total_cost: float = 0.0
    call_count: int = 0
    duplicate_streak: int = 0
    session_start: float = field(default_factory=time.monotonic)


class CircuitBreaker:
    """
    Protects VibeMarketolog API users from runaway agent loops.

    This middleware sits between the AI agent and the API client,
    analyzing patterns in API calls to detect infinite loops,
    semantic repetition, and budget overruns.

    Architecture:
        Agent (Claude/ChatGPT via MCP)
            │
            ▼
        CircuitBreaker.check()  ← HERE: analyzes patterns
            │
            ▼
        VibeClient.generate()   → VibeMarketolog API
    """

    def __init__(
        self,
        budget_cap: float = 500.0,
        max_duplicates: int = 3,
        similarity_threshold: float = 0.92,
        max_calls_per_minute: int = 20,
        history_size: int = 50,
    ):
        """
        Args:
            budget_cap: Maximum total spend (RUB) per session before tripping.
            max_duplicates: Max consecutive identical calls before tripping.
            similarity_threshold: Prompt similarity ratio (0-1) to detect semantic loops.
            max_calls_per_minute: Max API calls per 60-second sliding window.
            history_size: Number of recent calls to keep in memory.
        """
        self.budget_cap = budget_cap
        self.max_duplicates = max_duplicates
        self.similarity_threshold = similarity_threshold
        self.max_calls_per_minute = max_calls_per_minute

        self._history: deque[CallRecord] = deque(maxlen=history_size)
        self._state = CircuitBreakerState()

    @property
    def state(self) -> CircuitBreakerState:
        """Current circuit breaker state (read-only view)."""
        return self._state

    @property
    def is_tripped(self) -> bool:
        return self._state.is_tripped

    @property
    def trip_reason(self) -> str:
        return self._state.trip_message

    @property
    def total_cost(self) -> float:
        return self._state.total_cost

    def reset(self) -> None:
        """Reset the circuit breaker (e.g., after human intervention)."""
        self._state = CircuitBreakerState()
        self._history.clear()

    def check(self, endpoint: str, payload: dict, cost: float = 0.0) -> bool:
        """
        Check if the next API call is safe to execute.

        Args:
            endpoint: API endpoint (e.g., "/generate")
            payload: Request body
            cost: Estimated cost in RUB (from /generate/estimate)

        Returns:
            True if safe to proceed, False if circuit breaker tripped.
        """
        if self._state.is_tripped:
            return False

        prompt = payload.get("prompt", "")
        payload_hash = self._hash_payload(payload)

        # Check 1: Budget cap
        if self._state.total_cost + cost > self.budget_cap:
            self._trip(
                TripReason.BUDGET_EXCEEDED,
                f"Budget cap exceeded: {self._state.total_cost + cost:.0f}₽ > "
                f"{self.budget_cap:.0f}₽ limit. "
                f"Total calls: {self._state.call_count}."
            )
            return False

        # Check 2: Duplicate calls (same endpoint + same payload hash)
        if self._history:
            last = self._history[-1]
            if last.endpoint == endpoint and last.payload_hash == payload_hash:
                self._state.duplicate_streak += 1
                if self._state.duplicate_streak >= self.max_duplicates:
                    self._trip(
                        TripReason.DUPLICATE_CALLS,
                        f"Detected {self._state.duplicate_streak} identical calls to "
                        f"{endpoint}. Agent is stuck in a loop. "
                        f"Payload hash: {payload_hash[:16]}..."
                    )
                    return False
            else:
                self._state.duplicate_streak = 0

        # Check 3: Semantic similarity (prompt loops)
        if prompt and len(self._history) >= 2:
            recent_prompts = [
                r.prompt for r in list(self._history)[-5:]
                if r.prompt and r.endpoint == endpoint
            ]
            for prev_prompt in recent_prompts:
                similarity = self._text_similarity(prompt, prev_prompt)
                if similarity > self.similarity_threshold and prompt != prev_prompt:
                    self._trip(
                        TripReason.SEMANTIC_LOOP,
                        f"Semantic loop detected: last prompt is {similarity:.0%} similar "
                        f"to a recent prompt. Agent is rephrasing the same request. "
                        f"Consider reformulating your approach."
                    )
                    return False

        # Check 4: Rapid-fire rate limiting
        now = time.monotonic()
        recent_calls = sum(
            1 for r in self._history
            if (now - r.timestamp) < 60.0
        )
        if recent_calls >= self.max_calls_per_minute:
            self._trip(
                TripReason.RAPID_FIRE,
                f"Rate limit: {recent_calls} calls in the last 60 seconds "
                f"(limit: {self.max_calls_per_minute}). Slow down."
            )
            return False

        return True

    def record_success(self, endpoint: str, payload: dict,
                       cost: float = 0.0) -> None:
        """Record a successful API call after it completes."""
        prompt = payload.get("prompt", "")
        payload_hash = self._hash_payload(payload)

        self._history.append(CallRecord(
            endpoint=endpoint,
            payload_hash=payload_hash,
            prompt=prompt,
            cost=cost,
            timestamp=time.monotonic(),
        ))

        self._state.total_cost += cost
        self._state.call_count += 1

    def get_corrective_message(self) -> str:
        """
        Generate a corrective system message to inject into the agent's context.

        This message can be sent back to the LLM to break it out of a loop.
        """
        if not self._state.is_tripped:
            return ""

        reason = self._state.trip_reason
        base = (
            "⚠️ CIRCUIT BREAKER ACTIVATED ⚠️\n"
            f"Reason: {self._state.trip_message}\n\n"
        )

        if reason == TripReason.DUPLICATE_CALLS:
            base += (
                "You are calling the same API endpoint with identical parameters "
                "repeatedly. This is not productive. Please:\n"
                "1. Check the error response from the previous call\n"
                "2. Modify your approach or parameters\n"
                "3. Ask the user for clarification if needed"
            )
        elif reason == TripReason.SEMANTIC_LOOP:
            base += (
                "You are rephrasing the same prompt with minor variations. "
                "The results will be nearly identical. Please:\n"
                "1. Try a fundamentally different approach\n"
                "2. Change the model, style, or subject matter\n"
                "3. Ask the user if the current results are acceptable"
            )
        elif reason == TripReason.BUDGET_EXCEEDED:
            base += (
                "The session budget limit has been reached. "
                "No more generations can be performed. Please:\n"
                "1. Present the results generated so far to the user\n"
                "2. Ask the user if they want to increase the budget\n"
                "3. Summarize what was accomplished"
            )
        elif reason == TripReason.RAPID_FIRE:
            base += (
                "Too many API calls in a short time window. Please:\n"
                "1. Add delays between your API calls\n"
                "2. Batch your requests more efficiently\n"
                "3. Wait for current generations to complete before starting new ones"
            )

        return base

    def get_stats(self) -> dict:
        """Get session statistics."""
        return {
            "is_tripped": self._state.is_tripped,
            "trip_reason": self._state.trip_reason.value if self._state.trip_reason else None,
            "total_cost_rub": round(self._state.total_cost, 2),
            "budget_remaining_rub": round(self.budget_cap - self._state.total_cost, 2),
            "call_count": self._state.call_count,
            "duplicate_streak": self._state.duplicate_streak,
            "session_duration_sec": round(time.monotonic() - self._state.session_start, 1),
        }

    # ── Private Helpers ─────────────────────────────────────────────────

    def _trip(self, reason: TripReason, message: str) -> None:
        """Trip the circuit breaker."""
        self._state.is_tripped = True
        self._state.trip_reason = reason
        self._state.trip_message = message

    @staticmethod
    def _hash_payload(payload: dict) -> str:
        """Create a deterministic hash of the payload for duplicate detection."""
        # Remove non-deterministic fields
        clean = {k: v for k, v in sorted(payload.items())
                 if k not in ("idempotency_key", "callback_url", "strict")}
        raw = json.dumps(clean, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode()).hexdigest()

    @staticmethod
    def _text_similarity(a: str, b: str) -> float:
        """
        Fast text similarity using SequenceMatcher (stdlib, no ML deps).

        For production, replace with sentence-transformers or CLIP embeddings
        for true semantic similarity.
        """
        if not a or not b:
            return 0.0
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()
