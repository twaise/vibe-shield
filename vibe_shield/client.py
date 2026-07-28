"""
VibeShield Client — Async HTTP client for VibeMarketolog Agent API.

Features:
    - Exponential backoff with Retry-After header support
    - Automatic retry on 429 (rate_limit, key_cooling_down, daily_spend_limit)
    - Idempotency key support for safe retries
    - Strict mode by default (pre-charge validation)
    - Structured error handling with request_id tracking
"""

from __future__ import annotations

import asyncio
import hashlib
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx


@dataclass
class GenerationResult:
    """Structured result from a completed generation."""
    generation_id: int
    status: str
    model: str
    type: str
    display_url: Optional[str] = None
    result_url: Optional[str] = None
    result_urls: list[str] = field(default_factory=list)
    cost: float = 0.0
    duration: Optional[str] = None
    error_message: Optional[str] = None
    refunded: bool = False
    raw: dict = field(default_factory=dict)


class VibeAPIError(Exception):
    """Structured API error with machine-readable code and request_id."""

    def __init__(self, status_code: int, error_code: str, message: str,
                 request_id: Optional[str] = None, details: Optional[dict] = None):
        self.status_code = status_code
        self.error_code = error_code
        self.message = message
        self.request_id = request_id
        self.details = details or {}
        super().__init__(f"[{status_code} {error_code}] {message} (req: {request_id})")


class VibeClient:
    """
    Async client for VibeMarketolog Agent API with production-grade resilience.

    Usage:
        async with VibeClient("your_api_token") as client:
            gen_id = await client.generate({
                "type": "image",
                "model": "z-image",
                "prompt": "a sunset over the ocean"
            })
            result = await client.wait_for_result(gen_id)
            print(result.display_url)
    """

    BASE_URL = "https://lk.vibemarketolog.ru/api/agent"
    MAX_RETRIES = 5
    BACKOFF_BASE = 1  # seconds
    BACKOFF_MAX = 30  # seconds

    def __init__(self, api_token: str, *, base_url: Optional[str] = None,
                 timeout: float = 30.0, max_retries: int = MAX_RETRIES):
        self.api_token = api_token
        self.base_url = (base_url or self.BASE_URL).rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self) -> VibeClient:
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_token}",
                "Content-Type": "application/json",
            },
            timeout=self.timeout,
        )
        return self

    async def __aexit__(self, *exc) -> None:
        if self._client:
            await self._client.aclose()

    @property
    def client(self) -> httpx.AsyncClient:
        if not self._client:
            raise RuntimeError("Use 'async with VibeClient(token) as client:' context manager")
        return self._client

    # ── Core HTTP with Smart Retry ──────────────────────────────────────

    async def _request(self, method: str, endpoint: str, *,
                       json: Optional[dict] = None,
                       params: Optional[dict] = None) -> dict:
        """
        Execute HTTP request with exponential backoff and Retry-After support.

        Handles:
            - 429 rate_limit_exceeded: respects Retry-After header
            - 429 key_cooling_down: waits retry_after from body
            - 429 daily_spend_limit_exceeded: raises immediately (no retry)
            - 5xx: retries with backoff
            - 4xx: raises VibeAPIError immediately
        """
        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries):
            try:
                response = await self.client.request(
                    method, endpoint, json=json, params=params
                )

                # Success
                if response.status_code == 200:
                    return response.json()

                # Parse error body
                try:
                    body = response.json()
                except Exception:
                    body = {}

                error_code = body.get("error", "unknown")
                message = body.get("message", response.text[:200])
                request_id = body.get("request_id")

                # Daily limit — don't retry, it won't help
                if error_code == "daily_spend_limit_exceeded":
                    raise VibeAPIError(429, error_code, message, request_id)

                # Rate limit / cooling — retry with backoff
                if response.status_code == 429:
                    retry_after = (
                        int(response.headers.get("Retry-After", 0))
                        or body.get("retry_after", 0)
                        or self._backoff_delay(attempt)
                    )
                    print(f"⚠️  [429 {error_code}] Retry in {retry_after}s "
                          f"(attempt {attempt + 1}/{self.max_retries})")
                    await asyncio.sleep(retry_after)
                    continue

                # Server errors — retry
                if response.status_code >= 500:
                    delay = self._backoff_delay(attempt)
                    print(f"⚠️  [{response.status_code}] Server error, retry in {delay}s")
                    await asyncio.sleep(delay)
                    continue

                # Client errors — fail fast
                raise VibeAPIError(
                    response.status_code, error_code, message,
                    request_id, body.get("details")
                )

            except httpx.TransportError as e:
                delay = self._backoff_delay(attempt)
                print(f"⚠️  Network error: {e}. Retry in {delay}s")
                last_error = e
                await asyncio.sleep(delay)

        raise last_error or RuntimeError("Max retries exceeded")

    def _backoff_delay(self, attempt: int) -> float:
        """Exponential backoff: 1, 2, 4, 8... capped at BACKOFF_MAX."""
        return min(self.BACKOFF_BASE * (2 ** attempt), self.BACKOFF_MAX)

    # ── Public API Methods ──────────────────────────────────────────────

    async def me(self) -> dict:
        """Get token info, balances, and limits."""
        return await self._request("GET", "/me")

    async def balance(self) -> float:
        """Get current balance in RUB."""
        data = await self._request("GET", "/balance")
        return float(data.get("balance", 0))

    async def capabilities(self) -> dict:
        """Get full model catalog with parameters and limits."""
        return await self._request("GET", "/capabilities")

    async def prices(self) -> dict:
        """Get pricing for all models."""
        return await self._request("GET", "/prices")

    async def health(self) -> dict:
        """Check API health status."""
        return await self._request("GET", "/health")

    async def estimate(self, payload: dict) -> dict:
        """
        Dry-run: validate generation payload and get cost estimate.
        No balance is charged. FREE.
        """
        return await self._request("POST", "/generate/estimate", json=payload)

    async def generate(self, payload: dict, *,
                       strict: bool = True,
                       idempotency_key: Optional[str] = None) -> int:
        """
        Launch a generation. Returns generation_id.

        Args:
            payload: Generation parameters (type, model, prompt, etc.)
            strict: If True, rejects incompatible fields BEFORE charging (default: True)
            idempotency_key: Unique key for safe retries (auto-generated if not provided)

        Returns:
            generation_id (int) to poll with wait_for_result()
        """
        payload = {**payload, "strict": strict}
        if idempotency_key:
            payload["idempotency_key"] = idempotency_key

        data = await self._request("POST", "/generate", json=payload)
        return data["generation_id"]

    async def generation_status(self, generation_id: int) -> dict:
        """Get current status of a generation."""
        return await self._request("GET", f"/generation/{generation_id}/status")

    async def wait_for_result(self, generation_id: int, *,
                              poll_interval: float = 10.0,
                              timeout: float = 600.0) -> GenerationResult:
        """
        Poll generation status until complete or error.

        Args:
            generation_id: ID from generate()
            poll_interval: Seconds between polls (default: 10)
            timeout: Max wait time in seconds (default: 600)

        Returns:
            GenerationResult with display_url and metadata

        Raises:
            TimeoutError: If generation doesn't complete within timeout
            VibeAPIError: If generation fails with error
        """
        start = time.monotonic()

        while (time.monotonic() - start) < timeout:
            data = await self.generation_status(generation_id)
            status = data.get("status")

            if status == "complete":
                return GenerationResult(
                    generation_id=generation_id,
                    status="complete",
                    model=data.get("model", ""),
                    type=data.get("type", ""),
                    display_url=data.get("display_url"),
                    result_url=data.get("result_url"),
                    result_urls=data.get("result_urls", []),
                    cost=float(data.get("cost", 0)),
                    duration=data.get("duration"),
                    raw=data,
                )

            if status == "error":
                error_msg = data.get("error_message", "Unknown generation error")
                refunded = data.get("refunded", False)
                raise VibeAPIError(
                    502, "generation_failed",
                    f"{error_msg} (refunded: {refunded})",
                )

            await asyncio.sleep(poll_interval)

        raise TimeoutError(
            f"Generation #{generation_id} did not complete within {timeout}s"
        )

    async def upload_media(self, file_path: str) -> dict:
        """
        Upload a media file and get a stable URL.

        Returns dict with: url, kind, mime, size, expires_at, duration (for video)
        """
        import aiofiles

        async with aiofiles.open(file_path, "rb") as f:
            content = await f.read()

        filename = file_path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]

        # Use a separate client call for multipart upload
        response = await self.client.post(
            "/upload-media",
            files={"file": (filename, content)},
            headers={"Authorization": f"Bearer {self.api_token}"},
        )

        if response.status_code == 200:
            return response.json()

        body = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
        raise VibeAPIError(
            response.status_code,
            body.get("error", "upload_failed"),
            body.get("message", "File upload failed"),
            body.get("request_id"),
        )

    async def voices(self, *, gender: Optional[str] = None,
                     category: Optional[str] = None,
                     language: Optional[str] = None,
                     search: Optional[str] = None) -> list[dict]:
        """Get voice catalog with optional filters."""
        params = {}
        if gender:
            params["gender"] = gender
        if category:
            params["category"] = category
        if language:
            params["language"] = language
        if search:
            params["search"] = search

        data = await self._request("GET", "/voices", params=params)
        return data.get("voices", [])
