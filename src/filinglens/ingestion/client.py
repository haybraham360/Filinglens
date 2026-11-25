"""Shared SEC EDGAR HTTP client: User-Agent, rate limiting, retries and disk caching."""

from __future__ import annotations

import logging
import random
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Any

import httpx

from filinglens.config import Settings
from filinglens.ingestion.cache import CachedResponse, DiskCache, atomic_write_bytes
from filinglens.ingestion.ratelimit import RateLimiter

logger = logging.getLogger(__name__)

RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
CACHEABLE_STATUSES = frozenset({200, 404})


class EdgarError(RuntimeError):
    """A request to EDGAR failed after retries."""


class EdgarNotFoundError(EdgarError):
    """EDGAR returned 404 (e.g. a company with no XBRL facts)."""


class EdgarForbiddenError(EdgarError):
    """EDGAR returned 403: missing/invalid User-Agent or a temporary IP block. Never retried."""


@dataclass
class ClientStats:
    """Counters for one client's lifetime. Thread-safe."""

    network_requests: int = 0
    cache_hits: int = 0
    bytes_downloaded: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def add(self, *, requests: int = 0, hits: int = 0, nbytes: int = 0) -> None:
        """Increment counters atomically."""
        with self._lock:
            self.network_requests += requests
            self.cache_hits += hits
            self.bytes_downloaded += nbytes


def validate_identity(identity: str) -> str:
    """Return a cleaned User-Agent, or raise if it does not look like 'Name email'."""
    identity = identity.strip()
    if not identity or "@" not in identity:
        raise ValueError(
            "EDGAR_IDENTITY must be 'Your Name your@email.com'; "
            "SEC rejects requests without a declared User-Agent."
        )
    return identity


class EdgarClient:
    """The only path to SEC servers. Every request is paced, identified, retried and cached."""

    def __init__(
        self,
        identity: str,
        cache_dir: Path,
        *,
        max_rps: float = 8.0,
        rate_limiter: RateLimiter | None = None,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        max_retries: int = 5,
        backoff_base: float = 1.0,
        backoff_cap: float = 60.0,
        timeout: float = 60.0,
    ) -> None:
        """Create a client. ``rate_limiter``, ``transport`` and ``sleep`` are for tests."""
        self.user_agent = validate_identity(identity)
        self.cache = DiskCache(cache_dir)
        self.stats = ClientStats()
        self._limiter = rate_limiter or RateLimiter(max_rps)
        self._sleep = sleep
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        self._backoff_cap = backoff_cap
        self._http = httpx.Client(
            headers={"User-Agent": self.user_agent, "Accept-Encoding": "gzip, deflate"},
            timeout=timeout,
            follow_redirects=True,
            transport=transport,
        )

    @classmethod
    def from_settings(cls, settings: Settings) -> EdgarClient:
        """Build a client from application settings."""
        return cls(
            settings.require_edgar_identity(),
            settings.cache_dir / "http",
            max_rps=settings.edgar_max_rps,
        )

    # -- lifecycle ---------------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying connection pool."""
        self._http.close()

    def __enter__(self) -> EdgarClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # -- public API --------------------------------------------------------------------------

    def get(self, url: str, *, refresh: bool = False) -> CachedResponse:
        """GET ``url``, serving 200/404 responses from the disk cache unless ``refresh``."""
        if not refresh:
            cached = self.cache.get(url)
            if cached is not None:
                self.stats.add(hits=1)
                return cached
        response = self._fetch(url)
        if response.status_code in CACHEABLE_STATUSES:
            self.cache.put(url, response.status_code, response.content)
        return CachedResponse(url, response.status_code, response.content, from_cache=False)

    def get_json(self, url: str, *, refresh: bool = False) -> Any:
        """GET ``url`` and decode JSON. Raises :class:`EdgarNotFoundError` on 404."""
        response = self.get(url, refresh=refresh)
        if response.status_code == 404:
            raise EdgarNotFoundError(f"404 Not Found: {url}")
        if response.status_code != 200:
            raise EdgarError(f"HTTP {response.status_code}: {url}")
        return response.json()

    def download(self, url: str, dest: Path) -> bool:
        """Save ``url`` to ``dest``. Returns False without any request if ``dest`` exists."""
        if dest.exists():
            self.stats.add(hits=1)
            return False
        response = self._fetch(url)
        if response.status_code == 404:
            raise EdgarNotFoundError(f"404 Not Found: {url}")
        if response.status_code != 200:
            raise EdgarError(f"HTTP {response.status_code}: {url}")
        atomic_write_bytes(dest, response.content)
        return True

    # -- internals ---------------------------------------------------------------------------

    def _backoff(self, attempt: int, retry_after: str | None) -> float:
        if retry_after is not None:
            try:
                return min(self._backoff_cap, max(0.0, float(retry_after)))
            except ValueError:
                pass
        delay = self._backoff_base * (2**attempt) + random.uniform(0, self._backoff_base)
        return float(min(self._backoff_cap, delay))

    def _fetch(self, url: str) -> httpx.Response:
        """Send one paced GET, retrying 429/5xx and transport errors with exponential backoff."""
        for attempt in range(self._max_retries + 1):
            last = attempt == self._max_retries
            self._limiter.acquire()
            self.stats.add(requests=1)
            try:
                response = self._http.get(url)
            except httpx.TransportError as exc:
                if last:
                    raise EdgarError(f"Transport error after retries: {url}: {exc}") from exc
                delay = self._backoff(attempt, None)
                logger.warning("transport error on %s (%s); retrying in %.1fs", url, exc, delay)
                self._sleep(delay)
                continue

            status = response.status_code
            if status in RETRY_STATUSES:
                if last:
                    raise EdgarError(f"HTTP {status} after {attempt + 1} attempts: {url}")
                delay = self._backoff(attempt, response.headers.get("Retry-After"))
                logger.warning("HTTP %s on %s; retrying in %.1fs", status, url, delay)
                self._sleep(delay)
                continue
            if status == 403:
                raise EdgarForbiddenError(
                    f"HTTP 403 from SEC for {url}. Check EDGAR_IDENTITY, or you may be "
                    "temporarily blocked for exceeding 10 requests/second."
                )
            self.stats.add(nbytes=len(response.content))
            return response
        raise AssertionError("unreachable")  # pragma: no cover
