"""Per-IP rate limiting and an answer cache (both in-process)."""

from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable
from typing import Generic, TypeVar

V = TypeVar("V")


class RateLimiter:
    """Sliding-window limiter: at most ``limit`` hits per ``window_s`` per key.

    In-process, so with several workers each enforces its own window; the effective
    limit is ``limit * workers``. Enough for a demo; use Redis for anything bigger.
    """

    def __init__(
        self, limit: int, window_s: float = 60.0, clock: Callable[[], float] = time.monotonic
    ) -> None:
        """Create a limiter (``clock`` is injectable for tests)."""
        self.limit = limit
        self.window_s = window_s
        self._clock = clock
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str) -> float | None:
        """Record a hit; return ``None`` if allowed, else seconds until the next slot."""
        now = self._clock()
        with self._lock:
            hits = self._hits.setdefault(key, deque())
            while hits and now - hits[0] >= self.window_s:
                hits.popleft()
            if len(hits) >= self.limit:
                return max(0.0, self.window_s - (now - hits[0]))
            hits.append(now)
            if len(self._hits) > 10_000:  # forget idle clients
                for k in [k for k, v in self._hits.items() if not v][:5_000]:
                    del self._hits[k]
            return None


class TTLCache(Generic[V]):
    """Small thread-safe LRU cache with a time-to-live."""

    def __init__(
        self,
        max_items: int = 512,
        ttl_s: float = 86_400,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Create a cache (``ttl_s=0`` disables it)."""
        self.max_items = max_items
        self.ttl_s = ttl_s
        self._clock = clock
        self._data: OrderedDict[str, tuple[float, V]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> V | None:
        """Cached value, or ``None`` if absent or expired."""
        if self.ttl_s <= 0:
            return None
        with self._lock:
            item = self._data.get(key)
            if item is None:
                return None
            stored, value = item
            if self._clock() - stored > self.ttl_s:
                del self._data[key]
                return None
            self._data.move_to_end(key)
            return value

    def put(self, key: str, value: V) -> None:
        """Store a value, evicting the least recently used beyond ``max_items``."""
        if self.ttl_s <= 0:
            return
        with self._lock:
            self._data[key] = (self._clock(), value)
            self._data.move_to_end(key)
            while len(self._data) > self.max_items:
                self._data.popitem(last=False)


def cache_key(*parts: object) -> str:
    """Stable key; the question is case- and whitespace-normalized."""
    norm = [" ".join(str(p).lower().split()) if isinstance(p, str) else repr(p) for p in parts]
    return hashlib.sha256("\x1f".join(norm).encode()).hexdigest()
