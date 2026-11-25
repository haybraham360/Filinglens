"""Process-wide request pacing for SEC fair-access compliance."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable


class RateLimiter:
    """Thread-safe limiter that spaces permits at least ``1 / rate`` seconds apart.

    Strict spacing (a token bucket with a capacity of one) guarantees that no sliding
    one-second window ever holds more than ``rate`` requests. A larger burst capacity
    would allow up to twice the rate inside a window, which SEC would count against us.
    """

    def __init__(
        self,
        rate: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Create a limiter allowing ``rate`` permits per second.

        ``clock`` and ``sleep`` are injectable so tests can use a fake clock.
        """
        if rate <= 0:
            raise ValueError("rate must be positive")
        self.rate = rate
        self.interval = 1.0 / rate
        self._clock = clock
        self._sleep = sleep
        self._next_slot = float("-inf")
        self._lock = threading.Lock()

    def acquire(self) -> None:
        """Block until a request may be sent."""
        with self._lock:
            wait = self._next_slot - self._clock()
            if wait > 0:
                self._sleep(wait)
            slot = max(self._clock(), self._next_slot)
            self._next_slot = slot + self.interval
