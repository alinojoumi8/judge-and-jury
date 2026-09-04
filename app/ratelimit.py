"""Small, dependency-free limiters for the trial endpoint.

Every trial costs real money, and `/api/trial` is reachable by anyone who can reach
the server. These are deliberately simple: one process, one event loop, no
external store. That is the right size for a tool that runs on a laptop or a
single small host; a fleet behind a load balancer would want a shared limiter.
Both are pure objects with an injectable clock so they can be tested exactly.
"""

from __future__ import annotations

import time
from typing import Callable

# Beyond this many distinct clients, forget the ones that have been quiet longest
# so the per-client table cannot grow without bound.
_MAX_TRACKED_CLIENTS = 1000


class TokenBucket:
    """A token bucket per client key.

    Each key holds up to `rate_per_minute` tokens, refilled continuously at that
    rate; an allowed call spends one. A rate of zero or less disables the limit.
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._state: dict[str, tuple[float, float]] = {}  # key -> (tokens, last seen)

    def allow(self, key: str, rate_per_minute: float) -> tuple[bool, float]:
        """Return (allowed, seconds until the next token if refused)."""
        if rate_per_minute <= 0:
            return True, 0.0
        now = self._clock()
        tokens, last = self._state.get(key, (rate_per_minute, now))
        per_second = rate_per_minute / 60.0
        tokens = min(rate_per_minute, tokens + (now - last) * per_second)
        if tokens >= 1.0:
            self._state[key] = (tokens - 1.0, now)
            return True, 0.0
        self._state[key] = (tokens, now)
        self._prune(now)
        return False, (1.0 - tokens) / per_second

    def _prune(self, now: float) -> None:
        if len(self._state) <= _MAX_TRACKED_CLIENTS:
            return
        oldest = sorted(self._state.items(), key=lambda kv: kv[1][1])
        for key, _ in oldest[: len(self._state) - _MAX_TRACKED_CLIENTS]:
            del self._state[key]


class ConcurrencyGate:
    """A cap on trials in flight at once.

    Not thread-safe by design: FastAPI runs async handlers on one event loop, and
    acquire/release never await, so there is no interleaving to guard against.
    """

    def __init__(self) -> None:
        self.active = 0

    def try_acquire(self, limit: int) -> bool:
        if self.active >= limit:
            return False
        self.active += 1
        return True

    def release(self) -> None:
        self.active = max(0, self.active - 1)
