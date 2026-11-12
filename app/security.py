"""API-key check and per-key rate limit for the API."""

from __future__ import annotations

import hashlib
import secrets
import time
from collections import defaultdict, deque
from collections.abc import Callable

API_KEY_HEADER = "X-API-Key"
ANONYMOUS = "anonymous"


def key_id(key: str) -> str:
    """Short stable label of a key for logs and rate-limit buckets; the key itself is never kept."""
    return hashlib.sha256(key.encode()).hexdigest()[:8]


def match_key(presented: str | None, accepted: list[str]) -> str | None:
    """The accepted key equal to `presented`, compared in constant time; None if no match.

    Every accepted key is compared (no early exit) so the time does not reveal which one matched.
    """
    if presented is None:
        return None
    found = None
    for key in accepted:
        if secrets.compare_digest(presented.encode(), key.encode()):
            found = key
    return found


class RateLimiter:
    """Sliding window: at most `limit` calls per `window_s` seconds for each identity.

    In memory, per process. That is enough for one API process; a second replica would need a
    shared store (see the production roadmap).
    """

    def __init__(
        self, limit: int, window_s: float = 60.0, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self._limit = limit
        self._window_s = window_s
        self._clock = clock
        self._calls: dict[str, deque[float]] = defaultdict(deque)

    def check(self, identity: str) -> float | None:
        """Record a call and return None, or return the seconds to wait if the limit is hit."""
        now = self._clock()
        calls = self._calls[identity]
        while calls and calls[0] <= now - self._window_s:
            calls.popleft()
        if len(calls) >= self._limit:
            return calls[0] + self._window_s - now
        calls.append(now)
        return None
