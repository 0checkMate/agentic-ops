import asyncio
import math
import time
from collections.abc import Awaitable, Callable, Mapping

from pydantic import BaseModel, ConfigDict, Field

from agentic_ops.gateway.errors import GatewayConfigError, GatewayRateLimited

Clock = Callable[[], float]
Sleeper = Callable[[float], Awaitable[None]]

_EPSILON = 1e-9
_MAX_PENALTY = 86400.0


class RateLimit(BaseModel):
    """Budgets for one key; any dimension left as None is not enforced."""

    model_config = ConfigDict(frozen=True)

    requests_per_minute: int | None = Field(default=None, gt=0)
    tokens_per_minute: int | None = Field(default=None, gt=0)
    requests_per_day: int | None = Field(default=None, gt=0)


class _Bucket:
    """Token bucket: holds up to capacity units and refills continuously."""

    def __init__(self, capacity: float, refill_per_second: float, now: float) -> None:
        self._capacity = capacity
        self._refill_per_second = refill_per_second
        self._level = capacity
        self._updated = now

    def wait_for(self, amount: float, now: float) -> float:
        """Seconds until amount fits; math.inf if it exceeds the capacity."""
        if amount > self._capacity:
            return math.inf
        elapsed = max(0.0, now - self._updated)
        self._level = min(self._capacity, self._level + elapsed * self._refill_per_second)
        self._updated = now
        deficit = amount - self._level
        if deficit <= _EPSILON:
            return 0.0
        return deficit / self._refill_per_second

    def consume(self, amount: float) -> None:
        self._level -= amount


async def _real_sleep(seconds: float) -> None:
    await asyncio.sleep(seconds)


class RateLimiter:
    """Per-key limiter enforcing requests/minute, tokens/minute and requests/day together."""

    def __init__(
        self,
        limits: Mapping[str, RateLimit],
        *,
        clock: Clock = time.monotonic,
        sleep: Sleeper = _real_sleep,
    ) -> None:
        self._clock = clock
        self._sleep = sleep
        start = clock()
        self._buckets = {key: self._build(limit, start) for key, limit in limits.items()}
        self._blocked_until: dict[str, float] = {}

    @property
    def keys(self) -> frozenset[str]:
        """The keys that have a configured limit."""
        return frozenset(self._buckets)

    @staticmethod
    def _build(limit: RateLimit, now: float) -> dict[str, _Bucket]:
        buckets: dict[str, _Bucket] = {}
        if limit.requests_per_minute is not None:
            rpm = limit.requests_per_minute
            buckets['requests_minute'] = _Bucket(rpm, rpm / 60, now)
        if limit.tokens_per_minute is not None:
            tpm = limit.tokens_per_minute
            buckets['tokens_minute'] = _Bucket(tpm, tpm / 60, now)
        if limit.requests_per_day is not None:
            rpd = limit.requests_per_day
            buckets['requests_day'] = _Bucket(rpd, rpd / 86400, now)
        return buckets

    def penalize(self, key: str, seconds: float) -> None:
        """Block a key for seconds, capped at one day; a block is never shortened.

        Called when a provider returns HTTP 429: its answer outranks our local estimate.
        """
        if key not in self._buckets:
            raise GatewayConfigError(f'No rate limit configured for {key!r}')
        if seconds < 0:
            raise ValueError('seconds must be non-negative')
        deadline = self._clock() + min(seconds, _MAX_PENALTY)
        self._blocked_until[key] = max(self._blocked_until.get(key, 0.0), deadline)

    async def acquire(self, key: str, tokens: int = 0, *, max_wait: float) -> None:
        """Reserve one request and tokens of budget, waiting at most max_wait seconds.

        Raises GatewayRateLimited, without consuming anything, when the wait would
        exceed max_wait. Nothing awaits between the check and the consumption, so the
        reservation is atomic within one event loop; after any sleep it is re-checked.
        """
        buckets = self._buckets.get(key)
        if buckets is None:
            raise GatewayConfigError(f'No rate limit configured for {key!r}')
        costs = [
            (bucket, float(tokens) if name == 'tokens_minute' else 1.0)
            for name, bucket in buckets.items()
        ]
        waited = 0.0
        while True:
            now = self._clock()
            blocked = max(0.0, self._blocked_until.get(key, 0.0) - now)
            bucket_wait = max((bucket.wait_for(cost, now) for bucket, cost in costs), default=0.0)
            wait = max(bucket_wait, blocked)
            if wait == 0.0:
                for bucket, cost in costs:
                    bucket.consume(cost)
                return
            if wait > max_wait - waited:
                raise GatewayRateLimited(f'Rate limit reached for {key!r}', retry_after=wait)
            await self._sleep(wait)
            waited += wait
