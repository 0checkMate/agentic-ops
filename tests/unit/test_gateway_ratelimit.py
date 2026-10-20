import math

import pytest

from agentic_ops.gateway.errors import GatewayConfigError, GatewayRateLimited
from agentic_ops.gateway.ratelimit import RateLimit, RateLimiter


class FakeClock:
    """Deterministic time source: sleeping advances the clock instantly."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.now += seconds


def _limiter(clock: FakeClock, **limits: RateLimit) -> RateLimiter:
    return RateLimiter(limits, clock=clock, sleep=clock.sleep)


async def test_requests_within_capacity_do_not_wait() -> None:
    clock = FakeClock()
    limiter = _limiter(clock, p=RateLimit(requests_per_minute=3))
    for _ in range(3):
        await limiter.acquire('p', max_wait=0.0)
    assert clock.now == 0.0


async def test_request_over_capacity_waits_for_refill() -> None:
    clock = FakeClock()
    limiter = _limiter(clock, p=RateLimit(requests_per_minute=2))
    for _ in range(3):
        await limiter.acquire('p', max_wait=60.0)
    assert clock.now == pytest.approx(30.0)


async def test_token_limit_waits_proportionally() -> None:
    clock = FakeClock()
    limiter = _limiter(clock, p=RateLimit(tokens_per_minute=6000))
    await limiter.acquire('p', tokens=6000, max_wait=60.0)
    await limiter.acquire('p', tokens=1000, max_wait=60.0)
    assert clock.now == pytest.approx(10.0)


async def test_wait_beyond_max_wait_raises_without_sleeping() -> None:
    clock = FakeClock()
    limiter = _limiter(clock, p=RateLimit(requests_per_day=1))
    await limiter.acquire('p', max_wait=5.0)
    with pytest.raises(GatewayRateLimited) as excinfo:
        await limiter.acquire('p', max_wait=5.0)
    assert clock.now == 0.0
    assert excinfo.value.retry_after == pytest.approx(86400.0)


async def test_request_larger_than_token_capacity_can_never_succeed() -> None:
    clock = FakeClock()
    limiter = _limiter(clock, p=RateLimit(tokens_per_minute=1000))
    with pytest.raises(GatewayRateLimited) as excinfo:
        await limiter.acquire('p', tokens=2000, max_wait=3600.0)
    assert math.isinf(excinfo.value.retry_after)
    assert clock.now == 0.0


async def test_rejected_acquire_does_not_consume_other_buckets() -> None:
    clock = FakeClock()
    limiter = _limiter(clock, p=RateLimit(requests_per_minute=2, tokens_per_minute=100))
    await limiter.acquire('p', tokens=100, max_wait=0.0)
    with pytest.raises(GatewayRateLimited):
        await limiter.acquire('p', tokens=100, max_wait=0.0)
    clock.now = 30.0
    await limiter.acquire('p', tokens=50, max_wait=0.0)
    # Needs a second request slot: only free if the rejected call consumed nothing.
    await limiter.acquire('p', tokens=0, max_wait=0.0)


async def test_keys_are_independent() -> None:
    clock = FakeClock()
    one = RateLimit(requests_per_minute=1)
    limiter = _limiter(clock, a=one, b=one)
    await limiter.acquire('a', max_wait=0.0)
    await limiter.acquire('b', max_wait=0.0)
    assert clock.now == 0.0


async def test_unknown_key_is_a_configuration_error() -> None:
    limiter = _limiter(FakeClock(), p=RateLimit(requests_per_minute=1))
    with pytest.raises(GatewayConfigError):
        await limiter.acquire('missing', max_wait=0.0)
