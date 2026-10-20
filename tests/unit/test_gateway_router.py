import pytest

from agentic_ops.gateway.errors import GatewayConfigError, GatewayExhausted, ProviderUnavailable
from agentic_ops.gateway.models import Alias, ModelSpec, Skipped
from agentic_ops.gateway.ratelimit import RateLimit, RateLimiter
from agentic_ops.gateway.registry import ModelRegistry
from agentic_ops.gateway.router import Gateway, limiter_key

A = ModelSpec(provider='groq', model_id='model-a', supports_tools=True)
B = ModelSpec(provider='cerebras', model_id='model-b', supports_tools=True)
C = ModelSpec(provider='openrouter', model_id='model-c', supports_tools=False)
PLENTY = RateLimit(requests_per_minute=1000)


class Recorder:
    """Stands in for a provider call: records usage and fails on demand."""

    def __init__(self, failing: dict[str, Exception] | None = None) -> None:
        self.calls: list[str] = []
        self._failing = failing or {}

    async def __call__(self, spec: ModelSpec) -> str:
        self.calls.append(spec.model_id)
        error = self._failing.get(spec.model_id)
        if error is not None:
            raise error
        return f'ok:{spec.model_id}'


def _gateway(
    specs: list[ModelSpec],
    *,
    overrides: dict[str, RateLimit] | None = None,
    last_route_wait: float = 0.0,
) -> Gateway:
    limits = {limiter_key(spec): PLENTY for spec in specs}
    limits.update(overrides or {})
    registry = ModelRegistry({Alias.DEFAULT: specs})
    return Gateway(registry, RateLimiter(limits), last_route_wait=last_route_wait)


async def test_primary_serves_when_healthy() -> None:
    recorder = Recorder()
    outcome = await _gateway([A, B]).run(Alias.DEFAULT, recorder)
    assert outcome.value == 'ok:model-a'
    assert outcome.spec == A
    assert outcome.skipped == ()
    assert not outcome.fell_back
    assert recorder.calls == ['model-a']


async def test_provider_failure_falls_back_to_next_route() -> None:
    recorder = Recorder({'model-a': ProviderUnavailable('503')})
    outcome = await _gateway([A, B]).run(Alias.DEFAULT, recorder)
    assert outcome.spec == B
    assert outcome.skipped == (Skipped(spec=A, reason='unavailable'),)
    assert outcome.fell_back
    assert recorder.calls == ['model-a', 'model-b']


async def test_exhausted_budget_skips_route_without_calling_it() -> None:
    one = RateLimit(requests_per_minute=1)
    gateway = _gateway([A, B], overrides={limiter_key(A): one})
    recorder = Recorder()
    await gateway.run(Alias.DEFAULT, recorder)
    outcome = await gateway.run(Alias.DEFAULT, recorder)
    assert outcome.spec == B
    assert outcome.skipped == (Skipped(spec=A, reason='rate_limited'),)
    assert recorder.calls == ['model-a', 'model-b']


async def test_oversized_request_skips_route_that_can_never_fit() -> None:
    small = RateLimit(tokens_per_minute=1000)
    gateway = _gateway([A, B], overrides={limiter_key(A): small})
    outcome = await gateway.run(Alias.DEFAULT, Recorder(), estimated_tokens=2000)
    assert outcome.spec == B
    assert outcome.skipped == (Skipped(spec=A, reason='rate_limited'),)


async def test_all_routes_failing_raises_exhausted_with_reasons() -> None:
    recorder = Recorder(
        {'model-a': ProviderUnavailable('503'), 'model-b': ProviderUnavailable('timeout')}
    )
    with pytest.raises(GatewayExhausted) as excinfo:
        await _gateway([A, B]).run(Alias.DEFAULT, recorder)
    assert [skipped.reason for skipped in excinfo.value.skipped] == ['unavailable', 'unavailable']


async def test_require_tools_never_calls_tool_incapable_route() -> None:
    recorder = Recorder()
    outcome = await _gateway([C, A]).run(Alias.DEFAULT, recorder, require_tools=True)
    assert outcome.spec == A
    assert recorder.calls == ['model-a']


async def test_unexpected_errors_propagate_without_fallback() -> None:
    recorder = Recorder({'model-a': ValueError('programming error')})
    with pytest.raises(ValueError):
        await _gateway([A, B]).run(Alias.DEFAULT, recorder)
    assert recorder.calls == ['model-a']


async def test_last_route_waits_briefly_instead_of_failing() -> None:
    burst = RateLimit(tokens_per_minute=6_000_000)
    gateway = _gateway([A], overrides={limiter_key(A): burst}, last_route_wait=1.0)
    recorder = Recorder()
    await gateway.run(Alias.DEFAULT, recorder, estimated_tokens=6_000_000)
    outcome = await gateway.run(Alias.DEFAULT, recorder, estimated_tokens=10_000)
    assert outcome.value == 'ok:model-a'
    assert recorder.calls == ['model-a', 'model-a']


async def test_last_route_fails_fast_by_default() -> None:
    one = RateLimit(requests_per_minute=1)
    gateway = _gateway([A], overrides={limiter_key(A): one})
    await gateway.run(Alias.DEFAULT, Recorder())
    with pytest.raises(GatewayExhausted):
        await gateway.run(Alias.DEFAULT, Recorder())


def test_route_without_a_configured_limit_fails_at_construction() -> None:
    registry = ModelRegistry({Alias.DEFAULT: [A]})
    with pytest.raises(GatewayConfigError):
        Gateway(registry, RateLimiter({}))
