from collections.abc import Sequence
from typing import Any

import httpx2
import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
)

from agentic_ops.config import Settings
from agentic_ops.gateway.errors import (
    GatewayExhausted,
    ProviderRateLimited,
    ProviderUnavailable,
)
from agentic_ops.gateway.models import Alias, ModelSpec
from agentic_ops.gateway.provider_call import chat_call, translate_provider_error
from agentic_ops.gateway.ratelimit import RateLimit, RateLimiter
from agentic_ops.gateway.registry import ModelRegistry
from agentic_ops.gateway.router import Gateway

SPEC_A = ModelSpec(provider='groq', model_id='model-a', supports_tools=True)
SPEC_B = ModelSpec(provider='gemini', model_id='model-b', supports_tools=True)
_REQUEST = httpx2.Request('POST', 'https://example.test/v1/chat/completions')


def _status_error(
    cls: type[APIStatusError], status: int, headers: dict[str, str] | None = None
) -> APIStatusError:
    response = httpx2.Response(status, headers=headers, request=_REQUEST)
    return cls('boom', response=response, body=None)


# --- translate_provider_error ---


def test_rate_limit_with_retry_after_seconds() -> None:
    error = translate_provider_error(_status_error(RateLimitError, 429, {'retry-after': '7'}))
    assert isinstance(error, ProviderRateLimited)
    assert error.retry_after == 7.0


def test_rate_limit_without_header_has_no_retry_after() -> None:
    error = translate_provider_error(_status_error(RateLimitError, 429))
    assert isinstance(error, ProviderRateLimited)
    assert error.retry_after is None


@pytest.mark.parametrize('value', ['Wed, 21 Oct 2026 07:28:00 GMT', '-5', 'nan', 'inf', 'abc'])
def test_unusable_retry_after_becomes_none(value: str) -> None:
    error = translate_provider_error(_status_error(RateLimitError, 429, {'retry-after': value}))
    assert isinstance(error, ProviderRateLimited)
    assert error.retry_after is None


def test_connection_error_is_unavailable() -> None:
    error = translate_provider_error(APIConnectionError(request=_REQUEST))
    assert isinstance(error, ProviderUnavailable)


def test_timeout_is_unavailable() -> None:
    error = translate_provider_error(APITimeoutError(request=_REQUEST))
    assert isinstance(error, ProviderUnavailable)


@pytest.mark.parametrize('status', [500, 502, 503])
def test_server_errors_are_unavailable(status: int) -> None:
    error = translate_provider_error(_status_error(InternalServerError, status))
    assert isinstance(error, ProviderUnavailable)


@pytest.mark.parametrize(
    ('cls', 'status'),
    [
        (BadRequestError, 400),
        (AuthenticationError, 401),
        (PermissionDeniedError, 403),
        (NotFoundError, 404),
    ],
)
def test_client_errors_are_not_translated(cls: type[APIStatusError], status: int) -> None:
    assert translate_provider_error(_status_error(cls, status)) is None


def test_unrelated_exception_is_not_translated() -> None:
    assert translate_provider_error(ValueError('bug')) is None


# --- chat_call ---


class FakeModel:
    """Stands in for ChatOpenAI: returns a message or raises the configured exception."""

    def __init__(self, outcome: BaseMessage | Exception) -> None:
        self._outcome = outcome
        self.received: Sequence[BaseMessage] = ()

    async def ainvoke(self, messages: Sequence[BaseMessage]) -> BaseMessage:
        self.received = messages
        if isinstance(self._outcome, Exception):
            raise self._outcome
        return self._outcome


def _patch_build(monkeypatch: pytest.MonkeyPatch, model: FakeModel) -> list[ModelSpec]:
    built: list[ModelSpec] = []

    def fake_build(spec: ModelSpec, settings: Settings, **kwargs: Any) -> FakeModel:
        built.append(spec)
        return model

    monkeypatch.setattr('agentic_ops.gateway.provider_call.build_chat_model', fake_build)
    return built


_MESSAGES: list[BaseMessage] = [HumanMessage(content='hi')]


async def test_chat_call_returns_the_model_message_for_the_given_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = FakeModel(AIMessage(content='hello'))
    built = _patch_build(monkeypatch, model)
    result = await chat_call(Settings(), _MESSAGES)(SPEC_A)
    assert result.content == 'hello'
    assert built == [SPEC_A]
    assert model.received == _MESSAGES


async def test_chat_call_translates_a_provider_429(monkeypatch: pytest.MonkeyPatch) -> None:
    original = _status_error(RateLimitError, 429, {'retry-after': '3'})
    _patch_build(monkeypatch, FakeModel(original))
    with pytest.raises(ProviderRateLimited) as info:
        await chat_call(Settings(), _MESSAGES)(SPEC_A)
    assert info.value.retry_after == 3.0
    assert info.value.__cause__ is original


async def test_chat_call_translates_a_connection_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_build(monkeypatch, FakeModel(APIConnectionError(request=_REQUEST)))
    with pytest.raises(ProviderUnavailable):
        await chat_call(Settings(), _MESSAGES)(SPEC_A)


async def test_chat_call_lets_authentication_errors_propagate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_build(monkeypatch, FakeModel(_status_error(AuthenticationError, 401)))
    with pytest.raises(AuthenticationError):
        await chat_call(Settings(), _MESSAGES)(SPEC_A)


async def test_chat_call_lets_unrelated_exceptions_propagate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_build(monkeypatch, FakeModel(ValueError('bug')))
    with pytest.raises(ValueError, match='bug'):
        await chat_call(Settings(), _MESSAGES)(SPEC_A)


# --- Router handling of a provider-side 429 ---


def _gateway() -> Gateway:
    registry = ModelRegistry({Alias.DEFAULT: [SPEC_A, SPEC_B]})
    limiter = RateLimiter(
        {
            'groq:model-a': RateLimit(requests_per_minute=100),
            'gemini:model-b': RateLimit(requests_per_minute=100),
        }
    )
    return Gateway(registry, limiter)


async def test_router_skips_a_route_the_provider_rate_limited() -> None:
    async def call(spec: ModelSpec) -> str:
        if spec == SPEC_A:
            raise ProviderRateLimited('429', retry_after=5.0)
        return 'served'

    outcome = await _gateway().run(Alias.DEFAULT, call)
    assert outcome.value == 'served'
    assert outcome.spec == SPEC_B
    assert [(item.spec, item.reason) for item in outcome.skipped] == [(SPEC_A, 'rate_limited')]


async def test_router_exhausts_when_every_provider_rate_limits() -> None:
    async def call(spec: ModelSpec) -> str:
        raise ProviderRateLimited('429', retry_after=None)

    with pytest.raises(GatewayExhausted) as info:
        await _gateway().run(Alias.DEFAULT, call)
    assert [item.reason for item in info.value.skipped] == ['rate_limited', 'rate_limited']


class _RouterClock:
    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.now += seconds


def _penalty_gateway(clock: _RouterClock, *, default_penalty: float = 60.0) -> Gateway:
    registry = ModelRegistry({Alias.DEFAULT: [SPEC_A, SPEC_B]})
    limiter = RateLimiter(
        {
            'groq:model-a': RateLimit(requests_per_minute=100),
            'gemini:model-b': RateLimit(requests_per_minute=100),
        },
        clock=clock,
        sleep=clock.sleep,
    )
    return Gateway(registry, limiter, default_penalty=default_penalty)


async def test_provider_429_blocks_the_route_without_another_call() -> None:
    clock = _RouterClock()
    gateway = _penalty_gateway(clock)
    calls: list[ModelSpec] = []

    async def call(spec: ModelSpec) -> str:
        calls.append(spec)
        if spec == SPEC_A:
            raise ProviderRateLimited('429', retry_after=30.0)
        return 'served'

    await gateway.run(Alias.DEFAULT, call)
    calls.clear()
    outcome = await gateway.run(Alias.DEFAULT, call)
    assert calls == [SPEC_B]
    assert [(item.spec, item.reason) for item in outcome.skipped] == [(SPEC_A, 'rate_limited')]


async def test_route_is_retried_after_the_penalty_elapses() -> None:
    clock = _RouterClock()
    gateway = _penalty_gateway(clock)
    calls: list[ModelSpec] = []

    tripped = False

    async def call(spec: ModelSpec) -> str:
        nonlocal tripped
        calls.append(spec)
        if spec == SPEC_A and not tripped:
            tripped = True
            raise ProviderRateLimited('429', retry_after=30.0)
        return 'served'

    await gateway.run(Alias.DEFAULT, call)
    clock.now += 31
    calls.clear()
    outcome = await gateway.run(Alias.DEFAULT, call)
    assert outcome.spec == SPEC_A
    assert calls == [SPEC_A]


async def test_missing_retry_after_uses_the_default_penalty() -> None:
    clock = _RouterClock()
    gateway = _penalty_gateway(clock, default_penalty=45.0)
    calls: list[ModelSpec] = []

    tripped = False

    async def call(spec: ModelSpec) -> str:
        nonlocal tripped
        calls.append(spec)
        if spec == SPEC_A and not tripped:
            tripped = True
            raise ProviderRateLimited('429', retry_after=None)
        return 'served'

    await gateway.run(Alias.DEFAULT, call)
    clock.now += 40
    calls.clear()
    assert (await gateway.run(Alias.DEFAULT, call)).spec == SPEC_B
    clock.now += 10
    calls.clear()
    assert (await gateway.run(Alias.DEFAULT, call)).spec == SPEC_A
