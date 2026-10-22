from collections.abc import Sequence
from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolCall

from agentic_ops.bench.cases import BenchCase, ToolDef
from agentic_ops.bench.observer import (
    ToolCallRejected,
    build_observer,
    estimate_tokens,
    extract_calls,
)
from agentic_ops.bench.runner import Observe
from agentic_ops.bench.scoring import ObservedCall
from agentic_ops.gateway.cache import InMemoryCache
from agentic_ops.gateway.errors import GatewayConfigError, GatewayExhausted, ProviderUnavailable
from agentic_ops.gateway.models import ModelSpec
from agentic_ops.gateway.ratelimit import RateLimit
from agentic_ops.gateway.router import limiter_key

SPEC_A = ModelSpec(provider='groq', model_id='model-a', supports_tools=True)
SPEC_B = ModelSpec(provider='groq', model_id='model-b', supports_tools=True)
NO_TOOLS = ModelSpec(provider='groq', model_id='model-c', supports_tools=False)
LIMITS = {
    limiter_key(spec): RateLimit(requests_per_minute=1000) for spec in (SPEC_A, SPEC_B, NO_TOOLS)
}
TOOL = ToolDef(
    name='get_stock_level',
    description='Look up stock.',
    parameters={
        'type': 'object',
        'properties': {'sku': {'type': 'string'}},
        'required': ['sku'],
    },
)
CASE = BenchCase(id='c1', prompt='How much FLOUR-001?', expected=None)


def tool_call(name: str, args: dict[str, Any]) -> ToolCall:
    return ToolCall(name=name, args=args, id='call-1')


class FakeModel:
    """Stands in for the real tool-bound model; records what it was asked."""

    def __init__(self, outcome: Sequence[ToolCall] | Exception) -> None:
        self._outcome = outcome
        self.invocations = 0
        self.last_prompt = ''
        self.last_tools: tuple[str, ...] = ()

    async def __call__(self, spec: ModelSpec, prompt: str, tools: Sequence[ToolDef]) -> BaseMessage:
        self.invocations += 1
        self.last_prompt = prompt
        self.last_tools = tuple(tool.name for tool in tools)
        if isinstance(self._outcome, Exception):
            raise self._outcome
        return AIMessage(content='', tool_calls=list(self._outcome))


def observer_for(spec: ModelSpec, model: FakeModel, cache: InMemoryCache | None = None) -> Observe:
    shared = cache if cache is not None else InMemoryCache()
    return build_observer(spec, [TOOL], limits=LIMITS, cache=shared, call_model=model)


def test_extract_maps_tool_calls() -> None:
    message = AIMessage(content='', tool_calls=[tool_call('get_stock_level', {'sku': 'A'})])
    assert extract_calls(message) == (ObservedCall(name='get_stock_level', args={'sku': 'A'}),)


def test_extract_returns_nothing_for_a_plain_answer() -> None:
    assert extract_calls(AIMessage(content='HACCP is a food safety system.')) == ()


def test_extract_rejects_a_non_ai_message() -> None:
    with pytest.raises(TypeError):
        extract_calls(HumanMessage(content='hi'))


def test_estimate_grows_with_the_prompt() -> None:
    assert estimate_tokens('a' * 400, [TOOL]) > estimate_tokens('a' * 40, [TOOL])


def test_estimate_includes_the_tool_schemas() -> None:
    assert estimate_tokens('hi', [TOOL]) > estimate_tokens('hi', [])


def test_estimate_is_at_least_one() -> None:
    assert estimate_tokens('', []) >= 1


async def test_second_observation_is_served_from_the_cache() -> None:
    model = FakeModel([tool_call('get_stock_level', {'sku': 'FLOUR-001'})])
    observe = observer_for(SPEC_A, model)
    first = await observe(CASE)
    second = await observe(CASE)
    assert (first.cache_hit, second.cache_hit) == (False, True)
    assert model.invocations == 1
    assert second.calls == (ObservedCall(name='get_stock_level', args={'sku': 'FLOUR-001'}),)


async def test_an_abstain_result_is_cached_too() -> None:
    model = FakeModel([])
    observe = observer_for(SPEC_A, model)
    first = await observe(CASE)
    second = await observe(CASE)
    assert first.calls == () and second.calls == ()
    assert second.cache_hit is True
    assert model.invocations == 1


async def test_models_never_share_cache_entries() -> None:
    cache = InMemoryCache()
    model_a = FakeModel([tool_call('get_stock_level', {'sku': 'A'})])
    model_b = FakeModel([])
    await observer_for(SPEC_A, model_a, cache)(CASE)
    second = await observer_for(SPEC_B, model_b, cache)(CASE)
    assert second.cache_hit is False
    assert second.calls == ()
    assert model_b.invocations == 1


async def test_failures_are_not_cached() -> None:
    model = FakeModel(ProviderUnavailable('down'))
    observe = observer_for(SPEC_A, model)
    with pytest.raises(GatewayExhausted):
        await observe(CASE)
    with pytest.raises(GatewayExhausted):
        await observe(CASE)
    assert model.invocations == 2


async def test_model_receives_the_prompt_and_tool_schemas() -> None:
    model = FakeModel([])
    await observer_for(SPEC_A, model)(CASE)
    assert model.last_prompt == CASE.prompt
    assert model.last_tools == ('get_stock_level',)


def test_missing_rate_limit_fails_at_build_time() -> None:
    with pytest.raises(GatewayConfigError, match='No rate limit'):
        build_observer(SPEC_A, [TOOL], limits={}, cache=InMemoryCache(), call_model=FakeModel([]))


def test_model_without_tool_support_fails_at_build_time() -> None:
    with pytest.raises(GatewayConfigError, match='tool'):
        observer_for(NO_TOOLS, FakeModel([]))


async def test_rejected_call_is_reported_and_cached() -> None:
    model = FakeModel(ToolCallRejected('schema violation'))
    observe = observer_for(SPEC_A, model)
    first = await observe(CASE)
    second = await observe(CASE)
    assert first.rejected and second.rejected
    assert first.calls == ()
    assert second.cache_hit is True
    assert model.invocations == 1
