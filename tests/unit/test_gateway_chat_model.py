from collections.abc import Sequence
from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.utils.function_calling import convert_to_openai_tool

from agentic_ops.gateway.chat_model import (
    GatewayChatModel,
    estimate_request_tokens,
)
from agentic_ops.gateway.defaults import (
    GROQ_20B,
    GROQ_120B,
    OPENROUTER_120B,
    default_limits,
    default_registry,
)
from agentic_ops.gateway.errors import ProviderRateLimited
from agentic_ops.gateway.models import Alias, ModelSpec
from agentic_ops.gateway.ratelimit import RateLimiter
from agentic_ops.gateway.router import Gateway


def get_stock(sku: str) -> int:
    """Return units on hand for a SKU."""
    return 0


class Recorder:
    """Fake ChatCall: records every invocation and replies with a fixed AIMessage."""

    def __init__(self, *, reply: AIMessage | None = None, rate_limit_first: bool = False) -> None:
        self.reply = reply if reply is not None else AIMessage(content='ok')
        self.specs: list[ModelSpec] = []
        self.messages: list[list[BaseMessage]] = []
        self.tools: list[list[dict[str, Any]]] = []
        self._rate_limit_first = rate_limit_first

    async def __call__(
        self,
        spec: ModelSpec,
        messages: Sequence[BaseMessage],
        tools: Sequence[dict[str, Any]],
    ) -> BaseMessage:
        self.specs.append(spec)
        self.messages.append(list(messages))
        self.tools.append(list(tools))
        if self._rate_limit_first and len(self.specs) == 1:
            raise ProviderRateLimited('limited', retry_after=None)
        return self.reply


def make_model(recorder: Recorder, alias: Alias = Alias.DEFAULT) -> GatewayChatModel:
    gateway = Gateway(default_registry(), RateLimiter(default_limits()))
    return GatewayChatModel(gateway=gateway, call=recorder, route_alias=alias)


async def test_ainvoke_returns_the_call_reply() -> None:
    result = await make_model(Recorder(reply=AIMessage(content='42 units'))).ainvoke(
        [HumanMessage(content='stock?')]
    )
    assert result.content == '42 units'


async def test_messages_are_forwarded_to_the_call() -> None:
    recorder = Recorder()
    await make_model(recorder).ainvoke([HumanMessage(content='hello')])
    assert recorder.messages[0][0].content == 'hello'


async def test_unbound_model_sends_no_tools() -> None:
    recorder = Recorder()
    await make_model(recorder).ainvoke([HumanMessage(content='hello')])
    assert recorder.tools[0] == []


async def test_bind_tools_forwards_openai_schema() -> None:
    recorder = Recorder()
    bound = make_model(recorder).bind_tools([get_stock])
    await bound.ainvoke([HumanMessage(content='stock of A1?')])
    assert recorder.tools[0][0]['function']['name'] == 'get_stock'


async def test_tool_calls_are_preserved() -> None:
    reply = AIMessage(
        content='',
        tool_calls=[
            {'name': 'get_stock', 'args': {'sku': 'A1'}, 'id': 'call_1', 'type': 'tool_call'}
        ],
    )
    bound = make_model(Recorder(reply=reply)).bind_tools([get_stock])
    result = await bound.ainvoke([HumanMessage(content='stock of A1?')])
    assert isinstance(result, AIMessage)
    assert result.tool_calls[0]['args'] == {'sku': 'A1'}


async def test_rate_limited_route_falls_through_to_the_next() -> None:
    recorder = Recorder(rate_limit_first=True)
    result = await make_model(recorder).ainvoke([HumanMessage(content='hello')])
    assert result.content == 'ok'
    assert recorder.specs[:2] == [GROQ_120B, OPENROUTER_120B]


async def test_alias_selects_the_route_list() -> None:
    recorder = Recorder()
    await make_model(recorder, Alias.FAST).ainvoke([HumanMessage(content='hello')])
    assert recorder.specs[0] == GROQ_20B


def test_sync_invoke_is_not_supported() -> None:
    with pytest.raises(NotImplementedError):
        make_model(Recorder()).invoke([HumanMessage(content='hello')])


def test_estimate_grows_with_tool_schemas() -> None:
    messages: list[BaseMessage] = [HumanMessage(content='x' * 400)]
    schemas = [convert_to_openai_tool(get_stock)]
    assert estimate_request_tokens(messages, schemas) > estimate_request_tokens(messages, [])


def test_estimate_covers_a_quarter_of_the_characters() -> None:
    messages: list[BaseMessage] = [HumanMessage(content='x' * 400)]
    assert estimate_request_tokens(messages, []) > 100
