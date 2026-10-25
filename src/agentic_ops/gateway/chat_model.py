import json
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import ConfigDict

from agentic_ops.config import Settings
from agentic_ops.gateway.factory import build_chat_model
from agentic_ops.gateway.models import Alias, ModelSpec
from agentic_ops.gateway.provider_call import translate_provider_error
from agentic_ops.gateway.router import Gateway

ChatCall = Callable[
    [ModelSpec, Sequence[BaseMessage], Sequence[dict[str, Any]]], Awaitable[BaseMessage]
]

_OUTPUT_ALLOWANCE = 150


def estimate_request_tokens(
    messages: Sequence[BaseMessage], tools: Sequence[dict[str, Any]]
) -> int:
    """Conservative request size: four characters per token, schemas included, plus output."""
    text = ''.join(
        message.content
        if isinstance(message.content, str)
        else json.dumps(message.content, default=str)
        for message in messages
    )
    schemas = json.dumps(list(tools), separators=(',', ':'), default=str)
    return -(-(len(text) + len(schemas)) // 4) + _OUTPUT_ALLOWANCE


class GatewayChatModel(BaseChatModel):
    """LangChain chat model whose every request goes through Gateway.run.

    Routing, rate limits and fallback stay in the gateway; this class only translates
    LangChain messages and tool schemas into a ChatCall. Async only. Stop sequences and
    tool_choice are not supported.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    gateway: Gateway
    call: ChatCall
    route_alias: Alias = Alias.DEFAULT
    bound_tools: list[dict[str, Any]] = []

    @property
    def _llm_type(self) -> str:
        return 'agentic-ops-gateway'

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        if tool_choice is not None:
            raise NotImplementedError('tool_choice is not supported by GatewayChatModel')
        schemas = [convert_to_openai_tool(tool) for tool in tools]
        return self.model_copy(update={'bound_tools': schemas})

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise NotImplementedError('GatewayChatModel is async only; use ainvoke')

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        tools = self.bound_tools

        async def attempt(route: ModelSpec) -> BaseMessage:
            return await self.call(route, messages, tools)

        outcome = await self.gateway.run(
            self.route_alias,
            attempt,
            require_tools=bool(tools),
            estimated_tokens=estimate_request_tokens(messages, tools),
        )
        message = outcome.value
        if not isinstance(message, AIMessage):
            raise TypeError(f'Expected an AIMessage, got {type(message).__name__}')
        return ChatResult(generations=[ChatGeneration(message=message)])


def provider_chat_call(settings: Settings) -> ChatCall:
    """The real ChatCall: build the route's model, bind tools, translate provider failures."""

    async def call(
        spec: ModelSpec, messages: Sequence[BaseMessage], tools: Sequence[dict[str, Any]]
    ) -> BaseMessage:
        model = build_chat_model(spec, settings)
        try:
            if tools:
                return await model.bind_tools(list(tools)).ainvoke(list(messages))
            return await model.ainvoke(list(messages))
        except Exception as exc:
            translated = translate_provider_error(exc)
            if translated is None:
                raise
            raise translated from exc

    return call
