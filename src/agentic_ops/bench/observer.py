import json
import math
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from agentic_ops.bench.cases import BenchCase, ToolDef
from agentic_ops.bench.runner import Observation, Observe
from agentic_ops.bench.scoring import ObservedCall
from agentic_ops.config import Settings
from agentic_ops.gateway.cache import ResponseCache, cache_key, get_or_compute
from agentic_ops.gateway.errors import GatewayConfigError
from agentic_ops.gateway.factory import build_chat_model
from agentic_ops.gateway.models import Alias, ModelSpec
from agentic_ops.gateway.provider_call import translate_provider_error
from agentic_ops.gateway.ratelimit import RateLimit, RateLimiter
from agentic_ops.gateway.registry import ModelRegistry
from agentic_ops.gateway.router import Gateway, limiter_key

ModelCall = Callable[[ModelSpec, str, Sequence[ToolDef]], Awaitable[BaseMessage]]

TEMPERATURE = 0.0
_OUTPUT_ALLOWANCE = 150
_MAX_WAIT_SECONDS = 120.0


def extract_calls(message: BaseMessage) -> tuple[ObservedCall, ...]:
    """Decode the tool calls from a model reply; a plain answer yields no calls."""
    if not isinstance(message, AIMessage):
        raise TypeError(f'Expected an AIMessage, got {type(message).__name__}')
    return tuple(
        ObservedCall(name=call['name'], args=dict(call['args'])) for call in message.tool_calls
    )


def estimate_tokens(prompt: str, tools: Sequence[ToolDef]) -> int:
    """Conservative request size: four characters per token, schemas included, plus output."""
    schemas = json.dumps([tool.to_openai() for tool in tools], separators=(',', ':'))
    return math.ceil((len(prompt) + len(schemas)) / 4) + _OUTPUT_ALLOWANCE


def build_observer(
    spec: ModelSpec,
    tools: Sequence[ToolDef],
    *,
    limits: Mapping[str, RateLimit],
    cache: ResponseCache,
    call_model: ModelCall,
) -> Observe:
    """Build the per-model observer: cache first, then the gateway on a single-route registry."""
    key = limiter_key(spec)
    if not spec.supports_tools:
        raise GatewayConfigError(f'Model {spec.model_id!r} does not support tool calling')
    if key not in limits:
        raise GatewayConfigError(f'No rate limit configured for route {key!r}')
    gateway = Gateway(
        ModelRegistry({Alias.DEFAULT: [spec]}),
        RateLimiter({key: limits[key]}),
        last_route_wait=_MAX_WAIT_SECONDS,
    )
    schemas = [tool.to_openai() for tool in tools]

    async def observe(case: BenchCase) -> Observation:
        request: dict[str, Any] = {
            'prompt': case.prompt,
            'tools': schemas,
            'temperature': TEMPERATURE,
        }
        estimate = estimate_tokens(case.prompt, tools)

        async def compute() -> dict[str, Any]:
            async def call(route: ModelSpec) -> BaseMessage:
                return await call_model(route, case.prompt, tools)

            outcome = await gateway.run(
                Alias.DEFAULT, call, require_tools=True, estimated_tokens=estimate
            )
            calls = extract_calls(outcome.value)
            return {'calls': [item.model_dump(mode='json') for item in calls]}

        result = await get_or_compute(
            cache, cache_key(namespace=f'bench:{key}', request=request), compute
        )
        calls = tuple(ObservedCall.model_validate(item) for item in result.value['calls'])
        return Observation(calls=calls, cache_hit=result.hit)

    return observe


def tool_bound_call(settings: Settings) -> ModelCall:
    """The real model call: bind the tool schemas, invoke, translate provider failures."""

    async def call(spec: ModelSpec, prompt: str, tools: Sequence[ToolDef]) -> BaseMessage:
        model = build_chat_model(spec, settings, temperature=TEMPERATURE)
        bound = model.bind_tools([tool.to_openai() for tool in tools])
        try:
            return await bound.ainvoke([HumanMessage(content=prompt)])
        except Exception as exc:
            translated = translate_provider_error(exc)
            if translated is None:
                raise
            raise translated from exc

    return call
