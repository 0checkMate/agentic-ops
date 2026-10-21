import math
from collections.abc import Awaitable, Callable, Sequence

from langchain_core.messages import BaseMessage
from openai import APIConnectionError, APIStatusError, RateLimitError

from agentic_ops.config import Settings
from agentic_ops.gateway.errors import ProviderRateLimited, ProviderUnavailable
from agentic_ops.gateway.factory import build_chat_model
from agentic_ops.gateway.models import ModelSpec


def _parse_retry_after(value: str | None) -> float | None:
    """Seconds from a Retry-After header; None unless it is a finite, non-negative number.

    RFC 9110 also allows an HTTP-date. It is not supported: a malformed or unsupported
    header must never break the error path.
    """
    if value is None:
        return None
    try:
        seconds = float(value)
    except ValueError:
        return None
    if not math.isfinite(seconds) or seconds < 0:
        return None
    return seconds


def translate_provider_error(
    exc: Exception,
) -> ProviderRateLimited | ProviderUnavailable | None:
    """Map an SDK failure to the gateway's vocabulary, or None if another route cannot help.

    429 means this provider's budget is spent. Connection failures, timeouts and 5xx
    mean an outage. Everything else (bad request, bad key, bugs) is not translated, so
    it propagates instead of being replayed against every provider's quota.
    """
    name = type(exc).__name__
    if isinstance(exc, RateLimitError):
        retry_after = _parse_retry_after(exc.response.headers.get('retry-after'))
        return ProviderRateLimited(f'Provider rate limited ({name})', retry_after=retry_after)
    if isinstance(exc, APIConnectionError):
        return ProviderUnavailable(f'Provider unreachable ({name})')
    if isinstance(exc, APIStatusError) and exc.status_code >= 500:
        return ProviderUnavailable(f'Provider server error ({name}, {exc.status_code})')
    return None


def chat_call(
    settings: Settings, messages: Sequence[BaseMessage]
) -> Callable[[ModelSpec], Awaitable[BaseMessage]]:
    """Build the call(spec) coroutine that Gateway.run executes against each route."""

    async def call(spec: ModelSpec) -> BaseMessage:
        model = build_chat_model(spec, settings)
        try:
            return await model.ainvoke(list(messages))
        except Exception as exc:
            translated = translate_provider_error(exc)
            if translated is None:
                raise
            raise translated from exc

    return call
