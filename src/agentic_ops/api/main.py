"""Composition root: the only module that turns settings into live objects."""

from agentic_ops.config import Settings
from agentic_ops.gateway.chat_model import GatewayChatModel, provider_chat_call
from agentic_ops.gateway.defaults import default_limits, default_registry
from agentic_ops.gateway.ratelimit import RateLimiter
from agentic_ops.gateway.router import Gateway


def gateway_chat_model(settings: Settings) -> GatewayChatModel:
    """The production chat model: routing, rate limits and fallback on every request."""
    gateway = Gateway(default_registry(), RateLimiter(default_limits()))
    return GatewayChatModel(gateway=gateway, call=provider_chat_call(settings))
