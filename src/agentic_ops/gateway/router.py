from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from agentic_ops.gateway.errors import (
    GatewayConfigError,
    GatewayExhausted,
    GatewayRateLimited,
    ProviderUnavailable,
)
from agentic_ops.gateway.models import Alias, ModelSpec, Skipped
from agentic_ops.gateway.ratelimit import RateLimiter
from agentic_ops.gateway.registry import ModelRegistry


def limiter_key(spec: ModelSpec) -> str:
    """Limiter key for one route; per provider and model, as Groq and Cerebras limit."""
    return f'{spec.provider}:{spec.model_id}'


@dataclass(frozen=True)
class RouteOutcome[T]:
    """The result of a routed call, plus which route served it and which were skipped."""

    value: T
    spec: ModelSpec
    skipped: tuple[Skipped, ...]

    @property
    def fell_back(self) -> bool:
        return bool(self.skipped)


class Gateway:
    """Runs a call down an alias's ordered routes, gated by per-route budgets."""

    def __init__(
        self,
        registry: ModelRegistry,
        limiter: RateLimiter,
        *,
        last_route_wait: float = 0.0,
    ) -> None:
        missing = sorted({limiter_key(spec) for spec in registry.all_specs()} - limiter.keys)
        if missing:
            names = ', '.join(missing)
            raise GatewayConfigError(f'No rate limit configured for routes: {names}')
        self._registry = registry
        self._limiter = limiter
        self._last_route_wait = last_route_wait

    async def run[T](
        self,
        alias: Alias,
        call: Callable[[ModelSpec], Awaitable[T]],
        *,
        require_tools: bool = False,
        estimated_tokens: int = 0,
    ) -> RouteOutcome[T]:
        """Try each route in priority order until one serves the call.

        A route is skipped, not retried, when its budget is exhausted or the provider
        reports ProviderUnavailable. Any other exception is a bug or a bad request and
        propagates, so it is never replayed against every provider's quota.
        """
        route = self._registry.resolve(alias, require_tools=require_tools)
        skipped: list[Skipped] = []
        for index, spec in enumerate(route):
            is_last = index == len(route) - 1
            max_wait = self._last_route_wait if is_last else 0.0
            try:
                await self._limiter.acquire(limiter_key(spec), estimated_tokens, max_wait=max_wait)
            except GatewayRateLimited:
                skipped.append(Skipped(spec=spec, reason='rate_limited'))
                continue
            try:
                value = await call(spec)
            except ProviderUnavailable:
                skipped.append(Skipped(spec=spec, reason='unavailable'))
                continue
            return RouteOutcome(value=value, spec=spec, skipped=tuple(skipped))
        raise GatewayExhausted(
            f'All routes failed for alias {alias.value!r}', skipped=tuple(skipped)
        )
