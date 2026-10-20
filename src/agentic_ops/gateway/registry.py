from collections.abc import Mapping, Sequence

from agentic_ops.gateway.errors import GatewayConfigError
from agentic_ops.gateway.models import Alias, ModelSpec


class ModelRegistry:
    """Maps each alias to an ordered fallback route, validated at construction."""

    def __init__(self, routes: Mapping[Alias, Sequence[ModelSpec]]) -> None:
        validated: dict[Alias, tuple[ModelSpec, ...]] = {}
        for alias, specs in routes.items():
            if not specs:
                raise GatewayConfigError(f'Alias {alias.value!r} has an empty route')
            if len(set(specs)) != len(specs):
                raise GatewayConfigError(f'Alias {alias.value!r} has duplicate route entries')
            validated[alias] = tuple(specs)
        self._routes = validated

    def resolve(self, alias: Alias, *, require_tools: bool = False) -> list[ModelSpec]:
        """Return the ordered route for an alias, optionally limited to tool-capable models."""
        route = self._routes.get(alias)
        if route is None:
            raise GatewayConfigError(f'Alias {alias.value!r} is not configured')
        specs = [spec for spec in route if spec.supports_tools or not require_tools]
        if not specs:
            raise GatewayConfigError(f'Alias {alias.value!r} has no tool-capable model')
        return specs
