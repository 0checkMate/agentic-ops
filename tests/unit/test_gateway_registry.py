import pytest

from agentic_ops.gateway.errors import GatewayConfigError
from agentic_ops.gateway.models import Alias, ModelSpec
from agentic_ops.gateway.registry import ModelRegistry

# Placeholder identifiers: real model IDs are verified when the benchmark step is reached.
GROQ = ModelSpec(provider='groq', model_id='model-a', supports_tools=True)
CEREBRAS = ModelSpec(provider='cerebras', model_id='model-b', supports_tools=True)
NO_TOOLS = ModelSpec(provider='openrouter', model_id='model-c', supports_tools=False)


def test_resolve_preserves_priority_order() -> None:
    registry = ModelRegistry({Alias.DEFAULT: [GROQ, CEREBRAS, NO_TOOLS]})
    assert registry.resolve(Alias.DEFAULT) == [GROQ, CEREBRAS, NO_TOOLS]


def test_require_tools_filters_incapable_models() -> None:
    registry = ModelRegistry({Alias.DEFAULT: [NO_TOOLS, GROQ]})
    assert registry.resolve(Alias.DEFAULT, require_tools=True) == [GROQ]


def test_require_tools_with_no_capable_model_raises() -> None:
    registry = ModelRegistry({Alias.FAST: [NO_TOOLS]})
    with pytest.raises(GatewayConfigError):
        registry.resolve(Alias.FAST, require_tools=True)


def test_unconfigured_alias_raises() -> None:
    registry = ModelRegistry({Alias.DEFAULT: [GROQ]})
    with pytest.raises(GatewayConfigError):
        registry.resolve(Alias.REASONING)


def test_duplicate_route_entries_are_rejected() -> None:
    with pytest.raises(GatewayConfigError):
        ModelRegistry({Alias.DEFAULT: [GROQ, GROQ]})


def test_empty_route_is_rejected() -> None:
    with pytest.raises(GatewayConfigError):
        ModelRegistry({Alias.DEFAULT: []})


def test_resolve_returns_a_copy() -> None:
    registry = ModelRegistry({Alias.DEFAULT: [GROQ, CEREBRAS]})
    registry.resolve(Alias.DEFAULT).clear()
    assert registry.resolve(Alias.DEFAULT) == [GROQ, CEREBRAS]
