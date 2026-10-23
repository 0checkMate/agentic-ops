from agentic_ops.gateway.defaults import (
    GEMINI_LITE,
    GROQ_20B,
    GROQ_120B,
    OPENROUTER_120B,
    default_registry,
)
from agentic_ops.gateway.models import Alias


def test_fast_route_order() -> None:
    assert default_registry().resolve(Alias.FAST) == [GROQ_20B, OPENROUTER_120B, GEMINI_LITE]


def test_default_route_order() -> None:
    assert default_registry().resolve(Alias.DEFAULT) == [GROQ_120B, OPENROUTER_120B, GEMINI_LITE]


def test_reasoning_route_is_unchanged() -> None:
    assert default_registry().resolve(Alias.REASONING) == [GROQ_120B, OPENROUTER_120B]


def test_gemini_is_never_ahead_of_openrouter() -> None:
    registry = default_registry()
    for alias in Alias:
        providers = [spec.provider for spec in registry.resolve(alias)]
        if 'gemini' in providers and 'openrouter' in providers:
            assert providers.index('openrouter') < providers.index('gemini')
