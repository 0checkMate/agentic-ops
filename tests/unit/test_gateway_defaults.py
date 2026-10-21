from agentic_ops.gateway.defaults import default_limits, default_registry
from agentic_ops.gateway.models import Alias
from agentic_ops.gateway.providers import PROVIDERS
from agentic_ops.gateway.ratelimit import RateLimit, RateLimiter
from agentic_ops.gateway.router import Gateway, limiter_key


def test_every_alias_resolves_to_tool_capable_routes() -> None:
    registry = default_registry()
    for alias in Alias:
        assert registry.resolve(alias, require_tools=True)


def test_every_route_has_a_limit() -> None:
    routed = {limiter_key(spec) for spec in default_registry().all_specs()}
    assert routed <= set(default_limits())


def test_no_limit_is_orphaned() -> None:
    routed = {limiter_key(spec) for spec in default_registry().all_specs()}
    assert set(default_limits()) == routed


def test_gateway_accepts_the_defaults() -> None:
    Gateway(default_registry(), RateLimiter(default_limits()))


def test_routes_use_known_providers() -> None:
    for spec in default_registry().all_specs():
        assert spec.provider in PROVIDERS


def test_groq_is_primary_for_every_alias() -> None:
    registry = default_registry()
    for alias in Alias:
        assert registry.resolve(alias)[0].provider == 'groq'


def test_cerebras_is_not_routed() -> None:
    assert all(spec.provider != 'cerebras' for spec in default_registry().all_specs())


def test_groq_120b_limits_match_the_published_free_plan() -> None:
    assert default_limits()['groq:openai/gpt-oss-120b'] == RateLimit(
        requests_per_minute=30,
        tokens_per_minute=8000,
        requests_per_day=1000,
        tokens_per_day=200000,
    )


def test_openrouter_free_limits_assume_no_purchased_credits() -> None:
    limit = default_limits()['openrouter:openai/gpt-oss-120b:free']
    assert (limit.requests_per_minute, limit.requests_per_day) == (20, 50)
