import json
from typing import Any

from agentic_ops.gateway.audit import exhausted_payload, model_call_payload
from agentic_ops.gateway.errors import GatewayExhausted
from agentic_ops.gateway.models import Alias, ModelSpec, Skipped
from agentic_ops.gateway.router import RouteOutcome

SPEC_A = ModelSpec(provider='groq', model_id='model-a', supports_tools=True)
SPEC_B = ModelSpec(provider='gemini', model_id='model-b', supports_tools=True)


def _served(skipped: tuple[Skipped, ...] = ()) -> RouteOutcome[str]:
    return RouteOutcome(value='SECRET RESPONSE TEXT', spec=SPEC_B, skipped=skipped)


def _payload(outcome: RouteOutcome[Any] | None) -> dict[str, Any]:
    return model_call_payload(
        alias=Alias.DEFAULT, outcome=outcome, estimated_tokens=120, latency_ms=35.5
    )


def test_served_call_reports_route_and_metrics() -> None:
    assert _payload(_served()) == {
        'alias': 'default',
        'provider': 'gemini',
        'model_id': 'model-b',
        'fell_back': False,
        'skipped': [],
        'estimated_tokens': 120,
        'cache_hit': False,
        'latency_ms': 35.5,
    }


def test_fallback_records_skipped_routes_in_order_with_reasons() -> None:
    outcome = _served((Skipped(spec=SPEC_A, reason='rate_limited'),))
    payload = _payload(outcome)
    assert payload['fell_back'] is True
    assert payload['skipped'] == [
        {
            'spec': {'provider': 'groq', 'model_id': 'model-a', 'supports_tools': True},
            'reason': 'rate_limited',
        }
    ]


def test_cache_hit_has_no_serving_route() -> None:
    payload = _payload(None)
    assert payload['cache_hit'] is True
    assert payload['provider'] is None
    assert payload['model_id'] is None
    assert payload['fell_back'] is False
    assert payload['skipped'] == []


def test_payloads_are_strictly_json_safe() -> None:
    outcome = _served((Skipped(spec=SPEC_A, reason='unavailable'),))
    error = GatewayExhausted('all failed', skipped=(Skipped(spec=SPEC_A, reason='unavailable'),))
    for payload in (
        _payload(outcome),
        _payload(None),
        exhausted_payload(alias=Alias.FAST, error=error, estimated_tokens=1, latency_ms=2.0),
    ):
        json.dumps(payload, allow_nan=False)


def test_payload_never_contains_the_response_content() -> None:
    assert 'SECRET RESPONSE TEXT' not in json.dumps(_payload(_served()))


def test_exhausted_payload_records_every_skipped_route() -> None:
    error = GatewayExhausted(
        'all failed',
        skipped=(
            Skipped(spec=SPEC_A, reason='rate_limited'),
            Skipped(spec=SPEC_B, reason='unavailable'),
        ),
    )
    payload = exhausted_payload(alias=Alias.FAST, error=error, estimated_tokens=7, latency_ms=1.5)
    assert payload['alias'] == 'fast'
    assert payload['provider'] is None
    assert payload['model_id'] is None
    assert payload['cache_hit'] is False
    assert payload['fell_back'] is True
    assert [item['reason'] for item in payload['skipped']] == ['rate_limited', 'unavailable']


def test_served_payload_has_exactly_the_documented_keys() -> None:
    assert set(_payload(_served())) == {
        'alias',
        'provider',
        'model_id',
        'fell_back',
        'skipped',
        'estimated_tokens',
        'cache_hit',
        'latency_ms',
    }
