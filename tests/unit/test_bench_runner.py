import asyncio
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

import pytest

from agentic_ops.bench.cases import BenchCase, CaseSuite
from agentic_ops.bench.runner import CaseResult, Observation, run_suite, summarize
from agentic_ops.bench.scoring import ObservedCall, Verdict
from agentic_ops.gateway.errors import GatewayExhausted

STOCK = ObservedCall(name='get_stock_level', args={'sku': 'A'})
OTHER = ObservedCall(name='list_open_purchase_orders', args={})


def make_suite() -> CaseSuite:
    expected: dict[str, Any] = {'name': 'get_stock_level', 'args': {'sku': 'A'}}
    tool: dict[str, Any] = {
        'name': 'get_stock_level',
        'description': 'Look up stock.',
        'parameters': {
            'type': 'object',
            'properties': {'sku': {'type': 'string'}},
            'required': ['sku'],
        },
    }
    return CaseSuite.model_validate(
        {
            'version': 'test-1',
            'tools': [tool],
            'cases': [
                {'id': 'c1', 'prompt': 'p1', 'expected': expected},
                {'id': 'c2', 'prompt': 'p2', 'expected': expected},
                {'id': 'c3', 'prompt': 'p3', 'expected': None},
            ],
        }
    )


def seen(*calls: ObservedCall, cache_hit: bool = False) -> Observation:
    return Observation(calls=tuple(calls), cache_hit=cache_hit)


def scripted(
    outcomes: Mapping[str, Observation | Exception],
) -> Callable[[BenchCase], Awaitable[Observation]]:
    async def observe(case: BenchCase) -> Observation:
        outcome = outcomes[case.id]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    return observe


def flat_clock() -> float:
    return 0.0


def ticking(values: list[float]) -> Callable[[], float]:
    iterator = iter(values)
    return lambda: next(iterator)


def result(verdict: Verdict | None, *, latency: float = 100.0, hit: bool = False) -> CaseResult:
    error = None if verdict is not None else 'GatewayExhausted: boom'
    return CaseResult(case_id='x', verdict=verdict, latency_ms=latency, cache_hit=hit, error=error)


async def test_results_follow_case_order_and_carry_verdicts() -> None:
    observe = scripted({'c1': seen(STOCK), 'c2': seen(OTHER), 'c3': seen()})
    results = await run_suite(make_suite(), observe, clock=flat_clock)
    assert [item.case_id for item in results] == ['c1', 'c2', 'c3']
    assert [item.verdict for item in results] == [
        Verdict.CORRECT,
        Verdict.WRONG_TOOL,
        Verdict.CORRECT_ABSTAIN,
    ]


async def test_latency_is_measured_in_milliseconds() -> None:
    observe = scripted({'c1': seen(STOCK), 'c2': seen(STOCK), 'c3': seen()})
    clock = ticking([10.0, 10.25, 20.0, 20.5, 30.0, 30.001])
    results = await run_suite(make_suite(), observe, clock=clock)
    assert results[0].latency_ms == pytest.approx(250.0)
    assert results[1].latency_ms == pytest.approx(500.0)


async def test_cache_hit_flag_is_recorded() -> None:
    observe = scripted({'c1': seen(STOCK, cache_hit=True), 'c2': seen(STOCK), 'c3': seen()})
    results = await run_suite(make_suite(), observe, clock=flat_clock)
    assert [item.cache_hit for item in results] == [True, False, False]


async def test_gateway_error_becomes_an_error_result_and_the_run_continues() -> None:
    failure = GatewayExhausted('all routes failed', skipped=())
    observe = scripted({'c1': failure, 'c2': seen(STOCK), 'c3': seen()})
    results = await run_suite(make_suite(), observe, clock=flat_clock)
    assert results[0].verdict is None
    assert results[0].error is not None
    assert 'GatewayExhausted' in results[0].error
    assert results[1].verdict is Verdict.CORRECT


async def test_other_exceptions_propagate() -> None:
    observe = scripted({'c1': RuntimeError('bug'), 'c2': seen(STOCK), 'c3': seen()})
    with pytest.raises(RuntimeError):
        await run_suite(make_suite(), observe, clock=flat_clock)


async def test_cases_run_one_at_a_time() -> None:
    events: list[str] = []

    async def observe(case: BenchCase) -> Observation:
        events.append(f'start {case.id}')
        await asyncio.sleep(0)
        events.append(f'end {case.id}')
        return seen()

    await run_suite(make_suite(), observe, clock=flat_clock)
    assert events == ['start c1', 'end c1', 'start c2', 'end c2', 'start c3', 'end c3']


def test_accuracy_ignores_infrastructure_errors() -> None:
    summary = summarize('m', [result(Verdict.CORRECT), result(Verdict.WRONG_TOOL), result(None)])
    assert (summary.cases, summary.scored, summary.errors) == (3, 2, 1)
    assert summary.accuracy == pytest.approx(0.5)


def test_all_errors_leave_no_accuracy() -> None:
    summary = summarize('m', [result(None)])
    assert summary.accuracy is None
    assert summary.spurious_call_rate is None


def test_verdict_counts_cover_every_class() -> None:
    summary = summarize('m', [result(Verdict.CORRECT), result(Verdict.CORRECT)])
    assert summary.verdict_counts['correct'] == 2
    assert summary.verdict_counts['no_call'] == 0
    assert set(summary.verdict_counts) == {verdict.value for verdict in Verdict}


def test_spurious_call_rate_is_measured_over_abstain_cases() -> None:
    verdicts = [Verdict.CORRECT_ABSTAIN, Verdict.SPURIOUS_CALL, Verdict.CORRECT]
    summary = summarize('m', [result(verdict) for verdict in verdicts])
    assert summary.spurious_call_rate == pytest.approx(0.5)


def test_spurious_call_rate_is_none_without_abstain_cases() -> None:
    assert summarize('m', [result(Verdict.CORRECT)]).spurious_call_rate is None


def test_latency_percentiles_use_live_scored_calls_only() -> None:
    results = [result(Verdict.CORRECT, latency=value) for value in (100.0, 200.0, 300.0, 400.0)]
    results.append(result(Verdict.CORRECT, latency=1.0, hit=True))
    results.append(result(None, latency=9999.0))
    summary = summarize('m', results)
    assert summary.latency_p50_ms == pytest.approx(200.0)
    assert summary.latency_p95_ms == pytest.approx(400.0)


def test_latency_is_none_when_every_call_was_cached() -> None:
    summary = summarize('m', [result(Verdict.CORRECT, hit=True)])
    assert summary.latency_p50_ms is None
    assert summary.latency_p95_ms is None


def test_cache_hit_rate_covers_all_results() -> None:
    results = [result(Verdict.CORRECT, hit=True)] + [result(Verdict.CORRECT)] * 3
    assert summarize('m', results).cache_hit_rate == pytest.approx(0.25)


def test_summarising_nothing_is_an_error() -> None:
    with pytest.raises(ValueError):
        summarize('m', [])


async def test_rejected_observation_is_scored_by_what_was_expected() -> None:
    rejected = Observation(calls=(), cache_hit=False, rejected=True)
    observe = scripted({'c1': rejected, 'c2': seen(STOCK), 'c3': rejected})
    results = await run_suite(make_suite(), observe, clock=flat_clock)
    assert [item.verdict for item in results] == [
        Verdict.INVALID_CALL,
        Verdict.CORRECT,
        Verdict.SPURIOUS_CALL,
    ]
