from agentic_ops.bench.report import BenchReport, render_markdown, to_json
from agentic_ops.bench.runner import CaseResult, ModelSummary
from agentic_ops.bench.scoring import Verdict
from agentic_ops.gateway.ratelimit import RateLimit


def counts(**given: int) -> dict[str, int]:
    base = {verdict.value: 0 for verdict in Verdict}
    base.update(given)
    return base


def make_report() -> BenchReport:
    healthy = ModelSummary(
        model='groq:m1',
        cases=4,
        scored=4,
        errors=0,
        accuracy=0.75,
        spurious_call_rate=0.0,
        verdict_counts=counts(correct=3, wrong_tool=1),
        latency_p50_ms=100.0,
        latency_p95_ms=400.0,
        cache_hit_rate=0.25,
    )
    failed = ModelSummary(
        model='gemini:m2',
        cases=2,
        scored=0,
        errors=2,
        accuracy=None,
        spurious_call_rate=None,
        verdict_counts=counts(),
        latency_p50_ms=None,
        latency_p95_ms=None,
        cache_hit_rate=0.0,
    )
    row = CaseResult(
        case_id='c1', verdict=Verdict.CORRECT, latency_ms=100.0, cache_hit=False, error=None
    )
    return BenchReport(
        suite_version='procurement_v1',
        generated_at='2026-10-22T10:00:00+00:00',
        limits={
            'groq:m1': RateLimit(requests_per_minute=30, tokens_per_minute=8000),
            'gemini:m2': RateLimit(requests_per_minute=15, requests_per_day=1500),
        },
        summaries=[healthy, failed],
        results={'groq:m1': [row], 'gemini:m2': []},
    )


def test_heading_names_the_suite_and_the_time() -> None:
    text = render_markdown(make_report())
    assert '# Tool-calling benchmark: procurement_v1' in text
    assert 'Generated: 2026-10-22T10:00:00+00:00' in text


def test_summary_row_formats_percentages_and_latency() -> None:
    text = render_markdown(make_report())
    assert '| groq:m1 | 4 | 4 | 0 | 75.0% | 0.0% | 100 | 400 | 25.0% |' in text


def test_missing_metrics_render_as_not_available() -> None:
    text = render_markdown(make_report())
    assert '| gemini:m2 | 2 | 0 | 2 | n/a | n/a | n/a | n/a | 0.0% |' in text


def test_verdict_table_lists_every_class_in_enum_order() -> None:
    text = render_markdown(make_report())
    header = (
        '| Model | correct | wrong_tool | bad_arguments | no_call'
        ' | spurious_call | correct_abstain |'
    )
    assert header in text


def test_verdict_row_carries_the_counts() -> None:
    assert '| groq:m1 | 3 | 1 | 0 | 0 | 0 | 0 |' in render_markdown(make_report())


def test_limits_section_lists_enforced_dimensions_only() -> None:
    text = render_markdown(make_report())
    assert '- groq:m1: 30 requests/min, 8000 tokens/min' in text
    assert '- gemini:m2: 15 requests/min, 1500 requests/day' in text


def test_report_warns_about_small_samples() -> None:
    assert 'not statistically significant' in render_markdown(make_report())


def test_models_keep_their_order() -> None:
    text = render_markdown(make_report())
    assert text.index('| groq:m1 | 4') < text.index('| gemini:m2 | 2')


def test_json_round_trips() -> None:
    report = make_report()
    assert BenchReport.model_validate_json(to_json(report)) == report
