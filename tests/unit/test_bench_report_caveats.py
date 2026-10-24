from agentic_ops.bench.report import BenchReport, render_markdown


def test_report_warns_that_latency_includes_rate_limiter_waits() -> None:
    report = BenchReport(
        suite_version='test_suite',
        generated_at='2026-10-24T00:00:00Z',
        limits={},
        summaries=[],
        results={},
    )
    assert 'includes any local rate-limiter waits' in render_markdown(report)
