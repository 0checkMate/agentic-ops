from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from agentic_ops.bench.runner import CaseResult, ModelSummary
from agentic_ops.bench.scoring import Verdict
from agentic_ops.gateway.ratelimit import RateLimit


class BenchReport(BaseModel):
    """Everything needed to trace a published number back to the conditions that produced it."""

    model_config = ConfigDict(frozen=True)

    suite_version: str
    generated_at: str
    limits: dict[str, RateLimit]
    summaries: list[ModelSummary]
    results: dict[str, list[CaseResult]]


def _pct(value: float | None) -> str:
    return 'n/a' if value is None else f'{value:.1%}'


def _ms(value: float | None) -> str:
    return 'n/a' if value is None else f'{value:.0f}'


def _row(cells: Sequence[str]) -> str:
    return '| ' + ' | '.join(cells) + ' |'


def _rule(columns: int) -> str:
    return _row(['---'] * columns)


def _describe(limit: RateLimit) -> str:
    dimensions = [
        (limit.requests_per_minute, 'requests/min'),
        (limit.tokens_per_minute, 'tokens/min'),
        (limit.requests_per_day, 'requests/day'),
        (limit.tokens_per_day, 'tokens/day'),
    ]
    enforced = [f'{amount} {label}' for amount, label in dimensions if amount is not None]
    return ', '.join(enforced) if enforced else 'no limits enforced'


def render_markdown(report: BenchReport) -> str:
    """Render the report as Markdown; absent metrics show as n/a, never as zero."""
    summary_header = [
        'Model',
        'Cases',
        'Scored',
        'Errors',
        'Accuracy',
        'Spurious-call rate',
        'p50 ms',
        'p95 ms',
        'Cache hit rate',
    ]
    verdict_header = ['Model', *(verdict.value for verdict in Verdict)]
    lines = [
        f'# Tool-calling benchmark: {report.suite_version}',
        '',
        f'Generated: {report.generated_at}',
        '',
        '## Summary',
        '',
        _row(summary_header),
        _rule(len(summary_header)),
    ]
    for item in report.summaries:
        lines.append(
            _row(
                [
                    item.model,
                    str(item.cases),
                    str(item.scored),
                    str(item.errors),
                    _pct(item.accuracy),
                    _pct(item.spurious_call_rate),
                    _ms(item.latency_p50_ms),
                    _ms(item.latency_p95_ms),
                    _pct(item.cache_hit_rate),
                ]
            )
        )
    lines += ['', '## Verdicts', '', _row(verdict_header), _rule(len(verdict_header))]
    for item in report.summaries:
        counts = [str(item.verdict_counts.get(verdict.value, 0)) for verdict in Verdict]
        lines.append(_row([item.model, *counts]))
    lines += ['', '## Rate limits used', '']
    lines += [f'- {key}: {_describe(limit)}' for key, limit in report.limits.items()]
    lines += [
        '',
        '## Reading these numbers',
        '',
        '- With small case counts, differences of a few points are not statistically significant.',
        '- Errors are infrastructure failures; they are excluded from accuracy and latency.',
        '- Latency covers live scored calls only; cached calls are excluded.',
    ]
    return '\n'.join(lines) + '\n'


def to_json(report: BenchReport) -> str:
    """Serialise the full report, per-case results included, for audit and later comparison."""
    return report.model_dump_json(indent=2)
