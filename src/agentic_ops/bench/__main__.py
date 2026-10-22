import argparse
import asyncio
from datetime import UTC, datetime
from pathlib import Path

from agentic_ops.bench.cases import load_suite
from agentic_ops.bench.observer import build_observer, tool_bound_call
from agentic_ops.bench.report import BenchReport, render_markdown, to_json
from agentic_ops.bench.runner import CaseResult, ModelSummary, run_suite, summarize
from agentic_ops.config import Settings, get_settings
from agentic_ops.db.session import get_sessionmaker, make_engine
from agentic_ops.gateway.cache import InMemoryCache, ResponseCache
from agentic_ops.gateway.cache_pg import PostgresCache
from agentic_ops.gateway.defaults import GEMINI_LITE, GROQ_20B, GROQ_120B, default_limits
from agentic_ops.gateway.models import ModelSpec
from agentic_ops.gateway.providers import PROVIDERS
from agentic_ops.gateway.router import limiter_key

BENCH_MODELS = (GROQ_20B, GROQ_120B, GEMINI_LITE)


def _missing_keys(settings: Settings, specs: tuple[ModelSpec, ...]) -> list[str]:
    """Names (never values) of the API-key variables that are unset for the given models."""
    missing: set[str] = set()
    for spec in specs:
        field = PROVIDERS[spec.provider].key_field
        if getattr(settings, field) is None:
            missing.add(f'AGENTIC_{field.upper()}')
    return sorted(missing)


def _percent(value: float | None) -> str:
    return 'n/a' if value is None else f'{value:.1%}'


async def run(suite_path: Path, output_dir: Path, limit: int | None, fresh: bool) -> None:
    settings = get_settings()
    missing = _missing_keys(settings, BENCH_MODELS)
    if missing:
        raise SystemExit(f'Missing API keys: {", ".join(missing)}')
    suite = load_suite(suite_path)
    label = suite.version
    if limit is not None:
        label = f'{label}_smoke'
        suite = suite.model_copy(
            update={
                'cases': suite.cases[:limit],
                'version': f'{suite.version} (first {limit} cases only)',
            }
        )
    limits = default_limits()
    call_model = tool_bound_call(settings)
    engine = make_engine(settings.sqlalchemy_url)
    summaries: list[ModelSummary] = []
    results: dict[str, list[CaseResult]] = {}
    try:
        cache: ResponseCache = InMemoryCache() if fresh else PostgresCache(get_sessionmaker(engine))
        for spec in BENCH_MODELS:
            key = limiter_key(spec)
            print(f'Running {key} ...', flush=True)
            observe = build_observer(
                spec, suite.tools, limits=limits, cache=cache, call_model=call_model
            )
            outcome = await run_suite(suite, observe)
            summary = summarize(key, outcome)
            results[key] = outcome
            summaries.append(summary)
            print(
                f'  accuracy {_percent(summary.accuracy)}, errors {summary.errors}, '
                f'cache hits {_percent(summary.cache_hit_rate)}',
                flush=True,
            )
    finally:
        await engine.dispose()
    generated = datetime.now(UTC)
    report = BenchReport(
        suite_version=suite.version,
        generated_at=generated.isoformat(),
        limits={limiter_key(spec): limits[limiter_key(spec)] for spec in BENCH_MODELS},
        summaries=summaries,
        results=results,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / f'{label}_{generated.strftime("%Y%m%dT%H%M%SZ")}'
    stem.with_suffix('.md').write_text(render_markdown(report), encoding='utf-8')
    stem.with_suffix('.json').write_text(to_json(report), encoding='utf-8')
    print(f'Report written to {stem}.md and {stem}.json')


def main() -> None:
    parser = argparse.ArgumentParser(
        prog='python -m agentic_ops.bench', description='Run the tool-calling benchmark.'
    )
    parser.add_argument('--suite', type=Path, default=Path('benchmarks/procurement_v1.json'))
    parser.add_argument('--output-dir', type=Path, default=Path('benchmarks/results'))
    parser.add_argument('--limit', type=int, default=None, help='Run only the first N cases.')
    parser.add_argument('--fresh', action='store_true', help='Ignore and skip the Postgres cache.')
    args = parser.parse_args()
    suite_path: Path = args.suite
    output_dir: Path = args.output_dir
    limit: int | None = args.limit
    if limit is not None and limit < 1:
        parser.error('--limit must be at least 1')
    asyncio.run(run(suite_path, output_dir, limit, bool(args.fresh)))


if __name__ == '__main__':
    main()
