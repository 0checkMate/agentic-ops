import time
from collections.abc import Awaitable, Callable, Sequence

from pydantic import BaseModel, ConfigDict

from agentic_ops.bench.cases import BenchCase, CaseSuite
from agentic_ops.bench.scoring import ObservedCall, Verdict, score_case, tool_call_accuracy
from agentic_ops.gateway.errors import GatewayExhausted

Clock = Callable[[], float]


class Observation(BaseModel):
    """What one model call produced: the tool calls it made, and whether the cache served it."""

    model_config = ConfigDict(frozen=True)

    calls: tuple[ObservedCall, ...]
    cache_hit: bool
    rejected: bool = False


Observe = Callable[[BenchCase], Awaitable[Observation]]


class CaseResult(BaseModel):
    """One case's outcome; verdict None means infrastructure failed and nothing was scored."""

    model_config = ConfigDict(frozen=True)

    case_id: str
    verdict: Verdict | None
    latency_ms: float
    cache_hit: bool
    error: str | None


class ModelSummary(BaseModel):
    """Headline metrics for one model over one suite run."""

    model_config = ConfigDict(frozen=True)

    model: str
    cases: int
    scored: int
    errors: int
    accuracy: float | None
    spurious_call_rate: float | None
    verdict_counts: dict[str, int]
    latency_p50_ms: float | None
    latency_p95_ms: float | None
    cache_hit_rate: float


async def run_suite(
    suite: CaseSuite, observe: Observe, *, clock: Clock = time.perf_counter
) -> list[CaseResult]:
    """Run every case in order, one at a time, and score the observed tool calls.

    Only GatewayExhausted is recorded as an error result. Anything else, including
    GatewayConfigError, a provider 400 or a bug, propagates: it must be seen, not scored.
    """
    results: list[CaseResult] = []
    for case in suite.cases:
        start = clock()
        try:
            observation = await observe(case)
        except GatewayExhausted as exc:
            results.append(
                CaseResult(
                    case_id=case.id,
                    verdict=None,
                    latency_ms=(clock() - start) * 1000,
                    cache_hit=False,
                    error=f'{type(exc).__name__}: {exc}',
                )
            )
            continue
        results.append(
            CaseResult(
                case_id=case.id,
                verdict=score_case(case.expected, observation.calls, rejected=observation.rejected),
                latency_ms=(clock() - start) * 1000,
                cache_hit=observation.cache_hit,
                error=None,
            )
        )
    return results


def _percentile(sorted_values: Sequence[float], percent: int) -> float:
    """Nearest-rank percentile of a non-empty ascending sequence, using exact integer maths."""
    rank = max(1, -(-percent * len(sorted_values) // 100))
    return sorted_values[rank - 1]


def summarize(model: str, results: Sequence[CaseResult]) -> ModelSummary:
    """Aggregate case results; errors are counted but never enter accuracy or latency."""
    if not results:
        raise ValueError('Cannot summarise zero results')
    scored = [item for item in results if item.verdict is not None]
    verdicts = [item.verdict for item in scored if item.verdict is not None]
    counts = {verdict.value: 0 for verdict in Verdict}
    for verdict in verdicts:
        counts[verdict.value] += 1
    abstain_cases = counts[Verdict.CORRECT_ABSTAIN.value] + counts[Verdict.SPURIOUS_CALL.value]
    live = sorted(item.latency_ms for item in scored if not item.cache_hit)
    return ModelSummary(
        model=model,
        cases=len(results),
        scored=len(scored),
        errors=len(results) - len(scored),
        accuracy=tool_call_accuracy(verdicts) if verdicts else None,
        spurious_call_rate=(
            counts[Verdict.SPURIOUS_CALL.value] / abstain_cases if abstain_cases else None
        ),
        verdict_counts=counts,
        latency_p50_ms=_percentile(live, 50) if live else None,
        latency_p95_ms=_percentile(live, 95) if live else None,
        cache_hit_rate=sum(item.cache_hit for item in results) / len(results),
    )
