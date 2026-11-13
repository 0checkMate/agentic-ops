"""Synthetic weekly demand history and the naive baselines a forecast model must beat.

Own random stream: the ERP seed in demo_seed.py is never touched.
"""

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

from agentic_ops.erp.mock.demo_seed import ANCHOR, FINISHED_PRODUCT_NAMES

WEEKS = 104
SEASON_WEEKS = 52
HOLDOUT_WEEKS = 13

# Phase anchor: the seasonal curve crosses its mean at the start of the 13-week hold-out,
# so a last-value forecast drifts away from the truth while seasonal-naive stays close.
_HOLDOUT_PHASE = -2 * math.pi * (WEEKS - HOLDOUT_WEEKS - 0.5) / SEASON_WEEKS


@dataclass(frozen=True)
class DemandPoint:
    product_name: str
    week_start: datetime
    units: int


@dataclass(frozen=True)
class DemandDataset:
    points: tuple[DemandPoint, ...]


def build_demand_dataset(seed: int) -> DemandDataset:
    rng = random.Random(f'demand-{seed}')  # string seeding is deterministic across runs
    points: list[DemandPoint] = []
    for name in FINISHED_PRODUCT_NAMES:
        base = rng.uniform(300.0, 900.0)
        growth = rng.uniform(0.05, 0.10)  # fractional yearly growth
        amplitude = rng.uniform(0.30, 0.40)
        phase = _HOLDOUT_PHASE + rng.uniform(-0.4, 0.4)
        if rng.random() < 0.5:
            phase += math.pi
        for week in range(WEEKS):
            trend = 1.0 + growth * week / SEASON_WEEKS
            season = 1.0 + amplitude * math.sin(2 * math.pi * week / SEASON_WEEKS + phase)
            noise = max(-2.5, min(2.5, rng.gauss(0.0, 1.0)))  # clipped, as in demo_ops
            units = max(1, round(base * trend * season * (1.0 + 0.04 * noise)))
            week_start = ANCHOR - timedelta(weeks=WEEKS - week)
            points.append(DemandPoint(name, week_start, units))
    return DemandDataset(tuple(points))


def series(dataset: DemandDataset, product_name: str) -> list[float]:
    return [float(p.units) for p in dataset.points if p.product_name == product_name]


def last_value_forecast(history: Sequence[float], horizon: int) -> list[float]:
    return [history[-1]] * horizon


def seasonal_naive_forecast(
    history: Sequence[float], horizon: int, season: int = SEASON_WEEKS
) -> list[float]:
    if len(history) < season:
        raise ValueError('history must cover at least one full season')
    return [history[len(history) - season + (i % season)] for i in range(horizon)]


def mae(actual: Sequence[float], forecast: Sequence[float]) -> float:
    if len(actual) != len(forecast):
        raise ValueError('actual and forecast must have the same length')
    return sum(abs(a - f) for a, f in zip(actual, forecast, strict=True)) / len(actual)


def smape(actual: Sequence[float], forecast: Sequence[float]) -> float:
    if len(actual) != len(forecast):
        raise ValueError('actual and forecast must have the same length')
    total = 0.0
    for a, f in zip(actual, forecast, strict=True):
        denominator = abs(a) + abs(f)
        if denominator:
            total += 2 * abs(a - f) / denominator
    return total / len(actual)
