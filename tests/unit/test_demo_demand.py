"""Demand history: calendar, determinism, forecast helpers and a beatable baseline."""

from datetime import timedelta

import pytest

from agentic_ops.erp.mock.demo_demand import (
    HOLDOUT_WEEKS,
    SEASON_WEEKS,
    WEEKS,
    build_demand_dataset,
    last_value_forecast,
    mae,
    seasonal_naive_forecast,
    series,
    smape,
)
from agentic_ops.erp.mock.demo_seed import ANCHOR, FINISHED_PRODUCT_NAMES

DATASET = build_demand_dataset(42)


def test_finished_product_names_are_exposed() -> None:
    assert len(FINISHED_PRODUCT_NAMES) == 7
    assert len(set(FINISHED_PRODUCT_NAMES)) == 7


def test_same_seed_gives_identical_history() -> None:
    assert build_demand_dataset(42) == DATASET


def test_different_seed_changes_history() -> None:
    assert build_demand_dataset(43) != DATASET


def test_calendar_is_weekly_and_ends_at_anchor() -> None:
    assert {p.product_name for p in DATASET.points} == set(FINISHED_PRODUCT_NAMES)
    for name in FINISHED_PRODUCT_NAMES:
        weeks = [p.week_start for p in DATASET.points if p.product_name == name]
        assert len(weeks) == WEEKS
        assert weeks[-1] + timedelta(weeks=1) == ANCHOR
        steps = {later - earlier for earlier, later in zip(weeks, weeks[1:], strict=False)}
        assert steps == {timedelta(weeks=1)}


def test_units_are_positive_integers() -> None:
    assert all(isinstance(p.units, int) and p.units > 0 for p in DATASET.points)


def test_last_value_forecast_repeats_the_latest_week() -> None:
    assert last_value_forecast([1.0, 2.0, 3.0], 2) == [3.0, 3.0]


def test_seasonal_naive_forecast_repeats_the_season() -> None:
    history = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert seasonal_naive_forecast(history, 4, season=3) == [4.0, 5.0, 6.0, 4.0]


def test_seasonal_naive_needs_one_full_season() -> None:
    with pytest.raises(ValueError):
        seasonal_naive_forecast([1.0, 2.0], 1, season=3)


def test_mae_is_the_mean_absolute_error() -> None:
    assert mae([1.0, 2.0, 3.0], [2.0, 2.0, 5.0]) == pytest.approx(1.0)


def test_smape_is_a_symmetric_fraction() -> None:
    assert smape([100.0], [50.0]) == pytest.approx(2 / 3)
    assert smape([0.0], [0.0]) == 0.0


def test_metrics_reject_length_mismatch() -> None:
    with pytest.raises(ValueError):
        mae([1.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        smape([1.0], [1.0, 2.0])


def test_every_product_trends_upward() -> None:
    for name in FINISHED_PRODUCT_NAMES:
        values = series(DATASET, name)
        assert sum(values[-SEASON_WEEKS:]) > sum(values[:SEASON_WEEKS])


@pytest.mark.parametrize('name', FINISHED_PRODUCT_NAMES)
def test_seasonal_naive_beats_last_value_on_holdout(name: str) -> None:
    values = series(DATASET, name)
    train, actual = values[:-HOLDOUT_WEEKS], values[-HOLDOUT_WEEKS:]
    seasonal = mae(actual, seasonal_naive_forecast(train, HOLDOUT_WEEKS))
    naive = mae(actual, last_value_forecast(train, HOLDOUT_WEEKS))
    assert seasonal < naive
