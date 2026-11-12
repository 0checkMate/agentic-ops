"""Maintenance seed: equipment, daily sensor series, labelled anomalies, work orders."""

from collections import defaultdict
from datetime import datetime, timedelta

import pytest

from agentic_ops.erp.mock.demo_ops import (
    HISTORY_DAYS,
    AnomalyEpisode,
    AnomalyKind,
    MetricSpec,
    OpsDataset,
    SensorReading,
    build_ops_dataset,
)
from agentic_ops.erp.mock.demo_seed import ANCHOR
from agentic_ops.erp.models import MaintenanceKind

HISTORY_START = ANCHOR - timedelta(days=HISTORY_DAYS)
SUNDAY = 6


@pytest.fixture(scope='module')
def dataset() -> OpsDataset:
    return build_ops_dataset()


def _specs(data: OpsDataset) -> dict[tuple[str, str], MetricSpec]:
    return {(spec.equipment_ref, spec.metric): spec for spec in data.metrics}


def _z(spec: MetricSpec, reading: SensorReading) -> float:
    factor = spec.sunday_factor if reading.recorded_on.weekday() == SUNDAY else 1.0
    return (reading.value - spec.baseline * factor) / spec.noise


def _inside(episode: AnomalyEpisode, reading: SensorReading) -> bool:
    return (
        reading.equipment_ref == episode.equipment_ref
        and reading.metric == episode.metric
        and episode.start <= reading.recorded_on <= episode.end
    )


def _episode_readings(data: OpsDataset, episode: AnomalyEpisode) -> list[SensorReading]:
    found = [reading for reading in data.readings if _inside(episode, reading)]
    return sorted(found, key=lambda reading: reading.recorded_on)


def test_catalogue_has_equipment_and_metrics(dataset: OpsDataset) -> None:
    refs = [item.ref for item in dataset.equipment]
    assert len(refs) >= 5
    assert len(refs) == len(set(refs))
    specs = _specs(dataset)
    assert len(specs) == len(dataset.metrics)
    assert all(spec.noise > 0 for spec in dataset.metrics)
    metrics_per_equipment: defaultdict[str, int] = defaultdict(int)
    for spec in dataset.metrics:
        assert spec.equipment_ref in refs
        metrics_per_equipment[spec.equipment_ref] += 1
    assert all(metrics_per_equipment[ref] >= 2 for ref in refs)


def test_every_metric_has_one_reading_per_day(dataset: OpsDataset) -> None:
    specs = _specs(dataset)
    assert len(dataset.readings) == len(specs) * HISTORY_DAYS
    days_by_key: defaultdict[tuple[str, str], set[datetime]] = defaultdict(set)
    for reading in dataset.readings:
        key = (reading.equipment_ref, reading.metric)
        assert key in specs
        days_by_key[key].add(reading.recorded_on)
    expected_days = {HISTORY_START + timedelta(days=offset) for offset in range(HISTORY_DAYS)}
    assert all(days == expected_days for days in days_by_key.values())
    assert all(HISTORY_START <= reading.recorded_on < ANCHOR for reading in dataset.readings)


def test_same_seed_gives_identical_dataset() -> None:
    assert build_ops_dataset(7) == build_ops_dataset(7)
    assert build_ops_dataset(7) != build_ops_dataset(8)


def test_clean_readings_stay_inside_the_noise_band(dataset: OpsDataset) -> None:
    specs = _specs(dataset)
    episodes = dataset.truth.episodes
    for reading in dataset.readings:
        if any(_inside(episode, reading) for episode in episodes):
            continue
        assert abs(_z(specs[(reading.equipment_ref, reading.metric)], reading)) <= 3.6


def test_episode_catalogue(dataset: OpsDataset) -> None:
    episodes = dataset.truth.episodes
    specs = _specs(dataset)
    assert len(episodes) >= 9
    assert {episode.kind for episode in episodes} == set(AnomalyKind)
    assert len({episode.episode_id for episode in episodes}) == len(episodes)
    last_allowed = ANCHOR - timedelta(days=10)
    for episode in episodes:
        assert (episode.equipment_ref, episode.metric) in specs
        assert HISTORY_START + timedelta(days=30) <= episode.start <= episode.end <= last_allowed
        days = (episode.end - episode.start).days + 1
        if episode.kind is AnomalyKind.SPIKE:
            assert days == 1
        elif episode.kind is AnomalyKind.DRIFT:
            assert days >= 10
        else:
            assert 3 <= days <= 6
    for first in episodes:
        for second in episodes:
            if first is not second and (first.equipment_ref, first.metric) == (
                second.equipment_ref,
                second.metric,
            ):
                assert first.end < second.start or second.end < first.start


def test_spikes_are_extreme_single_readings(dataset: OpsDataset) -> None:
    specs = _specs(dataset)
    spikes = [e for e in dataset.truth.episodes if e.kind is AnomalyKind.SPIKE]
    assert spikes
    for episode in spikes:
        readings = _episode_readings(dataset, episode)
        assert len(readings) == 1
        assert abs(_z(specs[(episode.equipment_ref, episode.metric)], readings[0])) >= 5


def test_drifts_grow_until_the_end(dataset: OpsDataset) -> None:
    specs = _specs(dataset)
    drifts = [e for e in dataset.truth.episodes if e.kind is AnomalyKind.DRIFT]
    assert drifts
    for episode in drifts:
        spec = specs[(episode.equipment_ref, episode.metric)]
        readings = _episode_readings(dataset, episode)
        first, last = abs(_z(spec, readings[0])), abs(_z(spec, readings[-1]))
        assert last >= 5
        assert last > first


def test_stuck_sensors_flatline(dataset: OpsDataset) -> None:
    stuck = [e for e in dataset.truth.episodes if e.kind is AnomalyKind.STUCK]
    assert stuck
    for episode in stuck:
        values = {reading.value for reading in _episode_readings(dataset, episode)}
        assert len(values) == 1


def test_corrective_orders_follow_every_non_spike_episode(dataset: OpsDataset) -> None:
    orders = {order.ref: order for order in dataset.orders}
    linked: list[str] = []
    for episode in dataset.truth.episodes:
        if episode.kind is AnomalyKind.SPIKE:
            assert episode.work_order_ref is None
            continue
        assert episode.work_order_ref in orders
        order = orders[episode.work_order_ref]
        linked.append(order.ref)
        assert order.kind is MaintenanceKind.CORRECTIVE
        assert order.equipment_ref == episode.equipment_ref
        assert order.opened_at.date() == (episode.end + timedelta(days=1)).date()
    corrective = [o.ref for o in dataset.orders if o.kind is MaintenanceKind.CORRECTIVE]
    assert sorted(corrective) == sorted(linked)


def test_preventive_orders_follow_a_quarterly_cadence(dataset: OpsDataset) -> None:
    for item in dataset.equipment:
        opened = sorted(
            order.opened_at
            for order in dataset.orders
            if order.equipment_ref == item.ref and order.kind is MaintenanceKind.PREVENTIVE
        )
        assert len(opened) >= 6
        gaps = [(later - earlier).days for earlier, later in zip(opened, opened[1:], strict=False)]
        assert all(60 <= gap <= 120 for gap in gaps)


def test_order_integrity(dataset: OpsDataset) -> None:
    refs = [order.ref for order in dataset.orders]
    equipment = {item.ref for item in dataset.equipment}
    assert len(refs) == len(set(refs))
    for order in dataset.orders:
        assert order.equipment_ref in equipment
        assert HISTORY_START <= order.opened_at <= order.closed_at <= ANCHOR
