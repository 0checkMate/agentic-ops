"""
Maintenance seed: equipment, daily sensor series, labelled anomalies and work orders.

Independent of build_demo_dataset: it has its own random stream, so the ERP seed stays stable.
Anomalies are planted with known labels (the truth) and are never visible to the ERP port.
"""

import random
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from enum import StrEnum
from typing import NamedTuple

from agentic_ops.erp.mock.demo_seed import ANCHOR, DEFAULT_SEED
from agentic_ops.erp.models import Equipment, MaintenanceKind, MaintenanceOrder

HISTORY_DAYS = 730
_NOISE_CLIP = 3.5  # clean readings never exceed this many sigmas, so planted anomalies stand out
_SUNDAY = 6
_EDGE_DAYS = 30  # no episode in the first month, so a stuck sensor has a last good value
_TAIL_DAYS = 10  # no episode in the last days, so its corrective order closes before the anchor


class AnomalyKind(StrEnum):
    DRIFT = 'drift'
    SPIKE = 'spike'
    STUCK = 'stuck'


@dataclass(frozen=True)
class MetricSpec:
    equipment_ref: str
    metric: str
    unit: str
    baseline: float
    noise: float
    sunday_factor: float = 1.0  # load-dependent metrics run lower on the Sunday shutdown


@dataclass(frozen=True)
class SensorReading:
    equipment_ref: str
    metric: str
    recorded_on: datetime
    value: float


@dataclass(frozen=True)
class AnomalyEpisode:
    episode_id: str
    equipment_ref: str
    metric: str
    kind: AnomalyKind
    start: datetime
    end: datetime
    magnitude: float  # peak deviation in sigmas for drift and spike; 0.0 for stuck
    work_order_ref: str | None  # corrective order that resolved it; None for a self-clearing spike


@dataclass(frozen=True)
class OpsTruth:
    episodes: tuple[AnomalyEpisode, ...]


@dataclass(frozen=True)
class OpsDataset:
    equipment: tuple[Equipment, ...]
    metrics: tuple[MetricSpec, ...]
    readings: tuple[SensorReading, ...]
    orders: tuple[MaintenanceOrder, ...]
    truth: OpsTruth


_EQUIPMENT = (
    Equipment(ref='EQ-001', name='Cold press 1', kind='cold_press', location='Press hall'),
    Equipment(ref='EQ-002', name='HPP vessel', kind='hpp_vessel', location='HPP bay'),
    Equipment(ref='EQ-003', name='Filling line', kind='filler', location='Filling hall'),
    Equipment(ref='EQ-004', name='Cold room compressor', kind='chiller', location='Cold store'),
    Equipment(ref='EQ-005', name='CIP skid', kind='cip_skid', location='Utilities'),
)

_METRICS = (
    MetricSpec('EQ-001', 'vibration_mm_s', 'mm/s', 2.4, 0.18, 0.5),
    MetricSpec('EQ-001', 'oil_temp_c', 'degC', 48.0, 1.2, 0.8),
    MetricSpec('EQ-002', 'peak_pressure_mpa', 'MPa', 600.0, 3.0),
    MetricSpec('EQ-002', 'intensifier_temp_c', 'degC', 41.0, 1.0, 0.85),
    MetricSpec('EQ-003', 'fill_deviation_ml', 'ml', 0.4, 0.15),
    MetricSpec('EQ-003', 'motor_current_a', 'A', 14.0, 0.5, 0.3),
    MetricSpec('EQ-004', 'room_temp_c', 'degC', 3.2, 0.25),
    MetricSpec('EQ-004', 'compressor_current_a', 'A', 22.0, 0.8, 0.9),
    MetricSpec('EQ-005', 'caustic_conc_pct', '%', 2.0, 0.04),
    MetricSpec('EQ-005', 'supply_temp_c', 'degC', 75.0, 1.2, 0.9),
)

_EPISODE_PLAN = (
    (AnomalyKind.DRIFT, 5, (10, 20), (10.0, 14.0)),
    (AnomalyKind.SPIKE, 6, (1, 1), (9.0, 12.0)),
    (AnomalyKind.STUCK, 3, (3, 6), (0.0, 0.0)),
)
_PLACEMENT_ATTEMPTS = 500
_MIN_GAP_DAYS = 5


class _Plan(NamedTuple):
    key: tuple[str, str]
    kind: AnomalyKind
    start_offset: int
    length: int
    magnitude: float
    sign: float


def _overlaps(taken: list[_Plan], key: tuple[str, str], start: int, length: int) -> bool:
    for other in taken:
        if other.key != key:
            continue
        if start < other.start_offset + other.length + _MIN_GAP_DAYS and (
            other.start_offset < start + length + _MIN_GAP_DAYS
        ):
            return True
    return False


def _plan_episodes(rng: random.Random) -> list[_Plan]:
    keys = [(spec.equipment_ref, spec.metric) for spec in _METRICS]
    plans: list[_Plan] = []
    for kind, count, lengths, magnitudes in _EPISODE_PLAN:
        placed = 0
        for _ in range(_PLACEMENT_ATTEMPTS):
            if placed == count:
                break
            key = rng.choice(keys)
            length = rng.randint(*lengths)
            latest = HISTORY_DAYS - _TAIL_DAYS - length
            start = rng.randint(_EDGE_DAYS, latest)
            magnitude = round(rng.uniform(*magnitudes), 1)
            sign = rng.choice((-1.0, 1.0))
            if _overlaps(plans, key, start, length):
                continue
            plans.append(_Plan(key, kind, start, length, magnitude, sign))
            placed += 1
    return plans


def _bounded_noise(rng: random.Random) -> float:
    return max(-_NOISE_CLIP, min(_NOISE_CLIP, rng.gauss(0.0, 1.0)))


def _expected(spec: MetricSpec, day: datetime) -> float:
    return spec.baseline * (spec.sunday_factor if day.weekday() == _SUNDAY else 1.0)


def _series(
    rng: random.Random, spec: MetricSpec, start: datetime, plans: list[_Plan]
) -> list[float]:
    """One value per day. Planted anomalies are applied on top of the clean signal."""
    mine = [plan for plan in plans if plan.key == (spec.equipment_ref, spec.metric)]
    values: list[float] = []
    for offset in range(HISTORY_DAYS):
        day = start + timedelta(days=offset)
        value = _expected(spec, day) + _bounded_noise(rng) * spec.noise
        for plan in mine:
            step = offset - plan.start_offset
            if not 0 <= step < plan.length:
                continue
            if plan.kind is AnomalyKind.SPIKE:
                value += plan.sign * plan.magnitude * spec.noise
            elif plan.kind is AnomalyKind.DRIFT:
                value += plan.sign * plan.magnitude * spec.noise * (step + 1) / plan.length
            else:
                value = values[plan.start_offset - 1]
        values.append(round(value, 3))
    return values


class _Draft(NamedTuple):
    equipment_ref: str
    kind: MaintenanceKind
    opened_at: datetime
    closed_at: datetime
    description: str
    episode_key: int | None


def _preventive_drafts(rng: random.Random, start: datetime) -> list[_Draft]:
    drafts: list[_Draft] = []
    for item in _EQUIPMENT:
        offset = rng.randint(10, 60)
        while offset < HISTORY_DAYS - 5:
            opened = start + timedelta(days=offset, hours=8)
            drafts.append(
                _Draft(
                    item.ref,
                    MaintenanceKind.PREVENTIVE,
                    opened,
                    opened + timedelta(hours=rng.randint(2, 8)),
                    f'Quarterly preventive maintenance: {item.name}',
                    None,
                )
            )
            offset += rng.randint(75, 105)
    return drafts


def _corrective_drafts(rng: random.Random, start: datetime, plans: list[_Plan]) -> list[_Draft]:
    drafts: list[_Draft] = []
    for index, plan in enumerate(plans):
        if plan.kind is AnomalyKind.SPIKE:
            continue
        end_day = start + timedelta(days=plan.start_offset + plan.length - 1)
        opened = end_day + timedelta(days=1, hours=7)
        action = 'Replace sensor' if plan.kind is AnomalyKind.STUCK else 'Repair degraded unit'
        drafts.append(
            _Draft(
                plan.key[0],
                MaintenanceKind.CORRECTIVE,
                opened,
                opened + timedelta(hours=rng.randint(6, 60)),
                f'{action}: {plan.key[1]}',
                index,
            )
        )
    return drafts


def build_ops_dataset(seed: int = DEFAULT_SEED) -> OpsDataset:
    """Build the maintenance dataset. The same seed always yields an identical result."""
    rng = random.Random(seed)
    start = ANCHOR - timedelta(days=HISTORY_DAYS)
    plans = _plan_episodes(rng)
    readings: list[SensorReading] = []
    for spec in _METRICS:
        for offset, value in enumerate(_series(rng, spec, start, plans)):
            day = start + timedelta(days=offset)
            readings.append(SensorReading(spec.equipment_ref, spec.metric, day, value))
    drafts = _preventive_drafts(rng, start) + _corrective_drafts(rng, start, plans)
    drafts.sort(key=lambda draft: (draft.opened_at, draft.equipment_ref))
    orders: list[MaintenanceOrder] = []
    order_of_plan: dict[int, str] = {}
    for number, draft in enumerate(drafts, start=1):
        ref = f'MR-{number:04d}'
        orders.append(
            MaintenanceOrder(
                ref=ref,
                equipment_ref=draft.equipment_ref,
                kind=draft.kind,
                opened_at=draft.opened_at,
                closed_at=draft.closed_at,
                description=draft.description,
            )
        )
        if draft.episode_key is not None:
            order_of_plan[draft.episode_key] = ref
    episodes = [
        AnomalyEpisode(
            episode_id='',
            equipment_ref=plan.key[0],
            metric=plan.key[1],
            kind=plan.kind,
            start=start + timedelta(days=plan.start_offset),
            end=start + timedelta(days=plan.start_offset + plan.length - 1),
            magnitude=plan.magnitude,
            work_order_ref=order_of_plan.get(index),
        )
        for index, plan in enumerate(plans)
    ]
    episodes.sort(key=lambda episode: (episode.start, episode.equipment_ref, episode.metric))
    numbered = tuple(
        replace(episode, episode_id=f'AN-{number:03d}')
        for number, episode in enumerate(episodes, start=1)
    )
    return OpsDataset(
        equipment=_EQUIPMENT,
        metrics=_METRICS,
        readings=tuple(readings),
        orders=tuple(orders),
        truth=OpsTruth(episodes=numbered),
    )
