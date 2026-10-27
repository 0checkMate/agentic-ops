"""Deterministic demo dataset for a cold-pressed juice plant (illustrative prices in KES).

Pure data generation: no I/O, no clock. Planted answers live in DemoTruth, never in the
ERP-visible records, so agents cannot read them.
"""

import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import NamedTuple

from agentic_ops.erp.models import (
    Partner,
    Product,
    PurchaseOrder,
    PurchaseOrderLine,
    PurchaseOrderState,
    StockLevel,
)

ANCHOR = datetime(2026, 10, 1)
DEFAULT_SEED = 42

_ORDER_COUNT = 220
_BELOW_REORDER_COUNT = 8
_COLD = 'WH/Cold'
_AMBIENT = 'WH/Stock'
_CHEM = 'WH/Chem'


class _Spec(NamedTuple):
    name: str
    prefix: str
    uom: str
    price: str
    order_qty: int
    reorder: int
    location: str
    suppliers: tuple[int, ...]


# Supplier numbers refer to _SUPPLIERS (1-based); the first is the primary supplier.
_SPECS: tuple[_Spec, ...] = (
    _Spec('Mango Puree', 'ING', 'kg', '260', 400, 300, _COLD, (2,)),
    _Spec('Pineapple', 'ING', 'kg', '70', 500, 400, _COLD, (2, 1)),
    _Spec('Orange (Juicing)', 'ING', 'kg', '60', 800, 600, _COLD, (4, 1)),
    _Spec('Watermelon', 'ING', 'kg', '40', 800, 500, _COLD, (1,)),
    _Spec('Ginger Root', 'ING', 'kg', '180', 100, 80, _COLD, (3,)),
    _Spec('Turmeric Root', 'ING', 'kg', '220', 60, 40, _COLD, (3,)),
    _Spec('Passion Fruit Pulp', 'ING', 'kg', '480', 150, 100, _COLD, (2,)),
    _Spec('Lemon', 'ING', 'kg', '110', 300, 200, _COLD, (4,)),
    _Spec('Carrot', 'ING', 'kg', '55', 600, 400, _COLD, (3, 1)),
    _Spec('Beetroot', 'ING', 'kg', '65', 400, 250, _COLD, (3,)),
    _Spec('Apple', 'ING', 'kg', '120', 500, 350, _COLD, (1, 4)),
    _Spec('Spinach', 'ING', 'kg', '90', 150, 100, _COLD, (3,)),
    _Spec('Coconut Water', 'ING', 'L', '130', 400, 300, _COLD, (2,)),
    _Spec('Cane Sugar', 'ING', 'kg', '150', 500, 200, _AMBIENT, (5,)),
    _Spec('Honey', 'ING', 'kg', '650', 100, 60, _AMBIENT, (5,)),
    _Spec('Citric Acid', 'ING', 'kg', '210', 50, 25, _AMBIENT, (6,)),
    _Spec('Ascorbic Acid', 'ING', 'kg', '1400', 20, 10, _AMBIENT, (6,)),
    _Spec('Chia Seeds', 'ING', 'kg', '900', 50, 30, _AMBIENT, (7,)),
    _Spec('Moringa Powder', 'ING', 'kg', '1100', 25, 15, _AMBIENT, (7,)),
    _Spec('PET Bottle 250 ml', 'PKG', 'unit', '14', 10000, 6000, _AMBIENT, (8,)),
    _Spec('PET Bottle 500 ml', 'PKG', 'unit', '18', 20000, 12000, _AMBIENT, (8,)),
    _Spec('PET Bottle 1 L', 'PKG', 'unit', '26', 10000, 6000, _AMBIENT, (8,)),
    _Spec('Tamper-Evident Cap 38 mm', 'PKG', 'unit', '4', 40000, 25000, _AMBIENT, (9,)),
    _Spec('Front Label 500 ml', 'PKG', 'unit', '2', 40000, 25000, _AMBIENT, (10,)),
    _Spec('Shrink Sleeve 1 L', 'PKG', 'unit', '5', 15000, 8000, _AMBIENT, (10, 12)),
    _Spec('Carton 12 x 500 ml', 'PKG', 'unit', '90', 2000, 1200, _AMBIENT, (10,)),
    _Spec('Stretch Film Roll', 'PKG', 'unit', '1200', 40, 20, _AMBIENT, (12,)),
    _Spec('Peracetic Acid Sanitiser', 'CHM', 'L', '480', 100, 60, _CHEM, (11,)),
    _Spec('Caustic Soda (CIP)', 'CHM', 'kg', '160', 100, 50, _CHEM, (11,)),
    _Spec('Nitric Acid (CIP)', 'CHM', 'L', '300', 100, 50, _CHEM, (11,)),
)

_SUPPLIERS: tuple[tuple[str, str], ...] = (
    ('Rift Valley Fresh Produce Ltd', 'riftvalleyfresh'),
    ('Coastal Tropical Fruits Ltd', 'coastaltropical'),
    ('Highlands Roots & Greens', 'highlandsroots'),
    ('Naivasha Citrus Growers', 'naivashacitrus'),
    ('Sweet Source Sugar & Honey Ltd', 'sweetsource'),
    ('Pure Acids & Additives Ltd', 'pureacids'),
    ('Seed & Superfood Traders', 'seedsuperfood'),
    ('ClearPack PET Industries', 'clearpackpet'),
    ('SnapSeal Closures Ltd', 'snapseal'),
    ('BrightPrint Labels & Cartons', 'brightprint'),
    ('SafeClean Chemicals Ltd', 'safeclean'),
    ('Wrapwell Packaging Supplies', 'wrapwell'),
)

_STATES = tuple(PurchaseOrderState)
_STATE_WEIGHTS = (8, 8, 7, 20, 45, 12)  # draft, sent, to approve, purchase, done, cancel
_AGE_DAYS: dict[PurchaseOrderState, tuple[int, int]] = {
    PurchaseOrderState.DRAFT: (0, 14),
    PurchaseOrderState.SENT: (0, 21),
    PurchaseOrderState.TO_APPROVE: (0, 10),
    PurchaseOrderState.PURCHASE: (0, 60),
    PurchaseOrderState.DONE: (14, 240),
    PurchaseOrderState.CANCEL: (0, 240),
}

_StockPlan = tuple[tuple[StockLevel, ...], dict[str, Decimal], frozenset[str]]


@dataclass(frozen=True)
class DemoTruth:
    """Planted answers for evaluation. Never exposed through the ERP."""

    below_reorder: frozenset[str]


@dataclass(frozen=True)
class DemoDataset:
    products: tuple[Product, ...]
    partners: tuple[Partner, ...]
    stock: tuple[StockLevel, ...]
    purchase_orders: tuple[PurchaseOrder, ...]
    reorder_points: dict[str, Decimal]
    truth: DemoTruth


class _Draft(NamedTuple):
    ordered_at: datetime
    partner_ref: str
    state: PurchaseOrderState
    lines: tuple[PurchaseOrderLine, ...]


def _ref(position: int) -> str:
    return f'P-{position + 1:03d}'


def _products() -> tuple[Product, ...]:
    return tuple(
        Product(
            ref=_ref(position),
            name=spec.name,
            default_code=f'{spec.prefix}-{position + 1:03d}',
            uom=spec.uom,
        )
        for position, spec in enumerate(_SPECS)
    )


def _partners() -> tuple[Partner, ...]:
    return tuple(
        Partner(
            ref=f'V-{number:03d}',
            name=name,
            email=None if number % 6 == 0 else f'orders@{domain}.example',
            is_supplier=True,
        )
        for number, (name, domain) in enumerate(_SUPPLIERS, start=1)
    )


def _stock_and_reorder(rng: random.Random) -> _StockPlan:
    """Stock built so the below-reorder set is known by construction.

    Below-reorder products hold 30-90 % of their reorder point, the rest 150-300 %.
    Reservations never exceed 5 % of stock, so free stock cannot cross the threshold.
    """
    below = set(rng.sample(range(len(_SPECS)), _BELOW_REORDER_COUNT))
    levels: list[StockLevel] = []
    reorder_points: dict[str, Decimal] = {}
    for position, spec in enumerate(_SPECS):
        percent = rng.randint(30, 90) if position in below else rng.randint(150, 300)
        on_hand = spec.reorder * percent // 100
        reserved = on_hand // 20 if rng.random() < 0.4 else 0
        levels.append(
            StockLevel(
                product_ref=_ref(position),
                location=spec.location,
                quantity=Decimal(on_hand),
                reserved_quantity=Decimal(reserved),
            )
        )
        reorder_points[_ref(position)] = Decimal(spec.reorder)
    return tuple(levels), reorder_points, frozenset(_ref(position) for position in below)


def _products_by_supplier() -> dict[int, list[int]]:
    mapping: dict[int, list[int]] = {}
    for position, spec in enumerate(_SPECS):
        for supplier in spec.suppliers:
            mapping.setdefault(supplier, []).append(position)
    return mapping


def _line(rng: random.Random, position: int) -> PurchaseOrderLine:
    spec = _SPECS[position]
    quantity = spec.order_qty * rng.choice((1, 1, 2, 3))
    price = (Decimal(spec.price) * rng.randint(95, 108) / 100).quantize(Decimal('0.01'))
    return PurchaseOrderLine(
        product_ref=_ref(position), product_qty=Decimal(quantity), price_unit=price
    )


def _purchase_orders(rng: random.Random) -> tuple[PurchaseOrder, ...]:
    """Orders in every state; the first six are forced so each state always appears."""
    by_supplier = _products_by_supplier()
    suppliers = sorted(by_supplier)
    drafts: list[_Draft] = []
    for index in range(_ORDER_COUNT):
        if index < len(_STATES):
            state = _STATES[index]
        else:
            state = rng.choices(_STATES, weights=_STATE_WEIGHTS)[0]
        supplier = rng.choice(suppliers)
        positions = by_supplier[supplier]
        count = rng.randint(1, min(4, len(positions)))
        lines = tuple(_line(rng, position) for position in rng.sample(positions, count))
        low, high = _AGE_DAYS[state]
        ordered_at = ANCHOR - timedelta(days=rng.randint(low, high), hours=rng.randint(6, 17))
        drafts.append(_Draft(ordered_at, f'V-{supplier:03d}', state, lines))
    drafts.sort(key=lambda draft: draft.ordered_at)
    return tuple(
        PurchaseOrder(
            ref=f'PO-{number:04d}',
            name=f'PO{number:05d}',
            partner_ref=draft.partner_ref,
            state=draft.state,
            lines=draft.lines,
            date_order=draft.ordered_at,
        )
        for number, draft in enumerate(drafts, start=1)
    )


def build_demo_dataset(seed: int = DEFAULT_SEED) -> DemoDataset:
    """Build the full dataset. The same seed always yields an identical result."""
    rng = random.Random(seed)
    stock, reorder_points, below = _stock_and_reorder(rng)
    return DemoDataset(
        products=_products(),
        partners=_partners(),
        stock=stock,
        purchase_orders=_purchase_orders(rng),
        reorder_points=reorder_points,
        truth=DemoTruth(below_reorder=below),
    )
