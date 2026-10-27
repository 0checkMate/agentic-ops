"""Deterministic demo dataset for a cold-pressed juice plant (illustrative prices in KES).

Pure data generation: no I/O, no clock. Planted answers live in DemoTruth, never in the
ERP-visible records, so agents cannot read them.
"""

import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from enum import StrEnum
from typing import NamedTuple

from agentic_ops.erp.models import (
    BatchConsumption,
    Customer,
    FinishedLot,
    GoodsReceipt,
    GoodsReceiptLine,
    Partner,
    Product,
    ProductionBatch,
    PurchaseOrder,
    PurchaseOrderLine,
    PurchaseOrderState,
    Shipment,
    ShipmentLine,
    StockLevel,
    SupplierInvoice,
    SupplierInvoiceLine,
    SupplierLot,
)

ANCHOR = datetime(2026, 10, 1)
DEFAULT_SEED = 42

_ORDER_COUNT = 220
_BELOW_REORDER_COUNT = 8
_COLD = 'WH/Cold'
_AMBIENT = 'WH/Stock'
_CHEM = 'WH/Chem'

PRICE_TOLERANCE = Decimal('0.02')
_BILLABLE = frozenset({PurchaseOrderState.PURCHASE, PurchaseOrderState.DONE})
_SHORT_RATE = 0.25
_SPLIT_RATE = 0.15
_SHUTDOWN_WEEKDAY = 6  # no production on Sundays
_LOTTED_PREFIXES = ('ING-', 'PKG-')
_LAST_BATCH_DAYS_BEFORE_ANCHOR = 2


class DiscrepancyKind(StrEnum):
    QUANTITY_MISMATCH = 'quantity_mismatch'
    PRICE_MISMATCH = 'price_mismatch'
    NO_RECEIPT = 'no_receipt'
    UNORDERED_ITEM = 'unordered_item'
    DUPLICATE = 'duplicate'


_CORRUPTIONS = (
    DiscrepancyKind.QUANTITY_MISMATCH,
    DiscrepancyKind.PRICE_MISMATCH,
    DiscrepancyKind.UNORDERED_ITEM,
)


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
class RecallScope:
    """Everything a recall of one supplier lot must reach: finished lots and customers."""

    finished_lots: frozenset[str]
    customers: frozenset[str]


@dataclass(frozen=True)
class DemoTruth:
    """Planted answers for evaluation. Never exposed through the ERP."""

    below_reorder: frozenset[str]
    invoice_discrepancies: dict[str, DiscrepancyKind]
    recall_scope: dict[str, RecallScope]


@dataclass(frozen=True)
class DemoDataset:
    products: tuple[Product, ...]
    partners: tuple[Partner, ...]
    stock: tuple[StockLevel, ...]
    purchase_orders: tuple[PurchaseOrder, ...]
    receipts: tuple[GoodsReceipt, ...]
    invoices: tuple[SupplierInvoice, ...]
    reorder_points: dict[str, Decimal]
    supplier_lots: tuple[SupplierLot, ...]
    batches: tuple[ProductionBatch, ...]
    finished_lots: tuple[FinishedLot, ...]
    customers: tuple[Customer, ...]
    shipments: tuple[Shipment, ...]
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


class _ReceiptDraft(NamedTuple):
    received_at: datetime
    order_ref: str
    lines: tuple[GoodsReceiptLine, ...]


class _Base(NamedTuple):
    order: PurchaseOrder
    invoice_date: datetime
    lines: tuple[SupplierInvoiceLine, ...]


class _InvoiceDraft(NamedTuple):
    invoice_date: datetime
    sequence: int
    number: str
    order: PurchaseOrder
    lines: tuple[SupplierInvoiceLine, ...]
    kind: DiscrepancyKind | None


def _placed(order: PurchaseOrder) -> datetime:
    if order.date_order is None:
        raise ValueError(f'Order {order.ref} has no order date')
    return order.date_order


def _delay(rng: random.Random, start: datetime, low: int, high: int) -> datetime:
    """A moment after start, never later than the anchor."""
    return min(ANCHOR, start + timedelta(days=rng.randint(low, high), hours=rng.randint(1, 8)))


def _receive(
    rng: random.Random, order: PurchaseOrder
) -> tuple[list[_ReceiptDraft], dict[str, Decimal]]:
    """One or two goods receipts for an order; some deliveries are short, some split."""
    received: dict[str, Decimal] = {}
    any_short = False
    for line in order.lines:
        if rng.random() < _SHORT_RATE:
            any_short = True
            received[line.product_ref] = line.product_qty * rng.randint(50, 90) // 100
        else:
            received[line.product_ref] = line.product_qty
    first_at = _delay(rng, _placed(order), 2, 14)
    if any_short or rng.random() >= _SPLIT_RATE:
        lines = tuple(
            GoodsReceiptLine(product_ref=ref, quantity=quantity)
            for ref, quantity in received.items()
        )
        return [_ReceiptDraft(first_at, order.ref, lines)], received
    first = tuple(
        GoodsReceiptLine(product_ref=line.product_ref, quantity=line.product_qty * 60 // 100)
        for line in order.lines
    )
    rest = tuple(
        GoodsReceiptLine(
            product_ref=line.product_ref,
            quantity=line.product_qty - line.product_qty * 60 // 100,
        )
        for line in order.lines
    )
    second_at = _delay(rng, first_at, 2, 7)
    drafts = [_ReceiptDraft(first_at, order.ref, first), _ReceiptDraft(second_at, order.ref, rest)]
    return drafts, received


def _clean_lines(
    rng: random.Random, order: PurchaseOrder, received: dict[str, Decimal]
) -> tuple[SupplierInvoiceLine, ...]:
    """Bill what arrived, at the order price with occasional noise inside tolerance."""
    lines: list[SupplierInvoiceLine] = []
    for line in order.lines:
        factor = rng.randint(985, 1015) if rng.random() < 0.3 else 1000
        price = (line.price_unit * factor / 1000).quantize(Decimal('0.01'))
        lines.append(
            SupplierInvoiceLine(
                product_ref=line.product_ref,
                quantity=received[line.product_ref],
                price_unit=price,
            )
        )
    return tuple(lines)


def _corrupt(
    rng: random.Random,
    kind: DiscrepancyKind,
    order: PurchaseOrder,
    lines: tuple[SupplierInvoiceLine, ...],
) -> tuple[SupplierInvoiceLine, ...]:
    """Plant one defect in an otherwise clean invoice."""
    if kind is DiscrepancyKind.UNORDERED_ITEM:
        on_order = {line.product_ref for line in order.lines}
        position = rng.choice([p for p in range(len(_SPECS)) if _ref(p) not in on_order])
        spec = _SPECS[position]
        extra = SupplierInvoiceLine(
            product_ref=_ref(position),
            quantity=Decimal(spec.order_qty),
            price_unit=Decimal(spec.price),
        )
        return (*lines, extra)
    index = rng.randrange(len(lines))
    target = lines[index]
    if kind is DiscrepancyKind.QUANTITY_MISMATCH:
        changed = SupplierInvoiceLine(
            product_ref=target.product_ref,
            quantity=target.quantity + max(Decimal(1), target.quantity // 10),
            price_unit=target.price_unit,
        )
    elif kind is DiscrepancyKind.PRICE_MISMATCH:
        raised = (target.price_unit * rng.randint(110, 130) / 100).quantize(Decimal('0.01'))
        changed = SupplierInvoiceLine(
            product_ref=target.product_ref, quantity=target.quantity, price_unit=raised
        )
    else:
        raise ValueError(f'Cannot corrupt an invoice with {kind}')
    return (*lines[:index], changed, *lines[index + 1 :])


def _billing(
    rng: random.Random, orders: tuple[PurchaseOrder, ...]
) -> tuple[tuple[GoodsReceipt, ...], tuple[SupplierInvoice, ...], dict[str, DiscrepancyKind]]:
    """Receipts and invoices for billable orders, with every planted defect recorded."""
    receipt_drafts: list[_ReceiptDraft] = []
    bases: list[_Base] = []
    unreceipted: list[PurchaseOrder] = []
    for order in orders:
        if order.state not in _BILLABLE:
            continue
        done = order.state is PurchaseOrderState.DONE
        if rng.random() >= (0.92 if done else 0.5):
            unreceipted.append(order)
            continue
        drafts, received = _receive(rng, order)
        receipt_drafts.extend(drafts)
        if rng.random() < (0.9 if done else 0.3):
            last = max(draft.received_at for draft in drafts)
            invoice_date = _delay(rng, last, 1, 10)
            bases.append(_Base(order, invoice_date, _clean_lines(rng, order, received)))

    corrupted = rng.sample(range(len(bases)), max(3, len(bases) // 8))
    kind_by_base = {
        index: _CORRUPTIONS[position % len(_CORRUPTIONS)]
        for position, index in enumerate(corrupted)
    }
    clean = [index for index in range(len(bases)) if index not in kind_by_base]
    duplicated = rng.sample(clean, max(2, len(bases) // 20))

    invoice_drafts: list[_InvoiceDraft] = []
    for index, base in enumerate(bases):
        kind = kind_by_base.get(index)
        lines = base.lines if kind is None else _corrupt(rng, kind, base.order, base.lines)
        number = f'{base.order.partner_ref}-{1000 + index}'
        invoice_drafts.append(
            _InvoiceDraft(base.invoice_date, index, number, base.order, lines, kind)
        )
    sequence = len(bases)
    for index in duplicated:
        original = invoice_drafts[index]
        later = min(ANCHOR, original.invoice_date + timedelta(days=rng.randint(1, 9)))
        invoice_drafts.append(
            _InvoiceDraft(
                later,
                sequence,
                original.number,
                original.order,
                original.lines,
                DiscrepancyKind.DUPLICATE,
            )
        )
        sequence += 1
    for order in rng.sample(unreceipted, min(len(unreceipted), max(3, len(unreceipted) // 4))):
        lines = tuple(
            SupplierInvoiceLine(
                product_ref=line.product_ref, quantity=line.product_qty, price_unit=line.price_unit
            )
            for line in order.lines
        )
        invoice_date = _delay(rng, _placed(order), 3, 20)
        number = f'{order.partner_ref}-{1000 + sequence}'
        invoice_drafts.append(
            _InvoiceDraft(invoice_date, sequence, number, order, lines, DiscrepancyKind.NO_RECEIPT)
        )
        sequence += 1

    receipts = tuple(
        GoodsReceipt(
            ref=f'GR-{number:04d}',
            name=f'WH/IN/{number:05d}',
            purchase_order_ref=draft.order_ref,
            received_at=draft.received_at,
            lines=draft.lines,
        )
        for number, draft in enumerate(
            sorted(receipt_drafts, key=lambda draft: draft.received_at), start=1
        )
    )
    invoices: list[SupplierInvoice] = []
    truth: dict[str, DiscrepancyKind] = {}
    ordered = sorted(invoice_drafts, key=lambda draft: (draft.invoice_date, draft.sequence))
    for position, draft in enumerate(ordered, start=1):
        ref = f'BILL-{position:04d}'
        invoices.append(
            SupplierInvoice(
                ref=ref,
                number=draft.number,
                partner_ref=draft.order.partner_ref,
                purchase_order_ref=draft.order.ref,
                invoice_date=draft.invoice_date,
                lines=draft.lines,
            )
        )
        if draft.kind is not None:
            truth[ref] = draft.kind
    return receipts, tuple(invoices), truth


class _Recipe(NamedTuple):
    name: str
    ingredients: tuple[int, ...]  # positions in _SPECS; each draws a share of its order size
    packaging: tuple[int, ...]  # positions in _SPECS; one piece per finished unit


_RECIPES: tuple[_Recipe, ...] = (
    _Recipe('Tropical Mango-Pineapple 500 ml', (0, 1), (20, 22, 23)),
    _Recipe('Orange Fresh 500 ml', (2,), (20, 22, 23)),
    _Recipe('Green Detox 500 ml', (11, 10, 7), (20, 22, 23)),
    _Recipe('Ginger Turmeric Shot 250 ml', (4, 5, 7, 14), (19, 22)),
    _Recipe('Carrot Beet Roots 1 L', (8, 9, 10), (21, 22, 24)),
    _Recipe('Watermelon Cooler 500 ml', (3, 7), (20, 22, 23)),
    _Recipe('Passion Coconut 500 ml', (6, 12, 13), (20, 22, 23)),
)

_CUSTOMERS = (
    'Greenleaf Supermarkets',
    'Karura Hotels Group',
    'Mombasa Road Wholesalers',
    'Lakeside Hospitality Ltd',
    'FreshBasket Retail',
    'Nairobi Wellness Cafes',
    'Savanna Distributors',
    'Highrise Office Catering',
    'Kisumu Fresh Mart',
    'Coastline Resorts',
    'Everyday Minimarts',
    'Campus Juice Kiosks',
)


class _Produced(NamedTuple):
    batches: tuple[ProductionBatch, ...]
    finished: tuple[FinishedLot, ...]
    used_in: dict[str, set[str]]  # supplier lot ref -> refs of batches that consumed it


class _Trace(NamedTuple):
    supplier_lots: tuple[SupplierLot, ...]
    batches: tuple[ProductionBatch, ...]
    finished_lots: tuple[FinishedLot, ...]
    customers: tuple[Customer, ...]
    shipments: tuple[Shipment, ...]
    scope: dict[str, RecallScope]


def _supplier_lots(
    receipts: tuple[GoodsReceipt, ...], products: tuple[Product, ...]
) -> tuple[SupplierLot, ...]:
    """One lot per ingredient or packaging receipt line; chemicals are not lotted."""
    lotted = {
        product.ref
        for product in products
        if (product.default_code or '').startswith(_LOTTED_PREFIXES)
    }
    lots: list[SupplierLot] = []
    for receipt in receipts:
        for item in receipt.lines:
            if item.product_ref in lotted:
                lots.append(
                    SupplierLot(
                        ref=f'SL-{len(lots) + 1:05d}',
                        product_ref=item.product_ref,
                        receipt_ref=receipt.ref,
                        received_at=receipt.received_at,
                        quantity=item.quantity,
                    )
                )
    return tuple(lots)


def _draw(
    lots: list[SupplierLot], remaining: dict[str, Decimal], at: datetime, need: Decimal
) -> list[BatchConsumption] | None:
    """Take the quantity oldest-first (FIFO) from lots already received; None if short."""
    taken: list[BatchConsumption] = []
    still_needed = need
    for lot in lots:
        if lot.received_at > at or remaining[lot.ref] <= 0:
            continue
        quantity = min(remaining[lot.ref], still_needed)
        taken.append(BatchConsumption(supplier_lot_ref=lot.ref, quantity=quantity))
        still_needed -= quantity
        if still_needed <= 0:
            return taken
    return None


def _plan_batch(
    rng: random.Random,
    recipe: _Recipe,
    units: int,
    started: datetime,
    lots_by_product: dict[str, list[SupplierLot]],
    remaining: dict[str, Decimal],
) -> list[BatchConsumption] | None:
    """All consumption lines for one batch, or None if any component is short of stock."""
    plan: list[BatchConsumption] = []
    for position in recipe.ingredients:
        percent = rng.choice((25, 40, 50))
        need = Decimal(max(1, _SPECS[position].order_qty * percent // 100))
        drawn = _draw(lots_by_product.get(_ref(position), []), remaining, started, need)
        if drawn is None:
            return None
        plan.extend(drawn)
    for position in recipe.packaging:
        drawn = _draw(lots_by_product.get(_ref(position), []), remaining, started, Decimal(units))
        if drawn is None:
            return None
        plan.extend(drawn)
    return plan


def _produce(rng: random.Random, lots: tuple[SupplierLot, ...]) -> _Produced:
    """Daily production: stock is only committed when a whole batch can be supplied."""
    lots_by_product: dict[str, list[SupplierLot]] = {}
    for lot in lots:
        lots_by_product.setdefault(lot.product_ref, []).append(lot)
    remaining = {lot.ref: lot.quantity for lot in lots}
    batches: list[ProductionBatch] = []
    finished: list[FinishedLot] = []
    used_in: dict[str, set[str]] = {}
    earliest = min(lot.received_at for lot in lots)
    day = earliest.replace(hour=0, minute=0, second=0, microsecond=0)
    last_day = ANCHOR - timedelta(days=_LAST_BATCH_DAYS_BEFORE_ANCHOR)
    while day <= last_day:
        if day.weekday() != _SHUTDOWN_WEEKDAY:
            started = day + timedelta(hours=rng.randint(6, 14))
            for recipe in rng.sample(_RECIPES, len(_RECIPES)):
                units = rng.choice((2000, 3000, 4000))
                plan = _plan_batch(rng, recipe, units, started, lots_by_product, remaining)
                if plan is None:
                    continue
                number = len(batches) + 1
                batch_ref = f'B-{number:04d}'
                for use in plan:
                    remaining[use.supplier_lot_ref] -= use.quantity
                    used_in.setdefault(use.supplier_lot_ref, set()).add(batch_ref)
                batches.append(
                    ProductionBatch(
                        ref=batch_ref,
                        name=f'MO/{number:05d}',
                        started_at=started,
                        lines=tuple(plan),
                    )
                )
                finished.append(
                    FinishedLot(
                        ref=f'FL-{number:04d}',
                        batch_ref=batch_ref,
                        product_name=recipe.name,
                        produced_at=started + timedelta(hours=rng.randint(6, 10)),
                        quantity=Decimal(units),
                    )
                )
                break
        day += timedelta(days=1)
    return _Produced(tuple(batches), tuple(finished), used_in)


def _ship(
    rng: random.Random, finished: tuple[FinishedLot, ...], customers: tuple[Customer, ...]
) -> tuple[tuple[Shipment, ...], dict[str, set[str]]]:
    """Split each finished lot across 1-4 customers; recent lots may not have shipped yet."""
    lines_by_stop: dict[tuple[str, datetime], list[ShipmentLine]] = {}
    hour_by_stop: dict[tuple[str, datetime], int] = {}
    customers_of: dict[str, set[str]] = {}
    for lot in finished:
        delay_days = rng.randint(1, 6)
        shipped_total = lot.quantity * rng.randint(70, 100) // 100
        buyers = rng.sample(customers, rng.randint(1, 4))
        produced_day = lot.produced_at.replace(hour=0, minute=0, second=0, microsecond=0)
        dispatch_day = produced_day + timedelta(days=delay_days)
        if dispatch_day >= ANCHOR:
            continue
        share = shipped_total // len(buyers)
        for buyer in buyers:
            stop = (buyer.ref, dispatch_day)
            lines_by_stop.setdefault(stop, []).append(
                ShipmentLine(finished_lot_ref=lot.ref, quantity=share)
            )
            if stop not in hour_by_stop:
                hour_by_stop[stop] = rng.randint(7, 16)
            customers_of.setdefault(lot.ref, set()).add(buyer.ref)
    ordered = sorted(
        lines_by_stop, key=lambda stop: (stop[1] + timedelta(hours=hour_by_stop[stop]), stop[0])
    )
    shipments = tuple(
        Shipment(
            ref=f'SH-{number:04d}',
            customer_ref=stop[0],
            shipped_at=stop[1] + timedelta(hours=hour_by_stop[stop]),
            lines=tuple(lines_by_stop[stop]),
        )
        for number, stop in enumerate(ordered, start=1)
    )
    return shipments, customers_of


def _traceability(
    rng: random.Random, receipts: tuple[GoodsReceipt, ...], products: tuple[Product, ...]
) -> _Trace:
    """Lots, batches, finished lots, customers and shipments, plus each lot's recall scope."""
    lots = _supplier_lots(receipts, products)
    produced = _produce(rng, lots)
    customers = tuple(
        Customer(ref=f'C-{number:03d}', name=name)
        for number, name in enumerate(_CUSTOMERS, start=1)
    )
    shipments, customers_of = _ship(rng, produced.finished, customers)
    finished_of: dict[str, set[str]] = {}
    for finished_lot in produced.finished:
        finished_of.setdefault(finished_lot.batch_ref, set()).add(finished_lot.ref)
    scope: dict[str, RecallScope] = {}
    for lot in lots:
        reached = {
            finished_ref
            for batch_ref in produced.used_in.get(lot.ref, set())
            for finished_ref in finished_of[batch_ref]
        }
        reached_customers = {
            customer_ref
            for finished_ref in reached
            for customer_ref in customers_of.get(finished_ref, set())
        }
        scope[lot.ref] = RecallScope(frozenset(reached), frozenset(reached_customers))
    return _Trace(lots, produced.batches, produced.finished, customers, shipments, scope)


def build_demo_dataset(seed: int = DEFAULT_SEED) -> DemoDataset:
    """Build the full dataset. The same seed always yields an identical result."""
    rng = random.Random(seed)
    stock, reorder_points, below = _stock_and_reorder(rng)
    orders = _purchase_orders(rng)
    receipts, invoices, discrepancies = _billing(rng, orders)
    products = _products()
    trace = _traceability(rng, receipts, products)  # drawn last: earlier data stays stable
    return DemoDataset(
        products=products,
        partners=_partners(),
        stock=stock,
        purchase_orders=orders,
        receipts=receipts,
        invoices=invoices,
        reorder_points=reorder_points,
        supplier_lots=trace.supplier_lots,
        batches=trace.batches,
        finished_lots=trace.finished_lots,
        customers=trace.customers,
        shipments=trace.shipments,
        truth=DemoTruth(
            below_reorder=below,
            invoice_discrepancies=discrepancies,
            recall_scope=trace.scope,
        ),
    )
