"""The mock_erp schema: the generated dataset loaded as normalized, queryable tables."""

from typing import Any

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from agentic_ops.erp.mock.adapter import MockErp
from agentic_ops.erp.mock.demo_seed import build_demo_dataset
from agentic_ops.erp.mock.loader import SCHEMA, load_dataset

pytestmark = pytest.mark.integration

DATASET = build_demo_dataset()

EXPECTED_COUNTS = {
    'product': len(DATASET.products),
    'partner': len(DATASET.partners),
    'stock_level': len(DATASET.stock),
    'reorder_point': len(DATASET.reorder_points),
    'purchase_order': len(DATASET.purchase_orders),
    'purchase_order_line': sum(len(o.lines) for o in DATASET.purchase_orders),
    'goods_receipt': len(DATASET.receipts),
    'goods_receipt_line': sum(len(r.lines) for r in DATASET.receipts),
    'supplier_invoice': len(DATASET.invoices),
    'supplier_invoice_line': sum(len(i.lines) for i in DATASET.invoices),
    'supplier_lot': len(DATASET.supplier_lots),
    'production_batch': len(DATASET.batches),
    'batch_consumption': sum(len(b.lines) for b in DATASET.batches),
    'finished_lot': len(DATASET.finished_lots),
    'customer': len(DATASET.customers),
    'shipment': len(DATASET.shipments),
    'shipment_line': sum(len(s.lines) for s in DATASET.shipments),
}


async def _load(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:
        await load_dataset(conn, DATASET)


async def _counts(engine: AsyncEngine) -> dict[str, int]:
    async with engine.connect() as conn:
        counts: dict[str, int] = {}
        for table in EXPECTED_COUNTS:
            result = await conn.execute(text(f'SELECT count(*) FROM {SCHEMA}.{table}'))
            counts[table] = int(result.scalar_one())
        return counts


async def _rows(engine: AsyncEngine, sql: str, **params: Any) -> list[tuple[Any, ...]]:
    async with engine.connect() as conn:
        result = await conn.execute(text(sql), params)
        return [tuple(row) for row in result]


async def test_every_table_has_the_dataset_row_count(async_engine: AsyncEngine) -> None:
    await _load(async_engine)
    assert await _counts(async_engine) == EXPECTED_COUNTS


async def test_loading_twice_is_idempotent(async_engine: AsyncEngine) -> None:
    await _load(async_engine)
    await _load(async_engine)
    assert await _counts(async_engine) == EXPECTED_COUNTS


async def test_products_match_mock_erp(async_engine: AsyncEngine) -> None:
    await _load(async_engine)
    rows = await _rows(
        async_engine,
        f'SELECT ref, name, default_code, uom, is_storable FROM {SCHEMA}.product',
    )
    erp = MockErp.from_dataset(DATASET)
    expected = [
        (p.ref, p.name, p.default_code, p.uom, p.is_storable)
        for p in await erp.list_products(limit=1000)
    ]
    assert sorted(rows) == sorted(expected)


async def test_stock_matches_mock_erp_exactly(async_engine: AsyncEngine) -> None:
    await _load(async_engine)
    rows = await _rows(
        async_engine,
        f'SELECT product_ref, location, quantity, reserved_quantity FROM {SCHEMA}.stock_level',
    )
    erp = MockErp.from_dataset(DATASET)
    expected = [
        (s.product_ref, s.location, s.quantity, s.reserved_quantity)
        for s in await erp.get_stock_levels()
    ]
    assert sorted(rows) == sorted(expected)


async def test_reorder_points_are_stored(async_engine: AsyncEngine) -> None:
    await _load(async_engine)
    rows = await _rows(async_engine, f'SELECT product_ref, quantity FROM {SCHEMA}.reorder_point')
    assert dict(rows) == DATASET.reorder_points


async def test_recall_query_reproduces_the_planted_truth(async_engine: AsyncEngine) -> None:
    await _load(async_engine)
    lot_ref, scope = max(
        DATASET.truth.recall_scope.items(), key=lambda item: len(item[1].customers)
    )
    sql = f"""
        SELECT DISTINCT fl.ref, s.customer_ref
        FROM {SCHEMA}.batch_consumption bc
        JOIN {SCHEMA}.finished_lot fl ON fl.batch_ref = bc.batch_ref
        JOIN {SCHEMA}.shipment_line sl ON sl.finished_lot_ref = fl.ref
        JOIN {SCHEMA}.shipment s ON s.ref = sl.shipment_ref
        WHERE bc.supplier_lot_ref = :lot_ref
    """
    rows = await _rows(async_engine, sql, lot_ref=lot_ref)
    assert {row[0] for row in rows} == set(scope.finished_lots)
    assert {row[1] for row in rows} == set(scope.customers)


async def test_timestamps_have_no_time_zone(async_engine: AsyncEngine) -> None:
    await _load(async_engine)
    rows = await _rows(
        async_engine,
        'SELECT DISTINCT data_type FROM information_schema.columns '
        'WHERE table_schema = :schema AND column_name IN (:a, :b)',
        schema=SCHEMA,
        a='received_at',
        b='shipped_at',
    )
    assert rows == [('timestamp without time zone',)]
