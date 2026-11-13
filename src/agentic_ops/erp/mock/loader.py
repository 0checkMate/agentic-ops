"""
Load the generated demo dataset into a normalized mock_erp Postgres schema.

The schema is demo data owned by this loader: dropped and recreated on every load, never
migrated by Alembic (it lives on its own MetaData, not on the ORM Base).
The caller owns the transaction, as with record_event.
"""

import re
from collections.abc import Mapping, Sequence
from typing import Any

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    MetaData,
    Numeric,
    String,
    Table,
    text,
)
from sqlalchemy.ext.asyncio import AsyncConnection

from agentic_ops.erp.mock.demo_seed import DemoDataset

SCHEMA = 'mock_erp'
READONLY_ROLE = 'agentic_readonly'
_SAFE_PASSWORD = re.compile(r'[A-Za-z0-9_-]{16,128}')

metadata = MetaData(schema=SCHEMA)

_QTY = Numeric(18, 4)  # exact decimals: no floats for quantities or money

product = Table(
    'product',
    metadata,
    Column('ref', String, primary_key=True),
    Column('name', String, nullable=False),
    Column('default_code', String),
    Column('uom', String, nullable=False),
    Column('is_storable', Boolean, nullable=False),
)
partner = Table(
    'partner',
    metadata,
    Column('ref', String, primary_key=True),
    Column('name', String, nullable=False),
    Column('email', String),
    Column('is_supplier', Boolean, nullable=False),
)
stock_level = Table(
    'stock_level',
    metadata,
    Column('product_ref', String, ForeignKey('product.ref'), nullable=False),
    Column('location', String, nullable=False),
    Column('quantity', _QTY, nullable=False),
    Column('reserved_quantity', _QTY, nullable=False),
)
reorder_point = Table(
    'reorder_point',
    metadata,
    Column('product_ref', String, ForeignKey('product.ref'), primary_key=True),
    Column('quantity', _QTY, nullable=False),
)
purchase_order = Table(
    'purchase_order',
    metadata,
    Column('ref', String, primary_key=True),
    Column('name', String, nullable=False),
    Column('partner_ref', String, ForeignKey('partner.ref'), nullable=False),
    Column('state', String, nullable=False),
    Column('date_order', DateTime),
)
purchase_order_line = Table(
    'purchase_order_line',
    metadata,
    Column('purchase_order_ref', String, ForeignKey('purchase_order.ref'), primary_key=True),
    Column('line_no', Integer, primary_key=True),
    Column('product_ref', String, ForeignKey('product.ref'), nullable=False),
    Column('product_qty', _QTY, nullable=False),
    Column('price_unit', _QTY, nullable=False),
)
goods_receipt = Table(
    'goods_receipt',
    metadata,
    Column('ref', String, primary_key=True),
    Column('name', String, nullable=False),
    Column('purchase_order_ref', String, ForeignKey('purchase_order.ref'), nullable=False),
    Column('received_at', DateTime, nullable=False),
)
goods_receipt_line = Table(
    'goods_receipt_line',
    metadata,
    Column('goods_receipt_ref', String, ForeignKey('goods_receipt.ref'), primary_key=True),
    Column('line_no', Integer, primary_key=True),
    Column('product_ref', String, ForeignKey('product.ref'), nullable=False),
    Column('quantity', _QTY, nullable=False),
)
supplier_invoice = Table(
    'supplier_invoice',
    metadata,
    Column('ref', String, primary_key=True),
    Column('number', String, nullable=False),
    Column('partner_ref', String, ForeignKey('partner.ref'), nullable=False),
    Column('purchase_order_ref', String, ForeignKey('purchase_order.ref'), nullable=False),
    Column('invoice_date', DateTime, nullable=False),
)
supplier_invoice_line = Table(
    'supplier_invoice_line',
    metadata,
    Column('supplier_invoice_ref', String, ForeignKey('supplier_invoice.ref'), primary_key=True),
    Column('line_no', Integer, primary_key=True),
    Column('product_ref', String, ForeignKey('product.ref'), nullable=False),
    Column('quantity', _QTY, nullable=False),
    Column('price_unit', _QTY, nullable=False),
)
supplier_lot = Table(
    'supplier_lot',
    metadata,
    Column('ref', String, primary_key=True),
    Column('product_ref', String, ForeignKey('product.ref'), nullable=False),
    Column('receipt_ref', String, ForeignKey('goods_receipt.ref'), nullable=False),
    Column('received_at', DateTime, nullable=False),
    Column('quantity', _QTY, nullable=False),
)
production_batch = Table(
    'production_batch',
    metadata,
    Column('ref', String, primary_key=True),
    Column('name', String, nullable=False),
    Column('started_at', DateTime, nullable=False),
)
batch_consumption = Table(
    'batch_consumption',
    metadata,
    Column('batch_ref', String, ForeignKey('production_batch.ref'), primary_key=True),
    Column('line_no', Integer, primary_key=True),
    Column('supplier_lot_ref', String, ForeignKey('supplier_lot.ref'), nullable=False),
    Column('quantity', _QTY, nullable=False),
)
finished_lot = Table(
    'finished_lot',
    metadata,
    Column('ref', String, primary_key=True),
    Column('batch_ref', String, ForeignKey('production_batch.ref'), nullable=False),
    Column('product_name', String, nullable=False),
    Column('produced_at', DateTime, nullable=False),
    Column('quantity', _QTY, nullable=False),
    Column('uom', String, nullable=False),
)
customer = Table(
    'customer',
    metadata,
    Column('ref', String, primary_key=True),
    Column('name', String, nullable=False),
)
shipment = Table(
    'shipment',
    metadata,
    Column('ref', String, primary_key=True),
    Column('customer_ref', String, ForeignKey('customer.ref'), nullable=False),
    Column('shipped_at', DateTime, nullable=False),
)
shipment_line = Table(
    'shipment_line',
    metadata,
    Column('shipment_ref', String, ForeignKey('shipment.ref'), primary_key=True),
    Column('line_no', Integer, primary_key=True),
    Column('finished_lot_ref', String, ForeignKey('finished_lot.ref'), nullable=False),
    Column('quantity', _QTY, nullable=False),
)


async def _insert(conn: AsyncConnection, table: Table, rows: Sequence[Mapping[str, Any]]) -> None:
    if rows:  # an empty executemany is an error
        await conn.execute(table.insert(), list(rows))


async def load_dataset(conn: AsyncConnection, dataset: DemoDataset) -> None:
    """Drop and recreate the mock_erp schema, then load every table from the dataset."""
    await conn.exec_driver_sql(f'DROP SCHEMA IF EXISTS {SCHEMA} CASCADE')
    await conn.exec_driver_sql(f'CREATE SCHEMA {SCHEMA}')
    await conn.run_sync(metadata.create_all)

    await _insert(
        conn,
        product,
        [
            {
                'ref': p.ref,
                'name': p.name,
                'default_code': p.default_code,
                'uom': p.uom,
                'is_storable': p.is_storable,
            }
            for p in dataset.products
        ],
    )
    await _insert(
        conn,
        partner,
        [
            {'ref': p.ref, 'name': p.name, 'email': p.email, 'is_supplier': p.is_supplier}
            for p in dataset.partners
        ],
    )
    await _insert(
        conn,
        stock_level,
        [
            {
                'product_ref': s.product_ref,
                'location': s.location,
                'quantity': s.quantity,
                'reserved_quantity': s.reserved_quantity,
            }
            for s in dataset.stock
        ],
    )
    await _insert(
        conn,
        reorder_point,
        [{'product_ref': ref, 'quantity': qty} for ref, qty in dataset.reorder_points.items()],
    )

    await _insert(
        conn,
        purchase_order,
        [
            {
                'ref': o.ref,
                'name': o.name,
                'partner_ref': o.partner_ref,
                'state': o.state.value,
                'date_order': o.date_order,
            }
            for o in dataset.purchase_orders
        ],
    )
    await _insert(
        conn,
        purchase_order_line,
        [
            {
                'purchase_order_ref': o.ref,
                'line_no': n,
                'product_ref': line.product_ref,
                'product_qty': line.product_qty,
                'price_unit': line.price_unit,
            }
            for o in dataset.purchase_orders
            for n, line in enumerate(o.lines, start=1)
        ],
    )

    await _insert(
        conn,
        goods_receipt,
        [
            {
                'ref': r.ref,
                'name': r.name,
                'purchase_order_ref': r.purchase_order_ref,
                'received_at': r.received_at,
            }
            for r in dataset.receipts
        ],
    )
    await _insert(
        conn,
        goods_receipt_line,
        [
            {
                'goods_receipt_ref': r.ref,
                'line_no': n,
                'product_ref': line.product_ref,
                'quantity': line.quantity,
            }
            for r in dataset.receipts
            for n, line in enumerate(r.lines, start=1)
        ],
    )

    await _insert(
        conn,
        supplier_invoice,
        [
            {
                'ref': i.ref,
                'number': i.number,
                'partner_ref': i.partner_ref,
                'purchase_order_ref': i.purchase_order_ref,
                'invoice_date': i.invoice_date,
            }
            for i in dataset.invoices
        ],
    )
    await _insert(
        conn,
        supplier_invoice_line,
        [
            {
                'supplier_invoice_ref': i.ref,
                'line_no': n,
                'product_ref': line.product_ref,
                'quantity': line.quantity,
                'price_unit': line.price_unit,
            }
            for i in dataset.invoices
            for n, line in enumerate(i.lines, start=1)
        ],
    )

    await _insert(
        conn,
        supplier_lot,
        [
            {
                'ref': lot.ref,
                'product_ref': lot.product_ref,
                'receipt_ref': lot.receipt_ref,
                'received_at': lot.received_at,
                'quantity': lot.quantity,
            }
            for lot in dataset.supplier_lots
        ],
    )
    await _insert(
        conn,
        production_batch,
        [{'ref': b.ref, 'name': b.name, 'started_at': b.started_at} for b in dataset.batches],
    )
    await _insert(
        conn,
        batch_consumption,
        [
            {
                'batch_ref': b.ref,
                'line_no': n,
                'supplier_lot_ref': line.supplier_lot_ref,
                'quantity': line.quantity,
            }
            for b in dataset.batches
            for n, line in enumerate(b.lines, start=1)
        ],
    )
    await _insert(
        conn,
        finished_lot,
        [
            {
                'ref': f.ref,
                'batch_ref': f.batch_ref,
                'product_name': f.product_name,
                'produced_at': f.produced_at,
                'quantity': f.quantity,
                'uom': f.uom,
            }
            for f in dataset.finished_lots
        ],
    )
    await _insert(
        conn,
        customer,
        [{'ref': c.ref, 'name': c.name} for c in dataset.customers],
    )
    await _insert(
        conn,
        shipment,
        [
            {'ref': s.ref, 'customer_ref': s.customer_ref, 'shipped_at': s.shipped_at}
            for s in dataset.shipments
        ],
    )
    await _insert(
        conn,
        shipment_line,
        [
            {
                'shipment_ref': s.ref,
                'line_no': n,
                'finished_lot_ref': line.finished_lot_ref,
                'quantity': line.quantity,
            }
            for s in dataset.shipments
            for n, line in enumerate(s.lines, start=1)
        ],
    )


async def provision_readonly_role(conn: AsyncConnection, password: str) -> None:
    """Create or update the read-only SQL role and grant it SELECT on mock_erp only.

    Run after every load_dataset: dropping the schema also drops its grants.
    The password is interpolated into DDL because Postgres cannot bind parameters there,
    so only a strict character set is accepted.
    """
    if _SAFE_PASSWORD.fullmatch(password) is None:
        raise ValueError('password must be 16-128 characters from A-Z a-z 0-9 _ -')
    exists = (
        await conn.execute(
            text('SELECT 1 FROM pg_roles WHERE rolname = :name'), {'name': READONLY_ROLE}
        )
    ).first()
    verb = 'ALTER' if exists else 'CREATE'
    await conn.exec_driver_sql(f"{verb} ROLE {READONLY_ROLE} LOGIN PASSWORD '{password}'")
    await conn.exec_driver_sql(f'ALTER ROLE {READONLY_ROLE} SET default_transaction_read_only = on')
    await conn.exec_driver_sql(f'GRANT USAGE ON SCHEMA {SCHEMA} TO {READONLY_ROLE}')
    await conn.exec_driver_sql(f'GRANT SELECT ON ALL TABLES IN SCHEMA {SCHEMA} TO {READONLY_ROLE}')
