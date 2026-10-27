from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class DomainModel(BaseModel):
    model_config = ConfigDict(frozen=True)


class Product(DomainModel):
    ref: str
    name: str
    default_code: str | None = None
    uom: str
    is_storable: bool = True


class StockLevel(DomainModel):
    product_ref: str
    location: str
    quantity: Decimal
    reserved_quantity: Decimal = Decimal('0')


class Partner(DomainModel):
    ref: str
    name: str
    email: str | None = None
    is_supplier: bool = False


class PurchaseOrderState(StrEnum):
    DRAFT = 'draft'
    SENT = 'sent'
    TO_APPROVE = 'to approve'
    PURCHASE = 'purchase'
    DONE = 'done'
    CANCEL = 'cancel'


class PurchaseOrderLine(DomainModel):
    product_ref: str
    product_qty: Decimal = Field(gt=0)
    price_unit: Decimal = Field(ge=0)


class PurchaseOrder(DomainModel):
    ref: str
    name: str
    partner_ref: str
    state: PurchaseOrderState
    lines: tuple[PurchaseOrderLine, ...]
    date_order: datetime | None = None


class NewPurchaseOrder(DomainModel):
    partner_ref: str
    lines: tuple[PurchaseOrderLine, ...] = Field(min_length=1)
    idempotency_key: str


class GoodsReceiptLine(DomainModel):
    product_ref: str
    quantity: Decimal = Field(gt=0)


class GoodsReceipt(DomainModel):
    ref: str
    name: str
    purchase_order_ref: str
    received_at: datetime
    lines: tuple[GoodsReceiptLine, ...] = Field(min_length=1)


class SupplierInvoiceLine(DomainModel):
    product_ref: str
    quantity: Decimal = Field(gt=0)
    price_unit: Decimal = Field(ge=0)


class SupplierInvoice(DomainModel):
    ref: str
    number: str
    partner_ref: str
    purchase_order_ref: str
    invoice_date: datetime
    lines: tuple[SupplierInvoiceLine, ...] = Field(min_length=1)


class SupplierLot(DomainModel):
    ref: str
    product_ref: str
    receipt_ref: str
    received_at: datetime
    quantity: Decimal = Field(gt=0)


class BatchConsumption(DomainModel):
    supplier_lot_ref: str
    quantity: Decimal = Field(gt=0)


class ProductionBatch(DomainModel):
    ref: str
    name: str
    started_at: datetime
    lines: tuple[BatchConsumption, ...] = Field(min_length=1)


class FinishedLot(DomainModel):
    ref: str
    batch_ref: str
    product_name: str
    produced_at: datetime
    quantity: Decimal = Field(gt=0)
    uom: str = 'unit'


class Customer(DomainModel):
    ref: str
    name: str


class ShipmentLine(DomainModel):
    finished_lot_ref: str
    quantity: Decimal = Field(gt=0)


class Shipment(DomainModel):
    ref: str
    customer_ref: str
    shipped_at: datetime
    lines: tuple[ShipmentLine, ...] = Field(min_length=1)
