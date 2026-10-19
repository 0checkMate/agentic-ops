from decimal import Decimal

from agentic_ops.erp.models import Partner, Product, StockLevel

SEED_PRODUCTS = (
    Product(ref='P-001', name='Mango Puree', default_code='ING-001', uom='kg'),
    Product(ref='P-002', name='PET Bottle 500 ml', default_code='PKG-001', uom='unit'),
    Product(ref='P-003', name='Cane Sugar', default_code='ING-002', uom='kg'),
)

SEED_STOCK = (
    StockLevel(
        product_ref='P-001',
        location='WH/Stock',
        quantity=Decimal('120'),
        reserved_quantity=Decimal('20'),
    ),
    StockLevel(product_ref='P-001', location='WH/Cold', quantity=Decimal('40')),
    StockLevel(product_ref='P-002', location='WH/Stock', quantity=Decimal('5000')),
)

SEED_PARTNERS = (
    Partner(
        ref='V-001',
        name='Fresh Fruit Supplies Ltd',
        email='orders@freshfruit.example',
        is_supplier=True,
    ),
    Partner(ref='V-002', name='Packaging Co', is_supplier=True),
)
