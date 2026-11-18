"""Production wiring of MockErp: the deterministic demo dataset, no I/O."""

from agentic_ops.erp.mock.adapter import MockErp
from agentic_ops.erp.mock.demo_seed import DEFAULT_SEED, build_demo_dataset


def demo_erp(seed: int = DEFAULT_SEED) -> MockErp:
    """A fresh MockErp holding the demo catalogue, stock and suppliers."""
    dataset = build_demo_dataset(seed)
    return MockErp(products=dataset.products, stock=dataset.stock, partners=dataset.partners)
