from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class Alias(StrEnum):
    """Logical model tiers that agent code requests instead of vendor model names."""

    FAST = 'fast'
    DEFAULT = 'default'
    REASONING = 'reasoning'


class ModelSpec(BaseModel):
    """One concrete route: a provider, its model identifier and its capabilities."""

    model_config = ConfigDict(frozen=True)

    provider: str
    model_id: str
    supports_tools: bool
