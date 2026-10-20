class GatewayError(Exception):
    """Base class for all model gateway failures."""


class GatewayConfigError(GatewayError):
    """The gateway is misconfigured: a fault in setup, not a transient provider failure."""
