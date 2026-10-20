class GatewayError(Exception):
    """Base class for all model gateway failures."""


class GatewayConfigError(GatewayError):
    """The gateway is misconfigured: a fault in setup, not a transient provider failure."""


class GatewayRateLimited(GatewayError):
    """A provider budget is exhausted for longer than the caller is willing to wait.

    retry_after is the seconds until capacity returns; math.inf means the request can
    never fit (for example, a prompt larger than the per-minute token budget).
    """

    def __init__(self, message: str, *, retry_after: float) -> None:
        super().__init__(message)
        self.retry_after = retry_after
