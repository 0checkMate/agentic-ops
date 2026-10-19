class ErpError(Exception):
    """Base class for all errors raised by ERP adapters."""


class ErpNotFound(ErpError):
    """The requested record does not exist."""


class ErpValidationError(ErpError):
    """The ERP rejected the data as invalid."""


class ErpUnavailable(ErpError):
    """The ERP could not be reached or timed out."""
