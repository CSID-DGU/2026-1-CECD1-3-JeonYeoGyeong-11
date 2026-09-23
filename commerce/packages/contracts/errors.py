"""Shared local exceptions. No raw payload is kept in an error response."""
import re

from commerce.packages.contracts.validate import ERROR_CODE_ORDER


class ContractError(Exception):
    def __init__(self, code: str, field_path: str | None = None):
        if code not in ERROR_CODE_ORDER:
            raise ValueError("Unknown contract error code")
        if field_path is not None and re.fullmatch(r"(/(([^/~])|(~[01]))*)*", field_path) is None:
            raise ValueError("field_path must be a JSON pointer")
        self.code = code
        self.field_path = field_path
        super().__init__(code)

    def to_payload(self) -> dict:
        # Path segments can contain local identifiers: do not echo them over HTTP.
        return {
            "schema_version": "contract_error.v1", "code": self.code,
            "field_path": None, "expected": None, "received": None,
            "message": self.code,
        }


class FeatureNotImplemented(NotImplementedError):
    """Local scaffold failure; never add this to the v1 wire error enum."""


class JobBusyError(RuntimeError):
    """A local training/personalization task is already running."""
