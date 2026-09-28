"""Contract check shared by the adapters: reject with the validator's own v1 code."""
import functools

from commerce.packages.contracts.errors import ContractError
from commerce.packages.contracts.types import Payload
from commerce.packages.contracts.validate import load_schemas, validate_payload


@functools.cache
def _validators():
    return load_schemas()


def check_payload(contract: str, payload: Payload) -> None:
    verdict = validate_payload(_validators()[contract], contract, payload)
    if not verdict.ok:
        raise ContractError(verdict.code, verdict.field_path)
