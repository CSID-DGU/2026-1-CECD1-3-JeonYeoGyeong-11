"""Result JSON schema helpers."""

from __future__ import annotations

from argparse import Namespace
from typing import Any, Mapping


RESULT_SCHEMA_VERSION = "1"
LEGACY_RESULT_SCHEMA_VERSION = "v0"


def config_aliases_from_args(args: Namespace | None) -> list[str]:
    """Return config aliases recorded by ``config_io``."""
    if args is None:
        return []
    aliases = getattr(args, "_config_aliases_used", ())
    return [str(alias) for alias in aliases]


def unsupported_components_from_args(args: Namespace | None) -> list[str]:
    """Return unsupported component names recorded during execution."""
    if args is None:
        return []
    components = getattr(args, "_unsupported_components", ())
    return [str(component) for component in components]


def with_result_schema(
    payload: Mapping[str, Any],
    *,
    config_aliases_used: list[str] | tuple[str, ...] | None = None,
    unsupported_components: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Return a copy of ``payload`` with required result schema fields."""
    data = dict(payload)
    data.setdefault("result_schema_version", RESULT_SCHEMA_VERSION)
    data.setdefault(
        "config_aliases_used",
        [str(alias) for alias in (config_aliases_used or [])],
    )
    data.setdefault(
        "unsupported_components",
        [str(component) for component in (unsupported_components or [])],
    )
    return data


def result_schema_version(payload: Mapping[str, Any]) -> str:
    """Return the payload schema version, treating old results as v0."""
    value = payload.get("result_schema_version")
    if value is None:
        return LEGACY_RESULT_SCHEMA_VERSION
    return str(value)


def validate_result_schema(payload: Mapping[str, Any]) -> list[str]:
    """Return missing required fields for v1-style result payloads."""
    missing = []
    for key in ("result_schema_version", "config_aliases_used", "unsupported_components"):
        if key not in payload:
            missing.append(key)
    return missing


__all__ = [
    "LEGACY_RESULT_SCHEMA_VERSION",
    "RESULT_SCHEMA_VERSION",
    "config_aliases_from_args",
    "result_schema_version",
    "unsupported_components_from_args",
    "validate_result_schema",
    "with_result_schema",
]
