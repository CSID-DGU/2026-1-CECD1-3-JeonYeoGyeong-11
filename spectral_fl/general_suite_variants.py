"""Backward-compatible facade for vision FL suite variants."""

from graphfl_lab.experiments.suites.vision.variants import (
    build_base_cmd,
    parse_variant,
    variant_cmd,
)

__all__ = ["build_base_cmd", "parse_variant", "variant_cmd"]
