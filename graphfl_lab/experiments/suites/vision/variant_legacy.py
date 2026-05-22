"""Legacy compatibility variant parsing for vision suites."""

from __future__ import annotations

import re
from typing import List, Tuple

from graphfl_lab.experiments.suites.vision.variant_helpers import (
    legacy_residual_reweight_args,
)

ParsedVariant = Tuple[str, str, List[str]]


def parse_legacy_residual_variant(v: str) -> ParsedVariant | None:
    m = re.match(r"^ours_residual_reweight_knn_k(\d+)$", v)
    if m:
        k = m.group(1)
        return (
            "ours",
            v,
            legacy_residual_reweight_args("knn", k),
        )

    m = re.match(r"^ours_residual_reweight_random_matched_k(\d+)$", v)
    if m:
        k = m.group(1)
        return (
            "ours",
            v,
            legacy_residual_reweight_args("random", k),
        )

    if v == "ours_legacy_residual_reweight_dense":
        return (
            "ours",
            v,
            legacy_residual_reweight_args("dense"),
        )

    if v == "ours_legacy_residual_reweight_uniform":
        return (
            "ours",
            v,
            legacy_residual_reweight_args("uniform"),
        )

    m = re.match(r"^ours_legacy_residual_reweight_knn_k(\d+)$", v)
    if m:
        k = m.group(1)
        return (
            "ours",
            v,
            legacy_residual_reweight_args("knn", k),
        )

    m = re.match(r"^ours_legacy_residual_reweight_random_matched_k(\d+)$", v)
    if m:
        k = m.group(1)
        return (
            "ours",
            v,
            legacy_residual_reweight_args("random", k),
        )

    m = re.match(r"^ours_legacy_residual_reweight_magnitude_knn_k(\d+)$", v)
    if m:
        k = m.group(1)
        return (
            "ours",
            v,
            legacy_residual_reweight_args("magnitude_knn", k),
        )

    return None
