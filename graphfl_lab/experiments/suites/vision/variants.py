"""Vision FL suite variant token parsing."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

from graphfl_lab.experiments.suites.vision.variant_commands import build_base_cmd
from graphfl_lab.experiments.suites.vision.variant_core import parse_core_graph_variant
from graphfl_lab.experiments.suites.vision.variant_diagnostics import parse_diagnostic_variant
from graphfl_lab.experiments.suites.vision.variant_families import parse_baseline_variant
from graphfl_lab.experiments.suites.vision.variant_helpers import (
    result_path_for_variant,
)
from graphfl_lab.experiments.suites.vision.variant_legacy import (
    parse_legacy_residual_variant,
)
from graphfl_lab.experiments.suites.vision.variant_sources import parse_source_variant
from graphfl_lab.experiments.suites.vision.variant_suffixes import parse_suffix_variant
from graphfl_lab.experiments.suites.vision.variant_targets import parse_target_variant


def parse_variant(
    variant: str, args: argparse.Namespace
) -> Tuple[str, str, List[str]]:
    """Return (flower_method, row_label, extra_cli_args)."""
    default_knn_k = int(args.knn_k)
    v = variant.strip().lower()
    baseline = parse_baseline_variant(v, default_knn_k)
    if baseline is not None:
        return baseline

    suffix = parse_suffix_variant(variant, args, parse_variant)
    if suffix is not None:
        return suffix

    diagnostic = parse_diagnostic_variant(v, default_knn_k)
    if diagnostic is not None:
        return diagnostic

    legacy = parse_legacy_residual_variant(v)
    if legacy is not None:
        return legacy

    core = parse_core_graph_variant(v, default_knn_k)
    if core is not None:
        return core

    source = parse_source_variant(v)
    if source is not None:
        return source

    target = parse_target_variant(v)
    if target is not None:
        return target

    raise ValueError(f"Unknown general-FL variant: {variant!r}")


def variant_cmd(
    args: argparse.Namespace,
    variant: str,
    seed: int,
    suite_tag: str,
    out_dir: Path,
) -> Tuple[List[str], str, Path]:
    method, vlabel, extras = parse_variant(variant, args)
    # The suite directory already encodes the grid condition.  Keep per-run
    # tags short enough for Windows path limits.
    run_tag = f"{vlabel}_seed{seed}"
    cmd = (
        build_base_cmd(args)
        + ["--method", method, "--seed", str(seed), "--run-tag", run_tag]
        + extras
    )
    path = result_path_for_variant(out_dir, method, seed, run_tag)
    return cmd, method, path
