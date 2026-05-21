"""Vision FL suite variant token parsing."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Tuple

from graphfl_lab.experiments.suites.vision.variant_commands import build_base_cmd
from graphfl_lab.experiments.suites.vision.variant_core import parse_core_graph_variant
from graphfl_lab.experiments.suites.vision.variant_diagnostics import parse_diagnostic_variant
from graphfl_lab.experiments.suites.vision.variant_families import parse_baseline_variant
from graphfl_lab.experiments.suites.vision.variant_helpers import (
    result_path_for_variant,
    token_float as _token_float,
)
from graphfl_lab.experiments.suites.vision.variant_legacy import (
    parse_legacy_residual_variant,
)
from graphfl_lab.experiments.suites.vision.variant_sources import parse_source_variant


def parse_variant(
    variant: str, args: argparse.Namespace
) -> Tuple[str, str, List[str]]:
    """Return (flower_method, row_label, extra_cli_args)."""
    default_knn_k = int(args.knn_k)
    v = variant.strip().lower()
    baseline = parse_baseline_variant(v, default_knn_k)
    if baseline is not None:
        return baseline

    tau_suffix_args = {
        "fixed_tau": ["--disable-adaptive-tau", "true"],
        "norm_tau": ["--tau-source", "h_spec_normalized"],
        "normalized_tau": ["--tau-source", "h_spec_normalized"],
        "estd_tau": ["--tau-source", "e_std"],
        "e_std_tau": ["--tau-source", "e_std"],
        "norm_estd_tau": ["--tau-source", "h_spec_normalized_times_e_std"],
        "norm_e_std_tau": ["--tau-source", "h_spec_normalized_times_e_std"],
    }
    if v != "ours_fixed_tau":
        for suffix, tau_args in sorted(
            tau_suffix_args.items(), key=lambda item: -len(item[0])
        ):
            marker = f"_{suffix}"
            if v.endswith(marker):
                base = v[: -len(marker)]
                try:
                    method, _, extras = parse_variant(base, args)
                except ValueError:
                    continue
                if method != "ours":
                    continue
                return "ours", v, extras + tau_args

    for suffix in ("graph_filter_only", "filter_only", "spectral_only", "speconly"):
        marker = f"_{suffix}"
        if v.endswith(marker):
            base = v[: -len(marker)]
            method, _, extras = parse_variant(base, args)
            if method != "ours":
                raise ValueError(
                    f"Graph-filter-only suffix is only supported for Ours variants: {variant!r}"
                )
            return (
                "ours",
                v,
                extras
                + [
                    "--conflict-mix",
                    "0.0",
                    "--min-client-weight",
                    "0.0",
                    "--ours-server-learning-rate",
                    "1.0",
                    "--ours-server-momentum",
                    "0.0",
                ],
            )

    m = re.match(r"^(?P<base>.+)_lp(?P<strength>[0-9][0-9p.]*)$", v)
    if m:
        method, _, extras = parse_variant(m.group("base"), args)
        if method != "ours":
            raise ValueError(f"Low-pass suffix is only supported for Ours variants: {variant!r}")
        return (
            "ours",
            v,
            extras + ["--graph-filter-strength", _token_float(m.group("strength"))],
        )

    if v.endswith("_serverm"):
        base = v[: -len("_serverm")]
        method, _, extras = parse_variant(base, args)
        if method != "ours":
            raise ValueError(
                f"Server momentum suffix is only supported for Ours variants: {variant!r}"
            )
        return (
            "ours",
            v,
            extras
            + [
                "--ours-server-learning-rate",
                str(args.ours_server_learning_rate),
                "--ours-server-momentum",
                str(args.server_momentum),
            ],
        )

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

    if v == "ours_weight_agg":
        return "ours", "ours_weight_agg", ["--aggregation-target", "weight"]

    if v == "ours_graph_filtered_dense":
        return (
            "ours",
            "ours_graph_filtered_dense",
            ["--aggregation-target", "graph_filtered_update", "--graph-mode", "dense"],
        )

    if v == "ours_graph_filtered_uniform":
        return (
            "ours",
            "ours_graph_filtered_uniform",
            ["--aggregation-target", "graph_filtered_update", "--graph-mode", "uniform"],
        )

    m = re.match(r"^ours_graph_filtered_knn_k(\d+)$", v)
    if m:
        k = m.group(1)
        return (
            "ours",
            v,
            [
                "--aggregation-target",
                "graph_filtered_update",
                "--graph-mode",
                "knn",
                "--knn-k",
                k,
            ],
        )

    if v == "ours_graph_filtered_magnitude":
        return (
            "ours",
            "ours_graph_filtered_magnitude",
            [
                "--aggregation-target",
                "graph_filtered_update",
                "--graph-mode",
                "magnitude",
            ],
        )

    m = re.match(r"^ours_graph_filtered_magnitude_knn_k(\d+)$", v)
    if m:
        k = m.group(1)
        return (
            "ours",
            v,
            [
                "--aggregation-target",
                "graph_filtered_update",
                "--graph-mode",
                "magnitude_knn",
                "--knn-k",
                k,
            ],
        )

    if v == "ours_graph_filtered_rbf":
        return (
            "ours",
            "ours_graph_filtered_rbf",
            ["--aggregation-target", "graph_filtered_update", "--graph-mode", "rbf"],
        )

    m = re.match(r"^ours_graph_filtered_rbf_knn_k(\d+)$", v)
    if m:
        k = m.group(1)
        return (
            "ours",
            v,
            [
                "--aggregation-target",
                "graph_filtered_update",
                "--graph-mode",
                "rbf_knn",
                "--knn-k",
                k,
            ],
        )

    m = re.match(r"^ours_graph_filtered_random_matched_k(\d+)$", v)
    if m:
        k = m.group(1)
        return (
            "ours",
            v,
            [
                "--aggregation-target",
                "graph_filtered_update",
                "--graph-mode",
                "random",
                "--knn-k",
                k,
            ],
        )

    if v == "ours_spectral_filtered_dense":
        return (
            "ours",
            "ours_spectral_filtered_dense",
            ["--aggregation-target", "spectral_filtered_update", "--graph-mode", "dense"],
        )

    if v == "ours_spectral_filtered_uniform":
        return (
            "ours",
            "ours_spectral_filtered_uniform",
            ["--aggregation-target", "spectral_filtered_update", "--graph-mode", "uniform"],
        )

    m = re.match(r"^ours_spectral_filtered_knn_k(\d+)$", v)
    if m:
        k = m.group(1)
        return (
            "ours",
            v,
            [
                "--aggregation-target",
                "spectral_filtered_update",
                "--graph-mode",
                "knn",
                "--knn-k",
                k,
            ],
        )

    if v == "ours_spectral_filtered_magnitude":
        return (
            "ours",
            "ours_spectral_filtered_magnitude",
            [
                "--aggregation-target",
                "spectral_filtered_update",
                "--graph-mode",
                "magnitude",
            ],
        )

    m = re.match(r"^ours_spectral_filtered_magnitude_knn_k(\d+)$", v)
    if m:
        k = m.group(1)
        return (
            "ours",
            v,
            [
                "--aggregation-target",
                "spectral_filtered_update",
                "--graph-mode",
                "magnitude_knn",
                "--knn-k",
                k,
            ],
        )

    if v == "ours_spectral_filtered_rbf":
        return (
            "ours",
            "ours_spectral_filtered_rbf",
            ["--aggregation-target", "spectral_filtered_update", "--graph-mode", "rbf"],
        )

    m = re.match(r"^ours_spectral_filtered_rbf_knn_k(\d+)$", v)
    if m:
        k = m.group(1)
        return (
            "ours",
            v,
            [
                "--aggregation-target",
                "spectral_filtered_update",
                "--graph-mode",
                "rbf_knn",
                "--knn-k",
                k,
            ],
        )

    m = re.match(r"^ours_spectral_filtered_random_matched_k(\d+)$", v)
    if m:
        k = m.group(1)
        return (
            "ours",
            v,
            [
                "--aggregation-target",
                "spectral_filtered_update",
                "--graph-mode",
                "random",
                "--knn-k",
                k,
            ],
        )

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
