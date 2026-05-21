"""Vision FL suite variant token parsing."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Tuple

from graphfl_lab.experiments.suites.vision.variant_commands import build_base_cmd
from graphfl_lab.experiments.suites.vision.variant_diagnostics import parse_diagnostic_variant
from graphfl_lab.experiments.suites.vision.variant_families import parse_baseline_variant
from graphfl_lab.experiments.suites.vision.variant_helpers import (
    legacy_residual_reweight_args as _legacy_residual_reweight_args,
    result_path_for_variant,
    token_float as _token_float,
)


def parse_variant(
    variant: str, args: argparse.Namespace
) -> Tuple[str, str, List[str]]:
    """Return (flower_method, row_label, extra_cli_args)."""
    default_knn_k = int(args.knn_k)
    v = variant.strip().lower()
    baseline = parse_baseline_variant(v, default_knn_k)
    if baseline is not None:
        return baseline

    if v == "ours_default_graph":
        return "ours", v, ["--graph-method", "default_similarity_knn"]

    m = re.match(r"^ours_default_graph_k(\d+)$", v)
    if m:
        k = m.group(1)
        return (
            "ours",
            v,
            [
                "--graph-method",
                "default_similarity_knn",
                "--knn-k",
                k,
            ],
        )

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

    m = re.match(r"^ours_knn_k(\d+)$", v)
    if m:
        k = m.group(1)
        return "ours", v, ["--graph-mode", "knn", "--knn-k", k]

    if v == "ours_knn":
        return "ours", "ours_knn", ["--graph-mode", "knn", "--knn-k", str(default_knn_k)]

    if v in {"ours_update_graph_update_agg", "ours_grad_graph_grad_agg"}:
        return (
            "ours",
            v,
            ["--graph-source", "update", "--aggregation-target", "update"],
        )

    m = re.match(r"^ours_(?:update|grad)_graph_(?:update|grad)_agg_knn_k(\d+)$", v)
    if m:
        k = m.group(1)
        return (
            "ours",
            v,
            [
                "--graph-source",
                "update",
                "--aggregation-target",
                "update",
                "--graph-mode",
                "knn",
                "--knn-k",
                k,
            ],
        )

    m = re.match(r"^ours_residual_reweight_knn_k(\d+)$", v)
    if m:
        k = m.group(1)
        return (
            "ours",
            v,
            _legacy_residual_reweight_args("knn", k),
        )

    m = re.match(r"^ours_residual_reweight_random_matched_k(\d+)$", v)
    if m:
        k = m.group(1)
        return (
            "ours",
            v,
            _legacy_residual_reweight_args("random", k),
        )

    if v == "ours_legacy_residual_reweight_dense":
        return (
            "ours",
            v,
            _legacy_residual_reweight_args("dense"),
        )

    if v == "ours_legacy_residual_reweight_uniform":
        return (
            "ours",
            v,
            _legacy_residual_reweight_args("uniform"),
        )

    m = re.match(r"^ours_legacy_residual_reweight_knn_k(\d+)$", v)
    if m:
        k = m.group(1)
        return (
            "ours",
            v,
            _legacy_residual_reweight_args("knn", k),
        )

    m = re.match(r"^ours_legacy_residual_reweight_random_matched_k(\d+)$", v)
    if m:
        k = m.group(1)
        return (
            "ours",
            v,
            _legacy_residual_reweight_args("random", k),
        )

    m = re.match(r"^ours_legacy_residual_reweight_magnitude_knn_k(\d+)$", v)
    if m:
        k = m.group(1)
        return (
            "ours",
            v,
            _legacy_residual_reweight_args("magnitude_knn", k),
        )

    m = re.match(r"^ours_mutual_knn_k(\d+)$", v)
    if m:
        k = m.group(1)
        return "ours", v, ["--graph-mode", "mutual_knn", "--knn-k", k]

    if v == "ours_mutual_knn":
        return (
            "ours",
            "ours_mutual_knn",
            ["--graph-mode", "mutual_knn", "--knn-k", str(default_knn_k)],
        )

    if v == "ours_dense":
        return "ours", "ours_dense", ["--graph-mode", "dense"]

    if v == "ours_magnitude":
        return "ours", "ours_magnitude", ["--graph-mode", "magnitude"]

    m = re.match(r"^ours_magnitude_knn_k(\d+)$", v)
    if m:
        k = m.group(1)
        return "ours", v, ["--graph-mode", "magnitude_knn", "--knn-k", k]

    if v == "ours_global_alignment":
        return "ours", "ours_global_alignment", ["--graph-mode", "global_alignment"]

    if v == "ours_rbf":
        return "ours", "ours_rbf", ["--graph-mode", "rbf"]

    m = re.match(r"^ours_rbf_knn_k(\d+)$", v)
    if m:
        k = m.group(1)
        return "ours", v, ["--graph-mode", "rbf_knn", "--knn-k", k]

    if v in {"ours_learned_graph", "ours_learned_smooth"}:
        return "ours", v, ["--graph-mode", "learned_smooth"]

    m = re.match(r"^ours_learned_smooth_knn_k(\d+)$", v)
    if m:
        k = m.group(1)
        return "ours", v, ["--graph-mode", "learned_smooth_knn", "--knn-k", k]

    if v == "ours_signed_abs":
        return "ours", "ours_signed_abs", ["--graph-mode", "signed_abs"]

    m = re.match(r"^ours_signed_abs_knn_k(\d+)$", v)
    if m:
        k = m.group(1)
        return "ours", v, ["--graph-mode", "signed_abs_knn", "--knn-k", k]

    if v == "ours_negative":
        return "ours", "ours_negative", ["--graph-mode", "negative"]

    m = re.match(r"^ours_negative_knn_k(\d+)$", v)
    if m:
        k = m.group(1)
        return "ours", v, ["--graph-mode", "negative_knn", "--knn-k", k]

    if v == "ours_weight_graph":
        return (
            "ours",
            "ours_weight_graph",
            ["--graph-source", "weight", "--graph-mode", "dense"],
        )

    if v == "ours_weight_graph_weight_agg":
        return (
            "ours",
            "ours_weight_graph_weight_agg",
            [
                "--graph-source",
                "weight",
                "--aggregation-target",
                "weight",
                "--graph-mode",
                "dense",
            ],
        )

    if v == "ours_weight_graph_filtered_weight_agg":
        return (
            "ours",
            "ours_weight_graph_filtered_weight_agg",
            [
                "--graph-source",
                "weight",
                "--aggregation-target",
                "graph_filtered_weight",
                "--graph-mode",
                "dense",
            ],
        )

    if v == "ours_weight_graph_spectral_weight_agg":
        return (
            "ours",
            "ours_weight_graph_spectral_weight_agg",
            [
                "--graph-source",
                "weight",
                "--aggregation-target",
                "spectral_filtered_weight",
                "--graph-mode",
                "dense",
            ],
        )

    m = re.match(r"^ours_weight_graph_knn_k(\d+)$", v)
    if m:
        k = m.group(1)
        return (
            "ours",
            v,
            ["--graph-source", "weight", "--graph-mode", "knn", "--knn-k", k],
        )

    m = re.match(r"^ours_weight_graph_weight_agg_knn_k(\d+)$", v)
    if m:
        k = m.group(1)
        return (
            "ours",
            v,
            [
                "--graph-source",
                "weight",
                "--aggregation-target",
                "weight",
                "--graph-mode",
                "knn",
                "--knn-k",
                k,
            ],
        )

    m = re.match(r"^ours_weight_graph_filtered_weight_agg_knn_k(\d+)$", v)
    if m:
        k = m.group(1)
        return (
            "ours",
            v,
            [
                "--graph-source",
                "weight",
                "--aggregation-target",
                "graph_filtered_weight",
                "--graph-mode",
                "knn",
                "--knn-k",
                k,
            ],
        )

    m = re.match(r"^ours_weight_graph_spectral_weight_agg_knn_k(\d+)$", v)
    if m:
        k = m.group(1)
        return (
            "ours",
            v,
            [
                "--graph-source",
                "weight",
                "--aggregation-target",
                "spectral_filtered_weight",
                "--graph-mode",
                "knn",
                "--knn-k",
                k,
            ],
        )

    if v == "ours_layerwise_graph":
        return (
            "ours",
            "ours_layerwise_graph",
            ["--graph-source", "layerwise_update", "--graph-mode", "dense"],
        )

    m = re.match(r"^ours_layerwise_knn_k(\d+)$", v)
    if m:
        k = m.group(1)
        return (
            "ours",
            v,
            ["--graph-source", "layerwise_update", "--graph-mode", "knn", "--knn-k", k],
        )

    m = re.match(r"^ours_layerwise_weight_knn_k(\d+)$", v)
    if m:
        k = m.group(1)
        return (
            "ours",
            v,
            ["--graph-source", "layerwise_weight", "--graph-mode", "knn", "--knn-k", k],
        )

    m = re.match(r"^ours_head_graph_knn_k(\d+)$", v)
    if m:
        k = m.group(1)
        return (
            "ours",
            v,
            [
                "--graph-source",
                "classifier_head_update",
                "--graph-mode",
                "knn",
                "--knn-k",
                k,
            ],
        )

    m = re.match(r"^ours_head_ema_graph_knn_k(\d+)$", v)
    if m:
        k = m.group(1)
        return (
            "ours",
            v,
            [
                "--graph-source",
                "classifier_head_ema_update",
                "--graph-mode",
                "knn",
                "--knn-k",
                k,
            ],
        )

    m = re.match(r"^ours_head_weight_graph_knn_k(\d+)$", v)
    if m:
        k = m.group(1)
        return (
            "ours",
            v,
            [
                "--graph-source",
                "classifier_head_weight",
                "--graph-mode",
                "knn",
                "--knn-k",
                k,
            ],
        )

    m = re.match(r"^ours_head_weight_graph_filtered_weight_agg_knn_k(\d+)$", v)
    if m:
        k = m.group(1)
        return (
            "ours",
            v,
            [
                "--graph-source",
                "classifier_head_weight",
                "--aggregation-target",
                "graph_filtered_weight",
                "--graph-mode",
                "knn",
                "--knn-k",
                k,
            ],
        )

    m = re.match(r"^ours_head_weight_graph_spectral_weight_agg_knn_k(\d+)$", v)
    if m:
        k = m.group(1)
        return (
            "ours",
            v,
            [
                "--graph-source",
                "classifier_head_weight",
                "--aggregation-target",
                "spectral_filtered_weight",
                "--graph-mode",
                "knn",
                "--knn-k",
                k,
            ],
        )

    m = re.match(r"^ours_layerwise_head_graph_knn_k(\d+)$", v)
    if m:
        k = m.group(1)
        return (
            "ours",
            v,
            [
                "--graph-source",
                "layerwise_classifier_head_update",
                "--graph-mode",
                "knn",
                "--knn-k",
                k,
            ],
        )

    m = re.match(r"^ours_layerwise_head_ema_graph_knn_k(\d+)$", v)
    if m:
        k = m.group(1)
        return (
            "ours",
            v,
            [
                "--graph-source",
                "layerwise_classifier_head_ema_update",
                "--graph-mode",
                "knn",
                "--knn-k",
                k,
            ],
        )

    m = re.match(r"^ours_layerwise_head_weight_graph_knn_k(\d+)$", v)
    if m:
        k = m.group(1)
        return (
            "ours",
            v,
            [
                "--graph-source",
                "layerwise_classifier_head_weight",
                "--graph-mode",
                "knn",
                "--knn-k",
                k,
            ],
        )

    m = re.match(r"^ours_ema_graph_knn_k(\d+)$", v)
    if m:
        k = m.group(1)
        return (
            "ours",
            v,
            [
                "--graph-source",
                "ema_update",
                "--aggregation-target",
                "graph_filtered_update",
                "--graph-mode",
                "knn",
                "--knn-k",
                k,
            ],
        )

    m = re.match(r"^ours_ema_signal_knn_k(\d+)$", v)
    if m:
        k = m.group(1)
        return (
            "ours",
            v,
            [
                "--graph-source",
                "ema_update",
                "--aggregation-target",
                "graph_filtered_ema_update",
                "--graph-mode",
                "knn",
                "--knn-k",
                k,
            ],
        )

    m = re.match(r"^ours_tail_m(\d+)_knn_k(\d+)$", v)
    if m:
        start = f"-{m.group(1)}"
        k = m.group(2)
        return (
            "ours",
            v,
            [
                "--graph-source",
                "layer_slice_update",
                "--graph-layer-start",
                start,
                "--graph-mode",
                "knn",
                "--knn-k",
                k,
            ],
        )

    m = re.match(r"^ours_layerwise_tail_m(\d+)_knn_k(\d+)$", v)
    if m:
        start = f"-{m.group(1)}"
        k = m.group(2)
        return (
            "ours",
            v,
            [
                "--graph-source",
                "layerwise_slice_update",
                "--graph-layer-start",
                start,
                "--graph-mode",
                "knn",
                "--knn-k",
                k,
            ],
        )

    m = re.match(r"^ours_weight_tail_m(\d+)_knn_k(\d+)$", v)
    if m:
        start = f"-{m.group(1)}"
        k = m.group(2)
        return (
            "ours",
            v,
            [
                "--graph-source",
                "layer_slice_weight",
                "--graph-layer-start",
                start,
                "--graph-mode",
                "knn",
                "--knn-k",
                k,
            ],
        )

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

    if v == "ours_uniform":
        return "ours", "ours_uniform", ["--graph-mode", "uniform"]

    m = re.match(r"^ours_random_matched_k(\d+)$", v)
    if m:
        k = m.group(1)
        return "ours", v, ["--graph-mode", "random", "--knn-k", k]

    if v in ("ours_random", "ours_random_matched"):
        return "ours", v, ["--graph-mode", "random", "--knn-k", str(default_knn_k)]

    if v == "ours_threshold":
        return "ours", "ours_threshold", ["--graph-mode", "threshold"]

    if v == "ours_no_ema":
        return "ours", "ours_no_ema", ["--graph-mode", "dense", "--use-ema-graph", "false"]

    if v == "ours_fixed_tau":
        return ("ours", "ours_fixed_tau", ["--graph-mode", "dense", "--disable-adaptive-tau", "true"])

    custom_prefix = "ours_graph_mode_"
    if v.startswith(custom_prefix) and len(v) > len(custom_prefix):
        mode = v[len(custom_prefix):]
        return "ours", v, ["--graph-mode", mode]

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
