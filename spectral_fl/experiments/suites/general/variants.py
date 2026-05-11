"""General FL suite variant token parsing."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Tuple


def _token_float(value: str) -> str:
    """Convert compact variant floats such as ``0p01`` to CLI text ``0.01``."""
    return str(value).replace("p", ".")


def _legacy_residual_reweight_args(graph_mode: str, knn_k: str = "") -> List[str]:
    """Return CLI args for the pre-low-pass residual reweighting path.

    This preserves the earlier behavior where the graph spectral filter was
    used to score projected-update residuals, then raw update deltas were
    aggregated with shaped client weights.
    """
    out = [
        "--graph-source",
        "update",
        "--aggregation-target",
        "update",
        "--graph-mode",
        graph_mode,
        "--conflict-mix",
        "0.2",
        "--min-client-weight",
        "0.0",
    ]
    if knn_k:
        out += ["--knn-k", knn_k]
    return out


def parse_variant(
    variant: str, args: argparse.Namespace
) -> Tuple[str, str, List[str]]:
    """Return (flower_method, row_label, extra_cli_args)."""
    default_knn_k = int(args.knn_k)
    v = variant.strip().lower()
    if v == "fedavg":
        return "fedavg", "fedavg", []
    if v == "fedavgm":
        return "fedavgm", "fedavgm", []
    if v == "fedadagrad":
        return "fedadagrad", "fedadagrad", []
    if v == "fedadam":
        return "fedadam", "fedadam", []
    if v == "fedyogi":
        return "fedyogi", "fedyogi", []
    if v == "fednova":
        return "fednova", "fednova", []
    m = re.match(r"^(fedadagrad|fedadam|fedyogi)_eta([0-9][0-9p.]*)$", v)
    if m:
        return m.group(1), v, ["--fedopt-eta", _token_float(m.group(2))]
    m = re.match(r"^(fedadagrad|fedadam|fedyogi)_etal([0-9][0-9p.]*)$", v)
    if m:
        return m.group(1), v, ["--fedopt-eta-l", _token_float(m.group(2))]
    m = re.match(
        r"^(fedadam|fedyogi)_eta([0-9][0-9p.]*)_etal([0-9][0-9p.]*)$",
        v,
    )
    if m:
        return (
            m.group(1),
            v,
            [
                "--fedopt-eta",
                _token_float(m.group(2)),
                "--fedopt-eta-l",
                _token_float(m.group(3)),
            ],
        )
    m = re.match(r"^fednova_slr([0-9][0-9p.]*)$", v)
    if m:
        return "fednova", v, ["--server-learning-rate", _token_float(m.group(1))]
    if v == "fedprox":
        return "fedprox", "fedprox", []
    m = re.match(r"^fedprox_mu([0-9][0-9p.]*)$", v)
    if m:
        return "fedprox", v, ["--fedprox-mu", _token_float(m.group(1))]
    if v == "fedmedian":
        return "fedmedian", "fedmedian", []
    if v == "fedtrimmedavg":
        return "fedtrimmedavg", "fedtrimmedavg", []
    m = re.match(r"^fedtrimmedavg_beta([0-9][0-9p.]*)$", v)
    if m:
        return "fedtrimmedavg", v, ["--trimmed-beta", _token_float(m.group(1))]
    if v == "fedsim":
        return (
            "fedsim",
            "fedsim",
            ["--graph-mode", "knn", "--knn-k", str(default_knn_k)],
        )
    m = re.match(r"^fedsim_k(\d+)$", v)
    if m:
        k = m.group(1)
        return "fedsim", v, ["--graph-mode", "knn", "--knn-k", k]
    m = re.match(r"^fedsim_magnitude_knn_k(\d+)$", v)
    if m:
        k = m.group(1)
        return "fedsim", v, ["--graph-mode", "magnitude_knn", "--knn-k", k]
    m = re.match(r"^fedsim_rbf_knn_k(\d+)$", v)
    if m:
        k = m.group(1)
        return "fedsim", v, ["--graph-mode", "rbf_knn", "--knn-k", k]

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

    for suffix in ("spectral_only", "speconly"):
        marker = f"_{suffix}"
        if v.endswith(marker):
            base = v[: -len(marker)]
            method, _, extras = parse_variant(base, args)
            if method != "ours":
                raise ValueError(
                    f"Spectral-only suffix is only supported for Ours variants: {variant!r}"
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
            extras + ["--spectral-filter-strength", _token_float(m.group("strength"))],
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
                "spectral_filtered_update",
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
                "spectral_filtered_ema_update",
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

    raise ValueError(f"Unknown general-FL variant: {variant!r}")


def build_base_cmd(args: argparse.Namespace) -> List[str]:
    train_t = int(args.train_subset_size)
    test_t = int(args.test_subset_size)
    common = [
        args.python_bin,
        "run_general_experiment.py",
        "--dataset",
        str(args.dataset),
        "--num-clients",
        str(args.num_clients),
        "--rounds",
        str(args.rounds),
        "--local-epochs",
        str(args.local_epochs),
        "--batch-size",
        str(args.batch_size),
        "--model",
        str(args.model),
        "--lr",
        str(args.lr),
        "--momentum",
        str(args.momentum),
        "--weight-decay",
        str(args.weight_decay),
        "--partition",
        str(args.partition),
        "--dirichlet-alpha",
        str(args.dirichlet_alpha),
        "--data-root",
        str(args.data_root),
        "--projection-dim",
        str(args.projection_dim),
        "--compression-seed",
        str(args.compression_seed),
        "--ema-alpha",
        str(args.ema_alpha),
        "--tau-gain",
        str(args.tau_gain),
        "--tau-max",
        str(args.tau_max),
        "--conflict-mix",
        str(args.conflict_mix),
        "--warmup-rounds",
        str(args.warmup_rounds),
        "--graph-source",
        str(args.graph_source),
        "--aggregation-target",
        str(args.aggregation_target),
        "--graph-seed",
        str(args.graph_seed),
        "--use-ema-graph",
        str(args.use_ema_graph),
        "--disable-adaptive-tau",
        str(args.disable_adaptive_tau),
        "--fixed-tau",
        str(args.fixed_tau),
        "--tau-source",
        str(args.tau_source),
        "--spectral-filter-strength",
        str(args.spectral_filter_strength),
        "--client-update-ema-alpha",
        str(args.client_update_ema_alpha),
        "--diagnostic-only",
        str(args.diagnostic_only),
        "--correction-family",
        str(args.correction_family),
        "--control-graph-mode",
        str(args.control_graph_mode),
        "--cluster-method",
        str(args.cluster_method),
        "--cluster-k",
        str(args.cluster_k),
        "--cluster-auto-k",
        str(args.cluster_auto_k),
        "--graph-free-mode",
        str(args.graph_free_mode),
        "--graph-free-gamma",
        str(args.graph_free_gamma),
        "--clip-quantile",
        str(args.clip_quantile),
        "--contribution-cap",
        str(args.contribution_cap),
        "--diagnostics-enable",
        str(args.diagnostics_enable),
        "--save-round-graphs",
        str(args.save_round_graphs),
        "--graph-snapshot-rounds",
        str(args.graph_snapshot_rounds),
        "--save-update-arrays",
        str(args.save_update_arrays),
        "--loo-enabled",
        str(args.loo_enabled),
        "--edge-threshold",
        str(args.edge_threshold),
        "--graph-scale-sigma",
        str(args.graph_scale_sigma),
        "--learned-graph-lambda",
        str(args.learned_graph_lambda),
        "--graph-layer-start",
        str(args.graph_layer_start),
        "--graph-layer-end",
        str(args.graph_layer_end),
        "--e-std-threshold",
        str(args.e_std_threshold),
        "--min-client-weight",
        str(args.min_client_weight),
        "--server-learning-rate",
        str(args.server_learning_rate),
        "--server-momentum",
        str(args.server_momentum),
        "--ours-server-learning-rate",
        str(args.ours_server_learning_rate),
        "--ours-server-momentum",
        str(args.ours_server_momentum),
        "--fedprox-mu",
        str(args.fedprox_mu),
        "--fedopt-eta",
        str(args.fedopt_eta),
        "--fedopt-eta-l",
        str(args.fedopt_eta_l),
        "--fedopt-beta1",
        str(args.fedopt_beta1),
        "--fedopt-beta2",
        str(args.fedopt_beta2),
        "--fedopt-tau",
        str(args.fedopt_tau),
        "--trimmed-beta",
        str(args.trimmed_beta),
        "--out-dir",
        str(args.out_dir),
    ]
    if train_t > 0:
        common += ["--train-subset-size", str(train_t)]
    if test_t > 0:
        common += ["--test-subset-size", str(test_t)]
    return common


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
    tag_suffix = f"_{run_tag}"
    path = out_dir / f"result_general_{method}_seed{seed}{tag_suffix}.json"
    return cmd, method, path
