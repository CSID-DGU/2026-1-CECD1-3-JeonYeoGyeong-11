"""Variant token parsing for General FL suite runs."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Tuple


def parse_variant(
    variant: str, args: argparse.Namespace
) -> Tuple[str, str, List[str]]:
    """Return (flower_method, row_label, extra_cli_args)."""
    default_knn_k = int(args.knn_k)
    v = variant.strip().lower()
    if v == "fedavg":
        return "fedavg", "fedavg", []

    m = re.match(r"^ours_knn_k(\d+)$", v)
    if m:
        k = m.group(1)
        return "ours", v, ["--graph-mode", "knn", "--knn-k", k]

    if v == "ours_knn":
        return "ours", "ours_knn", ["--graph-mode", "knn", "--knn-k", str(default_knn_k)]

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

    if v == "ours_global_alignment":
        return "ours", "ours_global_alignment", ["--graph-mode", "global_alignment"]

    if v == "ours_weight_graph":
        return (
            "ours",
            "ours_weight_graph",
            ["--graph-source", "weight", "--graph-mode", "dense"],
        )

    m = re.match(r"^ours_weight_graph_knn_k(\d+)$", v)
    if m:
        k = m.group(1)
        return (
            "ours",
            v,
            ["--graph-source", "weight", "--graph-mode", "knn", "--knn-k", k],
        )

    if v == "ours_weight_agg":
        return "ours", "ours_weight_agg", ["--aggregation-target", "weight"]

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
        "--diagnostic-only",
        str(args.diagnostic_only),
        "--edge-threshold",
        str(args.edge_threshold),
        "--e-std-threshold",
        str(args.e_std_threshold),
        "--min-client-weight",
        str(args.min_client_weight),
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
    run_tag = f"{suite_tag}_{vlabel}_seed{seed}"
    cmd = (
        build_base_cmd(args)
        + ["--method", method, "--seed", str(seed), "--run-tag", run_tag]
        + extras
    )
    tag_suffix = f"_{run_tag}"
    path = out_dir / f"result_general_{method}_seed{seed}{tag_suffix}.json"
    return cmd, method, path
