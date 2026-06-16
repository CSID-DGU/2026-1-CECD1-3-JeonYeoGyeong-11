"""Command assembly helpers for vision suite variants."""

from __future__ import annotations

import argparse
import json
from typing import List


def build_base_cmd(args: argparse.Namespace) -> List[str]:
    train_t = int(args.train_subset_size)
    test_t = int(args.test_subset_size)
    user_arg_dests = set(getattr(args, "_user_arg_dests", ()))
    common = [
        args.python_bin,
        "run_vision_experiment.py",
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
        "--graph-seed",
        str(args.graph_seed),
        "--graph-plugin",
        str(getattr(args, "graph_plugin", "")),
        "--graph-preset",
        str(getattr(args, "graph_preset", "none")),
        "--aggregation-params",
        json.dumps(
            dict(getattr(args, "aggregation_params", {}) or {}),
            sort_keys=True,
        ),
        "--use-ema-graph",
        str(args.use_ema_graph),
        "--disable-adaptive-tau",
        str(args.disable_adaptive_tau),
        "--fixed-tau",
        str(args.fixed_tau),
        "--tau-source",
        str(args.tau_source),
        "--graph-filter-strength",
        str(args.graph_filter_strength),
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
    graph_method = str(getattr(args, "graph_method", "none"))
    if graph_method.strip().lower().replace("-", "_") not in {"", "none", "off", "disabled"}:
        common += ["--graph-method", graph_method]

    def append_user_arg(dest: str, flag: str, value: object) -> None:
        if dest in user_arg_dests:
            common.extend([flag, str(value)])

    append_user_arg("graph_source", "--graph-source", args.graph_source)
    append_user_arg("aggregation_target", "--aggregation-target", args.aggregation_target)
    append_user_arg("edge_threshold", "--edge-threshold", args.edge_threshold)
    append_user_arg("graph_scale_sigma", "--graph-scale-sigma", args.graph_scale_sigma)
    append_user_arg("learned_graph_lambda", "--learned-graph-lambda", args.learned_graph_lambda)
    append_user_arg("graph_layer_start", "--graph-layer-start", args.graph_layer_start)
    append_user_arg("graph_layer_end", "--graph-layer-end", args.graph_layer_end)

    if train_t > 0:
        common += ["--train-subset-size", str(train_t)]
    if test_t > 0:
        common += ["--test-subset-size", str(test_t)]
    return common
