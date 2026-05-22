"""Metadata helpers for vision suite result payloads."""

from __future__ import annotations

import math
import time
from argparse import Namespace
from datetime import datetime
from typing import Any, Dict, List

from graphfl_lab.config_io import public_args_dict


def build_suite_meta(
    args: Namespace,
    suite_tag: str,
    suite_started_at: datetime,
) -> Dict[str, Any]:
    return {
        "timestamp": suite_started_at.isoformat(),
        "track": "vision-fl",
        "suite_tag": suite_tag,
        "config": public_args_dict(args),
        "delta_baseline": "fedavg",
        "cross_track_variant_names": (
            "Tokens like ours_knn_k3 embed k in the name; FGL graph ablation uses ours_knn plus --knn-k. "
            "Tokens differ across tracks on purpose?match experiments by knn-k / graph-mode, not raw variant string equality."
        ),
        "matched_random_ablation": (
            "ours_random_matched_kK uses graph-mode random with the same --knn-k as FGL ours_random: matched edge count vs kNN "
            "(controls sparsity only). Compare variant ours_knn_kK to ours_random_matched_kK at equal K for graph-vs-random. "
            "If both variants share a suffix such as _fixed_tau, the generated kNN-vs-random CSV matches that suffix too. "
            "p-values / CI are not produced by this script?analyze seed<S>_delta or exports downstream."
        ),
        "delta_semantics": (
            "Per seed, delta = final distributed test accuracy(method) minus same-seed FedAvg. "
            "seed<S>_delta columns hold that gap for each seed S in --seeds. "
            "For non-fedavg rows, mean_delta is the unweighted mean of those gaps, min_delta is min(seed*_delta), "
            "max_delta is max(seed*_delta), std_delta is pstdev of the gaps, and "
            "win_rate is (# seeds with delta>0) / (number of seeds)."
        ),
        "trace_aggregate_semantics": (
            "mean_H_spec: mean over rounds of h_spec in round_trace "
            "(graph-update alignment diagnostic; not an absolute non-IID score). "
            "mean_H_spec_current: same diagnostic on the current-round graph. "
            "mean_low/high_frequency_energy_ratio: update energy split over the current graph spectrum. "
            "mean_spectral_filter_*: how much the configured low-pass filter keeps/removes from the graph signal; "
            "mean_update_spectral_filter_* applies to the full update when graph_filtered_update is used. "
            "mean_e_std: mean over rounds of client conflict-score spread (e_std / std_e in round_trace). "
            "mean_tau: mean over rounds of tau in round_trace (conflict-weight temperature / schedule). "
            "mean_graph_density: mean over rounds of graph_density in round_trace (similarity-graph edge density). "
            "mean_entropy_alpha: mean over rounds of entropy of normalized aggregation weights (entropy_alpha). "
            "mean_min_alpha / mean_max_alpha: mean over rounds of smallest/largest client aggregation mass share. "
            "mean_di_pre/post, mean_neff_pre/post, mean_alignment_pre/post, and mean_loo_pre/post "
            "track pre/post correction diagnostic changes in the same suite summary."
        ),
        "ranking_semantics": (
            "Summary sorted descending by rank_key (reverse=True): non-fedavg before fedavg; "
            "among methods ordered by mean_delta, then min_delta, then -std_delta (favors lower gap variance), "
            "then win_rate."
        ),
    }


def record_preloaded_fedavg_meta(
    suite_meta: Dict[str, Any],
    preload_dir: str,
    fedavg_acc_by_seed: Dict[int, float],
) -> None:
    if fedavg_acc_by_seed:
        suite_meta["preloaded_fedavg_dir"] = preload_dir
        suite_meta["preloaded_fedavg_seeds"] = sorted(fedavg_acc_by_seed.keys())


def record_training_data_note(suite_meta: Dict[str, Any], args: Namespace) -> None:
    train_size = int(args.train_subset_size)
    test_size = int(args.test_subset_size)
    suite_meta["training_data_note"] = (
        "full_dataset_splits"
        if train_size <= 0 and test_size <= 0
        else f"subset_train_{train_size}_test_{test_size}"
    )


def record_suite_timing(
    suite_meta: Dict[str, Any],
    suite_started_at: datetime,
    suite_start: float,
    rows: List[Dict[str, Any]],
) -> None:
    recorded_run_times = []
    for row in rows:
        try:
            value = float(row.get("run_wall_time_sec", float("nan")))
        except (TypeError, ValueError):
            continue
        if not math.isnan(value):
            recorded_run_times.append(value)
    suite_meta["timing"] = {
        "started_at": suite_started_at.isoformat(),
        "completed_at": datetime.now().isoformat(),
        "suite_wall_time_sec": float(time.perf_counter() - suite_start),
        "sum_recorded_run_wall_time_sec": (
            float(sum(recorded_run_times)) if recorded_run_times else float("nan")
        ),
    }
