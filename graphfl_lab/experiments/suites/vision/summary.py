"""Summary-row aggregation for vision suites."""

from __future__ import annotations

import math
import statistics
from argparse import Namespace
from typing import Any, Dict, List

from graphfl_lab.experiments.suites.stats import (
    safe_max,
    safe_mean,
    safe_min,
    safe_pstdev,
)
from graphfl_lab.experiments.suites.vision.features import rank_key


def build_summary_rows(rows: List[Dict[str, Any]], args: Namespace) -> List[Dict[str, Any]]:
    by_variant: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_variant.setdefault(row["variant"], []).append(row)

    summary_rows: List[Dict[str, Any]] = []
    seed_cols = sorted(set(args.seeds))

    for variant, group in by_variant.items():
        deltas = [x["delta"] for x in group if x["variant"] != "fedavg"]
        ours_acc_list = [x["ours_acc"] for x in group if x["variant"] != "fedavg"]
        fa_acc = [x["fedavg_acc"] for x in group]
        run_times = [x.get("run_wall_time_sec") for x in group]
        method_times = [x.get("result_method_wall_time_sec") for x in group]
        seconds_per_round = [x.get("seconds_per_round") for x in group]

        def gmean(key: str):
            return safe_mean([x.get(key) for x in group])

        def gfirst(key: str) -> str:
            for x in group:
                value = x.get(key)
                if value not in (None, ""):
                    return str(value)
            return ""

        row_base = {
            "dataset": str(args.dataset),
            "partition": str(args.partition),
            "dirichlet_alpha": float(args.dirichlet_alpha),
            "num_clients": int(args.num_clients),
            "variant": variant,
            "n_runs": len(group),
            "mean_run_wall_time_sec": safe_mean(run_times),
            "std_run_wall_time_sec": safe_pstdev(run_times),
            "min_run_wall_time_sec": safe_min(run_times),
            "max_run_wall_time_sec": safe_max(run_times),
            "mean_method_wall_time_sec": safe_mean(method_times),
            "mean_seconds_per_round": safe_mean(seconds_per_round),
            "reused_existing_result_count": sum(
                1 for x in group if bool(x.get("reused_existing_result", False))
            ),
            "timing_source": gfirst("timing_source"),
            "graph_mode": gfirst("graph_mode"),
            "graph_source": gfirst("graph_source"),
            "graph_source_used": gfirst("graph_source_used"),
            "aggregation_target": gfirst("aggregation_target"),
            "aggregation_target_used": gfirst("aggregation_target_used"),
            "server_optimizer": gfirst("server_optimizer"),
            "tau_source_used": gfirst("tau_source_used"),
            "graph_filter_strength": gfirst("graph_filter_strength"),
            "spectral_filter_strength": gfirst("spectral_filter_strength"),
            "client_update_ema_alpha": gfirst("client_update_ema_alpha"),
            "client_update_ema_source": gfirst("client_update_ema_source"),
            "server_learning_rate": gfirst("server_learning_rate"),
            "server_momentum": gfirst("server_momentum"),
            "graph_scale_sigma": gfirst("graph_scale_sigma"),
            "learned_graph_lambda": gfirst("learned_graph_lambda"),
            "graph_layer_start": gfirst("graph_layer_start"),
            "graph_layer_end": gfirst("graph_layer_end"),
            "mean_fedavg_acc": safe_mean(fa_acc),
            "mean_acc": safe_mean(ours_acc_list) if ours_acc_list else safe_mean(fa_acc),
            "std_acc": safe_pstdev(ours_acc_list) if len(ours_acc_list) > 1 else 0.0,
            "mean_delta": safe_mean(deltas) if deltas else 0.0,
            "min_delta": safe_min(deltas) if deltas else 0.0,
            "max_delta": safe_max(deltas) if deltas else 0.0,
            "std_delta": safe_pstdev(deltas) if deltas else 0.0,
            "win_rate": (
                (sum(1 for d in deltas if d > 0) / len(deltas)) if deltas else 0.0
            ),
            "mean_H_spec": gmean("mean_h_spec"),
            "mean_H_spec_normalized": gmean("mean_h_spec_normalized"),
            "mean_H_spec_current": gmean("mean_h_spec_current"),
            "mean_H_spec_raw_current_graph": gmean("mean_h_spec_raw_current_graph"),
            "mean_low_frequency_energy_ratio": gmean("mean_low_frequency_energy_ratio"),
            "mean_mid_frequency_energy_ratio": gmean("mean_mid_frequency_energy_ratio"),
            "mean_high_frequency_energy_ratio": gmean("mean_high_frequency_energy_ratio"),
            "mean_high_to_low_energy_ratio": gmean("mean_high_to_low_energy_ratio"),
            "mean_dominant_frequency_energy_ratio": gmean(
                "mean_dominant_frequency_energy_ratio"
            ),
            "mean_spectral_entropy": gmean("mean_spectral_entropy"),
            "mean_eigengap_max": gmean("mean_eigengap_max"),
            "mean_spectral_filter_gain": gmean("mean_spectral_filter_gain"),
            "mean_spectral_filter_output_energy_ratio": gmean(
                "mean_spectral_filter_output_energy_ratio"
            ),
            "mean_spectral_filter_residual_energy_ratio": gmean(
                "mean_spectral_filter_residual_energy_ratio"
            ),
            "mean_spectral_filter_suppressed_energy_ratio": gmean(
                "mean_spectral_filter_suppressed_energy_ratio"
            ),
            "mean_server_candidate_delta_norm": gmean(
                "mean_server_candidate_delta_norm"
            ),
            "mean_server_applied_delta_norm": gmean(
                "mean_server_applied_delta_norm"
            ),
            "mean_update_spectral_filter_output_energy_ratio": gmean(
                "mean_update_spectral_filter_output_energy_ratio"
            ),
            "mean_update_spectral_filter_residual_energy_ratio": gmean(
                "mean_update_spectral_filter_residual_energy_ratio"
            ),
            "mean_update_spectral_filter_suppressed_energy_ratio": gmean(
                "mean_update_spectral_filter_suppressed_energy_ratio"
            ),
            "mean_ema_update_spectral_filter_output_energy_ratio": gmean(
                "mean_ema_update_spectral_filter_output_energy_ratio"
            ),
            "mean_ema_update_spectral_filter_residual_energy_ratio": gmean(
                "mean_ema_update_spectral_filter_residual_energy_ratio"
            ),
            "mean_ema_update_spectral_filter_suppressed_energy_ratio": gmean(
                "mean_ema_update_spectral_filter_suppressed_energy_ratio"
            ),
            "mean_weight_spectral_filter_output_energy_ratio": gmean(
                "mean_weight_spectral_filter_output_energy_ratio"
            ),
            "mean_weight_spectral_filter_residual_energy_ratio": gmean(
                "mean_weight_spectral_filter_residual_energy_ratio"
            ),
            "mean_weight_spectral_filter_suppressed_energy_ratio": gmean(
                "mean_weight_spectral_filter_suppressed_energy_ratio"
            ),
            "mean_ema_delta_norm": gmean("mean_ema_delta_norm"),
            "mean_tau": gmean("mean_tau"),
            "mean_tau_source_signal": gmean("mean_tau_source_signal"),
            "mean_tau_source_ema": gmean("mean_tau_source_ema"),
            "mean_e_std": gmean("mean_e_std"),
            "mean_graph_density": gmean("mean_graph_density"),
            "mean_raw_current_graph_density": gmean("mean_raw_current_graph_density"),
            "mean_fedsim_num_clusters": gmean("mean_fedsim_num_clusters"),
            "mean_entropy_alpha": gmean("mean_entropy_alpha"),
            "mean_effective_clients": gmean("mean_effective_clients"),
            "mean_min_alpha": gmean("mean_min_alpha"),
            "mean_max_alpha": gmean("mean_max_alpha"),
            "mean_di_pre": gmean("mean_di_pre"),
            "mean_di_post": gmean("mean_di_post"),
            "mean_neff_pre": gmean("mean_neff_pre"),
            "mean_neff_post": gmean("mean_neff_post"),
            "mean_alignment_pre": gmean("mean_alignment_pre"),
            "mean_alignment_post": gmean("mean_alignment_post"),
            "mean_loo_pre": gmean("mean_loo_pre"),
            "mean_loo_post": gmean("mean_loo_post"),
        }
        seed_delta_vals: List[float] = []
        for seed in seed_cols:
            col = f"seed{seed}_delta"
            time_col = f"seed{seed}_wall_time_sec"
            match = [x for x in group if int(x["seed"]) == seed]
            val = float(match[0]["delta"]) if match else float("nan")
            time_val = (
                float(match[0].get("run_wall_time_sec", float("nan")))
                if match
                else float("nan")
            )
            row_base[col] = val
            row_base[time_col] = time_val
            if variant != "fedavg" and not math.isnan(val):
                seed_delta_vals.append(val)
        if variant == "fedavg":
            row_base["median_delta"] = 0.0
            row_base["number_of_positive_seeds"] = 0
        else:
            row_base["median_delta"] = (
                float(statistics.median(seed_delta_vals)) if seed_delta_vals else float("nan")
            )
            row_base["number_of_positive_seeds"] = sum(1 for v in seed_delta_vals if v > 0)
        summary_rows.append(row_base)

    summary_rows.sort(key=rank_key, reverse=True)
    return summary_rows
