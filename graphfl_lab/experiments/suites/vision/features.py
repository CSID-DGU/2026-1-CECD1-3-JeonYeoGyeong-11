"""Feature extraction helpers for vision suite result rows."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict

from graphfl_lab.experiments.suites.stats import (
    final_acc,
    load_json,
    round_trace_field,
    safe_mean,
)


def load_preloaded_fedavg_accs(dir_path: Path) -> Dict[int, float]:
    """Read FedAvg final accuracies from prior suite JSONs (resume support)."""
    out: Dict[int, float] = {}
    if not dir_path.is_dir():
        return out
    result_paths: dict[str, Path] = {}
    for p in sorted(dir_path.glob("result_general_fedavg_seed*.json")):
        result_paths[p.name.replace("result_general_", "", 1)] = p
    for p in sorted(dir_path.glob("result_vision_fedavg_seed*.json")):
        result_paths[p.name.replace("result_vision_", "", 1)] = p
    for p in sorted(result_paths.values()):
        try:
            obj = load_json(p)
            seed = obj.get("meta", {}).get("experiment", {}).get("seed")
            if seed is None:
                seed = obj.get("meta", {}).get("seed")
            if seed is None:
                continue
            out[int(seed)] = final_acc(obj, "fedavg")
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
    return out


def collect_run_features(result_obj: Dict[str, Any], method: str) -> Dict[str, Any]:
    if method not in {"ours", "fedsim"}:
        return {}
    trace = result_obj["results"].get(method, {}).get("round_trace", [])

    def trace_value(key: str, default: str = "") -> str:
        for row in reversed(trace):
            value = row.get(key)
            if value is not None:
                return str(value)
        return default

    return {
        "graph_mode": trace_value("graph_mode"),
        "graph_source": trace_value("graph_source"),
        "graph_source_used": trace_value("graph_source_used"),
        "aggregation_target": trace_value("aggregation_target"),
        "aggregation_target_used": trace_value("aggregation_target_used"),
        "server_optimizer": trace_value("server_optimizer"),
        "tau_source_used": trace_value("tau_source_used"),
        "graph_filter_strength": trace_value(
            "graph_filter_strength", trace_value("spectral_filter_strength")
        ),
        "spectral_filter_strength": trace_value(
            "spectral_filter_strength", trace_value("graph_filter_strength")
        ),
        "client_update_ema_alpha": trace_value("client_update_ema_alpha"),
        "client_update_ema_source": trace_value("client_update_ema_source"),
        "server_learning_rate": trace_value("server_learning_rate"),
        "server_momentum": trace_value("server_momentum"),
        "graph_scale_sigma": trace_value("graph_scale_sigma"),
        "learned_graph_lambda": trace_value("learned_graph_lambda"),
        "graph_layer_start": trace_value("graph_layer_start"),
        "graph_layer_end": trace_value("graph_layer_end"),
        "mean_h_spec": safe_mean(round_trace_field(trace, "h_spec")),
        "mean_h_spec_normalized": safe_mean(round_trace_field(trace, "h_spec_normalized")),
        "mean_h_spec_current": safe_mean(round_trace_field(trace, "h_spec_current")),
        "mean_h_spec_raw_current_graph": safe_mean(
            round_trace_field(trace, "h_spec_raw_current_graph")
        ),
        "mean_tau": safe_mean(round_trace_field(trace, "tau")),
        "mean_tau_source_signal": safe_mean(round_trace_field(trace, "tau_source_signal")),
        "mean_tau_source_ema": safe_mean(round_trace_field(trace, "tau_source_ema")),
        "mean_low_frequency_energy_ratio": safe_mean(
            round_trace_field(trace, "low_frequency_energy_ratio")
        ),
        "mean_mid_frequency_energy_ratio": safe_mean(
            round_trace_field(trace, "mid_frequency_energy_ratio")
        ),
        "mean_high_frequency_energy_ratio": safe_mean(
            round_trace_field(trace, "high_frequency_energy_ratio")
        ),
        "mean_high_to_low_energy_ratio": safe_mean(
            round_trace_field(trace, "high_to_low_energy_ratio")
        ),
        "mean_dominant_frequency_energy_ratio": safe_mean(
            round_trace_field(trace, "dominant_frequency_energy_ratio")
        ),
        "mean_spectral_entropy": safe_mean(round_trace_field(trace, "spectral_entropy")),
        "mean_eigengap_max": safe_mean(round_trace_field(trace, "eigengap_max")),
        "mean_spectral_filter_gain": safe_mean(
            round_trace_field(trace, "spectral_filter_gain_mean")
        ),
        "mean_spectral_filter_output_energy_ratio": safe_mean(
            round_trace_field(trace, "spectral_filter_output_energy_ratio")
        ),
        "mean_spectral_filter_residual_energy_ratio": safe_mean(
            round_trace_field(trace, "spectral_filter_residual_energy_ratio")
        ),
        "mean_spectral_filter_suppressed_energy_ratio": safe_mean(
            round_trace_field(trace, "spectral_filter_suppressed_energy_ratio")
        ),
        "mean_server_candidate_delta_norm": safe_mean(
            round_trace_field(trace, "server_candidate_delta_norm")
        ),
        "mean_server_applied_delta_norm": safe_mean(
            round_trace_field(trace, "server_applied_delta_norm")
        ),
        "mean_update_spectral_filter_output_energy_ratio": safe_mean(
            round_trace_field(trace, "update_spectral_filter_output_energy_ratio")
        ),
        "mean_update_spectral_filter_residual_energy_ratio": safe_mean(
            round_trace_field(trace, "update_spectral_filter_residual_energy_ratio")
        ),
        "mean_update_spectral_filter_suppressed_energy_ratio": safe_mean(
            round_trace_field(trace, "update_spectral_filter_suppressed_energy_ratio")
        ),
        "mean_ema_update_spectral_filter_output_energy_ratio": safe_mean(
            round_trace_field(trace, "ema_update_spectral_filter_output_energy_ratio")
        ),
        "mean_ema_update_spectral_filter_residual_energy_ratio": safe_mean(
            round_trace_field(trace, "ema_update_spectral_filter_residual_energy_ratio")
        ),
        "mean_ema_update_spectral_filter_suppressed_energy_ratio": safe_mean(
            round_trace_field(trace, "ema_update_spectral_filter_suppressed_energy_ratio")
        ),
        "mean_weight_spectral_filter_output_energy_ratio": safe_mean(
            round_trace_field(trace, "weight_spectral_filter_output_energy_ratio")
        ),
        "mean_weight_spectral_filter_residual_energy_ratio": safe_mean(
            round_trace_field(trace, "weight_spectral_filter_residual_energy_ratio")
        ),
        "mean_weight_spectral_filter_suppressed_energy_ratio": safe_mean(
            round_trace_field(trace, "weight_spectral_filter_suppressed_energy_ratio")
        ),
        "mean_ema_delta_norm": safe_mean(round_trace_field(trace, "ema_delta_norm_mean")),
        "mean_graph_density": safe_mean(round_trace_field(trace, "graph_density")),
        "mean_raw_current_graph_density": safe_mean(
            round_trace_field(trace, "raw_current_graph_density")
        ),
        "mean_fedsim_num_clusters": safe_mean(
            round_trace_field(trace, "fedsim_num_clusters")
        ),
        "mean_e_std": safe_mean(
            round_trace_field(trace, "e_std") or round_trace_field(trace, "std_e")
        ),
        "mean_entropy_alpha": safe_mean(round_trace_field(trace, "entropy_alpha")),
        "mean_effective_clients": safe_mean(round_trace_field(trace, "effective_clients")),
        "mean_min_alpha": safe_mean(round_trace_field(trace, "min_alpha")),
        "mean_max_alpha": safe_mean(round_trace_field(trace, "max_alpha")),
        "mean_di_pre": safe_mean(round_trace_field(trace, "di_pre")),
        "mean_di_post": safe_mean(round_trace_field(trace, "di_post")),
        "mean_neff_pre": safe_mean(round_trace_field(trace, "neff_pre")),
        "mean_neff_post": safe_mean(round_trace_field(trace, "neff_post")),
        "mean_alignment_pre": safe_mean(round_trace_field(trace, "alignment_mean_pre")),
        "mean_alignment_post": safe_mean(round_trace_field(trace, "alignment_mean_post")),
        "mean_loo_pre": safe_mean(round_trace_field(trace, "loo_mean_pre")),
        "mean_loo_post": safe_mean(round_trace_field(trace, "loo_mean_post")),
    }


def collect_timing_features(
    result_obj: Dict[str, Any],
    method: str,
    observed_wall_time_sec: float | None,
    reused_existing_result: bool,
) -> Dict[str, Any]:
    results = result_obj.get("results", {})
    method_obj = results.get(method, {}) if isinstance(results, dict) else {}
    method_timing = (
        method_obj.get("timing", {}) if isinstance(method_obj, dict) else {}
    )
    meta = result_obj.get("meta", {})
    meta_timing = meta.get("timing", {}) if isinstance(meta, dict) else {}

    def fget(obj: Dict[str, Any], key: str) -> float:
        try:
            return float(obj.get(key, float("nan")))
        except (TypeError, ValueError):
            return float("nan")

    method_wall = fget(method_timing, "wall_time_sec")
    total_wall = fget(meta_timing, "total_wall_time_sec")
    if observed_wall_time_sec is not None:
        run_wall = float(observed_wall_time_sec)
        source = "suite_observed"
    elif not math.isnan(method_wall):
        run_wall = method_wall
        source = "result_method_timing"
    elif not math.isnan(total_wall):
        run_wall = total_wall
        source = "result_total_timing"
    else:
        run_wall = float("nan")
        source = "missing"

    rounds = int(
        result_obj.get("meta", {})
        .get("experiment", {})
        .get("rounds", 0)
        or 0
    )
    return {
        "run_wall_time_sec": run_wall,
        "result_method_wall_time_sec": method_wall,
        "result_total_wall_time_sec": total_wall,
        "seconds_per_round": (
            float(run_wall / rounds)
            if rounds > 0 and not math.isnan(run_wall)
            else float("nan")
        ),
        "timing_source": source,
        "reused_existing_result": bool(reused_existing_result),
    }


def missing_timing_features(source: str) -> Dict[str, Any]:
    return {
        "run_wall_time_sec": float("nan"),
        "result_method_wall_time_sec": float("nan"),
        "result_total_wall_time_sec": float("nan"),
        "seconds_per_round": float("nan"),
        "timing_source": source,
        "reused_existing_result": False,
    }


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def rank_key(row: Dict[str, Any]):
    if row["variant"] == "fedavg":
        return (False, 0, 0, 0, 0)
    return (
        True,
        row.get("mean_delta", float("-inf")),
        row.get("min_delta", float("-inf")),
        -row.get("std_delta", 0.0),
        row.get("win_rate", 0.0),
    )
