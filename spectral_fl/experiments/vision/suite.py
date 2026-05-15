"""Vision FL variant-by-seed suite implementation.

Wraps ``run_vision_experiment.py`` with suite-level summaries
(``vision_suite_summary.json``, compatibility ``general_suite_summary.json``,
short ``suite_summary.json`` aliases, row files, ``knn_vs_random_matched.csv``,
and ``interpretation.md``).

--------------------------------------------------------------------
Canonical ``--variants`` tokens (case-insensitive; rows use lowercase)
--------------------------------------------------------------------

  fedavg
      FedAvg baseline.

  fedavgm
      FedAvg with server momentum from ``--server-momentum``.

  fedadagrad, fedadam, fedyogi
      FedOpt baselines. Use ``_eta{ETA}``, ``_etal{ETA_L}``, or for
      FedAdam/FedYogi ``_eta{ETA}_etal{ETA_L}`` to tune server/local steps.

  fednova, fednova_slr{LR}
      FedNova-style normalized averaging. Examples:
      ``fednova_slr0p5`` -> ``--server-learning-rate 0.5``.

  fedprox, fedprox_mu{MU}
      FedProx baseline. Examples: ``fedprox_mu0p01`` -> ``--fedprox-mu 0.01``.

  fedmedian
      Coordinate-wise median aggregation.

  fedtrimmedavg, fedtrimmedavg_beta{BETA}
      Coordinate-wise trimmed mean. Example: ``fedtrimmedavg_beta0p1``.

  fedsim, fedsim_k{K}
      FedSim-style similarity-guided cluster aggregation. Similar clients are
      clustered from the update graph, cluster-local updates are averaged, and
      clusters are then averaged with equal cluster mass.

  ours_default_graph, ours_default_graph_k{K}
      Representative default graph-FL path via
      ``--graph-method default_similarity_knn``: RBF update similarity,
      kNN topology, and graph-filtered update aggregation.

  ours_dense
      ``--graph-mode dense``. Same idea as FGL ``run_graph_ablation`` ``ours_dense``.

  ours_knn_k{K}   e.g. ours_knn_k2, ours_knn_k3
      ``--graph-mode knn --knn-k K``. FGL equivalent: ``ours_knn`` with matching ``--knn-k``.

  ours_knn
      ``--graph-mode knn --knn-k`` from suite ``--knn-k`` (default 3).

  ours_random_matched_k{K}   e.g. ours_random_matched_k3
      ``--graph-mode random --knn-k K`` (random graph matched to kNN edge count).
      FGL equivalent: ``ours_random`` with the same ``--knn-k``.

  ours_random, ours_random_matched
      Same as ``ours_random_matched_k{K}`` but ``K`` = suite ``--knn-k``.

  ours_uniform
      ``--graph-mode uniform``. FGL: ``ours_uniform``.

  ours_threshold
      ``--graph-mode threshold``; cosine cutoff from suite ``--edge-threshold``.
      FGL: ``ours_threshold``.

  ours_mutual_knn, ours_mutual_knn_k{K}
      ``--graph-mode mutual_knn`` with either suite ``--knn-k`` or explicit K.

  ours_magnitude
      ``--graph-mode magnitude``.

  ours_global_alignment
      ``--graph-mode global_alignment``.

  ours_weight_graph
      ``--graph-source weight --graph-mode dense``.

  ours_weight_graph_knn_k{K}
      ``--graph-source weight --graph-mode knn --knn-k K``.

  ours_weight_graph_filtered_weight_agg_knn_k{K}
      Build the graph from local weights and low-pass filter local weights
      before averaging.

  ours_head_graph_knn_k{K}, ours_head_weight_graph_knn_k{K}
      Build the graph from the final classifier head update or weight.

  ours_ema_graph_knn_k{K}
      Build the graph from client-update EMA, but filter the current update.

  ours_ema_signal_knn_k{K}
      Build the graph from client-update EMA and filter the EMA update signal.

  ours_real_graph, ours_real_graph_k{K}
      Current diagnostic protocol real graph:
      classifier-head update graph + kNN + graph-filtered update.

  ours_random_control, ours_shuffled_control, ours_uniform_control, ours_identity_control
      Matched control graph families for the current diagnostic protocol.
      Each also accepts ``_k{K}``.

  ours_cluster_only, ours_cluster_only_k{K}
      Clustering-only/block-uniform graph control.

  ours_graphfree_normclip, ours_graphfree_cap, ours_graphfree_reweight
      Graph-free correction controls with raw update aggregation.

  ours_weight_agg
      ``--aggregation-target weight``. This is mathematically close to update
      aggregation when alpha sums to one, but logs the target explicitly.

  ours_no_ema
      Dense graph with ``--use-ema-graph false``. FGL: ``ours_no_ema``.

  ours_fixed_tau
      Dense graph with ``--disable-adaptive-tau true``; magnitude from ``--fixed-tau``.
      FGL: ``ours_fixed_tau``.

Not wrapped here (run ``run_vision_experiment.py`` directly): grid_search-only knobs.

--------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from spectral_fl.config_io import public_args_dict
from spectral_fl.experiments.suites.vision.reporting import (
    _variant_k_number,
    append_validation_verdict,
    compute_best_knn_meta,
    duplicate_suite_summaries,
    write_dashboard_mockup,
    write_diagnostic_csv,
    write_interpretation_md,
    write_knn_vs_random_matched_csv,
    write_summary_markdown,
)
from spectral_fl.experiments.suites.vision.variants import variant_cmd
from spectral_fl.experiments.suites.stats import (
    final_acc,
    load_json,
    round_trace_field,
    safe_max,
    safe_mean,
    safe_min,
    safe_pstdev,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]


VARIANTS_EPILOG = """recognized variant tokens:
  fedavg fedavgm fedadagrad fedadam fedyogi fednova
  fedadam_eta0p05_etal0p01 fednova_slr0p5
  fedprox fedprox_mu0p01 fedmedian fedtrimmedavg_beta0p1 fedsim_k2
  ours_default_graph ours_default_graph_k2
  ours_dense ours_knn ours_knn_k2 ours_knn_k3 ...
  ours_random_matched_k3 ours_random ours_random_matched
  ours_uniform ours_threshold ours_mutual_knn_k3 ours_magnitude
  ours_magnitude_knn_k2 ours_rbf ours_rbf_knn_k2
  ours_head_graph_knn_k2 ours_head_weight_graph_knn_k2
  ours_ema_graph_knn_k2 ours_ema_signal_knn_k2
  ours_weight_graph_filtered_weight_agg_knn_k2
  ours_real_graph_k2 ours_random_control_k2 ours_shuffled_control_k2
  ours_uniform_control_k2 ours_identity_control_k2 ours_cluster_only_k2
  ours_graphfree_normclip ours_graphfree_cap ours_graphfree_reweight
  ours_learned_graph ours_learned_smooth_knn_k2
  ours_global_alignment ours_weight_graph ours_weight_agg
  ours_no_ema ours_fixed_tau
  ours_knn_k2_fixed_tau ours_knn_k2_norm_tau ours_knn_k2_estd_tau
  ours_graph_filtered_magnitude_knn_k1_lp0p5 ours_graph_filtered_magnitude_knn_k1_lp2p0
  ours_graph_filtered_magnitude_knn_k1_serverm_fixed_tau
  ours_layerwise_knn_k2 ours_signed_abs ours_negative
  ours_tail_m2_knn_k1 ours_layerwise_tail_m2_knn_k1
  ours_graph_filtered_knn_k2 ours_graph_filtered_uniform
  ours_graph_filtered_magnitude_knn_k2 ours_graph_filtered_rbf_knn_k2
  ours_graph_filtered_random_matched_k2
  ours_legacy_residual_reweight_knn_k2
  ours_legacy_residual_reweight_random_matched_k2
  ours_legacy_residual_reweight_magnitude_knn_k2
  ours_graph_mode_<custom_mode> (use with --graph-plugin and suite graph-source/aggregation-target)
(See module docstring for FGL name mapping.)"""



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


def run_cmd(cmd: List[str]) -> None:
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)


def execute_or_reuse_result(
    cmd: List[str], result_path: Path, reuse_existing: bool
) -> Tuple[bool, Optional[float]]:
    """Return (reused_existing_result, observed_subprocess_wall_time_sec)."""
    if reuse_existing and result_path.is_file():
        print(f"Reusing existing result: {result_path}")
        return True, None
    start = time.perf_counter()
    run_cmd(cmd)
    return False, float(time.perf_counter() - start)


def collect_timing_features(
    result_obj: Dict[str, Any],
    method: str,
    observed_wall_time_sec: Optional[float],
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


def run(args):
    suite_started_at = datetime.now()
    suite_start = time.perf_counter()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suite_tag = args.suite_tag.strip() or out_dir.name

    suite_meta = {
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
            "mean_min_alpha / mean_max_alpha: mean over rounds of smallest/largest client aggregation mass share."
        ),
        "ranking_semantics": (
            "Summary sorted descending by rank_key (reverse=True): non-fedavg before fedavg; "
            "among methods ordered by mean_delta, then min_delta, then -std_delta (favors lower gap variance), "
            "then win_rate."
        ),
    }

    fedavg_acc_by_seed: Dict[int, float] = {}
    preload_dir = (args.preload_fedavg_dir or "").strip()
    if preload_dir:
        loaded = load_preloaded_fedavg_accs(Path(preload_dir))
        fedavg_acc_by_seed.update(loaded)
        if loaded:
            print(f"Preloaded FedAvg accuracies for seeds {sorted(loaded.keys())} from {preload_dir}")
        elif "fedavg" not in args.variants:
            print(
                f"Warning: --preload-fedavg-dir {preload_dir!r} had no result_vision/result_general FedAvg JSONs; "
                "deltas vs FedAvg will be NaN unless you include fedavg in --variants."
            )
        if fedavg_acc_by_seed:
            suite_meta["preloaded_fedavg_dir"] = preload_dir
            suite_meta["preloaded_fedavg_seeds"] = sorted(fedavg_acc_by_seed.keys())

    rows: List[Dict[str, Any]] = []
    failed_runs: List[Dict[str, Any]] = []
    reuse_existing = truthy(args.reuse_existing_results)

    if "fedavg" in args.variants:
        for seed in args.seeds:
            cmd, method, result_path = variant_cmd(args, "fedavg", seed, suite_tag, out_dir)
            try:
                reused, observed_time = execute_or_reuse_result(
                    cmd=cmd,
                    result_path=result_path,
                    reuse_existing=reuse_existing,
                )
                result = load_json(result_path)
                acc = final_acc(result, method)
                fedavg_acc_by_seed[seed] = acc
                rows.append(
                    {
                        "variant": "fedavg",
                        "seed": int(seed),
                        "method": method,
                        "fedavg_acc": acc,
                        "ours_acc": float("nan"),
                        "delta": 0.0,
                        **collect_timing_features(
                            result_obj=result,
                            method=method,
                            observed_wall_time_sec=observed_time,
                            reused_existing_result=reused,
                        ),
                    }
                )
            except Exception as exc:
                print(f"!! fedavg seed={seed} failed: {exc}")
                failed_runs.append(
                    {
                        "variant": "fedavg",
                        "seed": int(seed),
                        "command": cmd,
                        "error": repr(exc),
                        "trace": traceback.format_exc(limit=4),
                    }
                )

    if "fedavg" not in args.variants and fedavg_acc_by_seed:
        for seed in args.seeds:
            acc = fedavg_acc_by_seed.get(int(seed))
            if acc is None or math.isnan(acc):
                continue
            rows.append(
                {
                    "variant": "fedavg",
                    "seed": int(seed),
                    "method": "fedavg",
                    "fedavg_acc": acc,
                    "ours_acc": float("nan"),
                    "delta": 0.0,
                    **missing_timing_features("preloaded_fedavg"),
                }
            )

    for variant in args.variants:
        if variant.strip().lower() == "fedavg":
            continue
        for seed in args.seeds:
            cmd, method, result_path = variant_cmd(args, variant, seed, suite_tag, out_dir)
            try:
                reused, observed_time = execute_or_reuse_result(
                    cmd=cmd,
                    result_path=result_path,
                    reuse_existing=reuse_existing,
                )
                result = load_json(result_path)
                acc = final_acc(result, method)
                fed_acc = fedavg_acc_by_seed.get(seed, float("nan"))
                feats = collect_run_features(result, method)
                timing = collect_timing_features(
                    result_obj=result,
                    method=method,
                    observed_wall_time_sec=observed_time,
                    reused_existing_result=reused,
                )
                delta = (acc - fed_acc) if not math.isnan(fed_acc) else float("nan")
                rows.append(
                    {
                        "variant": variant.strip().lower(),
                        "seed": int(seed),
                        "method": method,
                        "fedavg_acc": fed_acc,
                        "ours_acc": acc,
                        "delta": delta,
                        **feats,
                        **timing,
                    }
                )
            except Exception as exc:
                print(f"!! {variant} seed={seed} failed: {exc}")
                failed_runs.append(
                    {
                        "variant": variant,
                        "seed": int(seed),
                        "command": cmd,
                        "error": repr(exc),
                        "trace": traceback.format_exc(limit=4),
                    }
                )

    by_variant: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_variant.setdefault(r["variant"], []).append(r)

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
        }
        seed_delta_vals: List[float] = []
        for sd in seed_cols:
            col = f"seed{sd}_delta"
            time_col = f"seed{sd}_wall_time_sec"
            match = [x for x in group if int(x["seed"]) == sd]
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

    best_k_meta = compute_best_knn_meta(summary_rows)
    suite_meta["best_knn_by_meta"] = best_k_meta
    ts = int(args.train_subset_size)
    tst = int(args.test_subset_size)
    suite_meta["training_data_note"] = (
        "full_dataset_splits"
        if ts <= 0 and tst <= 0
        else f"subset_train_{ts}_test_{tst}"
    )
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

    suite_summary = {
        "meta": suite_meta,
        "summary": summary_rows,
        "failed_runs": failed_runs,
    }
    summary_json = out_dir / "general_suite_summary.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(suite_summary, f, indent=2, allow_nan=True)

    rows_path = out_dir / "general_suite_rows.json"
    with rows_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, allow_nan=True)

    csv_path = out_dir / "general_suite_summary.csv"
    if summary_rows:
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

    duplicate_suite_summaries(out_dir, suite_summary, summary_rows, rows)
    diagnostic_csv = write_diagnostic_csv(out_dir, summary_rows)
    knn_csv = write_knn_vs_random_matched_csv(out_dir, summary_rows)
    interp_md = write_interpretation_md(out_dir, summary_rows, suite_meta, args)
    dashboard_md = write_dashboard_mockup(
        out_dir,
        summary_rows,
        diagnostic_csv_path=diagnostic_csv,
    )
    append_validation_verdict(out_dir, summary_rows)

    print("\n=== Vision-FL suite summary (rank: mean_delta, min_delta, -std_delta, win_rate) ===")
    for row in summary_rows:
        if row["variant"] == "fedavg":
            continue
        print(
            f"  {row['variant']:<28} "
            f"mean_delta={row.get('mean_delta', 0):+.4f} "
            f"min_delta={row.get('min_delta', 0):+.4f} "
            f"std={row.get('std_delta', 0):.4f} "
            f"win_rate={row.get('win_rate', 0):.2f} "
            f"time={row.get('mean_run_wall_time_sec', float('nan')):.1f}s"
        )
    md_path = write_summary_markdown(out_dir, suite_tag, args, summary_rows)
    print(f"Saved: {md_path}")

    print(f"Saved: {summary_json}")
    print(f"Saved: {rows_path}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {diagnostic_csv}")
    print(f"Saved: {dashboard_md}")
    print(f"Saved: {out_dir / 'vision_suite_summary.json'} (canonical alias)")
    print(f"Saved: {out_dir / 'suite_summary.json'} (short alias)")
    if knn_csv:
        print(f"Saved: {knn_csv}")
    print(f"Saved: {interp_md}")
    if failed_runs:
        print(f"Failed runs: {len(failed_runs)} (see summary JSON)")
