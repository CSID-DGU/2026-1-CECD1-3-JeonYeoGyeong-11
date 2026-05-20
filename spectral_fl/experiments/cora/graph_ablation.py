"""Graph-construction ablation suite implementation.

Runs FedAvg plus seven Ours variants that share the spectral algorithm
but differ in either graph construction or tau/EMA configuration.  This
suite is the core falsification experiment for the "client similarity
graph carries useful structure" hypothesis.

``ours_random`` uses the same undirected edge count as ``ours_knn`` at the same k
so we can separate similarity-graph benefit from generic sparse-random regularization.

Variants (one row per seed):

    fedavg            -- baseline
    ours_dense        -- W = max(0, cos(z_i, z_j))                 (default)
    ours_knn          -- top-k positive cosine neighbors            (k=knn_k)
    ours_threshold    -- W_ij = cos>theta else 0
    ours_mutual_knn   -- mutual top-k positive cosine neighbors
    ours_magnitude    -- positive cosine down-weighted by magnitude mismatch
    ours_global_alignment -- positive cosine weighted by alignment to mean signal
    ours_weight_graph -- dense graph built from local model weights
    ours_weight_agg   -- alpha-weighted local weight average target
    ours_random       -- random binary edges with same edge count as kNN (same k)
    ours_uniform      -- W_ij = 1 for i!=j  (no client-specific structure)
    ours_no_ema       -- dense graph, EMA disabled
    ours_fixed_tau    -- dense graph, fixed tau (no tanh schedule)

Outputs in ``--out-dir``:

    raw result_*.json files (one per run)
    suite_<tag>_summary.json     -- per-variant aggregates, seed<S>_delta columns, ranking
    suite_<tag>_summary.csv      -- same content in CSV form
    suite_<tag>_rows.json        -- per-run rows used to build the summary
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from spectral_fl.config_io import public_args_dict
from spectral_fl.diagnostics.result_schema import (
    config_aliases_from_args,
    unsupported_components_from_args,
    with_result_schema,
)
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


# =============================================================================
# Argparse
# =============================================================================



# =============================================================================
# Helpers
# =============================================================================


def log_subprocess_run(variant: str, seed: int, result_path: Path, cmd: List[str]) -> None:
    """Echo upcoming subprocess so long suites show progress (worst-case debugging)."""
    try:
        joined = shlex.join([str(c) for c in cmd])
    except AttributeError:
        joined = " ".join(shlex.quote(str(c)) for c in cmd)
    print(f"--> {variant} seed={seed} -> {result_path.name}")
    print(f"    {joined}")


def variant_command(
    variant: str,
    args: argparse.Namespace,
    seed: int,
    out_dir: Path,
    run_tag: str,
) -> tuple:
    """Return (cmd, method, expected_result_path)."""
    common = [
        args.python_bin,
        "run_experiment.py",
        "--num-clients", str(args.num_clients),
        "--rounds", str(args.rounds),
        "--local-epochs", str(args.local_epochs),
        "--hidden-dim", str(args.hidden_dim),
        "--seed", str(seed),
        "--partition", str(args.partition),
        "--dirichlet-alpha", str(args.dirichlet_alpha),
        "--data-root", str(args.data_root),
        "--compression-dim", str(args.compression_dim),
        "--compression-seed", str(args.compression_seed),
        "--out-dir", str(out_dir),
        "--run-tag", run_tag,
    ]
    if variant == "fedavg":
        cmd = common + ["--method", "fedavg"]
        return cmd, "fedavg", out_dir / f"result_fedavg_seed{seed}_{run_tag}.json"

    ours = common + [
        "--method", "ours",
        "--warmup-rounds", str(args.warmup_rounds),
        "--tau-max", str(args.tau_max),
        "--tau-gain", str(args.tau_gain),
        "--conflict-mix", str(args.conflict_mix),
        "--ema-alpha", str(args.ema_alpha),
        "--graph-source", str(args.graph_source),
        "--aggregation-target", str(args.aggregation_target),
        "--graph-seed", str(args.graph_seed),
        "--knn-k", str(args.knn_k),
        "--edge-threshold", str(args.edge_threshold),
        "--diagnostic-only", str(args.diagnostic_only),
        "--e-std-threshold", str(args.e_std_threshold),
        "--min-client-weight", str(args.min_client_weight),
    ]
    if variant == "ours_dense":
        cmd = ours + ["--graph-mode", "dense"]
    elif variant == "ours_knn":
        cmd = ours + ["--graph-mode", "knn", "--knn-k", str(args.knn_k)]
    elif variant == "ours_mutual_knn":
        cmd = ours + ["--graph-mode", "mutual_knn", "--knn-k", str(args.knn_k)]
    elif variant == "ours_magnitude":
        cmd = ours + ["--graph-mode", "magnitude"]
    elif variant == "ours_global_alignment":
        cmd = ours + ["--graph-mode", "global_alignment"]
    elif variant == "ours_weight_graph":
        cmd = ours + ["--graph-source", "weight", "--graph-mode", "dense"]
    elif variant == "ours_weight_agg":
        cmd = ours + ["--aggregation-target", "weight"]
    elif variant == "ours_threshold":
        cmd = ours + [
            "--graph-mode", "threshold",
            "--edge-threshold", str(args.edge_threshold),
        ]
    elif variant == "ours_random":
        cmd = ours + ["--graph-mode", "random", "--knn-k", str(args.knn_k)]
    elif variant == "ours_uniform":
        cmd = ours + ["--graph-mode", "uniform"]
    elif variant == "ours_no_ema":
        cmd = ours + ["--graph-mode", "dense", "--use-ema-graph", "false"]
    elif variant == "ours_fixed_tau":
        cmd = ours + [
            "--graph-mode", "dense",
            "--disable-adaptive-tau", "true",
            "--fixed-tau", str(args.fixed_tau),
        ]
    else:
        raise ValueError(f"Unknown variant: {variant}")
    return cmd, "ours", out_dir / f"result_ours_seed{seed}_{run_tag}.json"


def collect_run_features(result_obj: Dict[str, Any], method: str) -> Dict[str, Any]:
    if method == "fedavg":
        return {}
    trace = result_obj["results"]["ours"].get("round_trace", [])

    def trace_value(key: str, default: str = "") -> str:
        for row in trace:
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
        "mean_h_spec": safe_mean(round_trace_field(trace, "h_spec")),
        "mean_h_spec_current": safe_mean(round_trace_field(trace, "h_spec_current")),
        "mean_h_spec_raw_current_graph": safe_mean(
            round_trace_field(trace, "h_spec_raw_current_graph")
        ),
        "mean_tau": safe_mean(round_trace_field(trace, "tau")),
        "mean_low_frequency_energy_ratio": safe_mean(
            round_trace_field(trace, "low_frequency_energy_ratio")
        ),
        "mean_high_frequency_energy_ratio": safe_mean(
            round_trace_field(trace, "high_frequency_energy_ratio")
        ),
        "mean_spectral_entropy": safe_mean(round_trace_field(trace, "spectral_entropy")),
        "mean_eigengap_max": safe_mean(round_trace_field(trace, "eigengap_max")),
        "mean_graph_density": safe_mean(round_trace_field(trace, "graph_density")),
        "mean_raw_current_graph_density": safe_mean(
            round_trace_field(trace, "raw_current_graph_density")
        ),
        "mean_e_std": safe_mean(
            round_trace_field(trace, "e_std")
            or round_trace_field(trace, "std_e")
        ),
        "mean_entropy_alpha": safe_mean(round_trace_field(trace, "entropy_alpha")),
        "min_entropy_alpha": safe_min(round_trace_field(trace, "entropy_alpha")),
        "mean_effective_clients": safe_mean(round_trace_field(trace, "effective_clients")),
        "min_effective_clients": safe_min(round_trace_field(trace, "effective_clients")),
        "mean_min_alpha": safe_mean(round_trace_field(trace, "min_alpha")),
        "global_min_alpha": safe_min(round_trace_field(trace, "min_alpha")),
        "mean_max_alpha": safe_mean(round_trace_field(trace, "max_alpha")),
        "n_graph_empty_rounds": sum(
            1 for r in trace if r.get("graph_empty") is True
        ),
    }


# =============================================================================
# Main
# =============================================================================


def run(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    suite_meta = {
        "timestamp": datetime.now().isoformat(),
        "suite_tag": args.suite_tag,
        "config": public_args_dict(args),
        "delta_baseline": "fedavg",
        "cross_track_variant_names": (
            "This suite labels kNN as variant ours_knn with --knn-k from CLI (no K in the name). "
            "General-FL / Fashion kNN sweep often use ours_knn_k{K} so k is visible in summary rows. "
            "Those strings differ by design; compare runs by graph-mode knn plus knn-k value, not only by token equality."
        ),
        "matched_random_ablation": (
            "ours_random samples a random binary graph with the same undirected edge count as ours_knn at the same k "
            "(spectral_fl.graph.sparsification.random_edges_matched_to_knn). Larger gaps vs FedAvg for ours_knn than for ours_random "
            "support client similarity carrying structure beyond generic sparse random graphs. "
            "Formal significance is not computed here; use per-seed seed<S>_delta externally if you need tests."
        ),
        "delta_semantics": (
            "Per seed, delta = final distributed test accuracy(variant) minus same-seed FedAvg. "
            "Columns seed<S>_delta store that gap for each run seed S. "
            "For non-fedavg variants, mean_delta is the (unweighted) mean of those per-seed gaps, "
            "min_delta is min(seed*_delta) (worst seed), max_delta is max (best seed), std_delta is pstdev of the gaps, "
            "and win_rate is (# seeds with delta>0) / (number of seeds), using the same per-seed gaps."
        ),
        "trace_aggregate_semantics": (
            "mean_H_spec: mean over rounds of h_spec in round_trace "
            "(graph-update alignment diagnostic; not an absolute non-IID score). "
            "mean_H_spec_current: same diagnostic on the current-round graph. "
            "mean_low/high_frequency_energy_ratio: update energy split over the current graph spectrum. "
            "mean_e_std: mean over rounds of conflict-score spread among clients (e_std / std_e in round_trace). "
            "mean_tau: mean over rounds of tau in round_trace (conflict-weight temperature / schedule). "
            "mean_graph_density: mean over rounds of graph_density in round_trace (similarity-graph edge density). "
            "mean_entropy_alpha / min_entropy_alpha: per-run mean or min-over-rounds of entropy_alpha; suite aggregates means/mins across seeds. "
            "mean_min_alpha / global_min_alpha / mean_max_alpha: per-run stats from min_alpha / max_alpha over rounds; suite aggregates across seeds. "
            "mean_effective_clients / min_effective_clients: mean or min over rounds of effective_clients in round_trace "
            "(clients counted as materially weighted after masking); suite aggregates across seeds. "
            "n_graph_empty_rounds: per seed-run count of rounds where graph_empty is true; suite mean across seeds."
        ),
        "ranking_semantics": (
            "Summary rows sorted ascending by (-min_delta, -mean_delta, std_delta, -win_rate): "
            "prefer larger worst-seed gap, then larger mean gap, then lower spread of per-seed gaps (std_delta), "
            "then higher win_rate; fedavg tuple (1,0,0,0) sorts last."
        ),
        "mean_H_spec_vs_mean_h_spec": (
            "Rows (suite_<tag>_rows.json) store mean_h_spec per seed: mean over rounds of h_spec "
            "for that run. Treat it as graph-update alignment, not standalone heterogeneity. "
            "Summary (suite_<tag>_summary.json) exposes mean_H_spec: mean of mean_h_spec across seeds for the variant. "
            "win_rate is only in summary (fraction of seeds with delta>0); same definition as delta_semantics."
        ),
        "trace_contract": (
            "Per-round diagnostics live under results.ours.round_trace (list of dicts). "
            "Authoritative keys are emitted by the graph-FL strategy diagnostics round_log row (spectral signals, tau, "
            "conflict vectors, graph stats). Typical numeric fields used by suite aggregates include h_spec, tau, "
            "graph_density, e_std or legacy std_e, entropy_alpha, min_alpha, max_alpha, effective_clients, graph_empty. "
            "See strategy round_log construction when adding new consumers."
        ),
    }

    fedavg_acc_by_seed: Dict[int, float] = {}
    rows: List[Dict[str, Any]] = []
    failed_runs: List[Dict[str, Any]] = []

    # 1) FedAvg first (reused for delta)
    if "fedavg" in args.variants:
        for seed in args.seeds:
            run_tag = f"{args.suite_tag}_fedavg_seed{seed}"
            cmd, method, result_path = variant_command(
                "fedavg", args, seed, out_dir, run_tag
            )
            try:
                log_subprocess_run("fedavg", seed, result_path, cmd)
                subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)
                acc = final_acc(load_json(result_path), method)
                fedavg_acc_by_seed[seed] = acc
                rows.append(
                    {
                        "variant": "fedavg",
                        "seed": int(seed),
                        "method": method,
                        "fedavg_acc": acc,
                        "ours_acc": float("nan"),
                        "delta": 0.0,
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

    # 2) Ours variants
    for variant in args.variants:
        if variant == "fedavg":
            continue
        for seed in args.seeds:
            run_tag = f"{args.suite_tag}_{variant}_seed{seed}"
            cmd, method, result_path = variant_command(
                variant, args, seed, out_dir, run_tag
            )
            try:
                log_subprocess_run(variant, seed, result_path, cmd)
                subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)
                result = load_json(result_path)
                acc = final_acc(result, method)
                fed_acc = fedavg_acc_by_seed.get(seed, float("nan"))
                feats = collect_run_features(result, method)
                rows.append(
                    {
                        "variant": variant,
                        "seed": int(seed),
                        "method": method,
                        "fedavg_acc": fed_acc,
                        "ours_acc": acc,
                        "delta": (acc - fed_acc) if not math.isnan(fed_acc) else float("nan"),
                        **feats,
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

    # 3) summary by variant
    by_variant: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_variant.setdefault(r["variant"], []).append(r)

    summary_rows: List[Dict[str, Any]] = []
    for variant, group in by_variant.items():
        deltas = [x["delta"] for x in group if x["variant"] != "fedavg"]
        ours_acc = [x["ours_acc"] for x in group if x["variant"] != "fedavg"]
        fa_acc = [x["fedavg_acc"] for x in group]

        def gmean(key: str):
            return safe_mean([x.get(key) for x in group])

        def gmin(key: str):
            return safe_min([x.get(key) for x in group])

        def gfirst(key: str) -> str:
            for x in group:
                value = x.get(key)
                if value not in (None, ""):
                    return str(value)
            return ""

        mean_o = safe_mean(ours_acc)
        mean_acc_row = mean_o if ours_acc else safe_mean(fa_acc)
        std_vals = (
            [x["fedavg_acc"] for x in group]
            if variant == "fedavg"
            else [x["ours_acc"] for x in group]
        )
        std_acc_row = safe_pstdev(std_vals)
        row_base = {
            "variant": variant,
            "n_runs": len(group),
            "graph_mode": gfirst("graph_mode"),
            "graph_source": gfirst("graph_source"),
            "graph_source_used": gfirst("graph_source_used"),
            "aggregation_target": gfirst("aggregation_target"),
            "aggregation_target_used": gfirst("aggregation_target_used"),
            "mean_fedavg_acc": safe_mean(fa_acc),
            "mean_ours_acc": mean_o,
            "mean_acc": mean_acc_row,
            "std_acc": std_acc_row,
            "mean_delta": safe_mean(deltas) if deltas else 0.0,
            "min_delta": safe_min(deltas) if deltas else 0.0,
            "max_delta": safe_max(deltas) if deltas else 0.0,
            "std_delta": safe_pstdev(deltas) if deltas else 0.0,
            "win_rate": (
                (sum(1 for d in deltas if d > 0) / len(deltas)) if deltas else 0.0
            ),
            "mean_H_spec": gmean("mean_h_spec"),
            "mean_H_spec_current": gmean("mean_h_spec_current"),
            "mean_H_spec_raw_current_graph": gmean("mean_h_spec_raw_current_graph"),
            "mean_low_frequency_energy_ratio": gmean("mean_low_frequency_energy_ratio"),
            "mean_high_frequency_energy_ratio": gmean("mean_high_frequency_energy_ratio"),
            "mean_spectral_entropy": gmean("mean_spectral_entropy"),
            "mean_eigengap_max": gmean("mean_eigengap_max"),
            "mean_tau": gmean("mean_tau"),
            "mean_graph_density": gmean("mean_graph_density"),
            "mean_raw_current_graph_density": gmean("mean_raw_current_graph_density"),
            "mean_e_std": gmean("mean_e_std"),
            "mean_entropy_alpha": gmean("mean_entropy_alpha"),
            "min_entropy_alpha": gmin("min_entropy_alpha"),
            "mean_effective_clients": gmean("mean_effective_clients"),
            "min_effective_clients": gmin("min_effective_clients"),
            "mean_min_alpha": gmean("mean_min_alpha"),
            "global_min_alpha": gmin("global_min_alpha"),
            "mean_max_alpha": gmean("mean_max_alpha"),
            "n_graph_empty_rounds": gmean("n_graph_empty_rounds"),
        }
        for sd in sorted(set(args.seeds)):
            col = f"seed{sd}_delta"
            match = [x for x in group if int(x["seed"]) == sd]
            row_base[col] = float(match[0]["delta"]) if match else float("nan")
        summary_rows.append(row_base)

    # rank: prefer variants whose worst seed is highest, then mean delta, then std
    def rank_key(row):
        if row["variant"] == "fedavg":
            return (1, 0, 0, 0)  # always last, doesn't matter
        return (-row["min_delta"], -row["mean_delta"], row["std_delta"], -row["win_rate"])

    summary_rows.sort(key=rank_key)

    # ---- save ----
    suite_summary = with_result_schema(
        {
            "meta": suite_meta,
            "summary": summary_rows,
            "failed_runs": failed_runs,
        },
        config_aliases_used=config_aliases_from_args(args),
        unsupported_components=unsupported_components_from_args(args),
    )
    summary_json = out_dir / f"suite_{args.suite_tag}_summary.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(suite_summary, f, indent=2)

    rows_path = out_dir / f"suite_{args.suite_tag}_rows.json"
    with rows_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    csv_path = out_dir / f"suite_{args.suite_tag}_summary.csv"
    if summary_rows:
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

    print("\n=== Graph-ablation summary (best worst-seed first) ===")
    for row in summary_rows:
        print(
            f"  {row['variant']:<18} "
            f"mean_delta={row.get('mean_delta', 0):+.4f} "
            f"min_delta={row.get('min_delta', 0):+.4f} "
            f"std={row.get('std_delta', 0):.4f} "
            f"win_rate={row.get('win_rate', 0):.2f} "
            f"mean_H_spec={row.get('mean_H_spec', float('nan')):.4f} "
            f"mean_e_std={row.get('mean_e_std', float('nan')):.4f} "
            f"mean_tau={row.get('mean_tau', float('nan')):.4f} "
            f"mean_graph_density={row.get('mean_graph_density', float('nan')):.4f} "
            f"mean_max_alpha={row.get('mean_max_alpha', float('nan')):.4f} "
            f"mean_eff_clients={row.get('mean_effective_clients', float('nan')):.4f}"
        )
    print(f"Saved: {summary_json}")
    print(f"Saved: {rows_path}")
    print(f"Saved: {csv_path}")
    if failed_runs:
        print(f"Failed runs: {len(failed_runs)} (see {summary_json})")
