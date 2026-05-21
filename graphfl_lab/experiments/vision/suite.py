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
import math
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from graphfl_lab.diagnostics.result_schema import (
    config_aliases_from_args,
    unsupported_components_from_args,
    with_result_schema,
)
from graphfl_lab.experiments.suites.vision.reporting import (
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
from graphfl_lab.experiments.suites.execution import execute_or_reuse_result
from graphfl_lab.experiments.suites.result_writer import write_csv_rows, write_json
from graphfl_lab.experiments.suites.vision.features import (
    collect_run_features,
    collect_timing_features,
    load_preloaded_fedavg_accs,
    missing_timing_features,
    truthy,
)
from graphfl_lab.experiments.suites.vision.metadata import (
    build_suite_meta,
    record_preloaded_fedavg_meta,
    record_suite_timing,
    record_training_data_note,
)
from graphfl_lab.experiments.suites.vision.summary import build_summary_rows
from graphfl_lab.experiments.suites.vision.variants import variant_cmd
from graphfl_lab.experiments.suites.stats import (
    final_acc,
    load_json,
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


def run(args):
    suite_started_at = datetime.now()
    suite_start = time.perf_counter()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suite_tag = args.suite_tag.strip() or out_dir.name

    suite_meta = build_suite_meta(args, suite_tag, suite_started_at)

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
            record_preloaded_fedavg_meta(suite_meta, preload_dir, fedavg_acc_by_seed)

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
                    cwd=PROJECT_ROOT,
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
                    cwd=PROJECT_ROOT,
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

    summary_rows = build_summary_rows(rows, args)

    best_k_meta = compute_best_knn_meta(summary_rows)
    suite_meta["best_knn_by_meta"] = best_k_meta
    record_training_data_note(suite_meta, args)
    record_suite_timing(suite_meta, suite_started_at, suite_start, rows)

    suite_summary = with_result_schema(
        {
            "meta": suite_meta,
            "summary": summary_rows,
            "failed_runs": failed_runs,
        },
        config_aliases_used=config_aliases_from_args(args),
        unsupported_components=unsupported_components_from_args(args),
    )
    summary_json = out_dir / "general_suite_summary.json"
    write_json(summary_json, suite_summary)

    rows_path = out_dir / "general_suite_rows.json"
    write_json(rows_path, rows)

    csv_path = out_dir / "general_suite_summary.csv"
    if summary_rows:
        write_csv_rows(csv_path, summary_rows, fieldnames=list(summary_rows[0].keys()))

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
