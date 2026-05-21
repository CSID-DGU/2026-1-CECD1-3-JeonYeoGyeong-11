"""Run matched vision FL suites across different client counts.

This wrapper keeps all experiment knobs fixed except ``num_clients``.  Each
client count gets its own ``run_vision_suite.py`` output directory, followed by
an aggregate client-count summary for quick comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[3]



def suite_cmd(args: argparse.Namespace, num_clients: int, out_dir: Path, tag: str) -> List[str]:
    cmd = [
        str(args.python_bin),
        "run_vision_suite.py",
        "--dataset",
        str(args.dataset),
        "--model",
        str(args.model),
        "--num-clients",
        str(num_clients),
        "--rounds",
        str(args.rounds),
        "--local-epochs",
        str(args.local_epochs),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--momentum",
        str(args.momentum),
        "--weight-decay",
        str(args.weight_decay),
        "--seeds",
        *[str(s) for s in args.seeds],
        "--partition",
        str(args.partition),
        "--dirichlet-alpha",
        str(args.dirichlet_alpha),
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
        "--knn-k",
        str(args.knn_k),
        "--graph-source",
        str(args.graph_source),
        "--aggregation-target",
        str(args.aggregation_target),
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
        "--graph-filter-strength",
        str(args.graph_filter_strength),
        "--client-update-ema-alpha",
        str(args.client_update_ema_alpha),
        "--diagnostic-only",
        str(args.diagnostic_only),
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
        "--data-root",
        str(args.data_root),
        "--train-subset-size",
        str(args.train_subset_size),
        "--test-subset-size",
        str(args.test_subset_size),
        "--out-dir",
        str(out_dir),
        "--suite-tag",
        tag,
        "--variants",
        *[str(v) for v in args.variants],
    ]
    return cmd


def read_summary_rows(path: Path, num_clients: int, suite_dir: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out = dict(row)
            out["num_clients"] = int(num_clients)
            out["suite_dir"] = str(suite_dir)
            rows.append(out)
    return rows


def csv_float(row: Dict[str, Any], key: str) -> float:
    try:
        value = float(row.get(key, "nan"))
    except (TypeError, ValueError):
        return float("nan")
    return value


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = [
        "num_clients",
        "variant",
        "n_runs",
        "mean_fedavg_acc",
        "mean_acc",
        "std_acc",
        "mean_delta",
        "min_delta",
        "max_delta",
        "std_delta",
        "win_rate",
        "graph_mode",
        "graph_source",
        "graph_source_used",
        "aggregation_target",
        "aggregation_target_used",
        "server_optimizer",
        "mean_H_spec",
        "mean_tau",
        "mean_graph_density",
        "mean_update_spectral_filter_output_energy_ratio",
        "mean_update_spectral_filter_residual_energy_ratio",
        "mean_update_spectral_filter_suppressed_energy_ratio",
        "mean_entropy_alpha",
        "mean_effective_clients",
        "suite_dir",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(rows: List[Dict[str, Any]], path: Path, args: argparse.Namespace) -> None:
    ordered = sorted(
        rows,
        key=lambda r: (
            int(r.get("num_clients", 0)),
            -csv_float(r, "mean_delta"),
            csv_float(r, "min_delta"),
        ),
    )
    lines = [
        "# Client Count Sweep Summary",
        "",
        f"- client counts: `{', '.join(str(x) for x in args.client_counts)}`",
        f"- rounds: `{args.rounds}`, warmup_rounds: `{args.warmup_rounds}`",
        f"- variants: `{', '.join(args.variants)}`",
        "",
        "| clients | variant | mean acc | mean delta | min delta | win rate | mean H_spec | eff clients |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in ordered:
        lines.append(
            "| "
            f"{row.get('num_clients')} | {row.get('variant')} | "
            f"{csv_float(row, 'mean_acc'):.4f} | "
            f"{csv_float(row, 'mean_delta'):+.4f} | "
            f"{csv_float(row, 'min_delta'):+.4f} | "
            f"{csv_float(row, 'win_rate'):.2f} | "
            f"{csv_float(row, 'mean_H_spec'):.4f} | "
            f"{csv_float(row, 'mean_effective_clients'):.2f} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args) -> None:
    root = Path(args.out_dir)
    root.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, Any]] = []
    for n_clients in args.client_counts:
        if int(n_clients) <= 0:
            raise SystemExit(f"Invalid client count: {n_clients}")
        tag = f"{args.sweep_tag}_clients{int(n_clients)}"
        suite_dir = root / f"clients_{int(n_clients)}"
        suite_dir.mkdir(parents=True, exist_ok=True)
        cmd = suite_cmd(args=args, num_clients=int(n_clients), out_dir=suite_dir, tag=tag)
        print(f"=== Running client-count suite: num_clients={n_clients} ===", flush=True)
        subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)
        all_rows.extend(
            read_summary_rows(
                (
                    suite_dir / "vision_suite_summary.csv"
                    if (suite_dir / "vision_suite_summary.csv").is_file()
                    else suite_dir / "general_suite_summary.csv"
                ),
                num_clients=int(n_clients),
                suite_dir=suite_dir,
            )
        )

    write_csv(all_rows, root / "client_count_sweep_summary.csv")
    (root / "client_count_sweep_summary.json").write_text(
        json.dumps(all_rows, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    write_markdown(all_rows, root / "client_count_sweep_summary.md", args)
    print(f"Saved client-count sweep summary under {root}")
