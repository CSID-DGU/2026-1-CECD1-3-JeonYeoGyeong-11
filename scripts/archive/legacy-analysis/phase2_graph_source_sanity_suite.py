"""Run graph-source sanity suite and aggregate source-level results.

This script repeatedly executes phase2_graph_informativeness.py with different
graph sources and summarizes whether each source beats graph controls.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[3]
LEGACY_REPORT_DIR = PROJECT_ROOT / "docs" / "previous" / "legacy_phase_reports"


def _safe_float(v, default=float("nan")) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _mean(values: List[float]) -> float:
    arr = [float(v) for v in values if not math.isnan(float(v))]
    if not arr:
        return float("nan")
    return float(statistics.mean(arr))


def _std(values: List[float]) -> float:
    arr = [float(v) for v in values if not math.isnan(float(v))]
    if len(arr) <= 1:
        return 0.0 if arr else float("nan")
    return float(statistics.pstdev(arr))


def _fmt(v: float, digits: int = 4) -> str:
    if v is None or math.isnan(float(v)):
        return "nan"
    return f"{float(v):.{digits}f}"


def _table(rows: List[List[str]]) -> str:
    if not rows:
        return ""
    lines = [
        "| " + " | ".join(rows[0]) + " |",
        "| " + " | ".join(["---"] * len(rows[0])) + " |",
    ]
    for row in rows[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run graph-source sanity suite.")
    p.add_argument("--python-bin", type=str, default=sys.executable)
    p.add_argument("--dataset", type=str, default="fashionmnist")
    p.add_argument("--partition", type=str, default="dirichlet")
    p.add_argument("--dirichlet-alpha", type=float, default=0.03)
    p.add_argument("--num-clients", type=int, default=5)
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--model", type=str, default="cnn")
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument(
        "--sources",
        type=str,
        nargs="+",
        default=[
            "update",
            "ema_update",
            "classifier_head_update",
            "classifier_head_ema_update",
        ],
    )
    p.add_argument("--graph-preset", type=str, default="none")
    p.add_argument("--graph-mode", type=str, default="dense")
    p.add_argument(
        "--variants",
        type=str,
        nargs="+",
        default=["update", "random", "shuffled", "uniform", "identity"],
    )
    p.add_argument("--graph-smoothing-lambda", type=float, default=0.05)
    p.add_argument("--graph-laplacian-type", type=str, default="unnormalized")
    p.add_argument("--graph-zero-diagonal", type=str, default="true")
    p.add_argument("--compression-dim", type=int, default=256)
    p.add_argument("--compression-seed", type=int, default=0)
    p.add_argument("--graph-seed", type=int, default=0)
    p.add_argument("--server-learning-rate", type=float, default=1.0)
    p.add_argument("--server-momentum", type=float, default=0.9)
    p.add_argument("--train-subset-size", type=int, default=0)
    p.add_argument("--test-subset-size", type=int, default=0)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "experiments_current" / "phase2_graph_source_sanity",
    )
    p.add_argument(
        "--suite-report-path",
        type=Path,
        default=LEGACY_REPORT_DIR / "PHASE2_GRAPH_SOURCE_SANITY_REPORT.md",
    )
    p.add_argument("--reuse-existing-results", type=str, default="true")
    return p.parse_args()


def run() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    args.suite_report_path.parent.mkdir(parents=True, exist_ok=True)
    reuse_existing = str(args.reuse_existing_results).strip().lower() in {
        "1",
        "true",
        "t",
        "yes",
        "y",
        "on",
    }

    commands: List[str] = []
    source_rows: List[Dict[str, object]] = []
    detail_rows: List[Dict[str, object]] = []

    # Clear stale Ray state before sequential Flower simulations.
    subprocess.run(
        [sys.executable, "-m", "ray", "stop", "--force"],
        cwd=str(PROJECT_ROOT),
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    for source in args.sources:
        source_dir = out_dir / source
        source_dir.mkdir(parents=True, exist_ok=True)
        source_report_path = source_dir / "PHASE2_GRAPH_INFORMATIVENESS_REPORT.md"
        cmd = [
            args.python_bin,
            "scripts/archive/legacy-analysis/phase2_graph_informativeness.py",
            "--dataset",
            args.dataset,
            "--partition",
            args.partition,
            "--dirichlet-alpha",
            str(args.dirichlet_alpha),
            "--num-clients",
            str(args.num_clients),
            "--rounds",
            str(args.rounds),
            "--local-epochs",
            str(args.local_epochs),
            "--batch-size",
            str(args.batch_size),
            "--model",
            args.model,
            "--lr",
            str(args.lr),
            "--momentum",
            str(args.momentum),
            "--weight-decay",
            str(args.weight_decay),
            "--seeds",
            *[str(s) for s in args.seeds],
            "--graph-preset",
            str(args.graph_preset),
            "--graph-mode",
            str(args.graph_mode),
            "--graph-source",
            source,
            "--variants",
            *args.variants,
            "--graph-smoothing-lambda",
            str(args.graph_smoothing_lambda),
            "--graph-laplacian-type",
            str(args.graph_laplacian_type),
            "--graph-zero-diagonal",
            str(args.graph_zero_diagonal),
            "--compression-dim",
            str(args.compression_dim),
            "--compression-seed",
            str(args.compression_seed),
            "--graph-seed",
            str(args.graph_seed),
            "--server-learning-rate",
            str(args.server_learning_rate),
            "--server-momentum",
            str(args.server_momentum),
            "--train-subset-size",
            str(args.train_subset_size),
            "--test-subset-size",
            str(args.test_subset_size),
            "--out-dir",
            str(source_dir),
            "--report-path",
            str(source_report_path),
            "--reuse-existing-results",
            "true" if reuse_existing else "false",
        ]
        commands.append(" ".join(cmd))
        subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)

        run_summary = _read_csv(source_dir / "phase2_run_summary.csv")
        by_variant: Dict[str, List[float]] = {}
        for row in run_summary:
            variant = str(row.get("variant", "")).strip()
            acc = _safe_float(row.get("final_accuracy"))
            by_variant.setdefault(variant, []).append(acc)
            detail_rows.append(
                {
                    "graph_source": source,
                    "variant": variant,
                    "seed": row.get("seed", ""),
                    "final_accuracy": acc,
                }
            )

        update_mean = _mean(by_variant.get("update", []))
        controls = ["random", "shuffled", "uniform", "identity"]
        control_means = {k: _mean(by_variant.get(k, [])) for k in controls}
        fedavgm_mean = _mean(by_variant.get("fedavgm_baseline", []))
        beats_all_controls = (
            not math.isnan(update_mean)
            and all(not math.isnan(control_means[k]) and update_mean > control_means[k] for k in controls)
        )
        source_rows.append(
            {
                "graph_source": source,
                "fedavgm_baseline_mean": fedavgm_mean,
                "update_mean": update_mean,
                "random_mean": control_means["random"],
                "shuffled_mean": control_means["shuffled"],
                "uniform_mean": control_means["uniform"],
                "identity_mean": control_means["identity"],
                "update_minus_best_control": update_mean - max(control_means.values()),
                "update_minus_fedavgm": update_mean - fedavgm_mean,
                "update_std": _std(by_variant.get("update", [])),
                "beats_all_controls": bool(beats_all_controls),
            }
        )

    source_rows.sort(
        key=lambda r: _safe_float(r.get("update_minus_best_control"), default=-1e9),
        reverse=True,
    )

    _write_csv(out_dir / "phase2_graph_source_sanity_summary.csv", source_rows)
    _write_csv(out_dir / "phase2_graph_source_sanity_detail.csv", detail_rows)
    (out_dir / "phase2_graph_source_sanity_summary.json").write_text(
        json.dumps(source_rows, indent=2, allow_nan=True),
        encoding="utf-8",
    )

    table_rows: List[List[str]] = [
        [
            "graph_source",
            "fedavgm",
            "update",
            "best_control",
            "update-best_control",
            "update-fedavgm",
            "beats_controls",
        ]
    ]
    for row in source_rows:
        best_control = max(
            _safe_float(row.get("random_mean")),
            _safe_float(row.get("shuffled_mean")),
            _safe_float(row.get("uniform_mean")),
            _safe_float(row.get("identity_mean")),
        )
        table_rows.append(
            [
                str(row["graph_source"]),
                _fmt(_safe_float(row.get("fedavgm_baseline_mean"))),
                _fmt(_safe_float(row.get("update_mean"))),
                _fmt(best_control),
                _fmt(_safe_float(row.get("update_minus_best_control"))),
                _fmt(_safe_float(row.get("update_minus_fedavgm"))),
                "yes" if bool(row.get("beats_all_controls")) else "no",
            ]
        )

    lines: List[str] = []
    lines.append("# PHASE2_GRAPH_SOURCE_SANITY_REPORT")
    lines.append("")
    lines.append("## Purpose")
    lines.append(
        "- Check whether graph source choice changes informativeness before complex operator tuning."
    )
    lines.append("")
    lines.append("## Suite Settings")
    lines.append(f"- dataset: `{args.dataset}`")
    lines.append(f"- partition: `{args.partition}` (alpha={args.dirichlet_alpha})")
    lines.append(f"- clients: `{args.num_clients}`")
    lines.append(f"- rounds: `{args.rounds}`")
    lines.append(f"- seeds: `{', '.join(str(s) for s in args.seeds)}`")
    lines.append(f"- variants: `{', '.join(args.variants)}`")
    lines.append("")
    lines.append("## Commands Run")
    for cmd in commands:
        lines.append(f"- `{cmd}`")
    lines.append("")
    lines.append("## Source-Level Summary")
    lines.append(_table(table_rows))
    lines.append("")
    lines.append("## Interpretation Guide")
    lines.append("- `update-best_control > 0`: source-specific update graph beats all control variants on mean.")
    lines.append("- `update-fedavgm > 0`: source-specific update graph exceeds FedAvgM baseline.")
    lines.append("")
    lines.append(f"_Generated at: {datetime.now().isoformat()}_")
    args.suite_report_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved: {args.suite_report_path}")
    print(f"Saved: {out_dir / 'phase2_graph_source_sanity_summary.csv'}")
    print(f"Saved: {out_dir / 'phase2_graph_source_sanity_detail.csv'}")


if __name__ == "__main__":
    run()
