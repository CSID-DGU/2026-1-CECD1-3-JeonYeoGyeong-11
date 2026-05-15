"""Phase 2 graph informativeness ablation runner and report generator."""

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
from typing import Any, Dict, Iterable, List, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[3]
LEGACY_REPORT_DIR = PROJECT_ROOT / "docs" / "previous" / "legacy_phase_reports"


def _safe_float(v: Any, default: float = float("nan")) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _final_metric_pairs(payload: Dict[str, Any], key: str) -> float:
    series = payload.get("metrics_distributed", {}).get(key, [])
    if not series:
        return float("nan")
    return _safe_float(series[-1][1])


def _write_json(path: Path, obj: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, allow_nan=True)


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                fields.append(k)
                seen.add(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _mk_cmd(args: argparse.Namespace, method: str, seed: int, run_tag: str, variant: str) -> List[str]:
    cmd = [
        args.python_bin,
        "run_vision_experiment.py",
        "--method",
        method,
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
        "--seed",
        str(seed),
        "--graph-preset",
        str(args.graph_preset),
        "--graph-mode",
        str(args.graph_mode),
        "--graph-source",
        args.graph_source,
        "--graph-variant",
        variant,
        "--graph-smoothing-lambda",
        str(args.graph_smoothing_lambda),
        "--graph-laplacian-type",
        args.graph_laplacian_type,
        "--graph-zero-diagonal",
        str(args.graph_zero_diagonal).lower(),
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
        str(args.out_dir),
        "--run-tag",
        run_tag,
    ]
    return cmd


def _result_path(out_dir: Path, method: str, seed: int, run_tag: str) -> Path:
    tag = f"_{run_tag}" if run_tag else ""
    return out_dir / f"result_general_{method}_seed{seed}{tag}.json"


def _run(cmd: Sequence[str]) -> None:
    def _cleanup_ray() -> None:
        subprocess.run(
            [sys.executable, "-m", "ray", "stop", "--force"],
            cwd=str(PROJECT_ROOT),
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    for attempt in range(2):
        proc = subprocess.run(
            list(cmd),
            cwd=str(PROJECT_ROOT),
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.stdout:
            print(proc.stdout, end="")
        if proc.stderr:
            print(proc.stderr, end="", file=sys.stderr)
        if proc.returncode == 0:
            return

        combined = f"{proc.stdout}\n{proc.stderr}".lower()
        recoverable = (
            "timed out during startup" in combined
            or "simulation engine crashed" in combined
            or "ending simulation" in combined
        )
        if attempt == 0 and recoverable:
            print(
                "[phase2] Detected Ray startup failure. "
                "Running `ray stop --force` and retrying once...",
                file=sys.stderr,
            )
            _cleanup_ray()
            continue

        raise subprocess.CalledProcessError(
            proc.returncode, list(cmd), output=proc.stdout, stderr=proc.stderr
        )


def _mean_std(vals: Iterable[float]) -> Tuple[float, float]:
    arr = [float(v) for v in vals if not math.isnan(float(v))]
    if not arr:
        return float("nan"), float("nan")
    if len(arr) == 1:
        return float(arr[0]), 0.0
    return float(statistics.mean(arr)), float(statistics.pstdev(arr))


def _load_result(path: Path, method: str) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["results"][method]


def _phase1_summary_block() -> str:
    return (
        "- Dataset: FashionMNIST\n"
        "- Partition: Dirichlet label skew (`alpha=0.03`)\n"
        "- Clients: 5\n"
        "- Seeds: 42, 43, 44\n"
        "- Rounds: 10\n"
        "- Methods: FedAvg / FedAvgM\n"
        "- FedAvg final accuracy mean: 0.7700\n"
        "- FedAvgM final accuracy mean: 0.8111\n"
        "- corr(final_acc, mean_CR) = -0.4657\n"
        "- corr(final_acc, mean_CA) = -0.0621\n"
        "- corr(final_acc, mean_DI) = -0.5984\n"
        "- Interpretation (tentative): dominance strongest, conflict secondary, cancellation weak, sample size small.\n"
    )


def _fmt(v: float) -> str:
    if v is None or math.isnan(float(v)):
        return "nan"
    return f"{float(v):.4f}"


def _table(rows: List[List[str]]) -> str:
    if not rows:
        return ""
    out = ["| " + " | ".join(rows[0]) + " |", "| " + " | ".join(["---"] * len(rows[0])) + " |"]
    for r in rows[1:]:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def build_report(
    *,
    args: argparse.Namespace,
    commands_run: List[str],
    summary_rows: List[Dict[str, Any]],
    round_rows: List[Dict[str, Any]],
    graph_rows: List[Dict[str, Any]],
    report_path: Path,
) -> None:
    per_variant: Dict[str, List[Dict[str, Any]]] = {}
    for row in summary_rows:
        per_variant.setdefault(str(row["variant"]), []).append(row)

    variant_stats: Dict[str, Dict[str, float]] = {}
    for variant, rows in per_variant.items():
        accs = [_safe_float(r.get("final_accuracy")) for r in rows]
        m, s = _mean_std(accs)
        variant_stats[variant] = {"mean": m, "std": s}

    controls = ["random", "shuffled", "uniform", "identity"]
    update_mean = variant_stats.get("update", {}).get("mean", float("nan"))
    better_than_all = all(
        (not math.isnan(update_mean))
        and (not math.isnan(variant_stats.get(c, {}).get("mean", float("nan"))))
        and (update_mean > variant_stats[c]["mean"])
        for c in controls
    )
    by_seed: Dict[int, Dict[str, float]] = {}
    for row in summary_rows:
        seed = int(row["seed"])
        by_seed.setdefault(seed, {})[str(row["variant"])] = _safe_float(row["final_accuracy"])
    robust_seed_wins = 0
    for seed, vals in by_seed.items():
        u = vals.get("update", float("nan"))
        if math.isnan(u):
            continue
        if all((c in vals) and (u > vals[c]) for c in controls):
            robust_seed_wins += 1
    claim_supported = better_than_all and (robust_seed_wins >= 2)

    baseline_diffs = [
        _safe_float(r.get("identity_minus_fedavgm"))
        for r in summary_rows
        if str(r.get("variant")) == "identity"
    ]
    base_m, base_s = _mean_std(baseline_diffs)

    seed_table = [["variant", "seed42", "seed43", "seed44", "mean", "std"]]
    for variant in ["fedavgm_baseline", "update", "random", "shuffled", "uniform", "identity"]:
        vals = {
            int(r["seed"]): _safe_float(r.get("final_accuracy"))
            for r in summary_rows
            if str(r["variant"]) == variant
        }
        m, s = _mean_std(vals.values())
        seed_table.append(
            [
                variant,
                _fmt(vals.get(42, float("nan"))),
                _fmt(vals.get(43, float("nan"))),
                _fmt(vals.get(44, float("nan"))),
                _fmt(m),
                _fmt(s),
            ]
        )

    content = []
    content.append("# PHASE2_GRAPH_INFORMATIVENESS_REPORT")
    content.append("")
    content.append("## 1) Phase 1 Summary")
    content.append(_phase1_summary_block())
    content.append("## 2) Exact Commands Used")
    for cmd in commands_run:
        content.append(f"- `{cmd}`")
    content.append("")
    content.append("## 3) Experiment Setting")
    content.append(f"- dataset: `{args.dataset}`")
    content.append(f"- partition: `{args.partition}`")
    content.append(f"- alpha: `{args.dirichlet_alpha}`")
    content.append(f"- num_clients: `{args.num_clients}`")
    content.append(f"- seeds: `{', '.join(str(s) for s in args.seeds)}`")
    content.append(f"- rounds: `{args.rounds}`")
    content.append("- base method: `FedAvgM` (plus identity/no-graph validity check)")
    content.append("")
    content.append("## 4) Graph Source")
    content.append(f"- graph source used: `{args.graph_source}`")
    content.append("## 5) Graph Variants")
    content.append("- `update`, `random`, `shuffled`, `uniform`, `identity`")
    content.append("")
    content.append("## 6) Correction Operator")
    content.append("- `G_corrected = (I - lambda L) G`")
    content.append("- `Delta_corrected = sum_i p_i g_i_corrected`")
    content.append("- FedAvgM server momentum applied after corrected aggregation.")
    content.append("")
    content.append("## 7) Lambda")
    content.append(f"- lambda: `{args.graph_smoothing_lambda}`")
    content.append(f"- laplacian: `{args.graph_laplacian_type}`")
    content.append("")
    content.append("## 8) Final Accuracy Table")
    content.append(_table(seed_table))
    content.append("")
    content.append("## 9) Round-wise Curves")
    content.append("- Full round-wise diagnostics saved in `phase2_round_diagnostics.csv` and `phase2_round_diagnostics.json`.")
    content.append("- Includes: accuracy, loss, CR, CR_weighted, CA, DI, corrected update norm.")
    content.append("")
    content.append("## 10) Graph Diagnostics")
    content.append("- Full graph diagnostics saved in `phase2_graph_diagnostics.csv`.")
    content.append("- Includes: density, edge-weight stats, degree stats, connected components, smoothness, `||A_t-A_{t-1}||_F`.")
    content.append("")
    content.append("## 11) Validity Check (Identity vs FedAvgM)")
    content.append(f"- identity - fedavgm (mean): `{_fmt(base_m)}`")
    content.append(f"- identity - fedavgm (std): `{_fmt(base_s)}`")
    content.append("- Expected behavior: near-zero gap indicates implementation consistency.")
    content.append("")
    content.append("## 12) Main Conclusion")
    if claim_supported:
        content.append(
            "- Graph-information hypothesis is **supported in this setting**: update graph outperforms random/shuffled/uniform/identity on mean and in most seeds."
        )
    else:
        content.append(
            "- Graph-information hypothesis is **weak in this setting**: update graph does not consistently beat random/shuffled/uniform/identity across seeds."
        )
    content.append("- This conclusion is specific to the current low-sample setup (3 seeds, 10 rounds).")
    content.append("")
    content.append("## 13) Recommendation")
    if claim_supported:
        content.append("- Proceed to Phase 3 correction ablation (conflict-only / dominance-only / cancellation-only / triggered).")
    else:
        content.append("- Do not proceed to complex graph correction yet.")
        content.append("- Prefer dominance-only correction, diagnostic-only direction, or generic regularization baselines first.")
    content.append("")
    content.append(f"_Generated at: {datetime.now().isoformat()}_")

    report_path.write_text("\n".join(content), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Phase 2 graph informativeness ablation.")
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
    p.add_argument("--graph-source", type=str, default="classifier_head_update")
    p.add_argument("--graph-preset", type=str, default="none")
    p.add_argument("--graph-mode", type=str, default="dense")
    p.add_argument("--compression-dim", type=int, default=256)
    p.add_argument("--compression-seed", type=int, default=0)
    p.add_argument("--graph-seed", type=int, default=0)
    p.add_argument("--graph-smoothing-lambda", type=float, default=0.05)
    p.add_argument("--graph-laplacian-type", type=str, default="unnormalized")
    p.add_argument("--graph-zero-diagonal", type=str, default="true")
    p.add_argument("--server-learning-rate", type=float, default=1.0)
    p.add_argument("--server-momentum", type=float, default=0.9)
    p.add_argument("--train-subset-size", type=int, default=0)
    p.add_argument("--test-subset-size", type=int, default=0)
    p.add_argument(
        "--variants",
        type=str,
        nargs="+",
        default=["update", "random", "shuffled", "uniform", "identity"],
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "experiments_current" / "phase2_graph_informativeness",
    )
    p.add_argument(
        "--report-path",
        type=Path,
        default=LEGACY_REPORT_DIR / "PHASE2_GRAPH_INFORMATIVENESS_REPORT.md",
        help="Markdown report output path.",
    )
    p.add_argument("--reuse-existing-results", type=str, default="true")
    return p.parse_args()


def run() -> None:
    args = parse_args()
    args.graph_zero_diagonal = str(args.graph_zero_diagonal).strip().lower() in {
        "1",
        "true",
        "t",
        "yes",
        "y",
        "on",
    }
    reuse_existing = str(args.reuse_existing_results).strip().lower() in {
        "1",
        "true",
        "t",
        "yes",
        "y",
        "on",
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    commands_run: List[str] = []
    summary_rows: List[Dict[str, Any]] = []
    round_rows: List[Dict[str, Any]] = []
    graph_rows: List[Dict[str, Any]] = []
    fedavgm_acc: Dict[int, float] = {}

    for seed in args.seeds:
        run_tag = f"phase2_fedavgm_seed{seed}"
        cmd = _mk_cmd(
            args=args,
            method="fedavgm",
            seed=int(seed),
            run_tag=run_tag,
            variant="identity",
        )
        path = _result_path(out_dir=out_dir, method="fedavgm", seed=int(seed), run_tag=run_tag)
        commands_run.append(" ".join(cmd))
        if not (reuse_existing and path.is_file()):
            _run(cmd)
        result = _load_result(path, "fedavgm")
        fedavgm_acc[int(seed)] = _final_metric_pairs(result, "accuracy")
        summary_rows.append(
            {
                "variant": "fedavgm_baseline",
                "seed": int(seed),
                "final_accuracy": fedavgm_acc[int(seed)],
                "final_loss": _safe_float(result.get("losses_distributed", [[None, float("nan")]])[-1][1]),
                "graph_source": args.graph_source,
                "graph_smoothing_lambda": float(args.graph_smoothing_lambda),
                "graph_laplacian_type": args.graph_laplacian_type,
            }
        )

    for variant in args.variants:
        for seed in args.seeds:
            run_tag = f"phase2_graphsmooth_{variant}_seed{seed}_l{str(args.graph_smoothing_lambda).replace('.', 'p')}"
            cmd = _mk_cmd(
                args=args,
                method="graph_smooth",
                seed=int(seed),
                run_tag=run_tag,
                variant=variant,
            )
            path = _result_path(
                out_dir=out_dir,
                method="graph_smooth",
                seed=int(seed),
                run_tag=run_tag,
            )
            commands_run.append(" ".join(cmd))
            if not (reuse_existing and path.is_file()):
                _run(cmd)
            result = _load_result(path, "graph_smooth")
            final_acc = _final_metric_pairs(result, "accuracy")
            final_loss = _safe_float(
                result.get("losses_distributed", [[None, float("nan")]])[-1][1]
            )
            id_gap = (
                final_acc - fedavgm_acc.get(int(seed), float("nan"))
                if variant == "identity"
                else float("nan")
            )
            summary_rows.append(
                {
                    "variant": variant,
                    "seed": int(seed),
                    "final_accuracy": final_acc,
                    "final_loss": final_loss,
                    "identity_minus_fedavgm": id_gap,
                    "graph_source": args.graph_source,
                    "graph_smoothing_lambda": float(args.graph_smoothing_lambda),
                    "graph_laplacian_type": args.graph_laplacian_type,
                }
            )

            trace = result.get("round_trace", [])
            for row in trace:
                rec = {
                    "variant": variant,
                    "seed": int(seed),
                    "round": int(row.get("round", -1)),
                    "accuracy": _safe_float(row.get("accuracy")),
                    "loss": _safe_float(row.get("loss")),
                    "CR_t": _safe_float(row.get("conflict_ratio")),
                    "CR_weighted_t": _safe_float(row.get("conflict_ratio_weighted")),
                    "CA_t": _safe_float(row.get("cancellation_ratio")),
                    "DI_t": _safe_float(row.get("dominance_ratio")),
                    "N_eff_t": _safe_float(row.get("effective_num_clients")),
                    "mean_cosine": _safe_float(row.get("pairwise_cosine_mean")),
                    "min_cosine": _safe_float(row.get("pairwise_cosine_min")),
                    "std_cosine": _safe_float(row.get("pairwise_cosine_std")),
                    "delta_norm": _safe_float(row.get("delta_norm")),
                    "delta_norm_ratio": _safe_float(
                        row.get("delta_norm_over_weighted_client_norm")
                    ),
                    "corrected_delta_norm": _safe_float(row.get("corrected_delta_norm")),
                }
                round_rows.append(rec)

                graph_rows.append(
                    {
                        "variant": variant,
                        "seed": int(seed),
                        "round": int(row.get("round", -1)),
                        "graph_density": _safe_float(row.get("graph_density")),
                        "edge_weight_mean": _safe_float(row.get("graph_edge_weight_mean")),
                        "edge_weight_std": _safe_float(row.get("graph_edge_weight_std")),
                        "edge_weight_min": _safe_float(row.get("graph_edge_weight_min")),
                        "edge_weight_max": _safe_float(row.get("graph_edge_weight_max")),
                        "degree_mean": _safe_float(row.get("graph_degree_mean")),
                        "degree_std": _safe_float(row.get("graph_degree_std")),
                        "connected_components": _safe_float(
                            row.get("graph_connected_components")
                        ),
                        "A_delta_fro": _safe_float(row.get("graph_adj_delta_fro")),
                        "graph_smoothness": _safe_float(row.get("graph_smoothness")),
                        "corrected_delta_norm": _safe_float(
                            row.get("corrected_delta_norm")
                        ),
                        "cos_fedavg_vs_corrected": _safe_float(
                            row.get("corrected_vs_fedavg_delta_cosine")
                        ),
                        "cos_corrected_vs_momentum": _safe_float(
                            row.get("corrected_vs_server_momentum_cosine")
                        ),
                    }
                )

    _write_csv(out_dir / "phase2_round_diagnostics.csv", round_rows)
    _write_json(out_dir / "phase2_round_diagnostics.json", round_rows)
    _write_csv(out_dir / "phase2_run_summary.csv", summary_rows)
    _write_json(out_dir / "phase2_run_summary.json", summary_rows)
    _write_csv(out_dir / "phase2_graph_diagnostics.csv", graph_rows)

    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    build_report(
        args=args,
        commands_run=commands_run,
        summary_rows=summary_rows,
        round_rows=round_rows,
        graph_rows=graph_rows,
        report_path=report_path,
    )
    print(f"Saved: {report_path}")
    print(f"Saved: {out_dir / 'phase2_round_diagnostics.csv'}")
    print(f"Saved: {out_dir / 'phase2_run_summary.csv'}")
    print(f"Saved: {out_dir / 'phase2_graph_diagnostics.csv'}")


if __name__ == "__main__":
    run()
