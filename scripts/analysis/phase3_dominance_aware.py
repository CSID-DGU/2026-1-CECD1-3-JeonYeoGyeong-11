"""Phase 3 dominance-aware server correction runner and report generator."""

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


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _safe_float(v: Any, default: float = float("nan")) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _mean(vals: Iterable[float]) -> float:
    arr = [float(v) for v in vals if not math.isnan(float(v))]
    if not arr:
        return float("nan")
    return float(sum(arr) / len(arr))


def _std(vals: Iterable[float]) -> float:
    arr = [float(v) for v in vals if not math.isnan(float(v))]
    if len(arr) <= 1:
        return 0.0 if arr else float("nan")
    return float(statistics.pstdev(arr))


def _corr(x_vals: Iterable[float], y_vals: Iterable[float]) -> float:
    pairs = [
        (float(x), float(y))
        for x, y in zip(x_vals, y_vals)
        if not (math.isnan(float(x)) or math.isnan(float(y)))
    ]
    if len(pairs) < 3:
        return float("nan")
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0.0 or vy <= 0.0:
        return float("nan")
    cov = sum((x - mx) * (y - my) for x, y in pairs)
    return float(cov / math.sqrt(vx * vy))


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


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, allow_nan=True), encoding="utf-8")


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _run(cmd: Sequence[str]) -> None:
    subprocess.run(list(cmd), cwd=str(PROJECT_ROOT), check=True)


def _result_path(out_dir: Path, method: str, seed: int, run_tag: str) -> Path:
    suffix = f"_{run_tag}" if run_tag else ""
    return out_dir / f"result_general_{method}_seed{seed}{suffix}.json"


def _method_configs(selected_labels: Sequence[str] | None = None) -> List[Dict[str, Any]]:
    all_configs = [
        {"name": "fedavgm", "method": "fedavgm", "extras": []},
        {
            "name": "uniform_weighting",
            "method": "dominance_aware",
            "extras": ["--dominance-mode", "uniform"],
        },
        {
            "name": "norm_clip_p75",
            "method": "dominance_aware",
            "extras": [
                "--dominance-mode",
                "norm_clip",
                "--dominance-clip-percentile",
                "0.75",
            ],
        },
        {
            "name": "contribution_cap_p75",
            "method": "dominance_aware",
            "extras": [
                "--dominance-mode",
                "contribution_cap",
                "--dominance-contribution-cap-percentile",
                "0.75",
            ],
        },
        {
            "name": "soft_tau0p5",
            "method": "dominance_aware",
            "extras": ["--dominance-mode", "soft_reweight", "--dominance-tau", "0.5"],
        },
        {
            "name": "soft_tau1p0",
            "method": "dominance_aware",
            "extras": ["--dominance-mode", "soft_reweight", "--dominance-tau", "1.0"],
        },
        {
            "name": "soft_tau2p0",
            "method": "dominance_aware",
            "extras": ["--dominance-mode", "soft_reweight", "--dominance-tau", "2.0"],
        },
        {
            "name": "soft_tau5p0",
            "method": "dominance_aware",
            "extras": ["--dominance-mode", "soft_reweight", "--dominance-tau", "5.0"],
        },
        {
            "name": "triggered_tau1_thr0p30",
            "method": "dominance_aware",
            "extras": [
                "--dominance-mode",
                "triggered_soft_reweight",
                "--dominance-tau",
                "1.0",
                "--dominance-threshold",
                "0.30",
            ],
        },
        {
            "name": "triggered_tau1_thr0p35",
            "method": "dominance_aware",
            "extras": [
                "--dominance-mode",
                "triggered_soft_reweight",
                "--dominance-tau",
                "1.0",
                "--dominance-threshold",
                "0.35",
            ],
        },
        {
            "name": "triggered_tau1_thr0p40",
            "method": "dominance_aware",
            "extras": [
                "--dominance-mode",
                "triggered_soft_reweight",
                "--dominance-tau",
                "1.0",
                "--dominance-threshold",
                "0.40",
            ],
        },
    ]
    if not selected_labels:
        return all_configs
    wanted = {str(x).strip().lower() for x in selected_labels if str(x).strip()}
    return [cfg for cfg in all_configs if str(cfg["name"]).strip().lower() in wanted]


def _base_cmd(args: argparse.Namespace, method: str, seed: int, run_tag: str) -> List[str]:
    return [
        args.python_bin,
        "run_general_experiment.py",
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
        "--server-learning-rate",
        str(args.server_learning_rate),
        "--server-momentum",
        str(args.server_momentum),
        "--out-dir",
        str(args.out_dir),
        "--run-tag",
        run_tag,
    ]


def _load_method_result(path: Path, method: str) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["results"][method]


def _build_report(
    *,
    report_path: Path,
    args: argparse.Namespace,
    commands: List[str],
    run_rows: List[Dict[str, Any]],
    round_rows: List[Dict[str, Any]],
    method_rows: List[Dict[str, Any]],
) -> None:
    by_method: Dict[str, List[Dict[str, Any]]] = {}
    for row in run_rows:
        by_method.setdefault(str(row["method_label"]), []).append(row)

    method_table = [["method", "seed42", "seed43", "seed44", "mean", "std"]]
    for method in sorted(by_method.keys()):
        seed_map = {int(r["seed"]): float(r["final_accuracy"]) for r in by_method[method]}
        values = [float(r["final_accuracy"]) for r in by_method[method]]
        method_table.append(
            [
                method,
                _fmt(seed_map.get(42, float("nan"))),
                _fmt(seed_map.get(43, float("nan"))),
                _fmt(seed_map.get(44, float("nan"))),
                _fmt(_mean(values)),
                _fmt(_std(values)),
            ]
        )

    corr_di_acc = _corr(
        [float(r["mean_DI_t_raw"]) for r in run_rows],
        [float(r["final_accuracy"]) for r in run_rows],
    )
    corr_neff_acc = _corr(
        [float(r["mean_N_eff_t_raw"]) for r in run_rows],
        [float(r["final_accuracy"]) for r in run_rows],
    )
    corr_di_reduction_acc = _corr(
        [float(r["mean_DI_reduction"]) for r in run_rows],
        [float(r["final_accuracy"]) for r in run_rows],
    )

    best_row = sorted(
        method_rows,
        key=lambda r: (_safe_float(r.get("final_accuracy_mean")), -_safe_float(r.get("final_accuracy_std"))),
        reverse=True,
    )[0]
    baseline = next((r for r in method_rows if r["method_label"] == "fedavgm"), None)

    lines: List[str] = []
    lines.append("# PHASE3_DOMINANCE_AWARE_REPORT")
    lines.append("")
    lines.append("## 1) Phase 1 and Phase 2 Conclusions")
    lines.append("- Phase 1: dominance showed the strongest negative association with final accuracy, conflict was weaker, cancellation was weak.")
    lines.append("- Phase 2: update-induced graph did not beat graph controls.")
    lines.append("- Phase 2 mean final accuracy: update 0.8008, random 0.8095, shuffled 0.8103, uniform 0.8087, identity 0.8108, FedAvgM 0.8123.")
    lines.append("")
    lines.append("## 2) Why Graph-Centered Direction Was Deprioritized")
    lines.append("- Graph-smoothing with update graph did not show consistent gains over random/shuffled/uniform/identity controls.")
    lines.append("- This phase focuses on direct dominance control in server aggregation without graph smoothing.")
    lines.append("")
    lines.append("## 3) Dominance Metrics")
    lines.append("- `q_i = p_i ||g_i^t||`")
    lines.append("- `qbar_i = q_i / (sum_j q_j + epsilon)`")
    lines.append("- `DI_t = max_i qbar_i`")
    lines.append("- `N_eff_t = 1 / (sum_i qbar_i^2 + epsilon)`")
    lines.append("")
    lines.append("## 4) Methods and Equations")
    lines.append("- FedAvgM baseline: `Delta_t = sum_i p_i g_i^t` (server momentum applied).")
    lines.append("- Uniform weighting: `Delta_t = sum_i (1/N) g_i^t`.")
    lines.append("- Norm clipping: `g_i_clipped = g_i * min(1, c / (||g_i|| + epsilon))`, `Delta_t = sum_i p_i g_i_clipped`.")
    lines.append("- Contribution cap: if `p_i||g_i|| > cap`, scale `g_i` to satisfy `p_i||g_i|| = cap`, then aggregate with `p_i`.")
    lines.append("- Soft reweighting: `alpha_i = p_i exp(-tau_d qbar_i) / sum_j p_j exp(-tau_d qbar_j)`, `Delta_t = sum_i alpha_i g_i`.")
    lines.append("- Triggered soft reweighting: same alpha only when `DI_t > threshold`, otherwise FedAvg weights.")
    lines.append("")
    lines.append("## 5) Commands Used")
    for cmd in commands:
        lines.append(f"- `{cmd}`")
    lines.append("")
    lines.append("## 6) Experiment Settings")
    lines.append(f"- dataset: `{args.dataset}`")
    lines.append(f"- partition: `{args.partition}` (`alpha={args.dirichlet_alpha}`)")
    lines.append(f"- num_clients: `{args.num_clients}`")
    lines.append(f"- rounds: `{args.rounds}`")
    lines.append(f"- seeds: `{', '.join(str(s) for s in args.seeds)}`")
    lines.append("")
    lines.append("## 7) Final Accuracy by Method and Seed")
    lines.append(_table(method_table))
    lines.append("")
    lines.append("## 8) Mean/Std Final Accuracy by Method")
    ms_rows = [["method", "mean_acc", "std_acc", "mean_DI_raw", "mean_DI_corrected", "mean_N_eff_raw", "mean_N_eff_corrected"]]
    for row in sorted(method_rows, key=lambda x: str(x["method_label"])):
        ms_rows.append(
            [
                str(row["method_label"]),
                _fmt(_safe_float(row.get("final_accuracy_mean"))),
                _fmt(_safe_float(row.get("final_accuracy_std"))),
                _fmt(_safe_float(row.get("mean_DI_t_raw"))),
                _fmt(_safe_float(row.get("mean_DI_t_corrected"))),
                _fmt(_safe_float(row.get("mean_N_eff_t_raw"))),
                _fmt(_safe_float(row.get("mean_N_eff_t_corrected"))),
            ]
        )
    lines.append(_table(ms_rows))
    lines.append("")
    lines.append("## 9) Round-wise Accuracy/Loss Curves")
    lines.append("- Round-level curves are saved in `experiments_current/phase3_dominance/phase3_round_diagnostics.csv` and `.json`.")
    lines.append("")
    lines.append("## 10) DI_t and N_eff_t Curves")
    lines.append("- Raw and corrected `DI_t` / `N_eff_t` are logged per round in the same round-diagnostics files.")
    lines.append("")
    lines.append("## 11) Whether Each Method Reduces Dominance")
    lines.append("- Positive `mean_DI_reduction` indicates lower dominance than raw FedAvg weighting.")
    lines.append("- Positive `mean_N_eff_gain` indicates broader effective participation.")
    lines.append(_table([["method", "mean_DI_reduction", "mean_N_eff_gain"]] + [[str(r["method_label"]), _fmt(_safe_float(r.get("mean_DI_reduction"))), _fmt(_safe_float(r.get("mean_N_eff_gain")))] for r in sorted(method_rows, key=lambda x: str(x["method_label"]))]))
    lines.append("")
    lines.append("## 12) Whether Reducing Dominance Improves Accuracy")
    lines.append(f"- corr(final_accuracy, mean_DI_t_raw): `{_fmt(corr_di_acc)}`")
    lines.append(f"- corr(final_accuracy, mean_N_eff_t_raw): `{_fmt(corr_neff_acc)}`")
    lines.append(f"- corr(final_accuracy, mean_DI_reduction): `{_fmt(corr_di_reduction_acc)}`")
    lines.append("")
    lines.append("## 13) Comparison Against Simple Baselines")
    if baseline is not None:
        lines.append(f"- FedAvgM mean accuracy: `{_fmt(_safe_float(baseline.get('final_accuracy_mean')))} ± {_fmt(_safe_float(baseline.get('final_accuracy_std')))}.`")
    lines.append("- Included simple baselines: uniform weighting, norm clipping, contribution cap.")
    lines.append("")
    lines.append("## 14) Recommendation")
    if baseline is None:
        lines.append("- Baseline row missing; rerun Phase 3 before interpretation.")
    else:
        best_name = str(best_row["method_label"])
        best_mean = _safe_float(best_row.get("final_accuracy_mean"))
        base_mean = _safe_float(baseline.get("final_accuracy_mean"))
        if best_name != "fedavgm" and best_mean > base_mean:
            lines.append(f"- Continue dominance-aware direction: `{best_name}` is currently best and exceeds FedAvgM.")
            lines.append("- Next: expand settings (alpha/client-count/seed sweep) and verify robustness.")
        else:
            lines.append("- Dominance-aware correction is weak in this setting (no clear gain over FedAvgM/simple baselines).")
            lines.append("- Prefer honest diagnostic-first reporting and consider conflict-aware add-on only after expanded checks.")
    lines.append("")
    lines.append("## Raw Outputs")
    lines.append("- `experiments_current/phase3_dominance/phase3_round_diagnostics.csv`")
    lines.append("- `experiments_current/phase3_dominance/phase3_run_summary.csv`")
    lines.append("- `experiments_current/phase3_dominance/phase3_method_comparison.csv`")
    lines.append("- `experiments_current/phase3_dominance/phase3_round_diagnostics.json`")
    lines.append("- `experiments_current/phase3_dominance/phase3_run_summary.json`")
    lines.append("")
    lines.append(f"_Generated at: {datetime.now().isoformat()}_")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Phase 3 dominance-aware FL ablation.")
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
    p.add_argument("--server-learning-rate", type=float, default=1.0)
    p.add_argument("--server-momentum", type=float, default=0.9)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "experiments_current" / "phase3_dominance",
    )
    p.add_argument("--reuse-existing-results", type=str, default="true")
    p.add_argument(
        "--from-existing-only",
        type=str,
        default="false",
        help="If true, do not run missing experiments; summarize only existing result files.",
    )
    p.add_argument(
        "--method-labels",
        type=str,
        nargs="*",
        default=[],
        help="Optional subset of method labels to execute/export.",
    )
    return p.parse_args()


def run() -> None:
    args = parse_args()
    reuse_existing = str(args.reuse_existing_results).strip().lower() in {
        "1",
        "true",
        "t",
        "yes",
        "y",
        "on",
    }
    from_existing_only = str(args.from_existing_only).strip().lower() in {
        "1",
        "true",
        "t",
        "yes",
        "y",
        "on",
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    commands: List[str] = []
    round_rows: List[Dict[str, Any]] = []
    run_rows: List[Dict[str, Any]] = []
    method_rows: List[Dict[str, Any]] = []

    selected = [m for m in args.method_labels if str(m).strip()]
    for cfg in _method_configs(selected_labels=selected):
        method_label = str(cfg["name"])
        flower_method = str(cfg["method"])
        extras = list(cfg["extras"])
        for seed in args.seeds:
            run_tag = f"phase3_{method_label}_seed{seed}"
            cmd = _base_cmd(
                args=args,
                method=flower_method,
                seed=int(seed),
                run_tag=run_tag,
            ) + extras
            commands.append(" ".join(cmd))
            result_path = _result_path(
                out_dir=out_dir,
                method=flower_method,
                seed=int(seed),
                run_tag=run_tag,
            )
            if not (reuse_existing and result_path.is_file()):
                if from_existing_only and (not result_path.is_file()):
                    continue
                _run(cmd)

            res = _load_method_result(result_path, flower_method)
            acc_series = res.get("metrics_distributed", {}).get("accuracy", [])
            loss_series = res.get("losses_distributed", [])
            final_acc = _safe_float(acc_series[-1][1]) if acc_series else float("nan")
            final_loss = _safe_float(loss_series[-1][1]) if loss_series else float("nan")
            trace = res.get("round_trace", [])

            di_raw_vals: List[float] = []
            di_corr_vals: List[float] = []
            neff_raw_vals: List[float] = []
            neff_corr_vals: List[float] = []
            dominance_switch_count = 0
            dominance_trigger_count = 0
            for row in trace:
                di_raw = _safe_float(row.get("dominance_ratio_raw", row.get("dominance_ratio")))
                di_corr = _safe_float(
                    row.get("dominance_ratio_corrected", row.get("dominance_ratio"))
                )
                neff_raw = _safe_float(
                    row.get("effective_num_clients_raw", row.get("effective_num_clients"))
                )
                neff_corr = _safe_float(
                    row.get(
                        "effective_num_clients_corrected", row.get("effective_num_clients")
                    )
                )
                di_raw_vals.append(di_raw)
                di_corr_vals.append(di_corr)
                neff_raw_vals.append(neff_raw)
                neff_corr_vals.append(neff_corr)
                if bool(row.get("dominant_client_changed", False)):
                    dominance_switch_count += 1
                if bool(row.get("dominance_triggered", False)):
                    dominance_trigger_count += 1
                round_rows.append(
                    {
                        "method_label": method_label,
                        "method": flower_method,
                        "seed": int(seed),
                        "round": int(row.get("round", -1)),
                        "accuracy": _safe_float(row.get("accuracy")),
                        "loss": _safe_float(row.get("loss")),
                        "final_accuracy": final_acc,
                        "final_loss": final_loss,
                        "DI_t_raw": di_raw,
                        "N_eff_t_raw": neff_raw,
                        "DI_t_corrected": di_corr,
                        "N_eff_t_corrected": neff_corr,
                        "max_qbar_i_raw": _safe_float(
                            row.get("dominance_ratio_raw", row.get("dominance_ratio"))
                        ),
                        "dominant_client_id": str(
                            row.get("dominant_client_id", row.get("dominant_client_index", ""))
                        ),
                        "dominant_client_index": _safe_float(row.get("dominant_client_index")),
                        "dominant_client_changed": bool(
                            row.get("dominant_client_changed", False)
                        ),
                        "dominance_triggered": bool(row.get("dominance_triggered", False)),
                        "qbar_i_distribution_raw": row.get(
                            "client_contribution_normalized_raw",
                            row.get("client_contribution_normalized", []),
                        ),
                        "qbar_i_distribution_corrected": row.get(
                            "client_contribution_normalized_corrected",
                            row.get("client_contribution_normalized", []),
                        ),
                        "client_update_norm_mean": _safe_float(row.get("client_update_norm_mean")),
                        "client_update_norm_std": _safe_float(row.get("client_update_norm_std")),
                        "client_update_norm_max": _safe_float(row.get("client_update_norm_max")),
                        "delta_norm_raw_fedavg": _safe_float(
                            row.get("fedavg_delta_norm", row.get("delta_norm"))
                        ),
                        "delta_norm_corrected": _safe_float(
                            row.get("corrected_delta_norm", row.get("delta_norm"))
                        ),
                        "delta_norm_ratio_corrected_over_raw": _safe_float(
                            row.get("delta_corrected_over_fedavg_norm")
                        ),
                        "cosine_raw_vs_corrected": _safe_float(
                            row.get("cosine_raw_vs_corrected_delta")
                        ),
                        "cosine_dominant_vs_raw": _safe_float(
                            row.get("cosine_dominant_vs_raw_delta")
                        ),
                        "cosine_dominant_vs_corrected": _safe_float(
                            row.get("cosine_dominant_vs_corrected_delta")
                        ),
                        "conflict_ratio": _safe_float(row.get("conflict_ratio")),
                        "cancellation_ratio": _safe_float(row.get("cancellation_ratio")),
                        "pairwise_cosine_mean": _safe_float(row.get("pairwise_cosine_mean")),
                        "pairwise_cosine_min": _safe_float(row.get("pairwise_cosine_min")),
                        "pairwise_cosine_std": _safe_float(row.get("pairwise_cosine_std")),
                    }
                )

            run_rows.append(
                {
                    "method_label": method_label,
                    "method": flower_method,
                    "seed": int(seed),
                    "final_accuracy": final_acc,
                    "final_loss": final_loss,
                    "mean_DI_t_raw": _mean(di_raw_vals),
                    "mean_N_eff_t_raw": _mean(neff_raw_vals),
                    "mean_DI_t_corrected": _mean(di_corr_vals),
                    "mean_N_eff_t_corrected": _mean(neff_corr_vals),
                    "mean_DI_reduction": _mean(di_raw_vals) - _mean(di_corr_vals),
                    "mean_N_eff_gain": _mean(neff_corr_vals) - _mean(neff_raw_vals),
                    "dominant_switch_count": int(dominance_switch_count),
                    "trigger_count": int(dominance_trigger_count),
                    "round_count": int(len(trace)),
                }
            )

    by_method: Dict[str, List[Dict[str, Any]]] = {}
    for row in run_rows:
        by_method.setdefault(str(row["method_label"]), []).append(row)
    for method_label, rows in by_method.items():
        method_rows.append(
            {
                "method_label": method_label,
                "n_runs": len(rows),
                "final_accuracy_mean": _mean([_safe_float(r["final_accuracy"]) for r in rows]),
                "final_accuracy_std": _std([_safe_float(r["final_accuracy"]) for r in rows]),
                "final_loss_mean": _mean([_safe_float(r["final_loss"]) for r in rows]),
                "mean_DI_t_raw": _mean([_safe_float(r["mean_DI_t_raw"]) for r in rows]),
                "mean_DI_t_corrected": _mean(
                    [_safe_float(r["mean_DI_t_corrected"]) for r in rows]
                ),
                "mean_N_eff_t_raw": _mean([_safe_float(r["mean_N_eff_t_raw"]) for r in rows]),
                "mean_N_eff_t_corrected": _mean(
                    [_safe_float(r["mean_N_eff_t_corrected"]) for r in rows]
                ),
                "mean_DI_reduction": _mean([_safe_float(r["mean_DI_reduction"]) for r in rows]),
                "mean_N_eff_gain": _mean([_safe_float(r["mean_N_eff_gain"]) for r in rows]),
                "mean_trigger_count": _mean([_safe_float(r["trigger_count"]) for r in rows]),
            }
        )

    _write_csv(out_dir / "phase3_round_diagnostics.csv", round_rows)
    _write_json(out_dir / "phase3_round_diagnostics.json", round_rows)
    _write_csv(out_dir / "phase3_run_summary.csv", run_rows)
    _write_json(out_dir / "phase3_run_summary.json", run_rows)
    _write_csv(out_dir / "phase3_method_comparison.csv", method_rows)

    report_path = PROJECT_ROOT / "PHASE3_DOMINANCE_AWARE_REPORT.md"
    _build_report(
        report_path=report_path,
        args=args,
        commands=commands,
        run_rows=run_rows,
        round_rows=round_rows,
        method_rows=method_rows,
    )
    print(f"Saved: {report_path}")
    print(f"Saved: {out_dir / 'phase3_round_diagnostics.csv'}")
    print(f"Saved: {out_dir / 'phase3_run_summary.csv'}")
    print(f"Saved: {out_dir / 'phase3_method_comparison.csv'}")


if __name__ == "__main__":
    run()
