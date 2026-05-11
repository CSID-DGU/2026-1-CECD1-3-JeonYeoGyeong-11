"""Phase 2.5 batch runner with resumable incremental summaries."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _safe_float(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _mean(vals: Iterable[float]) -> float:
    arr = [float(v) for v in vals if not math.isnan(float(v))]
    if not arr:
        return float("nan")
    return float(sum(arr) / len(arr))


def _std(vals: Iterable[float]) -> float:
    arr = [float(v) for v in vals if not math.isnan(float(v))]
    if len(arr) < 2:
        return 0.0 if arr else float("nan")
    return float(statistics.pstdev(arr))


def _fmt(v: float) -> str:
    if math.isnan(float(v)):
        return "nan"
    return f"{float(v):.4f}"


def _table(rows: List[List[str]]) -> str:
    if not rows:
        return ""
    out = [
        "| " + " | ".join(rows[0]) + " |",
        "| " + " | ".join(["---"] * len(rows[0])) + " |",
    ]
    for row in rows[1:]:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, allow_nan=True), encoding="utf-8")


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: List[str] = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _read_json_list(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def _dedupe_run_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    keep: Dict[Tuple[str, str, str, int], Dict[str, Any]] = {}
    for row in rows:
        key = (
            str(row.get("experiment")),
            str(row.get("setting")),
            str(row.get("method_label")),
            int(row.get("seed", -1)),
        )
        keep[key] = row
    return [keep[k] for k in sorted(keep.keys())]


def _dedupe_round_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    keep: Dict[Tuple[str, str, str, int, int], Dict[str, Any]] = {}
    for row in rows:
        key = (
            str(row.get("experiment")),
            str(row.get("setting")),
            str(row.get("method_label")),
            int(row.get("seed", -1)),
            int(row.get("round", -1)),
        )
        keep[key] = row
    return [keep[k] for k in sorted(keep.keys())]


def _effect_rows(run_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_key: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        by_key[
            (str(row.get("experiment")), str(row.get("setting")), str(row.get("method_label")))
        ].append(row)
    out: List[Dict[str, Any]] = []
    for (exp, setting, method), rows in sorted(by_key.items()):
        out.append(
            {
                "experiment": exp,
                "setting": setting,
                "method_label": method,
                "n_runs": len(rows),
                "final_accuracy_mean": _mean([_safe_float(r.get("final_accuracy")) for r in rows]),
                "final_accuracy_std": _std([_safe_float(r.get("final_accuracy")) for r in rows]),
                "direction_change_mean": _mean(
                    [_safe_float(r.get("mean_direction_change")) for r in rows]
                ),
                "direction_change_std": _std(
                    [_safe_float(r.get("mean_direction_change")) for r in rows]
                ),
                "DI_reduction_mean": _mean([_safe_float(r.get("mean_DI_reduction")) for r in rows]),
                "N_eff_gain_mean": _mean([_safe_float(r.get("mean_N_eff_gain")) for r in rows]),
            }
        )
    return out


def _result_path(out_dir: Path, method: str, seed: int, run_tag: str) -> Path:
    return out_dir / f"result_general_{method}_seed{seed}_{run_tag}.json"


def _to_cli(args: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for k, v in args.items():
        out += [f"--{k}", str(v)]
    return out


def _base_args(dataset: str, alpha: float, n_clients: int, rounds: int, out_dir: Path) -> Dict[str, Any]:
    return {
        "dataset": dataset,
        "partition": "dirichlet",
        "dirichlet-alpha": alpha,
        "num-clients": n_clients,
        "rounds": rounds,
        "local-epochs": 1,
        "batch-size": 64,
        "model": "cnn",
        "lr": 0.01,
        "momentum": 0.9,
        "weight-decay": 5e-4,
        "graph-source": "classifier_head_update",
        "compression-dim": 256,
        "compression-seed": 0,
        "graph-seed": 0,
        "server-learning-rate": 1.0,
        "server-momentum": 0.9,
        "out-dir": str(out_dir),
    }


def _reset_ray(python_bin: str) -> None:
    # Best-effort cleanup between configurations.
    subprocess.run(
        [python_bin, "-m", "ray", "stop", "--force"],
        cwd=str(PROJECT_ROOT),
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _run_one(cmd: Sequence[str], python_bin: str) -> None:
    env = dict(os.environ)
    env.update({"RAY_raylet_start_wait_time_s": "120"})
    _reset_ray(python_bin)
    subprocess.run(list(cmd), cwd=str(PROJECT_ROOT), check=True, env=env)


def _collect_rows_from_result(
    payload: Dict[str, Any],
    *,
    experiment: str,
    setting: str,
    method_label: str,
    operator: str,
    graph_variant: str,
    graph_source: str,
    graph_lambda: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    meta = payload.get("meta", {}).get("experiment", {})
    method_key = (
        "graph_smooth"
        if "graph_smooth" in payload.get("results", {})
        else ("dominance_aware" if "dominance_aware" in payload.get("results", {}) else "fedavgm")
    )
    result = payload.get("results", {}).get(method_key, {})
    seed = int(meta.get("seed", -1))
    alpha = _safe_float(meta.get("dirichlet_alpha"))
    n_clients = int(meta.get("num_clients", -1))
    trace = result.get("round_trace", [])
    acc_series = result.get("metrics_distributed", {}).get("accuracy", [])
    loss_series = result.get("losses_distributed", [])
    final_acc = _safe_float(acc_series[-1][1]) if acc_series else float("nan")
    final_loss = _safe_float(loss_series[-1][1]) if loss_series else float("nan")

    round_rows: List[Dict[str, Any]] = []
    di_raw: List[float] = []
    di_corr: List[float] = []
    neff_raw: List[float] = []
    neff_corr: List[float] = []
    direction: List[float] = []
    rel_change: List[float] = []

    for row in trace:
        di_r = _safe_float(row.get("dominance_ratio_raw", row.get("dominance_ratio")))
        di_c = _safe_float(row.get("dominance_ratio_corrected", row.get("dominance_ratio")))
        ne_r = _safe_float(row.get("effective_num_clients_raw", row.get("effective_num_clients")))
        ne_c = _safe_float(row.get("effective_num_clients_corrected", row.get("effective_num_clients")))
        cos_rc = _safe_float(
            row.get("corrected_vs_fedavg_delta_cosine", row.get("cosine_raw_vs_corrected_delta"))
        )
        dir_chg = _safe_float(row.get("direction_change_one_minus_cosine", 1.0 - cos_rc))
        rel = _safe_float(row.get("relative_delta_change"))
        di_raw.append(di_r)
        di_corr.append(di_c)
        neff_raw.append(ne_r)
        neff_corr.append(ne_c)
        direction.append(dir_chg)
        rel_change.append(rel)
        round_rows.append(
            {
                "phase": "phase2_5",
                "experiment": experiment,
                "setting": setting,
                "seed": seed,
                "round": int(row.get("round", -1)),
                "method_label": method_label,
                "operator": operator,
                "graph_variant": graph_variant,
                "graph_source": graph_source,
                "graph_lambda": graph_lambda,
                "accuracy": _safe_float(row.get("accuracy")),
                "loss": _safe_float(row.get("loss")),
                "CR_t": _safe_float(row.get("conflict_ratio")),
                "CR_weighted_t": _safe_float(row.get("conflict_ratio_weighted")),
                "CA_t": _safe_float(row.get("cancellation_ratio")),
                "DI_t_raw": di_r,
                "DI_t_corrected": di_c,
                "N_eff_t_raw": ne_r,
                "N_eff_t_corrected": ne_c,
                "pairwise_cos_mean": _safe_float(row.get("pairwise_cosine_mean")),
                "pairwise_cos_min": _safe_float(row.get("pairwise_cosine_min")),
                "pairwise_cos_std": _safe_float(row.get("pairwise_cosine_std")),
                "client_update_norm_mean": _safe_float(row.get("client_update_norm_mean")),
                "client_update_norm_std": _safe_float(row.get("client_update_norm_std")),
                "client_update_norm_max": _safe_float(row.get("client_update_norm_max")),
                "delta_raw_norm": _safe_float(row.get("fedavg_delta_norm", row.get("delta_norm"))),
                "delta_corr_norm": _safe_float(row.get("corrected_delta_norm", row.get("delta_norm"))),
                "delta_corr_raw_norm_ratio": _safe_float(
                    row.get("corrected_over_raw_norm_ratio", row.get("delta_corrected_over_fedavg_norm"))
                ),
                "delta_corr_raw_cosine": cos_rc,
                "direction_change": dir_chg,
                "relative_delta_change": rel,
                "graph_density": _safe_float(row.get("graph_density")),
                "graph_edge_weight_mean": _safe_float(row.get("graph_edge_weight_mean")),
                "graph_edge_weight_std": _safe_float(row.get("graph_edge_weight_std")),
                "graph_edge_weight_min": _safe_float(row.get("graph_edge_weight_min")),
                "graph_edge_weight_max": _safe_float(row.get("graph_edge_weight_max")),
                "graph_degree_mean": _safe_float(row.get("graph_degree_mean")),
                "graph_degree_std": _safe_float(row.get("graph_degree_std")),
                "graph_smoothness": _safe_float(row.get("graph_smoothness")),
            }
        )

    run_row = {
        "phase": "phase2_5",
        "experiment": experiment,
        "setting": setting,
        "seed": seed,
        "alpha": alpha,
        "num_clients": n_clients,
        "method_label": method_label,
        "operator": operator,
        "graph_variant": graph_variant,
        "graph_lambda": graph_lambda,
        "final_accuracy": final_acc,
        "final_loss": final_loss,
        "mean_DI_t_raw": _mean(di_raw),
        "mean_DI_t_corrected": _mean(di_corr),
        "mean_N_eff_t_raw": _mean(neff_raw),
        "mean_N_eff_t_corrected": _mean(neff_corr),
        "mean_DI_reduction": _mean(di_raw) - _mean(di_corr),
        "mean_N_eff_gain": _mean(neff_corr) - _mean(neff_raw),
        "mean_direction_change": _mean(direction),
        "mean_relative_delta_change": _mean(rel_change),
    }
    return round_rows, run_row


def _load_phase2_l05(
    phase2_dir: Path,
    *,
    variant: str,
    seed: int,
) -> Dict[str, Any] | None:
    p = phase2_dir / (
        f"result_general_graph_smooth_seed{seed}_"
        f"phase2_graphsmooth_{variant}_seed{seed}_l0p05.json"
    )
    if p.is_file():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


def _save_all(
    *,
    out_dir: Path,
    report_path: Path,
    run_rows: List[Dict[str, Any]],
    round_rows: List[Dict[str, Any]],
    status: Dict[str, Any],
) -> None:
    run_rows = _dedupe_run_rows(run_rows)
    round_rows = _dedupe_round_rows(round_rows)
    effect_rows = _effect_rows(run_rows)

    _write_json(out_dir / "phase2_5_run_summary.json", run_rows)
    _write_csv(out_dir / "phase2_5_run_summary.csv", run_rows)
    _write_json(out_dir / "phase2_5_round_diagnostics.json", round_rows)
    _write_csv(out_dir / "phase2_5_round_diagnostics.csv", round_rows)
    _write_csv(out_dir / "phase2_5_effect_size.csv", effect_rows)
    _write_json(out_dir / "phase2_5_batch_status.json", status)

    completed = sorted([k for k, v in status.get("batches", {}).items() if v == "complete"])
    pending = sorted([k for k, v in status.get("batches", {}).items() if v != "complete"])
    lines: List[str] = []
    lines.append("# PHASE2_5_SMOOTHING_FAILURE_ANALYSIS")
    lines.append("")
    lines.append("## Batch Status")
    lines.append(f"- completed: {', '.join(completed) if completed else 'none'}")
    lines.append(f"- pending: {', '.join(pending) if pending else 'none'}")
    if status.get("failed_runs"):
        lines.append(f"- failed_runs: {len(status['failed_runs'])} (see `phase2_5_batch_status.json`)")
    lines.append("")
    lines.append("## Effect Summary")
    lines.append(
        _table(
            [["experiment", "setting", "method", "mean_acc", "std_acc", "direction_change"]]
            + [
                [
                    str(r["experiment"]),
                    str(r["setting"]),
                    str(r["method_label"]),
                    _fmt(_safe_float(r["final_accuracy_mean"])),
                    _fmt(_safe_float(r["final_accuracy_std"])),
                    _fmt(_safe_float(r["direction_change_mean"])),
                ]
                for r in effect_rows
            ]
        )
    )
    lines.append("")
    lines.append("## Notes")
    lines.append("- Report is incremental and resumable.")
    lines.append("- Each batch merges into summary files without rerunning existing result files.")
    lines.append("")
    lines.append(f"_Generated at: {datetime.now().isoformat()}_")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 2.5 batch runner.")
    p.add_argument("--python-bin", type=str, default=sys.executable)
    p.add_argument("--dataset", type=str, default="fashionmnist")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--batch", type=str, default="A", choices=["A", "B", "C", "D", "E", "all"])
    p.add_argument("--phase2-dir", type=Path, default=PROJECT_ROOT / "experiments_current" / "phase2_graph_informativeness")
    p.add_argument("--reuse-existing-results", type=str, default="true")
    p.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "experiments_current" / "phase2_5_smoothing_failure")
    p.add_argument("--report-path", type=Path, default=PROJECT_ROOT / "PHASE2_5_SMOOTHING_FAILURE_ANALYSIS.md")
    return p.parse_args()


def _execute_one(
    *,
    run_rows: List[Dict[str, Any]],
    round_rows: List[Dict[str, Any]],
    status: Dict[str, Any],
    out_dir: Path,
    report_path: Path,
    reuse: bool,
    python_bin: str,
    method: str,
    seed: int,
    run_tag: str,
    cmd_args: Dict[str, Any],
    row_meta: Dict[str, Any],
) -> None:
    rp = _result_path(out_dir, method, seed, run_tag)
    cmd = [python_bin, "run_general_experiment.py"] + _to_cli(cmd_args)
    try:
        if not (reuse and rp.is_file()):
            _run_one(cmd, python_bin)
        payload = json.loads(rp.read_text(encoding="utf-8"))
        rr, r = _collect_rows_from_result(payload, **row_meta)
        round_rows.extend(rr)
        run_rows.append(r)
    except Exception as exc:
        status.setdefault("failed_runs", []).append(
            {
                "run_tag": run_tag,
                "method": method,
                "seed": seed,
                "error": repr(exc),
                "command": " ".join(cmd),
            }
        )
    finally:
        _save_all(
            out_dir=out_dir,
            report_path=report_path,
            run_rows=run_rows,
            round_rows=round_rows,
            status=status,
        )


def main() -> None:
    args = parse_args()
    reuse = str(args.reuse_existing_results).strip().lower() in {"1", "true", "t", "yes", "y", "on"}
    args.out_dir.mkdir(parents=True, exist_ok=True)

    run_rows = _read_json_list(args.out_dir / "phase2_5_run_summary.json")
    round_rows = _read_json_list(args.out_dir / "phase2_5_round_diagnostics.json")
    status = {
        "batches": {"A": "pending", "B": "pending", "C": "pending", "D": "pending", "E": "pending"},
        "failed_runs": [],
    }
    status_path = args.out_dir / "phase2_5_batch_status.json"
    if status_path.is_file():
        try:
            loaded = json.loads(status_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                status.update(loaded)
        except json.JSONDecodeError:
            pass

    batches = ["A", "B", "C", "D", "E"] if args.batch == "all" else [args.batch]
    variants = ["update", "shuffled", "uniform", "identity"]

    for batch in batches:
        if status.get("batches", {}).get(batch) == "complete":
            continue
        if batch == "E":
            prereq = status.get("batches", {})
            if not (
                prereq.get("A") == "complete"
                and prereq.get("B") == "complete"
                and prereq.get("C") == "complete"
                and prereq.get("D") == "complete"
            ):
                status["batches"]["E"] = "pending_prereq"
                _save_all(
                    out_dir=args.out_dir,
                    report_path=args.report_path,
                    run_rows=run_rows,
                    round_rows=round_rows,
                    status=status,
                )
                continue

        try:
            if batch == "A":
                base = _base_args(args.dataset, 0.03, 5, args.rounds, args.out_dir)
                for seed in args.seeds:
                    tag = f"p25_A_fedavgm_s{seed}"
                    _execute_one(
                        run_rows=run_rows,
                        round_rows=round_rows,
                        status=status,
                        out_dir=args.out_dir,
                        report_path=args.report_path,
                        reuse=reuse,
                        python_bin=args.python_bin,
                        method="fedavgm",
                        seed=seed,
                        run_tag=tag,
                        cmd_args={**base, "method": "fedavgm", "seed": seed, "run-tag": tag},
                        row_meta={
                            "experiment": "A_laplacian",
                            "setting": "fashion_alpha0p03_n5",
                            "method_label": "A_fedavgm_baseline",
                            "operator": "fedavgm",
                            "graph_variant": "identity",
                            "graph_source": "none",
                            "graph_lambda": float("nan"),
                        },
                    )
                for lam in [0.2, 0.5]:
                    for variant in variants:
                        for seed in args.seeds:
                            tag = f"p25_A_lap_l{str(lam).replace('.', 'p')}_{variant}_s{seed}"
                            _execute_one(
                                run_rows=run_rows,
                                round_rows=round_rows,
                                status=status,
                                out_dir=args.out_dir,
                                report_path=args.report_path,
                                reuse=reuse,
                                python_bin=args.python_bin,
                                method="graph_smooth",
                                seed=seed,
                                run_tag=tag,
                                cmd_args={
                                    **base,
                                    "method": "graph_smooth",
                                    "seed": seed,
                                    "run-tag": tag,
                                    "graph-variant": variant,
                                    "graph-smoothing-operator": "laplacian",
                                    "graph-smoothing-lambda": lam,
                                    "graph-dominance-mode": "sample",
                                },
                                row_meta={
                                    "experiment": "A_laplacian",
                                    "setting": "fashion_alpha0p03_n5",
                                    "method_label": f"A_lap_l{str(lam).replace('.', 'p')}_{variant}",
                                    "operator": "laplacian",
                                    "graph_variant": variant,
                                    "graph_source": "classifier_head_update",
                                    "graph_lambda": lam,
                                },
                            )
            elif batch == "B":
                base = _base_args(args.dataset, 0.03, 5, args.rounds, args.out_dir)
                for variant in variants:
                    for seed in args.seeds:
                        tag = f"p25_B_res_l0p5_{variant}_s{seed}"
                        _execute_one(
                            run_rows=run_rows,
                            round_rows=round_rows,
                            status=status,
                            out_dir=args.out_dir,
                            report_path=args.report_path,
                            reuse=reuse,
                            python_bin=args.python_bin,
                            method="graph_smooth",
                            seed=seed,
                            run_tag=tag,
                            cmd_args={
                                **base,
                                "method": "graph_smooth",
                                "seed": seed,
                                "run-tag": tag,
                                "graph-variant": variant,
                                "graph-smoothing-operator": "residual",
                                "graph-smoothing-lambda": 0.5,
                                "graph-dominance-mode": "sample",
                            },
                            row_meta={
                                "experiment": "B_residual",
                                "setting": "fashion_alpha0p03_n5",
                                "method_label": f"B_res_l0p5_{variant}",
                                "operator": "residual",
                                "graph_variant": variant,
                                "graph_source": "classifier_head_update",
                                "graph_lambda": 0.5,
                            },
                        )
            elif batch == "C":
                base = _base_args(args.dataset, 0.03, 5, args.rounds, args.out_dir)
                for seed in args.seeds:
                    tag = f"p25_C_plain_res_update_s{seed}"
                    _execute_one(
                        run_rows=run_rows,
                        round_rows=round_rows,
                        status=status,
                        out_dir=args.out_dir,
                        report_path=args.report_path,
                        reuse=reuse,
                        python_bin=args.python_bin,
                        method="graph_smooth",
                        seed=seed,
                        run_tag=tag,
                        cmd_args={
                            **base,
                            "method": "graph_smooth",
                            "seed": seed,
                            "run-tag": tag,
                            "graph-variant": "update",
                            "graph-smoothing-operator": "residual",
                            "graph-smoothing-lambda": 0.5,
                            "graph-dominance-mode": "sample",
                        },
                        row_meta={
                            "experiment": "C_dominance_smoothing",
                            "setting": "fashion_alpha0p03_n5",
                            "method_label": "C_plain_update_residual",
                            "operator": "residual",
                            "graph_variant": "update",
                            "graph_source": "classifier_head_update",
                            "graph_lambda": 0.5,
                        },
                    )
                for variant, label in [
                    ("update", "C_update_dom"),
                    ("shuffled", "C_shuffled_dom"),
                    ("identity", "C_identity"),
                ]:
                    for seed in args.seeds:
                        if variant == "identity":
                            tag = f"p25_C_identity_s{seed}"
                            cmd_args = {
                                **base,
                                "method": "graph_smooth",
                                "seed": seed,
                                "run-tag": tag,
                                "graph-variant": "identity",
                                "graph-smoothing-operator": "residual",
                                "graph-smoothing-lambda": 0.5,
                                "graph-dominance-mode": "sample",
                            }
                            op = "identity"
                        else:
                            tag = f"p25_C_domres_{variant}_s{seed}"
                            cmd_args = {
                                **base,
                                "method": "graph_smooth",
                                "seed": seed,
                                "run-tag": tag,
                                "graph-variant": variant,
                                "graph-smoothing-operator": "dominance_residual",
                                "graph-smoothing-lambda": 0.5,
                                "graph-dominance-gamma": 1.0,
                                "graph-dominance-mode": "sample",
                            }
                            op = "dominance_residual"
                        _execute_one(
                            run_rows=run_rows,
                            round_rows=round_rows,
                            status=status,
                            out_dir=args.out_dir,
                            report_path=args.report_path,
                            reuse=reuse,
                            python_bin=args.python_bin,
                            method="graph_smooth",
                            seed=seed,
                            run_tag=tag,
                            cmd_args=cmd_args,
                            row_meta={
                                "experiment": "C_dominance_smoothing",
                                "setting": "fashion_alpha0p03_n5",
                                "method_label": label,
                                "operator": op,
                                "graph_variant": variant,
                                "graph_source": "classifier_head_update",
                                "graph_lambda": 0.5,
                            },
                        )
            elif batch == "D":
                base = _base_args(args.dataset, 0.03, 5, args.rounds, args.out_dir)
                for seed in args.seeds:
                    tag = f"p25_D_fedavgm_s{seed}"
                    _execute_one(
                        run_rows=run_rows,
                        round_rows=round_rows,
                        status=status,
                        out_dir=args.out_dir,
                        report_path=args.report_path,
                        reuse=reuse,
                        python_bin=args.python_bin,
                        method="fedavgm",
                        seed=seed,
                        run_tag=tag,
                        cmd_args={**base, "method": "fedavgm", "seed": seed, "run-tag": tag},
                        row_meta={
                            "experiment": "D_dominance_only",
                            "setting": "fashion_alpha0p03_n5",
                            "method_label": "D_fedavgm",
                            "operator": "fedavgm",
                            "graph_variant": "identity",
                            "graph_source": "none",
                            "graph_lambda": float("nan"),
                        },
                    )
                configs = [
                    ("uniform", "D_uniform", 0.0, 0.0),
                    ("contribution_cap", "D_cap_k2", 0.0, 2.0),
                    ("soft_reweight", "D_soft_tau5", 5.0, 0.0),
                    ("soft_reweight", "D_soft_tau10", 10.0, 0.0),
                ]
                for mode, label, tau, kappa in configs:
                    for seed in args.seeds:
                        tag = f"p25_D_{label}_s{seed}"
                        _execute_one(
                            run_rows=run_rows,
                            round_rows=round_rows,
                            status=status,
                            out_dir=args.out_dir,
                            report_path=args.report_path,
                            reuse=reuse,
                            python_bin=args.python_bin,
                            method="dominance_aware",
                            seed=seed,
                            run_tag=tag,
                            cmd_args={
                                **base,
                                "method": "dominance_aware",
                                "seed": seed,
                                "run-tag": tag,
                                "dominance-mode": mode,
                                "dominance-tau": tau,
                                "dominance-contribution-cap-kappa": kappa,
                            },
                            row_meta={
                                "experiment": "D_dominance_only",
                                "setting": "fashion_alpha0p03_n5",
                                "method_label": label,
                                "operator": mode,
                                "graph_variant": "identity",
                                "graph_source": "none",
                                "graph_lambda": float("nan"),
                            },
                        )
            elif batch == "E":
                # Pick most informative operator from completed A/B.
                effects = _effect_rows(run_rows)
                candidates = [e for e in effects if str(e.get("experiment")) in {"A_laplacian", "B_residual"}]
                chosen = "residual"
                if candidates:
                    top = sorted(
                        candidates,
                        key=lambda x: _safe_float(x.get("direction_change_mean")),
                        reverse=True,
                    )[0]
                    chosen = "laplacian" if str(top.get("experiment")) == "A_laplacian" else "residual"
                for alpha, n_clients in [(0.1, 5), (0.1, 10)]:
                    setting = f"fashion_alpha{str(alpha).replace('.', 'p')}_n{n_clients}"
                    base = _base_args(args.dataset, alpha, n_clients, args.rounds, args.out_dir)
                    for seed in args.seeds:
                        tag = f"p25_E_fedavgm_a{str(alpha).replace('.', 'p')}_n{n_clients}_s{seed}"
                        _execute_one(
                            run_rows=run_rows,
                            round_rows=round_rows,
                            status=status,
                            out_dir=args.out_dir,
                            report_path=args.report_path,
                            reuse=reuse,
                            python_bin=args.python_bin,
                            method="fedavgm",
                            seed=seed,
                            run_tag=tag,
                            cmd_args={**base, "method": "fedavgm", "seed": seed, "run-tag": tag},
                            row_meta={
                                "experiment": "E_robustness",
                                "setting": setting,
                                "method_label": "E_fedavgm_baseline",
                                "operator": "fedavgm",
                                "graph_variant": "identity",
                                "graph_source": "none",
                                "graph_lambda": float("nan"),
                            },
                        )
                    for variant in variants:
                        for seed in args.seeds:
                            tag = f"p25_E_{chosen}_a{str(alpha).replace('.', 'p')}_n{n_clients}_{variant}_s{seed}"
                            _execute_one(
                                run_rows=run_rows,
                                round_rows=round_rows,
                                status=status,
                                out_dir=args.out_dir,
                                report_path=args.report_path,
                                reuse=reuse,
                                python_bin=args.python_bin,
                                method="graph_smooth",
                                seed=seed,
                                run_tag=tag,
                                cmd_args={
                                    **base,
                                    "method": "graph_smooth",
                                    "seed": seed,
                                    "run-tag": tag,
                                    "graph-variant": variant,
                                    "graph-smoothing-operator": chosen,
                                    "graph-smoothing-lambda": 0.5,
                                    "graph-dominance-mode": "sample",
                                },
                                row_meta={
                                    "experiment": "E_robustness",
                                    "setting": setting,
                                    "method_label": f"E_{chosen}_l0p5_{variant}",
                                    "operator": chosen,
                                    "graph_variant": variant,
                                    "graph_source": "classifier_head_update",
                                    "graph_lambda": 0.5,
                                },
                            )
            status["batches"][batch] = "complete"
        except Exception as exc:
            status.setdefault("failed_runs", []).append({"batch": batch, "error": repr(exc)})
            status["batches"][batch] = "failed"
        finally:
            _save_all(
                out_dir=args.out_dir,
                report_path=args.report_path,
                run_rows=run_rows,
                round_rows=round_rows,
                status=status,
            )

    print(f"Saved: {args.report_path}")
    print(f"Saved: {args.out_dir / 'phase2_5_run_summary.csv'}")
    print(f"Saved: {args.out_dir / 'phase2_5_round_diagnostics.csv'}")
    print(f"Saved: {args.out_dir / 'phase2_5_effect_size.csv'}")


if __name__ == "__main__":
    main()
