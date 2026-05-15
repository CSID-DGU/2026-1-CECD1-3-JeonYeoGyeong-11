"""Build Phase-1 interaction diagnostics report from result JSON files."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out


def _mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if not math.isnan(float(v))]
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


def _std(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if not math.isnan(float(v))]
    if len(vals) < 2:
        return 0.0 if vals else float("nan")
    mu = sum(vals) / len(vals)
    return float(math.sqrt(sum((v - mu) ** 2 for v in vals) / len(vals)))


def _corr(x: List[float], y: List[float]) -> float:
    pairs = [(float(a), float(b)) for a, b in zip(x, y) if not (math.isnan(a) or math.isnan(b))]
    if len(pairs) < 3:
        return float("nan")
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    vx = sum((v - mx) ** 2 for v in xs)
    vy = sum((v - my) ** 2 for v in ys)
    if vx <= 0.0 or vy <= 0.0:
        return float("nan")
    cov = sum((a - mx) * (b - my) for a, b in pairs)
    return float(cov / math.sqrt(vx * vy))


def _load_result(path: Path) -> Tuple[Dict[str, Any], str]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    methods = list((obj.get("results") or {}).keys())
    if len(methods) != 1:
        raise ValueError(f"Expected exactly one method in {path}, got {methods}")
    return obj, methods[0]


def _format(v: float, digits: int = 4) -> str:
    if math.isnan(v):
        return "nan"
    return f"{v:.{digits}f}"


def build_report(
    input_dir: Path,
    report_path: Path,
    round_csv_path: Path,
    run_csv_path: Path,
    command_used: str,
) -> None:
    result_files = sorted(input_dir.glob("result_general_*.json"))
    if not result_files:
        raise FileNotFoundError(f"No result files found in {input_dir}")

    round_rows: List[Dict[str, Any]] = []
    run_rows: List[Dict[str, Any]] = []
    by_seed_method: Dict[Tuple[int, str], Dict[str, Any]] = {}

    for path in result_files:
        obj, method = _load_result(path)
        meta = obj.get("meta", {})
        exp = meta.get("experiment", {})
        seed = int(exp.get("seed"))
        dataset = str(exp.get("dataset"))
        alpha = float(exp.get("dirichlet_alpha"))
        num_clients = int(exp.get("num_clients"))
        rounds = int(exp.get("rounds"))

        result = obj["results"][method]
        acc_pairs = result.get("metrics_distributed", {}).get("accuracy", [])
        loss_pairs = result.get("losses_distributed", [])
        final_acc = _safe_float(acc_pairs[-1][1]) if acc_pairs else float("nan")
        final_loss = _safe_float(loss_pairs[-1][1]) if loss_pairs else float("nan")
        round_trace = result.get("round_trace", [])

        per_round_acc = []
        per_round_cr = []
        per_round_ca = []
        per_round_di = []
        per_round_neff = []
        per_round_neg = []

        for row in round_trace:
            rr = {
                "dataset": dataset,
                "alpha": alpha,
                "num_clients": num_clients,
                "method": method,
                "seed": seed,
                "round": int(row.get("round", 0)),
                "accuracy": _safe_float(row.get("accuracy")),
                "loss": _safe_float(row.get("loss")),
                "conflict_ratio": _safe_float(row.get("conflict_ratio")),
                "conflict_ratio_weighted": _safe_float(row.get("conflict_ratio_weighted")),
                "pairwise_cosine_mean": _safe_float(row.get("pairwise_cosine_mean")),
                "pairwise_cosine_min": _safe_float(row.get("pairwise_cosine_min")),
                "pairwise_cosine_max": _safe_float(row.get("pairwise_cosine_max")),
                "pairwise_cosine_std": _safe_float(row.get("pairwise_cosine_std")),
                "pairwise_cosine_fraction_negative": _safe_float(
                    row.get("pairwise_cosine_fraction_negative")
                ),
                "cancellation_ratio": _safe_float(row.get("cancellation_ratio")),
                "dominance_ratio": _safe_float(row.get("dominance_ratio")),
                "effective_num_clients": _safe_float(row.get("effective_num_clients")),
                "client_update_norm_mean": _safe_float(row.get("client_update_norm_mean")),
                "client_update_norm_max": _safe_float(row.get("client_update_norm_max")),
                "client_update_norm_std": _safe_float(row.get("client_update_norm_std")),
                "delta_norm": _safe_float(row.get("delta_norm")),
                "delta_norm_over_weighted_client_norm": _safe_float(
                    row.get("delta_norm_over_weighted_client_norm")
                ),
                "train_loss_mean": _safe_float(row.get("train_loss_mean")),
                "train_accuracy_mean": _safe_float(row.get("train_accuracy_mean")),
            }
            round_rows.append(rr)
            per_round_acc.append(rr["accuracy"])
            per_round_cr.append(rr["conflict_ratio"])
            per_round_ca.append(rr["cancellation_ratio"])
            per_round_di.append(rr["dominance_ratio"])
            per_round_neff.append(rr["effective_num_clients"])
            per_round_neg.append(rr["pairwise_cosine_fraction_negative"])

        run_summary = {
            "dataset": dataset,
            "alpha": alpha,
            "num_clients": num_clients,
            "method": method,
            "seed": seed,
            "rounds": rounds,
            "final_accuracy": final_acc,
            "final_loss": final_loss,
            "mean_conflict_ratio": _mean(per_round_cr),
            "max_conflict_ratio": max(per_round_cr) if per_round_cr else float("nan"),
            "mean_cancellation_ratio": _mean(per_round_ca),
            "max_cancellation_ratio": max(per_round_ca) if per_round_ca else float("nan"),
            "mean_dominance_ratio": _mean(per_round_di),
            "max_dominance_ratio": max(per_round_di) if per_round_di else float("nan"),
            "mean_effective_num_clients": _mean(per_round_neff),
            "min_effective_num_clients": min(per_round_neff) if per_round_neff else float("nan"),
            "mean_negative_pair_fraction": _mean(per_round_neg),
            "corr_round_acc_vs_cr": _corr(per_round_acc, per_round_cr),
            "corr_round_acc_vs_ca": _corr(per_round_acc, per_round_ca),
            "corr_round_acc_vs_di": _corr(per_round_acc, per_round_di),
        }
        run_rows.append(run_summary)
        by_seed_method[(seed, method)] = run_summary

    round_fieldnames = list(round_rows[0].keys())
    with round_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=round_fieldnames)
        writer.writeheader()
        writer.writerows(round_rows)

    run_fieldnames = list(run_rows[0].keys())
    with run_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=run_fieldnames)
        writer.writeheader()
        writer.writerows(run_rows)

    run_rows_sorted = sorted(run_rows, key=lambda x: (x["method"], x["seed"]))
    final_accs = [r["final_accuracy"] for r in run_rows_sorted]
    mean_crs = [r["mean_conflict_ratio"] for r in run_rows_sorted]
    mean_cas = [r["mean_cancellation_ratio"] for r in run_rows_sorted]
    mean_dis = [r["mean_dominance_ratio"] for r in run_rows_sorted]
    min_neffs = [r["min_effective_num_clients"] for r in run_rows_sorted]
    corr_final_vs_mean_cr = _corr(final_accs, mean_crs)
    corr_final_vs_mean_ca = _corr(final_accs, mean_cas)
    corr_final_vs_mean_di = _corr(final_accs, mean_dis)
    corr_final_vs_min_neff = _corr(final_accs, min_neffs)

    methods = sorted({r["method"] for r in run_rows_sorted})
    method_stats: Dict[str, Dict[str, float]] = {}
    for m in methods:
        rows = [r for r in run_rows_sorted if r["method"] == m]
        method_stats[m] = {
            "final_acc_mean": _mean([r["final_accuracy"] for r in rows]),
            "final_acc_std": _std([r["final_accuracy"] for r in rows]),
            "mean_cr": _mean([r["mean_conflict_ratio"] for r in rows]),
            "mean_ca": _mean([r["mean_cancellation_ratio"] for r in rows]),
            "mean_di": _mean([r["mean_dominance_ratio"] for r in rows]),
            "mean_neff": _mean([r["mean_effective_num_clients"] for r in rows]),
        }

    seed_rows = []
    seeds = sorted({r["seed"] for r in run_rows_sorted})
    for seed in seeds:
        fa = by_seed_method.get((seed, "fedavg"))
        fm = by_seed_method.get((seed, "fedavgm"))
        if fa is None or fm is None:
            continue
        seed_rows.append(
            {
                "seed": seed,
                "fedavg_final_acc": fa["final_accuracy"],
                "fedavgm_final_acc": fm["final_accuracy"],
                "acc_gap_m_minus_a": fm["final_accuracy"] - fa["final_accuracy"],
                "fedavg_mean_cr": fa["mean_conflict_ratio"],
                "fedavg_mean_ca": fa["mean_cancellation_ratio"],
                "fedavg_mean_di": fa["mean_dominance_ratio"],
                "fedavg_min_neff": fa["min_effective_num_clients"],
            }
        )

    gap_vs_cr = _corr(
        [r["acc_gap_m_minus_a"] for r in seed_rows],
        [r["fedavg_mean_cr"] for r in seed_rows],
    )
    gap_vs_ca = _corr(
        [r["acc_gap_m_minus_a"] for r in seed_rows],
        [r["fedavg_mean_ca"] for r in seed_rows],
    )
    gap_vs_di = _corr(
        [r["acc_gap_m_minus_a"] for r in seed_rows],
        [r["fedavg_mean_di"] for r in seed_rows],
    )

    strongest = [
        ("conflict", abs(corr_final_vs_mean_cr)),
        ("cancellation", abs(corr_final_vs_mean_ca)),
        ("dominance", abs(corr_final_vs_mean_di)),
    ]
    strongest.sort(key=lambda x: x[1], reverse=True)
    strongest_name, strongest_abs = strongest[0]
    if math.isnan(strongest_abs) or strongest_abs < 0.25:
        pathology_verdict = "none"
        phase2_decision = "interaction-pathology hypothesis is weak in the current setting."
    else:
        pathology_verdict = strongest_name
        phase2_decision = "at least one interaction pathology shows signal; Phase 2 is justified."

    lines: List[str] = []
    lines.append("# PHASE 1 Diagnostics Report")
    lines.append("")
    lines.append("## Exact Commands Used")
    lines.append("")
    lines.append("```bash")
    lines.append(command_used)
    lines.append("```")
    lines.append("")
    lines.append("## Dataset / Configuration")
    lines.append("")
    if run_rows_sorted:
        cfg = run_rows_sorted[0]
        lines.append(f"- dataset: fashionmnist")
        lines.append(f"- partition: dirichlet")
        lines.append(f"- alpha: {_format(float(cfg['alpha']), 3)}")
        lines.append(f"- num_clients: {int(cfg['num_clients'])}")
        lines.append(f"- rounds: {int(cfg['rounds'])}")
        lines.append(f"- seeds: {', '.join(str(s) for s in seeds)}")
        lines.append(f"- methods: {', '.join(methods)}")
    lines.append("")
    lines.append("## FedAvg / FedAvgM Final Results")
    lines.append("")
    lines.append("| method | mean final acc | std final acc | mean CR | mean CA | mean DI | mean N_eff |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for m in methods:
        s = method_stats[m]
        lines.append(
            f"| {m} | {_format(s['final_acc_mean'])} | {_format(s['final_acc_std'])} | "
            f"{_format(s['mean_cr'])} | {_format(s['mean_ca'])} | {_format(s['mean_di'])} | {_format(s['mean_neff'])} |"
        )
    lines.append("")
    lines.append("## Seed-by-Seed Comparison")
    lines.append("")
    lines.append("| seed | FedAvg acc | FedAvgM acc | gap (M-A) | FedAvg mean CR | FedAvg mean CA | FedAvg mean DI | FedAvg min N_eff |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in seed_rows:
        lines.append(
            f"| {r['seed']} | {_format(r['fedavg_final_acc'])} | {_format(r['fedavgm_final_acc'])} | "
            f"{_format(r['acc_gap_m_minus_a'])} | {_format(r['fedavg_mean_cr'])} | {_format(r['fedavg_mean_ca'])} | "
            f"{_format(r['fedavg_mean_di'])} | {_format(r['fedavg_min_neff'])} |"
        )
    lines.append("")
    lines.append("## Per-Round Diagnostic Summaries (mean across seeds)")
    lines.append("")
    lines.append("| method | mean pair-neg frac (CR) | mean cancellation (CA) | mean dominance (DI) | mean effective clients |")
    lines.append("|---|---:|---:|---:|---:|")
    for m in methods:
        method_rounds = [r for r in round_rows if r["method"] == m]
        lines.append(
            f"| {m} | {_format(_mean([r['conflict_ratio'] for r in method_rounds]))} | "
            f"{_format(_mean([r['cancellation_ratio'] for r in method_rounds]))} | "
            f"{_format(_mean([r['dominance_ratio'] for r in method_rounds]))} | "
            f"{_format(_mean([r['effective_num_clients'] for r in method_rounds]))} |"
        )
    lines.append("")
    lines.append("## Correlation Checks")
    lines.append("")
    lines.append("- corr(final_acc, mean_CR): " + _format(corr_final_vs_mean_cr))
    lines.append("- corr(final_acc, mean_CA): " + _format(corr_final_vs_mean_ca))
    lines.append("- corr(final_acc, mean_DI): " + _format(corr_final_vs_mean_di))
    lines.append("- corr(final_acc, min_N_eff): " + _format(corr_final_vs_min_neff))
    lines.append("- corr(FedAvgM gain, FedAvg mean_CR): " + _format(gap_vs_cr))
    lines.append("- corr(FedAvgM gain, FedAvg mean_CA): " + _format(gap_vs_ca))
    lines.append("- corr(FedAvgM gain, FedAvg mean_DI): " + _format(gap_vs_di))
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "High CR/CA/DI should coincide with lower final accuracy or unstable rounds if the interaction-pathology hypothesis is strong."
    )
    lines.append(
        f"In this run set, the largest absolute cross-run correlation among CR/CA/DI is for **{pathology_verdict}** "
        f"(abs corr = {_format(strongest_abs)})."
    )
    lines.append("")
    lines.append(f"- most important observed pathology: **{pathology_verdict}**")
    lines.append(f"- Phase 2 decision: **{phase2_decision}**")
    lines.append("")
    lines.append("## Raw Diagnostic Exports")
    lines.append("")
    lines.append(f"- round-level CSV: `{round_csv_path.as_posix()}`")
    lines.append(f"- run-level CSV: `{run_csv_path.as_posix()}`")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--report-path", type=Path, required=True)
    p.add_argument("--round-csv-path", type=Path, required=True)
    p.add_argument("--run-csv-path", type=Path, required=True)
    p.add_argument("--command-used", type=str, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    build_report(
        input_dir=args.input_dir,
        report_path=args.report_path,
        round_csv_path=args.round_csv_path,
        run_csv_path=args.run_csv_path,
        command_used=args.command_used,
    )


if __name__ == "__main__":
    main()
