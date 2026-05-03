"""Aggregate raw result_*.json files produced by run_graph_ablation.py.

This helper reproduces the suite-level summary
(``suite_<tag>_summary.json|csv|rows.json``) from an output directory
containing raw result files. It is useful when summary files are missing
or when result JSONs need to be re-aggregated after analysis changes.

Usage::

    python scripts/aggregate_graph_ablation.py \
        --in-dir experiments_current/graph_ablation \
        --suite-tag graph_ablation \
        --variants fedavg ours_dense ours_knn ours_random ours_uniform

The naming convention parsed is::

    result_<method>_seed<seed>_<suite_tag>_<variant>_seed<seed>.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir", required=True, type=str)
    p.add_argument("--suite-tag", required=True, type=str)
    p.add_argument("--variants", nargs="+", default=[
        "fedavg", "ours_dense", "ours_random", "ours_uniform", "ours_no_ema",
    ])
    return p.parse_args()


def safe_mean(values, default=float("nan")):
    values = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return statistics.mean(values) if values else default


def safe_min(values, default=float("nan")):
    values = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return min(values) if values else default


def safe_max(values, default=float("nan")):
    values = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return max(values) if values else default


def safe_pstdev(values, default=0.0):
    values = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if len(values) < 2:
        return default
    return statistics.pstdev(values)


def round_trace_field(trace, key):
    out = []
    for row in trace:
        v = row.get(key)
        if v is None:
            continue
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            continue
    return out


def final_acc(result_obj: Dict[str, Any], method: str) -> float:
    acc = result_obj["results"][method]["metrics_distributed"]["accuracy"]
    return float(acc[-1][1]) if acc else float("nan")


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


def discover_files(in_dir: Path, suite_tag: str, variants: List[str]):
    """Return list of (variant, seed, method, path)."""
    out = []
    pat_fedavg = re.compile(
        rf"^result_fedavg_seed(\d+)_{re.escape(suite_tag)}_fedavg_seed(\d+)\.json$"
    )
    pat_ours = re.compile(
        rf"^result_ours_seed(\d+)_{re.escape(suite_tag)}_(ours_[a-z0-9_]+)_seed(\d+)\.json$"
    )
    for p in sorted(in_dir.glob("result_*.json")):
        m = pat_fedavg.match(p.name)
        if m:
            seed = int(m.group(1))
            if "fedavg" in variants:
                out.append(("fedavg", seed, "fedavg", p))
            continue
        m = pat_ours.match(p.name)
        if m:
            seed = int(m.group(1))
            variant = m.group(2)
            if variant in variants:
                out.append((variant, seed, "ours", p))
            continue
    return out


def main():
    args = parse_args()
    in_dir = Path(args.in_dir)

    fed_acc_by_seed: Dict[int, float] = {}
    rows: List[Dict[str, Any]] = []

    found = discover_files(in_dir, args.suite_tag, args.variants)

    # First pass: fedavg accuracies (needed for delta).
    for variant, seed, method, path in found:
        if variant != "fedavg":
            continue
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        acc = final_acc(obj, "fedavg")
        fed_acc_by_seed[seed] = acc
        rows.append(
            {
                "variant": "fedavg",
                "seed": int(seed),
                "method": "fedavg",
                "fedavg_acc": acc,
                "ours_acc": float("nan"),
                "delta": 0.0,
            }
        )

    # Second pass: ours_* variants
    for variant, seed, method, path in found:
        if variant == "fedavg":
            continue
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        acc = final_acc(obj, "ours")
        fed_acc = fed_acc_by_seed.get(seed, float("nan"))
        feats = collect_run_features(obj, "ours")
        rows.append(
            {
                "variant": variant,
                "seed": int(seed),
                "method": "ours",
                "fedavg_acc": fed_acc,
                "ours_acc": acc,
                "delta": (acc - fed_acc) if not math.isnan(fed_acc) else float("nan"),
                **feats,
            }
        )

    by_variant: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_variant.setdefault(r["variant"], []).append(r)

    seed_cols = sorted({int(r["seed"]) for r in rows})

    summary_rows: List[Dict[str, Any]] = []
    for variant, group in by_variant.items():
        deltas = [x["delta"] for x in group if x["variant"] != "fedavg"]
        ours_acc = [x["ours_acc"] for x in group if x["variant"] != "fedavg"]
        fa_acc = [x["fedavg_acc"] for x in group]

        def gmean(key: str, group=group):
            return safe_mean([x.get(key) for x in group])

        def gmin(key: str, group=group):
            return safe_min([x.get(key) for x in group])

        def gfirst(key: str, group=group) -> str:
            for x in group:
                value = x.get(key)
                if value not in (None, ""):
                    return str(value)
            return ""

        row_base = {
            "variant": variant,
            "n_runs": len(group),
            "graph_mode": gfirst("graph_mode"),
            "graph_source": gfirst("graph_source"),
            "graph_source_used": gfirst("graph_source_used"),
            "aggregation_target": gfirst("aggregation_target"),
            "aggregation_target_used": gfirst("aggregation_target_used"),
            "mean_fedavg_acc": safe_mean(fa_acc),
            "mean_ours_acc": safe_mean(ours_acc),
            "std_ours_acc": safe_pstdev(ours_acc),
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
        for sd in seed_cols:
            col = f"seed{sd}_delta"
            match = [x for x in group if int(x["seed"]) == sd]
            row_base[col] = float(match[0]["delta"]) if match else float("nan")
        summary_rows.append(row_base)

    # rank: prefer variants whose worst seed is highest, then mean delta, then std
    def rank_key(row):
        if row["variant"] == "fedavg":
            return (1, 0, 0, 0)
        return (-row["min_delta"], -row["mean_delta"], row["std_delta"], -row["win_rate"])

    summary_rows.sort(key=rank_key)

    suite_summary = {
        "meta": {
            "timestamp": datetime.now().isoformat(),
            "suite_tag": args.suite_tag,
            "in_dir": str(in_dir),
            "regenerated_from_raw_results": True,
            "variants": args.variants,
        },
        "summary": summary_rows,
        "failed_runs": [],
    }
    summary_json = in_dir / f"suite_{args.suite_tag}_summary.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(suite_summary, f, indent=2)

    rows_path = in_dir / f"suite_{args.suite_tag}_rows.json"
    with rows_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    csv_path = in_dir / f"suite_{args.suite_tag}_summary.csv"
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
            f"mean_acc={row.get('mean_ours_acc', float('nan')):.4f}"
        )
    print(f"Saved: {summary_json}")
    print(f"Saved: {rows_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
