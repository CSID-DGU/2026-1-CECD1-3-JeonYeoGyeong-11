"""General FL (FashionMNIST / MNIST / CIFAR10) variant-by-seed suite.

Wraps ``run_general_experiment.py`` with suite-level summaries
(``general_suite_summary.json`` / ``suite_summary.json`` aliases,
``general_suite_rows.json`` / ``suite_rows.json``, ``knn_vs_random_matched.csv``,
``interpretation.md``).

--------------------------------------------------------------------
Canonical ``--variants`` tokens (case-insensitive; rows use lowercase)
--------------------------------------------------------------------

  fedavg
      FedAvg baseline.

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

  ours_weight_agg
      ``--aggregation-target weight``. This is mathematically close to update
      aggregation when alpha sums to one, but logs the target explicitly.

  ours_no_ema
      Dense graph with ``--use-ema-graph false``. FGL: ``ours_no_ema``.

  ours_fixed_tau
      Dense graph with ``--disable-adaptive-tau true``; magnitude from ``--fixed-tau``.
      FGL: ``ours_fixed_tau``.

Not wrapped here (run ``run_general_experiment.py`` directly): grid_search-only knobs.

--------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from spectral_fl.config_io import add_config_argument, parse_args_with_config
from spectral_fl.general_suite_variants import variant_cmd
from spectral_fl.suite_stats import (
    final_acc,
    load_json,
    round_trace_field,
    safe_max,
    safe_mean,
    safe_min,
    safe_pstdev,
)


VARIANTS_EPILOG = """recognized variant tokens:
  fedavg ours_dense ours_knn ours_knn_k2 ours_knn_k3 ...
  ours_random_matched_k3 ours_random ours_random_matched
  ours_uniform ours_threshold ours_mutual_knn_k3 ours_magnitude
  ours_global_alignment ours_weight_graph ours_weight_agg
  ours_no_ema ours_fixed_tau
(See module docstring for FGL name mapping.)"""


def parse_args():
    p = argparse.ArgumentParser(
        description="General FL variant-by-seed suite; writes general_suite_summary.json",
        epilog=VARIANTS_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--python-bin", type=str, default=sys.executable)
    add_config_argument(p)
    p.add_argument("--dataset", type=str, default="fashionmnist")
    p.add_argument("--num-clients", type=int, default=20)
    p.add_argument("--rounds", type=int, default=30)
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--model", type=str, default="cnn")
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=5e-4)

    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    p.add_argument("--partition", type=str, default="dirichlet", choices=["iid", "dirichlet"])
    p.add_argument("--dirichlet-alpha", type=float, default=0.1)

    p.add_argument("--projection-dim", type=int, default=256)
    p.add_argument("--compression-seed", type=int, default=0)
    p.add_argument("--ema-alpha", type=float, default=0.8)
    p.add_argument("--tau-gain", type=float, default=2.0)
    p.add_argument("--tau-max", type=float, default=2.0)
    p.add_argument("--conflict-mix", type=float, default=0.2)
    p.add_argument("--warmup-rounds", type=int, default=1)
    p.add_argument("--knn-k", type=int, default=3, help="Default k for ours_knn / ours_random if unsuffixed")
    p.add_argument(
        "--graph-source",
        type=str,
        default="update",
        choices=["update", "normalized_update", "weight"],
        help="Default graph source forwarded to run_general_experiment.",
    )
    p.add_argument(
        "--aggregation-target",
        type=str,
        default="update",
        choices=["update", "weight"],
        help="Default aggregation target forwarded to run_general_experiment.",
    )

    p.add_argument(
        "--edge-threshold",
        type=float,
        default=0.0,
        help="Cosine cutoff for variant ours_threshold (graph-mode threshold).",
    )

    p.add_argument("--graph-seed", type=int, default=0)
    p.add_argument("--use-ema-graph", type=str, default="true")
    p.add_argument("--disable-adaptive-tau", type=str, default="false")
    p.add_argument("--fixed-tau", type=float, default=1.0)
    p.add_argument(
        "--diagnostic-only",
        type=str,
        default="false",
        help="Forward to run_general_experiment: log diagnostics but keep FedAvg aggregation weights.",
    )
    p.add_argument("--e-std-threshold", type=float, default=0.0)
    p.add_argument("--min-client-weight", type=float, default=0.0)

    p.add_argument("--data-root", type=str, default="./data/torchvision")
    p.add_argument("--train-subset-size", type=int, default=0)
    p.add_argument("--test-subset-size", type=int, default=0)

    p.add_argument("--out-dir", type=str, default="./experiments_current/general_suite")
    p.add_argument(
        "--suite-tag",
        type=str,
        default="",
        help="Defaults to last path segment of --out-dir when empty.",
    )
    p.add_argument(
        "--variants",
        type=str,
        nargs="+",
        metavar="NAME",
        default=[
            "fedavg",
            "ours_dense",
            "ours_knn_k2",
            "ours_knn_k3",
            "ours_knn_k5",
            "ours_random_matched_k3",
            "ours_uniform",
        ],
        help="Suite variant tokens (see module docstring). Example: fedavg ours_knn_k3 ours_random_matched_k3",
    )
    p.add_argument(
        "--preload-fedavg-dir",
        type=str,
        default="",
        help=(
            "Directory containing completed result_general_fedavg_seed*.json runs. "
            "Loads final FedAcc per seed for delta computation when FedAvg is omitted from --variants "
            "(resume after partial suite). Same-dir --out-dir is typical."
        ),
    )
    return parse_args_with_config(p)


def load_preloaded_fedavg_accs(dir_path: Path) -> Dict[int, float]:
    """Read FedAvg final accuracies from prior suite JSONs (resume support)."""
    out: Dict[int, float] = {}
    if not dir_path.is_dir():
        return out
    for p in sorted(dir_path.glob("result_general_fedavg_seed*.json")):
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
        "mean_graph_density": safe_mean(round_trace_field(trace, "graph_density")),
        "mean_raw_current_graph_density": safe_mean(
            round_trace_field(trace, "raw_current_graph_density")
        ),
        "mean_e_std": safe_mean(
            round_trace_field(trace, "e_std") or round_trace_field(trace, "std_e")
        ),
        "mean_entropy_alpha": safe_mean(round_trace_field(trace, "entropy_alpha")),
        "mean_effective_clients": safe_mean(round_trace_field(trace, "effective_clients")),
        "mean_min_alpha": safe_mean(round_trace_field(trace, "min_alpha")),
        "mean_max_alpha": safe_mean(round_trace_field(trace, "max_alpha")),
    }


def run_cmd(cmd: List[str]) -> None:
    subprocess.run(cmd, check=True)


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


def _variant_k_number(variant: str) -> int | None:
    m = re.match(r"^ours_knn_k(\d+)$", variant)
    return int(m.group(1)) if m else None


def compute_best_knn_meta(summary_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Best k among ``ours_knn_k*`` variants by mean Δ, min Δ, win_rate."""
    knn_rows = [
        r for r in summary_rows if _variant_k_number(r.get("variant", "") or "") is not None
    ]
    if not knn_rows:
        return {}

    def tb(r: Dict[str, Any]):
        return (
            r.get("mean_delta", float("-inf")),
            r.get("min_delta", float("-inf")),
            -r.get("std_delta", 0.0),
            r.get("win_rate", 0.0),
        )

    by_mean = max(knn_rows, key=tb)
    by_min = max(
        knn_rows,
        key=lambda r: (
            r.get("min_delta", float("-inf")),
            r.get("mean_delta", float("-inf")),
            -r.get("std_delta", 0.0),
            r.get("win_rate", 0.0),
        ),
    )
    by_wr = max(
        knn_rows,
        key=lambda r: (
            r.get("win_rate", 0.0),
            r.get("mean_delta", float("-inf")),
            r.get("min_delta", float("-inf")),
            -r.get("std_delta", 0.0),
        ),
    )
    return {
        "best_k_by_mean_delta": _variant_k_number(by_mean["variant"]),
        "best_knn_variant_by_mean_delta": by_mean["variant"],
        "best_k_by_min_delta": _variant_k_number(by_min["variant"]),
        "best_knn_variant_by_min_delta": by_min["variant"],
        "best_k_by_win_rate": _variant_k_number(by_wr["variant"]),
        "best_knn_variant_by_win_rate": by_wr["variant"],
    }


def _classify_knn_vs_random(knn: Dict[str, Any], rnd: Dict[str, Any]) -> str:
    """One of similarity_graph_helpful | sparsity_only_possible | random_better | inconclusive."""
    eps = 0.003
    km = knn.get("mean_delta", float("nan"))
    rm = rnd.get("mean_delta", float("nan"))
    if math.isnan(km) or math.isnan(rm):
        return "inconclusive"
    diff = km - rm
    if rm > km + eps:
        return "random_better"
    if abs(diff) <= eps:
        return "sparsity_only_possible"
    if km > rm + eps:
        return "similarity_graph_helpful"
    return "inconclusive"


def write_knn_vs_random_matched_csv(
    out_dir: Path, summary_rows: List[Dict[str, Any]]
) -> Path | None:
    by_v = {r["variant"]: r for r in summary_rows}
    fieldnames = [
        "k",
        "knn_mean_delta",
        "random_mean_delta",
        "difference_mean_delta",
        "knn_min_delta",
        "random_min_delta",
        "difference_min_delta",
        "knn_win_rate",
        "random_win_rate",
        "knn_graph_density",
        "random_graph_density",
        "interpretation",
    ]
    rows_out: List[Dict[str, Any]] = []
    for k in (2, 3, 5):
        vk = f"ours_knn_k{k}"
        rk = f"ours_random_matched_k{k}"
        kn = by_v.get(vk)
        rn = by_v.get(rk)
        if not kn or not rn:
            rows_out.append(
                {
                    "k": k,
                    "knn_mean_delta": float("nan"),
                    "random_mean_delta": float("nan"),
                    "difference_mean_delta": float("nan"),
                    "knn_min_delta": float("nan"),
                    "random_min_delta": float("nan"),
                    "difference_min_delta": float("nan"),
                    "knn_win_rate": float("nan"),
                    "random_win_rate": float("nan"),
                    "knn_graph_density": float("nan"),
                    "random_graph_density": float("nan"),
                    "interpretation": "inconclusive",
                }
            )
            continue
        km = kn.get("mean_delta", float("nan"))
        rm = rn.get("mean_delta", float("nan"))
        diff_m = km - rm if not (math.isnan(km) or math.isnan(rm)) else float("nan")
        kmin = kn.get("min_delta", float("nan"))
        rmin = rn.get("min_delta", float("nan"))
        diff_min = kmin - rmin if not (math.isnan(kmin) or math.isnan(rmin)) else float("nan")
        interp = _classify_knn_vs_random(kn, rn)
        rows_out.append(
            {
                "k": k,
                "knn_mean_delta": km,
                "random_mean_delta": rm,
                "difference_mean_delta": diff_m,
                "knn_min_delta": kmin,
                "random_min_delta": rmin,
                "difference_min_delta": diff_min,
                "knn_win_rate": kn.get("win_rate", float("nan")),
                "random_win_rate": rn.get("win_rate", float("nan")),
                "knn_graph_density": kn.get("mean_graph_density", float("nan")),
                "random_graph_density": rn.get("mean_graph_density", float("nan")),
                "interpretation": interp,
            }
        )
    path = out_dir / "knn_vs_random_matched.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_out)
    return path


def write_interpretation_md(
    out_dir: Path,
    summary_rows: List[Dict[str, Any]],
    suite_meta: Dict[str, Any],
    args: argparse.Namespace,
) -> Path:
    """Conservative A–E style verdict; does not claim FedAvg wins."""
    by_v = {r["variant"]: r for r in summary_rows}
    fed = by_v.get("fedavg", {})
    dense = by_v.get("ours_dense", {})
    best_meta = suite_meta.get("best_knn_by_meta") or {}
    lines = [
        "# FashionMNIST / general-FL suite interpretation",
        "",
        "## Run configuration (data usage)",
        "",
    ]
    ts = int(args.train_subset_size)
    tst = int(args.test_subset_size)
    if ts <= 0 and tst <= 0:
        lines.append(
            "- **Training**: full dataset train split (`train_subset_size=0`). "
            "**Test**: full test split (`test_subset_size=0`)."
        )
    else:
        lines.append(
            f"- **Subset mode**: `train_subset_size={ts}`, `test_subset_size={tst}` "
            "(document this when comparing to full-data runs)."
        )
    lines.extend(
        [
            "",
            "## Headline comparisons (cautious)",
            "",
            f"- **FedAvg** mean_acc ≈ {fed.get('mean_acc', float('nan')):.4f} (reference).",
            f"- **Ours-dense** mean_delta vs FedAvg ≈ {dense.get('mean_delta', float('nan')):+.4f}.",
            "",
            "### kNN vs density-matched random (same k)",
            "",
            "See `knn_vs_random_matched.csv`. "
            "If kNN mean_delta exceeds random-matched by a clear margin, similarity structure "
            "may matter beyond sparsity alone; if not, benefit may be sparsity-only or inconclusive.",
            "",
            "### Verdict rubric (not a proof)",
            "",
            "- **A (strong positive — rare):** kNN mean_delta > 0, min_delta near ≥ 0, "
            "win_rate ≥ 4/5, and kNN beats matched random → *promising sparse similarity regime*.",
            "- **B (weak positive):** kNN mean_delta > 0 but min_delta < 0 or random close → "
            "*encouraging but not established*.",
            "- **C (graph signal only):** kNN beats random-matched but not FedAvg → "
            "*graph structure appears meaningful; aggregation gain weak*.",
            "- **D (sparsity-only):** random-matched ≈ kNN → *benefit may be sparsity/regularization*.",
            "- **E (negative):** all Ours variants mean_delta < 0 → "
            "*aggregation benefit unproven; diagnostic framing strengthens*.",
            "",
            "### Auto-summary from this suite",
            "",
        ]
    )
    knn_variants = sorted(
        [v for v in by_v if _variant_k_number(v)],
        key=lambda v: (_variant_k_number(v) or 0),
    )
    if knn_variants:
        best_k = max(
            knn_variants,
            key=lambda v: (
                by_v[v].get("mean_delta", float("-inf")),
                by_v[v].get("min_delta", float("-inf")),
                -by_v[v].get("std_delta", 0.0),
                by_v[v].get("win_rate", 0.0),
            ),
        )
        bk = by_v[best_k]
        lines.append(
            f"- Best kNN variant in this run (mean Δ, tie-break min Δ, −std Δ, win_rate): "
            f"`{best_k}` with mean_delta={bk.get('mean_delta'):+.4f}, "
            f"min_delta={bk.get('min_delta'):+.4f}, win_rate={bk.get('win_rate'):.2f}."
        )
        if best_meta.get("best_k_by_mean_delta") is not None:
            lines.append(
                f"- Meta `best_k_by_mean_delta` = {best_meta.get('best_k_by_mean_delta')} "
                f"(`{best_meta.get('best_knn_variant_by_mean_delta')}`)."
            )
    lines.append("")
    lines.append(
        "_This file is generated by `run_general_suite.py`. "
        "Do not treat it as establishing superiority over FedAvg without the full seed table "
        "and kNN-vs-random matched comparison._"
    )
    path = out_dir / "interpretation.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def append_validation_verdict(out_dir: Path, summary_rows: List[Dict[str, Any]]) -> None:
    """Append alpha-sweep / positioning hints from summary statistics (rules 1–4)."""
    by_v = {r["variant"]: r for r in summary_rows}
    ours_rows = [r for r in summary_rows if r["variant"] != "fedavg"]
    if not ours_rows:
        return

    lines = ["", "## Automated gate for alpha sweep (heuristic)", ""]
    eps = 0.003

    def flush():
        path = out_dir / "interpretation.md"
        path.write_text(path.read_text(encoding="utf-8") + "\n".join(lines), encoding="utf-8")

    finite = [
        r for r in ours_rows if not math.isnan(float(r.get("mean_delta", float("nan"))))
    ]
    if finite and all(r.get("mean_delta", 0) < 0 for r in finite):
        lines.append(
            "**Gate (rule 4):** Every Ours variant has mean_delta < 0 vs FedAvg → "
            "do **not** run the alpha sweep yet; emphasize graph/spectrum interpretation before performance claims."
        )
        flush()
        return

    knn_candidates = [
        (k, by_v[f"ours_knn_k{k}"]) for k in (2, 3, 5) if f"ours_knn_k{k}" in by_v
    ]
    if not knn_candidates:
        lines.append("No `ours_knn_k*` rows in summary — cannot classify kNN gate.")
        flush()
        return

    best_k, best_knn = max(knn_candidates, key=lambda x: x[1].get("mean_delta", float("-inf")))
    rnd = by_v.get(f"ours_random_matched_k{best_k}")
    dense = by_v.get("ours_dense")
    if rnd is None:
        lines.append("Missing `ours_random_matched_k*` for kNN k — inconclusive.")
        flush()
        return

    md_k = float(best_knn.get("mean_delta", float("nan")))
    md_r = float(rnd.get("mean_delta", float("nan")))
    similar = math.isnan(md_k) or math.isnan(md_r) or abs(md_k - md_r) <= eps
    kn_beats_r = not similar and md_k > md_r + eps
    min_ok = best_knn.get("min_delta", -999) >= -0.03
    wr_ok = best_knn.get("win_rate", 0) >= 0.6

    if similar:
        lines.append(
            f"**Gate (rule 3):** `ours_knn_k{best_k}` ≈ matched random at the same k → "
            "**skip alpha sweep** (similarity may not matter beyond sparsity)."
        )
    elif kn_beats_r and md_k > 0 and min_ok and wr_ok:
        lines.append(
            f"**Gate (rule 1):** `ours_knn_k{best_k}` has mean_delta > 0, tolerable min_delta, "
            "beats matched random → **alpha sweep reasonable next** "
            "(still not a general-FL *contribution* without FedProx / SCAFFOLD / etc.)."
        )
    elif kn_beats_r and md_k <= 0:
        lines.append(
            "**Gate (rule 2):** kNN beats matched random on mean_delta but **not FedAvg** "
            "(mean_delta ≤ 0) → **graph-structured signal, weak aggregation gain**; "
            "alpha sweep **optional**."
        )
    else:
        lines.append(
            "**Gate:** Mixed outcome — inspect per-seed deltas and `knn_vs_random_matched.csv`."
        )

    if dense is not None:
        lines.append(
            f"- Versus dense: `ours_knn_k{best_k}` mean_delta {md_k:+.4f} vs `ours_dense` "
            f"{dense.get('mean_delta', float('nan')):+.4f}."
        )
    flush()


def duplicate_suite_summaries(
    out_dir: Path, suite_summary: Dict[str, Any], summary_rows: List[Dict[str, Any]], rows: List[Dict[str, Any]]
) -> None:
    """Also write ``suite_summary.*`` / ``suite_rows.json`` (aliases of general_suite_*)."""
    with (out_dir / "suite_summary.json").open("w", encoding="utf-8") as f:
        json.dump(suite_summary, f, indent=2, allow_nan=True)
    with (out_dir / "suite_rows.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, allow_nan=True)
    if summary_rows:
        csv_path = out_dir / "suite_summary.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suite_tag = args.suite_tag.strip() or out_dir.name

    suite_meta = {
        "timestamp": datetime.now().isoformat(),
        "track": "general-fl",
        "suite_tag": suite_tag,
        "config": vars(args),
        "delta_baseline": "fedavg",
        "cross_track_variant_names": (
            "Tokens like ours_knn_k3 embed k in the name; FGL graph ablation uses ours_knn plus --knn-k. "
            "Tokens differ across tracks on purpose—match experiments by knn-k / graph-mode, not raw variant string equality."
        ),
        "matched_random_ablation": (
            "ours_random_matched_kK uses graph-mode random with the same --knn-k as FGL ours_random: matched edge count vs kNN "
            "(controls sparsity only). Compare variant ours_knn_kK to ours_random_matched_kK at equal K for spectral-vs-random. "
            "p-values / CI are not produced by this script—analyze seed<S>_delta or exports downstream."
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
                f"Warning: --preload-fedavg-dir {preload_dir!r} had no result_general_fedavg_seed*.json; "
                "deltas vs FedAvg will be NaN unless you include fedavg in --variants."
            )
        if fedavg_acc_by_seed:
            suite_meta["preloaded_fedavg_dir"] = preload_dir
            suite_meta["preloaded_fedavg_seeds"] = sorted(fedavg_acc_by_seed.keys())

    rows: List[Dict[str, Any]] = []
    failed_runs: List[Dict[str, Any]] = []

    if "fedavg" in args.variants:
        for seed in args.seeds:
            cmd, method, result_path = variant_cmd(args, "fedavg", seed, suite_tag, out_dir)
            try:
                run_cmd(cmd)
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
                }
            )

    for variant in args.variants:
        if variant.strip().lower() == "fedavg":
            continue
        for seed in args.seeds:
            cmd, method, result_path = variant_cmd(args, variant, seed, suite_tag, out_dir)
            try:
                run_cmd(cmd)
                result = load_json(result_path)
                acc = final_acc(result, method)
                fed_acc = fedavg_acc_by_seed.get(seed, float("nan"))
                feats = collect_run_features(result, method)
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
            "graph_mode": gfirst("graph_mode"),
            "graph_source": gfirst("graph_source"),
            "graph_source_used": gfirst("graph_source_used"),
            "aggregation_target": gfirst("aggregation_target"),
            "aggregation_target_used": gfirst("aggregation_target_used"),
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
            "mean_tau": gmean("mean_tau"),
            "mean_e_std": gmean("mean_e_std"),
            "mean_graph_density": gmean("mean_graph_density"),
            "mean_raw_current_graph_density": gmean("mean_raw_current_graph_density"),
            "mean_entropy_alpha": gmean("mean_entropy_alpha"),
            "mean_effective_clients": gmean("mean_effective_clients"),
            "mean_min_alpha": gmean("mean_min_alpha"),
            "mean_max_alpha": gmean("mean_max_alpha"),
        }
        seed_delta_vals: List[float] = []
        for sd in seed_cols:
            col = f"seed{sd}_delta"
            match = [x for x in group if int(x["seed"]) == sd]
            val = float(match[0]["delta"]) if match else float("nan")
            row_base[col] = val
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
    knn_csv = write_knn_vs_random_matched_csv(out_dir, summary_rows)
    interp_md = write_interpretation_md(out_dir, summary_rows, suite_meta, args)
    append_validation_verdict(out_dir, summary_rows)

    print("\n=== General-FL suite summary (rank: mean_delta, min_delta, -std_delta, win_rate) ===")
    for row in summary_rows:
        if row["variant"] == "fedavg":
            continue
        print(
            f"  {row['variant']:<28} "
            f"mean_delta={row.get('mean_delta', 0):+.4f} "
            f"min_delta={row.get('min_delta', 0):+.4f} "
            f"std={row.get('std_delta', 0):.4f} "
            f"win_rate={row.get('win_rate', 0):.2f}"
        )
    md_path = out_dir / "general_suite_summary.md"
    lines = [
        "# General FL suite summary",
        "",
        f"- Suite tag: `{suite_tag}`",
        f"- Dataset: `{args.dataset}`, Dirichlet α={args.dirichlet_alpha}, clients={args.num_clients}",
        "",
        "## Ranking (mean Δ ↓, min Δ ↓, std Δ ↑, win rate ↓)",
        "",
        "| variant | mean_acc | std_acc | mean Δ | min Δ | max Δ | std Δ | win_rate | mean H_spec | mean τ | mean ρ | mean_ent_α | mean_eff |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row.get('variant')} | {row.get('mean_acc', float('nan')):.4f} | "
            f"{row.get('std_acc', float('nan')):.4f} | {row.get('mean_delta', float('nan')):+.4f} | "
            f"{row.get('min_delta', float('nan')):+.4f} | {row.get('max_delta', float('nan')):+.4f} | "
            f"{row.get('std_delta', float('nan')):.4f} | {row.get('win_rate', 0):.2f} | "
            f"{row.get('mean_H_spec', float('nan')):.4f} | {row.get('mean_tau', float('nan')):.4f} | "
            f"{row.get('mean_graph_density', float('nan')):.4f} | "
            f"{row.get('mean_entropy_alpha', float('nan')):.4f} | "
            f"{row.get('mean_effective_clients', float('nan')):.4f} |"
        )
    lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {md_path}")

    print(f"Saved: {summary_json}")
    print(f"Saved: {rows_path}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {out_dir / 'suite_summary.json'} (alias)")
    if knn_csv:
        print(f"Saved: {knn_csv}")
    print(f"Saved: {interp_md}")
    if failed_runs:
        print(f"Failed runs: {len(failed_runs)} (see summary JSON)")


if __name__ == "__main__":
    main()
