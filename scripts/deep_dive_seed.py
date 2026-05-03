"""Deep-dive analysis for a single Ours+FedAvg result JSON.

Given one ``result_*.json`` produced by ``run_experiment.py --method both``
this script extracts:

    round_summary.csv         - per-round Ours/FedAvg accuracy and signals
    client_round_trace.csv    - one row per (round, client)
    suppressed_clients.csv    - per-client suppression statistics
    deep_dive_report.md       - markdown report with client-level diagnostics

The "suppression" definition (used in suppressed_clients.csv and the
report) is: alpha_i < FedAvg-equivalent data-size weight (n_i / sum n_i).
A client is "suppressed in round r" when its Ours alpha falls below that
data-size weight.  This is robust to the warmup rounds where Ours uses
FedAvg weights anyway (no false positives, by definition).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# =============================================================================
# Argparse
# =============================================================================


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--path", type=str, required=True, help="result_*.json path")
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--method", type=str, default="ours")
    p.add_argument("--compare-method", type=str, default="fedavg")
    return p.parse_args()


# =============================================================================
# Helpers
# =============================================================================


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_corr(xs: List[float], ys: List[float]) -> Optional[float]:
    pairs = [
        (float(x), float(y))
        for x, y in zip(xs, ys)
        if x is not None and y is not None and not math.isnan(x) and not math.isnan(y)
    ]
    if len(pairs) < 2:
        return None
    xs2 = [p[0] for p in pairs]
    ys2 = [p[1] for p in pairs]
    if len(set(xs2)) < 2 or len(set(ys2)) < 2:
        return None
    try:
        return float(statistics.correlation(xs2, ys2))
    except Exception:
        return None


def safe_mean(values, default=float("nan")):
    values = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return statistics.mean(values) if values else default


def safe_min(values, default=float("nan")):
    values = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return min(values) if values else default


def safe_max(values, default=float("nan")):
    values = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return max(values) if values else default


def acc_map_from_pairs(pairs) -> Dict[int, float]:
    return {int(r): float(v) for r, v in pairs}


def fmt_corr(v):
    if v is None:
        return "N/A"
    return f"{v:+.3f}"


def fmt_float(v, fmt="+.4f"):
    if v is None:
        return "N/A"
    if isinstance(v, float) and math.isnan(v):
        return "N/A"
    return format(v, fmt)


# =============================================================================
# Extraction
# =============================================================================


def extract_round_summary(
    result: Dict[str, Any], method: str, compare_method: str
) -> List[Dict[str, Any]]:
    main_trace = result["results"].get(method, {}).get("round_trace", [])
    cmp_acc = acc_map_from_pairs(
        result["results"].get(compare_method, {})
        .get("metrics_distributed", {})
        .get("accuracy", [])
    )
    cmp_loss = acc_map_from_pairs(
        result["results"].get(compare_method, {}).get("losses_distributed", [])
    )
    rows: List[Dict[str, Any]] = []
    for row in main_trace:
        r = int(row["round"])
        ours_acc = row.get("accuracy")
        ours_loss = row.get("loss")
        fa_acc = cmp_acc.get(r)
        fa_loss = cmp_loss.get(r)
        delta = None
        if ours_acc is not None and fa_acc is not None:
            delta = float(ours_acc) - float(fa_acc)
        rows.append(
            {
                "round": r,
                "fedavg_acc": fa_acc,
                "ours_acc": ours_acc,
                "delta_acc": delta,
                "fedavg_loss": fa_loss,
                "ours_loss": ours_loss,
                "h_spec": row.get("h_spec"),
                "h_spec_ema": row.get("h_spec_ema"),
                "tau": row.get("tau"),
                "min_alpha": row.get("min_alpha"),
                "max_alpha": row.get("max_alpha"),
                "entropy_alpha": row.get("entropy_alpha"),
                "effective_clients": row.get("effective_clients"),
                "graph_density": row.get("graph_density"),
                "number_of_edges": row.get("number_of_edges"),
                "graph_empty": row.get("graph_empty"),
                "e_mean": row.get("e_mean", row.get("mean_e")),
                "e_std": row.get("e_std", row.get("std_e")),
                "alpha_mode": row.get("alpha_mode"),
                "conflict_penalty_disabled_due_to_estd": row.get(
                    "conflict_penalty_disabled_due_to_estd"
                ),
            }
        )
    return rows


def extract_client_round_trace(
    result: Dict[str, Any], method: str
) -> List[Dict[str, Any]]:
    trace = result["results"].get(method, {}).get("round_trace", [])
    rows: List[Dict[str, Any]] = []
    label_hist_meta = (
        result.get("meta", {}).get("client_class_distribution")
        or result.get("meta", {}).get("client_label_hist")
        or []
    )
    cid_to_idx = _build_cid_to_index(trace)
    for row in trace:
        r = int(row["round"])
        cids = row.get("cids") or []
        alpha = row.get("alpha_norm_list") or row.get("alpha_list") or []
        e = row.get("e_list") or []
        e_z = row.get("e_z_list") or []
        cw = row.get("conflict_weight_list") or []
        delta_norm = row.get("delta_norm_list") or []
        z_norm = row.get("z_norm_list") or []
        n_examples = row.get("client_num_examples") or []
        per_client_eval = {
            str(p.get("cid")): p for p in row.get("per_client_eval", [])
        }
        train_acc = row.get("client_train_accuracy_list") or [None] * len(alpha)
        train_loss = row.get("client_train_loss_list") or [None] * len(alpha)
        for i, cid in enumerate(cids):
            sc = str(cid)
            cid_int = cid_to_idx.get(sc, i)
            label_hist = (
                label_hist_meta[i] if i < len(label_hist_meta) else None
            )
            eval_row = per_client_eval.get(sc)
            if eval_row is not None:
                # General-FL currently evaluates the global test set only on
                # client 0. Other clients return num_examples=0 as dummy eval
                # rows, which must not be treated as per-client accuracy.
                try:
                    if int(eval_row.get("num_examples", 0) or 0) <= 0:
                        eval_row = None
                except (TypeError, ValueError):
                    eval_row = None
            rows.append(
                {
                    "round": r,
                    "client_id": cid_int,
                    "raw_cid": sc,
                    "alpha": float(alpha[i]) if i < len(alpha) else None,
                    "e": float(e[i]) if i < len(e) else None,
                    "e_z": float(e_z[i]) if i < len(e_z) else None,
                    "conflict_weight": float(cw[i]) if i < len(cw) else None,
                    "delta_norm": float(delta_norm[i]) if i < len(delta_norm) else None,
                    "z_norm": float(z_norm[i]) if i < len(z_norm) else None,
                    "num_examples": int(n_examples[i]) if i < len(n_examples) else None,
                    "label_hist": label_hist,
                    "client_train_acc": train_acc[i] if i < len(train_acc) else None,
                    "client_train_loss": train_loss[i] if i < len(train_loss) else None,
                    "client_eval_acc": (
                        float(eval_row["accuracy"]) if eval_row else None
                    ),
                    "client_eval_loss": (
                        float(eval_row["loss"]) if eval_row else None
                    ),
                    "alpha_mode": row.get("alpha_mode"),
                }
            )
    return rows


def fedavg_data_weight(
    n_examples_seq: List[List[int]],
) -> List[float]:
    """For each round return the n_i / sum_n_i vector (FedAvg-equivalent weights)."""
    out = []
    for row in n_examples_seq:
        s = float(sum(row)) if row else 0.0
        if s <= 0.0:
            out.append([0.0 for _ in row])
        else:
            out.append([float(x) / s for x in row])
    return out


def _build_cid_to_index(trace: List[Dict[str, Any]]) -> Dict[str, int]:
    """Stable mapping from Flower proxy cid (string) to a 0..n-1 client index.

    Flower's ClientProxy.cid is generally an opaque string identifier
    (e.g. a hash or a UUID), not a 0..n-1 integer.  For analysis we want
    a small integer index per client.  Strategy:

      1. If every cid parses as an int and they are 0..n-1, reuse them.
      2. Otherwise sort the unique cids lexicographically and assign 0..n-1.
    """
    seen: List[str] = []
    for row in trace:
        for c in row.get("cids") or []:
            sc = str(c)
            if sc not in seen:
                seen.append(sc)
    # Try to use parsed ints if they form a contiguous 0..n-1 range.
    try:
        as_int = sorted(int(c) for c in seen)
        if as_int == list(range(len(seen))):
            return {str(c): int(c) for c in seen}
    except (TypeError, ValueError):
        pass
    seen_sorted = sorted(seen)
    return {c: i for i, c in enumerate(seen_sorted)}


def extract_suppression_stats(
    result: Dict[str, Any], method: str
) -> List[Dict[str, Any]]:
    """Aggregate per-client metrics including suppression count.

    Suppression definition: alpha_i < (n_i / sum n_j) (i.e. below the
    FedAvg-equivalent data-size weight) in that round.
    """
    trace = result["results"].get(method, {}).get("round_trace", [])
    label_hist_meta = (
        result.get("meta", {}).get("client_class_distribution")
        or result.get("meta", {}).get("client_label_hist")
        or []
    )
    if not trace:
        return []
    cid_to_idx = _build_cid_to_index(trace)
    n_clients = len(cid_to_idx)
    if n_clients == 0:
        return []

    by_client_alpha: Dict[int, List[float]] = {i: [] for i in range(n_clients)}
    by_client_e: Dict[int, List[float]] = {i: [] for i in range(n_clients)}
    by_client_ez: Dict[int, List[float]] = {i: [] for i in range(n_clients)}
    by_client_supp: Dict[int, int] = {i: 0 for i in range(n_clients)}
    by_client_n: Dict[int, int] = {}
    by_client_label_hist: Dict[int, Optional[List[int]]] = {}

    for row in trace:
        cids = row.get("cids") or []
        alpha = row.get("alpha_norm_list") or row.get("alpha_list") or []
        e = row.get("e_list") or []
        e_z = row.get("e_z_list") or []
        n_examples = row.get("client_num_examples") or []
        s = float(sum(n_examples)) if n_examples else 0.0
        fa_w = [float(x) / s if s > 0 else 0.0 for x in n_examples]
        for i, c in enumerate(cids):
            sc = str(c)
            cid_int = cid_to_idx.get(sc, i)
            if i < len(alpha):
                by_client_alpha[cid_int].append(float(alpha[i]))
            if i < len(e):
                by_client_e[cid_int].append(float(e[i]))
            if i < len(e_z):
                by_client_ez[cid_int].append(float(e_z[i]))
            if i < len(n_examples):
                by_client_n[cid_int] = int(n_examples[i])
            if i < len(label_hist_meta):
                by_client_label_hist[cid_int] = label_hist_meta[i]
            if i < len(alpha) and i < len(fa_w):
                if float(alpha[i]) + 1e-9 < float(fa_w[i]):
                    by_client_supp[cid_int] += 1

    out: List[Dict[str, Any]] = []
    for cid in range(n_clients):
        out.append(
            {
                "client_id": cid,
                "num_examples": by_client_n.get(cid),
                "label_hist": by_client_label_hist.get(cid),
                "mean_alpha": safe_mean(by_client_alpha.get(cid, [])),
                "min_alpha": safe_min(by_client_alpha.get(cid, [])),
                "max_alpha": safe_max(by_client_alpha.get(cid, [])),
                "mean_e": safe_mean(by_client_e.get(cid, [])),
                "mean_e_z": safe_mean(by_client_ez.get(cid, [])),
                "max_e_z": safe_max(by_client_ez.get(cid, [])),
                "suppression_count": int(by_client_supp.get(cid, 0)),
                "n_total_rounds": len(trace),
            }
        )
    return out


# =============================================================================
# Markdown report
# =============================================================================


def write_markdown_report(
    result: Dict[str, Any],
    round_summary: List[Dict[str, Any]],
    client_trace: List[Dict[str, Any]],
    suppressed: List[Dict[str, Any]],
    out_path: Path,
    method: str,
    compare_method: str,
) -> None:
    meta = result.get("meta", {})
    seed = meta.get("seed")
    partition = meta.get("partition")
    rounds = meta.get("rounds")
    warmup = meta.get("warmup_rounds")

    # Q1: rounds with the largest gap
    gap_rows = sorted(
        [r for r in round_summary if r["delta_acc"] is not None],
        key=lambda r: r["delta_acc"],
    )
    worst_rounds = gap_rows[:3]

    # Q2: most-suppressed clients
    most_suppressed = sorted(
        suppressed, key=lambda x: x["suppression_count"], reverse=True
    )[:3]

    # Q3: minority labels
    label_hist_lines = []
    for c in suppressed:
        label_hist_lines.append(
            f"- client {c['client_id']}: n={c['num_examples']}, "
            f"label_hist={c['label_hist']}, "
            f"suppression_count={c['suppression_count']}/{c['n_total_rounds']}, "
            f"mean_alpha={fmt_float(c['mean_alpha'], '.4f')}, "
            f"mean_e_z={fmt_float(c['mean_e_z'], '+.3f')}"
        )

    # Q4-Q5 correlations from per-(round, client) data
    e_z_vals = [r["e_z"] for r in client_trace]
    alpha_vals = [r["alpha"] for r in client_trace]
    cw_vals = [r["conflict_weight"] for r in client_trace]
    delta_norm_vals = [r["delta_norm"] for r in client_trace]
    eval_acc_vals = [r["client_eval_acc"] for r in client_trace]
    train_acc_vals = [r["client_train_acc"] for r in client_trace]

    corr_ez_alpha = safe_corr(e_z_vals, alpha_vals)
    corr_ez_cw = safe_corr(e_z_vals, cw_vals)
    corr_ez_eval = safe_corr(e_z_vals, eval_acc_vals)
    corr_alpha_eval = safe_corr(alpha_vals, eval_acc_vals)
    corr_ez_delta_norm = safe_corr(e_z_vals, delta_norm_vals)
    corr_ez_train = safe_corr(e_z_vals, train_acc_vals)
    corr_alpha_train = safe_corr(alpha_vals, train_acc_vals)

    # Q6: weight collapse
    min_eff = safe_min([r.get("effective_clients") for r in round_summary])
    min_entropy = safe_min([r.get("entropy_alpha") for r in round_summary])

    # Q7: graph density / stability
    densities = [r.get("graph_density") for r in round_summary]
    n_empty = sum(1 for r in round_summary if r.get("graph_empty"))

    # Q8: tau trend
    tau_curve = [(r["round"], r.get("tau")) for r in round_summary]

    # Q9: warmup boundary check
    onset_round = (warmup + 1) if warmup is not None else None
    onset_row = next(
        (r for r in round_summary if r["round"] == onset_round), None
    )

    # Q10: useful-heterogeneity hypothesis
    if corr_ez_eval is not None and corr_ez_eval > 0.3:
        useful_het_verdict = (
            "Plausible: high e_z correlates *positively* with eval acc, "
            "suggesting suppressed clients may carry useful heterogeneity."
        )
    elif corr_ez_eval is not None and corr_ez_eval < -0.3:
        useful_het_verdict = (
            "Less likely: high e_z correlates *negatively* with eval acc, "
            "suggesting the penalty mostly suppresses noisy/poor updates."
        )
    else:
        useful_het_verdict = (
            "Inconclusive from this single run: |corr(e_z, eval_acc)| is small "
            "or N/A. Larger-scale analysis needed."
        )

    lines: List[str] = []
    lines.append(f"# Deep-dive report: seed={seed}, partition={partition}")
    lines.append("")
    lines.append(f"- result file: `{meta.get('output_path', 'unknown')}`")
    lines.append(f"- rounds: {rounds}, warmup: {warmup}")
    lines.append(
        f"- method: `{method}`, compare: `{compare_method}`"
    )
    lines.append("")
    lines.append("## Headline numbers")
    lines.append("")
    if round_summary:
        last = round_summary[-1]
        lines.append(
            f"- final {compare_method} acc: {fmt_float(last['fedavg_acc'], '.4f')}"
        )
        lines.append(
            f"- final {method} acc: {fmt_float(last['ours_acc'], '.4f')}"
        )
        lines.append(f"- final delta: {fmt_float(last['delta_acc'])}")
    lines.append("")
    lines.append("## Q1. Rounds with the largest accuracy gap")
    lines.append("")
    if worst_rounds:
        lines.append("| round | fedavg | ours | delta | tau | min_alpha | entropy_alpha | graph_density | e_std |")
        lines.append("|-------|--------|------|-------|-----|-----------|---------------|---------------|-------|")
        for r in worst_rounds:
            lines.append(
                f"| {r['round']} | {fmt_float(r['fedavg_acc'], '.4f')} | "
                f"{fmt_float(r['ours_acc'], '.4f')} | {fmt_float(r['delta_acc'])} | "
                f"{fmt_float(r.get('tau'), '.3f')} | {fmt_float(r.get('min_alpha'), '.3f')} | "
                f"{fmt_float(r.get('entropy_alpha'), '.3f')} | "
                f"{fmt_float(r.get('graph_density'), '.3f')} | "
                f"{fmt_float(r.get('e_std'), '.4f')} |"
            )
    else:
        lines.append("No rounds with comparable accuracy data.")
    lines.append("")
    lines.append("## Q2. Most-suppressed clients (alpha below FedAvg data-size weight)")
    lines.append("")
    lines.append("Definition: a round counts as 'suppressed' for client i when "
                 "alpha_i < n_i / sum_j n_j.  This is robust to warmup (where "
                 "Ours uses FedAvg weights and contributes 0 to the count).")
    lines.append("")
    lines.append("| client | suppression_count | mean_alpha | mean_e_z | label_hist | n |")
    lines.append("|--------|-------------------|------------|----------|------------|---|")
    for c in most_suppressed:
        lines.append(
            f"| {c['client_id']} | {c['suppression_count']}/{c['n_total_rounds']} | "
            f"{fmt_float(c['mean_alpha'], '.4f')} | "
            f"{fmt_float(c['mean_e_z'], '+.3f')} | "
            f"{c['label_hist']} | {c['num_examples']} |"
        )
    lines.append("")
    lines.append("## Q3. Did suppressed clients contain minority/rare labels?")
    lines.append("")
    lines.extend(label_hist_lines)
    lines.append("")
    lines.append(
        "If a suppressed client has a label histogram concentrated in a class "
        "that other clients see rarely (e.g. zeros across other rows of the "
        "client_class_distribution), useful-heterogeneity suppression is the "
        "leading hypothesis."
    )
    lines.append("")
    lines.append("## Q4-Q5. Correlation between e_z and weights / accuracy")
    lines.append("")
    lines.append(f"- corr(e_z, alpha)             = {fmt_corr(corr_ez_alpha)}  (expected: < 0; high conflict ⇒ low weight)")
    lines.append(f"- corr(e_z, conflict_weight)   = {fmt_corr(corr_ez_cw)}  (expected: ≈ -1 by construction outside estd-skip)")
    lines.append(f"- corr(e_z, client_eval_acc)   = {fmt_corr(corr_ez_eval)}  (positive ⇒ useful-heterogeneity suppression)")
    lines.append(f"- corr(alpha, client_eval_acc) = {fmt_corr(corr_alpha_eval)}  (positive ⇒ Ours up-weights better clients)")
    lines.append(f"- corr(e_z, delta_norm)        = {fmt_corr(corr_ez_delta_norm)}  (positive ⇒ outliers in update space are penalized)")
    lines.append(f"- corr(e_z, client_train_acc)  = {fmt_corr(corr_ez_train)}")
    lines.append(f"- corr(alpha, client_train_acc) = {fmt_corr(corr_alpha_train)}")
    lines.append("")
    lines.append("## Q6. Weight collapse?")
    lines.append("")
    lines.append(
        f"- min(effective_clients) over rounds = {fmt_float(min_eff, '.3f')}"
    )
    lines.append(
        f"- min(entropy_alpha) over rounds     = {fmt_float(min_entropy, '.3f')}"
    )
    lines.append(
        "Collapse would manifest as effective_clients dropping toward 1 and "
        "entropy_alpha dropping toward 0."
    )
    lines.append("")
    lines.append("## Q7. Was the graph dense, sparse, or unstable?")
    lines.append("")
    lines.append(
        f"- mean graph_density = {fmt_float(safe_mean(densities), '.3f')}, "
        f"min = {fmt_float(safe_min(densities), '.3f')}, "
        f"max = {fmt_float(safe_max(densities), '.3f')}"
    )
    lines.append(f"- rounds with graph_empty = True: {n_empty}")
    lines.append("")
    lines.append("## Q8. Did tau increase before performance degradation?")
    lines.append("")
    lines.append("| round | tau | delta_acc |")
    lines.append("|-------|-----|-----------|")
    for r in round_summary:
        lines.append(
            f"| {r['round']} | {fmt_float(r.get('tau'), '.3f')} | "
            f"{fmt_float(r.get('delta_acc'))} |"
        )
    lines.append("")
    lines.append("## Q9. Did suppression begin right after warmup?")
    lines.append("")
    if onset_row is not None:
        lines.append(
            f"- first post-warmup round is r={onset_round} "
            f"(alpha_mode={onset_row.get('alpha_mode')}, "
            f"min_alpha={fmt_float(onset_row.get('min_alpha'), '.3f')}, "
            f"delta_acc={fmt_float(onset_row.get('delta_acc'))}, "
            f"tau={fmt_float(onset_row.get('tau'), '.3f')})"
        )
    else:
        lines.append("- no post-warmup round found in trace.")
    lines.append("")
    lines.append("## Q10. Is this likely a useful-heterogeneity suppression case?")
    lines.append("")
    lines.append(useful_het_verdict)
    lines.append("")
    lines.append(
        "Interpretation rule: this report is *one* run.  Treat all answers "
        "as descriptive evidence, not proof.  See EXPERIMENT_REPORT_FULL.md "
        "for the broader interpretation framework."
    )

    out_path.write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# CSV writers
# =============================================================================


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            safe_row = {}
            for k, v in row.items():
                if isinstance(v, list):
                    safe_row[k] = json.dumps(v)
                else:
                    safe_row[k] = v
            writer.writerow(safe_row)


# =============================================================================
# Main
# =============================================================================


def main():
    args = parse_args()
    in_path = Path(args.path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    result = load_json(in_path)

    round_summary = extract_round_summary(
        result, method=args.method, compare_method=args.compare_method
    )
    client_trace = extract_client_round_trace(result, method=args.method)
    suppressed = extract_suppression_stats(result, method=args.method)

    write_csv(round_summary, out_dir / "round_summary.csv")
    write_csv(client_trace, out_dir / "client_round_trace.csv")
    write_csv(suppressed, out_dir / "suppressed_clients.csv")
    write_markdown_report(
        result=result,
        round_summary=round_summary,
        client_trace=client_trace,
        suppressed=suppressed,
        out_path=out_dir / "deep_dive_report.md",
        method=args.method,
        compare_method=args.compare_method,
    )

    print(f"Saved: {out_dir / 'round_summary.csv'}")
    print(f"Saved: {out_dir / 'client_round_trace.csv'}")
    print(f"Saved: {out_dir / 'suppressed_clients.csv'}")
    print(f"Saved: {out_dir / 'deep_dive_report.md'}")


if __name__ == "__main__":
    main()
