"""Report how client graphs align with client label distributions.

This script uses data already written to result JSON files:

* ``meta.client_label_hist`` / ``meta.client_class_distribution``
* per-round ``round_trace[*].W_matrix`` from Ours runs

It compares graph edge weights against pairwise label-histogram similarity.
The goal is diagnostic: if update-space neighbors do not match label-space
neighbors at all, the graph source/projection/similarity choice needs scrutiny
before making performance claims.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute graph-vs-label-histogram alignment from result JSON files."
    )
    p.add_argument("--result", type=str, nargs="+", required=True)
    p.add_argument("--method", type=str, default="ours")
    p.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="Defaults to <first_result_stem>_graph_label_alignment.",
    )
    return p.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def label_hist_from_meta(meta: Dict[str, Any]) -> List[List[float]]:
    hist = (
        meta.get("client_label_hist")
        or meta.get("client_class_distribution")
        or meta.get("experiment", {}).get("client_label_hist")
    )
    if not hist:
        raise ValueError("Result JSON does not contain client_label_hist")
    return [[float(v) for v in row] for row in hist]


def ordered_label_hist(hist: List[List[float]], cids: Iterable[Any]) -> np.ndarray:
    cids_l = list(cids)
    if len(cids_l) != len(hist):
        return np.asarray(hist, dtype=np.float64)
    try:
        idx = [int(str(cid)) for cid in cids_l]
    except ValueError:
        return np.asarray(hist, dtype=np.float64)
    if any(i < 0 or i >= len(hist) for i in idx):
        return np.asarray(hist, dtype=np.float64)
    return np.asarray([hist[i] for i in idx], dtype=np.float64)


def pairwise_label_metrics(hist: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    counts = hist.astype(np.float64, copy=False)
    norm = np.linalg.norm(counts, axis=1) + 1e-12
    cosine = (counts @ counts.T) / (norm[:, None] * norm[None, :])
    total = np.sum(counts, axis=1, keepdims=True) + 1e-12
    probs = counts / total
    overlap = np.minimum(probs[:, None, :], probs[None, :, :]).sum(axis=2)
    np.fill_diagonal(cosine, 1.0)
    np.fill_diagonal(overlap, 1.0)
    return cosine.astype(np.float64), overlap.astype(np.float64)


def upper_values(mat: np.ndarray) -> np.ndarray:
    if mat.shape[0] <= 1:
        return np.array([], dtype=np.float64)
    return mat[np.triu_indices(mat.shape[0], k=1)].astype(np.float64)


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    x0 = x - float(np.mean(x))
    y0 = y - float(np.mean(y))
    den = float(np.linalg.norm(x0) * np.linalg.norm(y0))
    if den <= 1e-12:
        return float("nan")
    return float(np.dot(x0, y0) / den)


def rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, x.size + 1, dtype=np.float64)
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    return pearson(rankdata(x), rankdata(y))


def safe_mean(vals: Iterable[float]) -> float:
    xs = [float(v) for v in vals if v is not None and not math.isnan(float(v))]
    return float(sum(xs) / len(xs)) if xs else float("nan")


def round_rows(result_path: Path, method: str) -> List[Dict[str, Any]]:
    obj = load_json(result_path)
    meta = obj.get("meta", {})
    hist = label_hist_from_meta(meta)
    trace = obj.get("results", {}).get(method, {}).get("round_trace", [])
    rows: List[Dict[str, Any]] = []
    for row in trace:
        w_raw = row.get("W_matrix")
        if not w_raw:
            continue
        w = np.asarray(w_raw, dtype=np.float64)
        if w.ndim != 2 or w.shape[0] != w.shape[1]:
            continue
        hist_ordered = ordered_label_hist(hist, row.get("cids", []))
        if hist_ordered.shape[0] != w.shape[0]:
            continue

        label_cos, label_overlap = pairwise_label_metrics(hist_ordered)
        w_upper = upper_values(w)
        cos_upper = upper_values(label_cos)
        overlap_upper = upper_values(label_overlap)
        edge_mask = w_upper > 1e-12
        nonedge_mask = ~edge_mask
        if bool(np.any(edge_mask)):
            weighted_cos = float(np.average(cos_upper, weights=w_upper + 1e-12))
            weighted_overlap = float(
                np.average(overlap_upper, weights=w_upper + 1e-12)
            )
        else:
            weighted_cos = float("nan")
            weighted_overlap = float("nan")
        rows.append(
            {
                "result": str(result_path),
                "run_tag": meta.get("run_tag", ""),
                "method": method,
                "round": int(row.get("round", -1)),
                "graph_mode": row.get("graph_mode", ""),
                "graph_source_used": row.get("graph_source_used", ""),
                "aggregation_target_used": row.get("aggregation_target_used", ""),
                "graph_density": float(row.get("graph_density", float("nan"))),
                "number_of_edges": int(row.get("number_of_edges", 0)),
                "pearson_w_label_cosine": pearson(w_upper, cos_upper),
                "spearman_w_label_cosine": spearman(w_upper, cos_upper),
                "pearson_w_label_overlap": pearson(w_upper, overlap_upper),
                "spearman_w_label_overlap": spearman(w_upper, overlap_upper),
                "weighted_label_cosine": weighted_cos,
                "weighted_label_overlap": weighted_overlap,
                "edge_mean_label_cosine": safe_mean(cos_upper[edge_mask]),
                "nonedge_mean_label_cosine": safe_mean(cos_upper[nonedge_mask]),
                "edge_mean_label_overlap": safe_mean(overlap_upper[edge_mask]),
                "nonedge_mean_label_overlap": safe_mean(overlap_upper[nonedge_mask]),
            }
        )
    return rows


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(rows: List[Dict[str, Any]], out_dir: Path) -> None:
    by_key: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row.get("graph_mode", "")),
            str(row.get("graph_source_used", "")),
            str(row.get("aggregation_target_used", "")),
        )
        by_key.setdefault(key, []).append(row)
    summary = []
    for (graph_mode, graph_source, agg_target), group in sorted(by_key.items()):
        summary.append(
            {
                "graph_mode": graph_mode,
                "graph_source_used": graph_source,
                "aggregation_target_used": agg_target,
                "n_rounds": len(group),
                "mean_graph_density": safe_mean(r["graph_density"] for r in group),
                "mean_pearson_w_label_cosine": safe_mean(
                    r["pearson_w_label_cosine"] for r in group
                ),
                "mean_spearman_w_label_cosine": safe_mean(
                    r["spearman_w_label_cosine"] for r in group
                ),
                "mean_edge_label_cosine_minus_nonedge": safe_mean(
                    r["edge_mean_label_cosine"] - r["nonedge_mean_label_cosine"]
                    for r in group
                    if not math.isnan(float(r["edge_mean_label_cosine"]))
                    and not math.isnan(float(r["nonedge_mean_label_cosine"]))
                ),
                "mean_weighted_label_cosine": safe_mean(
                    r["weighted_label_cosine"] for r in group
                ),
                "mean_weighted_label_overlap": safe_mean(
                    r["weighted_label_overlap"] for r in group
                ),
            }
        )
    (out_dir / "graph_label_alignment_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    write_csv(summary, out_dir / "graph_label_alignment_summary.csv")

    lines = [
        "# Graph Label Alignment Report",
        "",
        "`pearson_w_label_cosine` and `spearman_w_label_cosine` compare graph",
        "edge weights against pairwise client label-histogram cosine similarity.",
        "Positive values mean update-space neighbors tend to share label mix.",
        "",
        "| graph_mode | source | n_rounds | mean density | mean Spearman | edge-label lift |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['graph_mode']} | {row['graph_source_used']} | "
            f"{row['n_rounds']} | {row['mean_graph_density']:.4f} | "
            f"{row['mean_spearman_w_label_cosine']:.4f} | "
            f"{row['mean_edge_label_cosine_minus_nonedge']:.4f} |"
        )
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    result_paths = [Path(p) for p in args.result]
    if args.out_dir.strip():
        out_dir = Path(args.out_dir)
    else:
        out_dir = result_paths[0].with_suffix("").parent / (
            result_paths[0].with_suffix("").name + "_graph_label_alignment"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for path in result_paths:
        rows.extend(round_rows(path, method=args.method))
    write_csv(rows, out_dir / "graph_label_alignment_rounds.csv")
    write_summary(rows, out_dir)
    print(f"Saved graph-label alignment report under {out_dir}")


if __name__ == "__main__":
    main()
