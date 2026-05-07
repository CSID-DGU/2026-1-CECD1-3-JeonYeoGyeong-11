"""Export round/client spectral decomposition tables from one result JSON."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--result", type=Path, required=True)
    p.add_argument("--method", type=str, default="ours")
    p.add_argument("--out-dir", type=Path, default=None)
    return p.parse_args()


def safe_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(x):
        return None
    return x


def safe_mean(values: Iterable[Any]) -> float:
    xs = [x for x in (safe_float(v) for v in values) if x is not None]
    return float(sum(xs) / len(xs)) if xs else float("nan")


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def list_at(xs: Any, i: int) -> Any:
    if isinstance(xs, list) and i < len(xs):
        return xs[i]
    return None


def cid_to_index(cid: Any, fallback: int) -> int:
    try:
        return int(cid)
    except (TypeError, ValueError):
        return int(fallback)


def source_path_for_readme(result_path: Path, readme_dir: Path) -> str:
    """Path for README (POSIX slashes), relative to readme_dir when possible."""
    try:
        rel = os.path.relpath(result_path.resolve(), readme_dir.resolve())
    except (OSError, ValueError):
        return result_path.as_posix()
    return Path(rel).as_posix()


def main() -> None:
    args = parse_args()
    result = json.loads(args.result.read_text(encoding="utf-8"))
    method_obj = result.get("results", {}).get(args.method, {})
    trace = method_obj.get("round_trace", [])
    out_dir = args.out_dir or (args.result.parent / f"{args.result.stem}_spectral_decomposition")
    out_dir.mkdir(parents=True, exist_ok=True)

    label_hist = (
        result.get("meta", {}).get("client_class_distribution")
        or result.get("meta", {}).get("client_label_hist")
        or []
    )

    round_rows: List[Dict[str, Any]] = []
    client_rows: List[Dict[str, Any]] = []
    for row in trace:
        round_rows.append(
            {
                "round": row.get("round"),
                "graph_mode": row.get("graph_mode"),
                "graph_source": row.get("graph_source"),
                "graph_source_used": row.get("graph_source_used"),
                "aggregation_target": row.get("aggregation_target"),
                "aggregation_target_used": row.get("aggregation_target_used"),
                "knn_k": row.get("knn_k"),
                "graph_used_source": row.get("graph_used_source"),
                "h_spec": row.get("h_spec"),
                "h_spec_raw_current_graph": row.get("h_spec_raw_current_graph"),
                "low_frequency_energy_ratio": row.get("low_frequency_energy_ratio"),
                "mid_frequency_energy_ratio": row.get("mid_frequency_energy_ratio"),
                "high_frequency_energy_ratio": row.get("high_frequency_energy_ratio"),
                "high_to_low_energy_ratio": row.get("high_to_low_energy_ratio"),
                "spectral_entropy": row.get("spectral_entropy"),
                "eigengap_max": row.get("eigengap_max"),
                "dominant_frequency_mode_index": row.get("dominant_frequency_mode_index"),
                "dominant_frequency_mode_lambda": row.get("dominant_frequency_mode_lambda"),
                "dominant_frequency_energy_ratio": row.get("dominant_frequency_energy_ratio"),
                "graph_density": row.get("graph_density"),
                "raw_current_graph_density": row.get("raw_current_graph_density"),
                "alpha_mode": row.get("alpha_mode"),
                "effective_clients": row.get("effective_clients"),
                "entropy_alpha": row.get("entropy_alpha"),
            }
        )

        cids = row.get("cids") or []
        low_client = row.get("low_frequency_component_norm_ratio_list") or []
        mid_client = row.get("mid_frequency_component_norm_ratio_list") or []
        high_client = row.get("high_frequency_component_norm_ratio_list") or []
        e_list = row.get("e_list") or []
        e_z_list = row.get("e_z_list") or []
        alpha = row.get("alpha_norm_list") or row.get("alpha_list") or []
        n_examples = row.get("client_num_examples") or []
        train_acc = row.get("client_train_accuracy_list") or []
        train_loss = row.get("client_train_loss_list") or []
        for i, cid in enumerate(cids):
            client_index = cid_to_index(cid, i)
            client_rows.append(
                {
                    "round": row.get("round"),
                    "graph_mode": row.get("graph_mode"),
                    "graph_source": row.get("graph_source"),
                    "aggregation_target": row.get("aggregation_target"),
                    "client_id": client_index,
                    "flower_cid": cid,
                    "label_hist": json.dumps(
                        list_at(label_hist, client_index), ensure_ascii=False
                    ),
                    "num_examples": list_at(n_examples, i),
                    "low_component_norm_ratio": list_at(low_client, i),
                    "mid_component_norm_ratio": list_at(mid_client, i),
                    "high_component_norm_ratio": list_at(high_client, i),
                    "spectral_residual_e": list_at(e_list, i),
                    "spectral_residual_e_z": list_at(e_z_list, i),
                    "alpha": list_at(alpha, i),
                    "client_train_acc": list_at(train_acc, i),
                    "client_train_loss": list_at(train_loss, i),
                }
            )

    write_csv(round_rows, out_dir / "round_frequency_decomposition.csv")
    write_csv(client_rows, out_dir / "client_frequency_decomposition.csv")

    lines = [
        "# Spectral Frequency Decomposition Report",
        "",
        f"- Source (relative to this folder): `{source_path_for_readme(args.result, out_dir)}`",
        f"- Method: `{args.method}`",
        f"- Rounds: {len(round_rows)}",
        "",
        "## Round Means",
        "",
        f"- mean low energy ratio: {safe_mean(r.get('low_frequency_energy_ratio') for r in round_rows):.4f}",
        f"- mean mid energy ratio: {safe_mean(r.get('mid_frequency_energy_ratio') for r in round_rows):.4f}",
        f"- mean high energy ratio: {safe_mean(r.get('high_frequency_energy_ratio') for r in round_rows):.4f}",
        f"- mean high/low ratio: {safe_mean(r.get('high_to_low_energy_ratio') for r in round_rows):.4f}",
        f"- mean spectral entropy: {safe_mean(r.get('spectral_entropy') for r in round_rows):.4f}",
        f"- mean effective clients: {safe_mean(r.get('effective_clients') for r in round_rows):.4f}",
        "",
        "## Files",
        "",
        "- `round_frequency_decomposition.csv`",
        "- `client_frequency_decomposition.csv`",
    ]
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {out_dir}")


if __name__ == "__main__":
    main()
