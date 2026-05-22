"""Reporting helpers for vision-FL suite outputs.

Migration B policy: keep historical variant tags intact, but let reporting pair
canonical ``ours_graph_filtered_*`` families with legacy ``ours_spectral_filtered_*``
families when comparing kNN vs matched-random summaries. Do not collapse those
families into one prefix during Gate 5/C5 cleanup.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from graphfl_lab.experiments.suites.result_writer import write_csv_rows, write_json


BASE_KNN_PAIR_PREFIX = "ours_knn_k"
GRAPH_FILTERED_KNN_PAIR_PREFIX = "ours_graph_filtered_knn_k"
LEGACY_SPECTRAL_FILTERED_KNN_PAIR_PREFIX = "ours_spectral_filtered_knn_k"
LEGACY_RESIDUAL_KNN_PAIR_PREFIX = "ours_legacy_residual_reweight_knn_k"

BASE_RANDOM_PAIR_PREFIX = "ours_random_matched_k"
GRAPH_FILTERED_RANDOM_PAIR_PREFIX = "ours_graph_filtered_random_matched_k"
LEGACY_SPECTRAL_FILTERED_RANDOM_PAIR_PREFIX = "ours_spectral_filtered_random_matched_k"
LEGACY_RESIDUAL_RANDOM_PAIR_PREFIX = "ours_legacy_residual_reweight_random_matched_k"

KNN_PAIR_PREFIXES = (
    BASE_KNN_PAIR_PREFIX,
    GRAPH_FILTERED_KNN_PAIR_PREFIX,
    LEGACY_SPECTRAL_FILTERED_KNN_PAIR_PREFIX,
    LEGACY_RESIDUAL_KNN_PAIR_PREFIX,
)
RANDOM_PAIR_PREFIXES = (
    BASE_RANDOM_PAIR_PREFIX,
    GRAPH_FILTERED_RANDOM_PAIR_PREFIX,
    LEGACY_SPECTRAL_FILTERED_RANDOM_PAIR_PREFIX,
    LEGACY_RESIDUAL_RANDOM_PAIR_PREFIX,
)


def _variant_k_pair_key(variant: str, prefix: str) -> Tuple[int, str] | None:
    m = re.match(rf"^{re.escape(prefix)}(\d+)(?P<suffix>(?:_.+)?)$", variant)
    if not m:
        return None
    return int(m.group(1)), m.group("suffix") or ""


def _variant_k_pair_key_any(
    variant: str, prefixes: Tuple[str, ...]
) -> Tuple[int, str] | None:
    for prefix in prefixes:
        key = _variant_k_pair_key(variant, prefix)
        if key is not None:
            return key
    return None


def _variant_k_number(variant: str) -> int | None:
    key = _variant_k_pair_key_any(variant, KNN_PAIR_PREFIXES)
    return key[0] if key else None


def compute_best_knn_meta(summary_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Best k among ``ours_knn_k*`` variants by delta statistics."""
    knn_rows = [
        r
        for r in summary_rows
        if _variant_k_number(r.get("variant", "") or "") is not None
    ]
    if not knn_rows:
        return {}

    def by_mean(r: Dict[str, Any]):
        return (
            r.get("mean_delta", float("-inf")),
            r.get("min_delta", float("-inf")),
            -r.get("std_delta", 0.0),
            r.get("win_rate", 0.0),
        )

    mean_row = max(knn_rows, key=by_mean)
    min_row = max(
        knn_rows,
        key=lambda r: (
            r.get("min_delta", float("-inf")),
            r.get("mean_delta", float("-inf")),
            -r.get("std_delta", 0.0),
            r.get("win_rate", 0.0),
        ),
    )
    win_row = max(
        knn_rows,
        key=lambda r: (
            r.get("win_rate", 0.0),
            r.get("mean_delta", float("-inf")),
            r.get("min_delta", float("-inf")),
            -r.get("std_delta", 0.0),
        ),
    )
    return {
        "best_k_by_mean_delta": _variant_k_number(mean_row["variant"]),
        "best_knn_variant_by_mean_delta": mean_row["variant"],
        "best_k_by_min_delta": _variant_k_number(min_row["variant"]),
        "best_knn_variant_by_min_delta": min_row["variant"],
        "best_k_by_win_rate": _variant_k_number(win_row["variant"]),
        "best_knn_variant_by_win_rate": win_row["variant"],
    }


def _classify_knn_vs_random(knn: Dict[str, Any], rnd: Dict[str, Any]) -> str:
    eps = 0.003
    km = float(knn.get("mean_delta", float("nan")))
    rm = float(rnd.get("mean_delta", float("nan")))
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
    by_variant = {r["variant"]: r for r in summary_rows}
    knn_by_key: Dict[Tuple[int, str], Dict[str, Any]] = {}
    random_by_key: Dict[Tuple[int, str], Dict[str, Any]] = {}
    for row in summary_rows:
        variant = row.get("variant", "") or ""
        knn_key = _variant_k_pair_key_any(variant, KNN_PAIR_PREFIXES)
        if knn_key is not None:
            knn_by_key[knn_key] = row
            continue
        random_key = _variant_k_pair_key_any(variant, RANDOM_PAIR_PREFIXES)
        if random_key is not None:
            random_by_key[random_key] = row

    fieldnames = [
        "k",
        "variant_suffix",
        "knn_variant",
        "random_variant",
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
    keys = sorted(
        set(knn_by_key) | set(random_by_key), key=lambda item: (item[1], item[0])
    )
    if not keys:
        keys = [(k, "") for k in (2, 3, 5)]

    rows_out: List[Dict[str, Any]] = []
    for k, suffix in keys:
        knn_name = f"ours_knn_k{k}{suffix}"
        random_name = f"ours_random_matched_k{k}{suffix}"
        knn = knn_by_key.get((k, suffix), by_variant.get(knn_name))
        rnd = random_by_key.get((k, suffix), by_variant.get(random_name))
        if not knn or not rnd:
            rows_out.append(
                {
                    "k": k,
                    "variant_suffix": suffix,
                    "knn_variant": knn_name if knn else "",
                    "random_variant": random_name if rnd else "",
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

        km = float(knn.get("mean_delta", float("nan")))
        rm = float(rnd.get("mean_delta", float("nan")))
        kmin = float(knn.get("min_delta", float("nan")))
        rmin = float(rnd.get("min_delta", float("nan")))
        rows_out.append(
            {
                "k": k,
                "variant_suffix": suffix,
                "knn_variant": knn.get("variant", knn_name),
                "random_variant": rnd.get("variant", random_name),
                "knn_mean_delta": km,
                "random_mean_delta": rm,
                "difference_mean_delta": (
                    km - rm if not (math.isnan(km) or math.isnan(rm)) else float("nan")
                ),
                "knn_min_delta": kmin,
                "random_min_delta": rmin,
                "difference_min_delta": (
                    kmin - rmin
                    if not (math.isnan(kmin) or math.isnan(rmin))
                    else float("nan")
                ),
                "knn_win_rate": knn.get("win_rate", float("nan")),
                "random_win_rate": rnd.get("win_rate", float("nan")),
                "knn_graph_density": knn.get("mean_graph_density", float("nan")),
                "random_graph_density": rnd.get("mean_graph_density", float("nan")),
                "interpretation": _classify_knn_vs_random(knn, rnd),
            }
        )

    path = out_dir / "knn_vs_random_matched.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)
    return path


def write_interpretation_md(
    out_dir: Path,
    summary_rows: List[Dict[str, Any]],
    suite_meta: Dict[str, Any],
    args: argparse.Namespace,
) -> Path:
    """Write a conservative generated interpretation note."""
    by_variant = {r["variant"]: r for r in summary_rows}
    fed = by_variant.get("fedavg", {})
    dense = by_variant.get("ours_dense", {})
    best_meta = suite_meta.get("best_knn_by_meta") or {}
    train_subset = int(args.train_subset_size)
    test_subset = int(args.test_subset_size)

    lines = [
        "# Vision-FL suite interpretation",
        "",
        "## Run Configuration",
        "",
    ]
    if train_subset <= 0 and test_subset <= 0:
        lines.append(
            "- Training uses the full train split and test uses the full test split."
        )
    else:
        lines.append(
            f"- Subset mode: train_subset_size={train_subset}, "
            f"test_subset_size={test_subset}."
        )
    lines.extend(
        [
            "",
            "## Headline Comparisons",
            "",
            f"- FedAvg mean_acc: {fed.get('mean_acc', float('nan')):.4f}.",
            f"- Ours-dense mean_delta vs FedAvg: {dense.get('mean_delta', float('nan')):+.4f}.",
            "",
            "## kNN vs Matched Random",
            "",
            "Use `knn_vs_random_matched.csv` to separate similarity-graph benefit "
            "from sparse-random regularization.",
            "",
            "## Auto Summary",
            "",
        ]
    )

    knn_variants = sorted(
        [v for v in by_variant if _variant_k_number(v)],
        key=lambda v: (_variant_k_number(v) or 0),
    )
    if knn_variants:
        best_variant = max(
            knn_variants,
            key=lambda v: (
                by_variant[v].get("mean_delta", float("-inf")),
                by_variant[v].get("min_delta", float("-inf")),
                -by_variant[v].get("std_delta", 0.0),
                by_variant[v].get("win_rate", 0.0),
            ),
        )
        best = by_variant[best_variant]
        lines.append(
            f"- Best kNN variant by mean_delta: `{best_variant}` "
            f"(mean_delta={best.get('mean_delta'):+.4f}, "
            f"min_delta={best.get('min_delta'):+.4f}, "
            f"win_rate={best.get('win_rate'):.2f})."
        )
        if best_meta.get("best_k_by_mean_delta") is not None:
            lines.append(
                f"- best_k_by_mean_delta: {best_meta.get('best_k_by_mean_delta')} "
                f"(`{best_meta.get('best_knn_variant_by_mean_delta')}`)."
            )
    lines.append("")
    lines.append(
        "Generated automatically. Treat this as a navigation aid, not a claim "
        "of superiority without inspecting per-seed deltas and controls."
    )
    path = out_dir / "interpretation.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def append_validation_verdict(out_dir: Path, summary_rows: List[Dict[str, Any]]) -> None:
    by_variant = {r["variant"]: r for r in summary_rows}
    ours_rows = [r for r in summary_rows if r["variant"] != "fedavg"]
    if not ours_rows:
        return

    lines = ["", "## Automated Alpha-Sweep Gate", ""]
    eps = 0.003

    def flush() -> None:
        path = out_dir / "interpretation.md"
        path.write_text(
            path.read_text(encoding="utf-8") + "\n".join(lines),
            encoding="utf-8",
        )

    finite = [
        r
        for r in ours_rows
        if not math.isnan(float(r.get("mean_delta", float("nan"))))
    ]
    if finite and all(r.get("mean_delta", 0) < 0 for r in finite):
        lines.append(
            "Every Ours variant has mean_delta < 0 vs FedAvg; skip the alpha "
            "sweep until graph/spectrum diagnostics explain the regime."
        )
        flush()
        return

    random_by_key = {
        key: by_variant[variant]
        for variant in by_variant
        for key in [_variant_k_pair_key_any(variant, RANDOM_PAIR_PREFIXES)]
        if key is not None
    }
    knn_candidates = [
        (key, by_variant[variant])
        for variant in by_variant
        for key in [_variant_k_pair_key_any(variant, KNN_PAIR_PREFIXES)]
        if key is not None
    ]
    if not knn_candidates:
        lines.append("No kNN rows found; cannot classify the alpha-sweep gate.")
        flush()
        return

    best_key, best_knn = max(
        knn_candidates, key=lambda item: item[1].get("mean_delta", float("-inf"))
    )
    rnd = random_by_key.get(best_key)
    best_k, suffix = best_key
    best_variant = best_knn.get("variant", f"ours_knn_k{best_k}{suffix}")
    if rnd is None:
        lines.append(f"Missing matched random row for `{best_variant}`; inconclusive.")
        flush()
        return

    random_variant = rnd.get("variant", f"ours_random_matched_k{best_k}{suffix}")
    mean_knn = float(best_knn.get("mean_delta", float("nan")))
    mean_random = float(rnd.get("mean_delta", float("nan")))
    similar = (
        math.isnan(mean_knn)
        or math.isnan(mean_random)
        or abs(mean_knn - mean_random) <= eps
    )
    knn_beats_random = not similar and mean_knn > mean_random + eps
    min_ok = best_knn.get("min_delta", -999) >= -0.03
    win_ok = best_knn.get("win_rate", 0) >= 0.6

    if similar:
        lines.append(
            f"`{best_variant}` is close to `{random_variant}` at the same k; "
            "skip the alpha sweep for now."
        )
    elif knn_beats_random and mean_knn > 0 and min_ok and win_ok:
        lines.append(
            f"`{best_variant}` beats matched random with positive mean_delta; "
            "an alpha sweep is a reasonable next experiment."
        )
    elif knn_beats_random and mean_knn <= 0:
        lines.append(
            "kNN beats matched random but not FedAvg; graph structure appears "
            "meaningful, while aggregation gain remains weak."
        )
    else:
        lines.append(
            f"Mixed outcome for `{best_variant}` vs `{random_variant}`; inspect "
            "`knn_vs_random_matched.csv` and per-seed deltas."
        )
    dense = by_variant.get("ours_dense")
    if dense is not None:
        lines.append(
            f"- Dense comparison: `{best_variant}` mean_delta {mean_knn:+.4f} "
            f"vs `ours_dense` {dense.get('mean_delta', float('nan')):+.4f}."
        )
    flush()


def write_suite_summary_artifacts(
    out_dir: Path,
    suite_summary: Dict[str, Any],
    summary_rows: List[Dict[str, Any]],
    rows: List[Dict[str, Any]],
) -> tuple[Path, Path, Path]:
    """Write canonical suite artifacts and compatibility mirrors.

    Returns ``(summary_json, rows_path, csv_path)`` using canonical filenames.
    """
    summary_json = out_dir / "vision_suite_summary.json"
    rows_path = out_dir / "vision_suite_rows.json"
    csv_path = out_dir / "vision_suite_summary.csv"

    write_json(summary_json, suite_summary)
    write_json(rows_path, rows)
    write_json(out_dir / "suite_summary.json", suite_summary)
    write_json(out_dir / "suite_rows.json", rows)
    write_json(out_dir / "general_suite_summary.json", suite_summary)
    write_json(out_dir / "general_suite_rows.json", rows)

    if summary_rows:
        fieldnames = list(summary_rows[0].keys())
        write_csv_rows(csv_path, summary_rows, fieldnames=fieldnames)
        write_csv_rows(
            out_dir / "general_suite_summary.csv",
            summary_rows,
            fieldnames=fieldnames,
        )
        write_csv_rows(
            out_dir / "suite_summary.csv",
            summary_rows,
            fieldnames=fieldnames,
        )

    return summary_json, rows_path, csv_path


def duplicate_suite_summaries(
    out_dir: Path,
    suite_summary: Dict[str, Any],
    summary_rows: List[Dict[str, Any]],
    rows: List[Dict[str, Any]],
) -> None:
    """Compatibility alias for ``write_suite_summary_artifacts``."""
    write_suite_summary_artifacts(out_dir, suite_summary, summary_rows, rows)


def write_summary_markdown(
    out_dir: Path,
    suite_tag: str,
    args: argparse.Namespace,
    summary_rows: List[Dict[str, Any]],
) -> Path:
    path = out_dir / "vision_suite_summary.md"
    lines = [
        "# Vision FL suite summary",
        "",
        f"- Suite tag: `{suite_tag}`",
        f"- Dataset: `{args.dataset}`, Dirichlet alpha={args.dirichlet_alpha}, clients={args.num_clients}",
        "",
        "## Ranking",
        "",
        "| variant | mean_acc | std_acc | mean_delta | min_delta | max_delta | std_delta | win_rate | mean_H_spec | mean_tau | mean_graph_density | mean_entropy_alpha | mean_effective_clients |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row.get('variant')} | {row.get('mean_acc', float('nan')):.4f} | "
            f"{row.get('std_acc', float('nan')):.4f} | "
            f"{row.get('mean_delta', float('nan')):+.4f} | "
            f"{row.get('min_delta', float('nan')):+.4f} | "
            f"{row.get('max_delta', float('nan')):+.4f} | "
            f"{row.get('std_delta', float('nan')):.4f} | "
            f"{row.get('win_rate', 0):.2f} | "
            f"{row.get('mean_H_spec', float('nan')):.4f} | "
            f"{row.get('mean_tau', float('nan')):.4f} | "
            f"{row.get('mean_graph_density', float('nan')):.4f} | "
            f"{row.get('mean_entropy_alpha', float('nan')):.4f} | "
            f"{row.get('mean_effective_clients', float('nan')):.4f} |"
        )
    lines.append("")
    lines.extend(
        [
            "## Timing",
            "",
            "| variant | mean wall time(s) | std wall time(s) | sec/round | timing source | reused |",
            "|---|---:|---:|---:|---|---:|",
        ]
    )
    for row in summary_rows:
        lines.append(
            f"| {row.get('variant')} | "
            f"{row.get('mean_run_wall_time_sec', float('nan')):.1f} | "
            f"{row.get('std_run_wall_time_sec', float('nan')):.1f} | "
            f"{row.get('mean_seconds_per_round', float('nan')):.1f} | "
            f"{row.get('timing_source', '')} | "
            f"{row.get('reused_existing_result_count', 0)} |"
        )
    lines.append("")
    lines.extend(
        [
            "## Notes",
            "",
            "- `mean_delta`, `min_delta`, and `win_rate` are computed against same-seed FedAvg.",
            "- `mean_H_spec` is a graph-update alignment diagnostic, not a standalone non-IID score.",
            "- Check `knn_vs_random_matched.csv` before attributing gains to graph similarity.",
        ]
    )
    content = "\n".join(lines)
    path.write_text(content, encoding="utf-8")
    (out_dir / "general_suite_summary.md").write_text(content, encoding="utf-8")
    return path


def write_diagnostic_csv(
    out_dir: Path,
    summary_rows: List[Dict[str, Any]],
) -> Path:
    """Write compact diagnostic summary used by dashboard/plot scripts."""
    path = out_dir / "diagnostic_summary.csv"
    fieldnames = [
        "variant",
        "seeds",
        "mean_final_acc",
        "mean_delta_vs_fedavg",
        "mean_di_drop",
        "mean_neff_gain",
        "mean_alignment_gain",
        "mean_loo_drop",
        "win_rate",
    ]
    rows: List[Dict[str, Any]] = []
    for row in summary_rows:
        if row.get("variant") == "fedavg":
            continue
        di_pre = float(row.get("mean_di_pre", float("nan")))
        di_post = float(row.get("mean_di_post", float("nan")))
        neff_pre = float(row.get("mean_neff_pre", float("nan")))
        neff_post = float(row.get("mean_neff_post", float("nan")))
        align_pre = float(row.get("mean_alignment_pre", float("nan")))
        align_post = float(row.get("mean_alignment_post", float("nan")))
        loo_pre = float(row.get("mean_loo_pre", float("nan")))
        loo_post = float(row.get("mean_loo_post", float("nan")))
        rows.append(
            {
                "variant": row.get("variant", ""),
                "seeds": int(row.get("n_runs", 0)),
                "mean_final_acc": float(row.get("mean_acc", float("nan"))),
                "mean_delta_vs_fedavg": float(row.get("mean_delta", float("nan"))),
                "mean_di_drop": (
                    di_pre - di_post if not (math.isnan(di_pre) or math.isnan(di_post)) else float("nan")
                ),
                "mean_neff_gain": (
                    neff_post - neff_pre if not (math.isnan(neff_pre) or math.isnan(neff_post)) else float("nan")
                ),
                "mean_alignment_gain": (
                    align_post - align_pre
                    if not (math.isnan(align_pre) or math.isnan(align_post))
                    else float("nan")
                ),
                "mean_loo_drop": (
                    loo_pre - loo_post if not (math.isnan(loo_pre) or math.isnan(loo_post)) else float("nan")
                ),
                "win_rate": float(row.get("win_rate", float("nan"))),
            }
        )
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_dashboard_mockup(
    out_dir: Path,
    summary_rows: List[Dict[str, Any]],
    diagnostic_csv_path: Path | None = None,
) -> Path:
    """Write a static dashboard mockup markdown file."""
    lines = [
        "# Diagnostic Dashboard Mockup",
        "",
        "## Overview",
        "",
        f"- Variants: {max(len(summary_rows) - 1, 0)} (excluding fedavg)",
        f"- Generated files: `vision_suite_summary.csv`, `general_suite_summary.csv`, `knn_vs_random_matched.csv`",
    ]
    if diagnostic_csv_path is not None:
        lines.append(f"- Diagnostic summary: `{diagnostic_csv_path.name}`")
    lines.extend(
        [
            "",
            "## Top Variants",
            "",
            "| variant | mean_delta | win_rate | mean_graph_density |",
            "|---|---:|---:|---:|",
        ]
    )
    ranked = [r for r in summary_rows if r.get("variant") != "fedavg"]
    ranked = sorted(
        ranked,
        key=lambda r: (
            float(r.get("mean_delta", float("-inf"))),
            float(r.get("win_rate", float("-inf"))),
        ),
        reverse=True,
    )
    for row in ranked[:5]:
        lines.append(
            f"| {row.get('variant')} | {float(row.get('mean_delta', float('nan'))):+.4f} | "
            f"{float(row.get('win_rate', float('nan'))):.2f} | "
            f"{float(row.get('mean_graph_density', float('nan'))):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Linked Artifacts",
            "",
            "- `vision_suite_summary.md`",
            "- `general_suite_summary.md`",
            "- `interpretation.md`",
            "- `knn_vs_random_matched.csv`",
            "- `diagnostic_summary.csv`",
            "",
            "This mockup is intentionally static and lightweight for CI-safe generation.",
        ]
    )
    path = out_dir / "dashboard_mockup.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


__all__ = [
    "append_validation_verdict",
    "compute_best_knn_meta",
    "duplicate_suite_summaries",
    "write_suite_summary_artifacts",
    "write_dashboard_mockup",
    "write_diagnostic_csv",
    "write_interpretation_md",
    "write_knn_vs_random_matched_csv",
    "write_summary_markdown",
    "_variant_k_number",
]
