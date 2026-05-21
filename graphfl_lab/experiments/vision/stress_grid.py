"""Run vision FL stress suites across clients, data size, and k.

This wrapper is for regimes where FedAvg is expected to wobble under strong
label skew.  Each grid point runs ``run_vision_suite.py`` once, with all
requested k values expanded into the variant list.  This avoids rerunning
FedAvg separately for every k.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from graphfl_lab.experiments.suites.execution import run_cmd
from graphfl_lab.experiments.suites.result_writer import write_json


PROJECT_ROOT = Path(__file__).resolve().parents[3]


DEFAULT_VARIANT_TEMPLATES = [
    "fedavg",
    "ours_knn_k{k}_fixed_tau",
    "ours_tail_m2_knn_k{k}_fixed_tau",
    "ours_layerwise_tail_m2_knn_k{k}_fixed_tau",
    "ours_random_matched_k{k}_fixed_tau",
]



def str_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def as_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def float_tag(value: float) -> str:
    text = f"{float(value):g}"
    return text.replace("-", "m").replace(".", "p")


def expand_variant_templates(templates: Sequence[str], knn_ks: Sequence[int]) -> List[str]:
    variants: List[str] = []
    seen = set()
    for template in templates:
        expanded: Iterable[str]
        if "{k}" in template:
            expanded = (template.format(k=int(k)) for k in knn_ks)
        else:
            expanded = (template,)
        for variant in expanded:
            key = str(variant).strip().lower()
            if key and key not in seen:
                seen.add(key)
                variants.append(key)
    return variants


def variant_k_from_label(variant: str) -> int | None:
    match = re.search(r"_k(\d+)(?:_|$)", str(variant))
    return int(match.group(1)) if match else None


def iter_grid(args: argparse.Namespace) -> Iterable[Dict[str, Any]]:
    for alpha, local_epochs, n_clients, train_size, test_size in itertools.product(
        as_list(args.dirichlet_alphas),
        as_list(args.local_epochs),
        as_list(args.client_counts),
        as_list(args.train_subset_sizes),
        as_list(args.test_subset_sizes),
    ):
        min_client_weight = float(args.min_client_weight)
        yield {
            "dirichlet_alpha": float(alpha),
            "local_epochs": int(local_epochs),
            "num_clients": int(n_clients),
            "train_subset_size": int(train_size),
            "test_subset_size": int(test_size),
            "min_client_weight": min_client_weight,
            "weight_floor_total": min_client_weight * float(n_clients),
        }


def suite_name(args: argparse.Namespace, combo: Dict[str, Any]) -> str:
    return (
        f"{args.grid_tag}"
        f"_a{float_tag(combo['dirichlet_alpha'])}"
        f"_le{combo['local_epochs']}"
        f"_n{combo['num_clients']}"
        f"_tr{combo['train_subset_size']}"
        f"_te{combo['test_subset_size']}"
    )


def suite_cmd(
    args: argparse.Namespace,
    combo: Dict[str, Any],
    suite_dir: Path,
    suite_tag: str,
    variants: Sequence[str],
) -> List[str]:
    return [
        str(args.python_bin),
        "run_vision_suite.py",
        "--dataset",
        str(args.dataset),
        "--model",
        str(args.model),
        "--num-clients",
        str(combo["num_clients"]),
        "--rounds",
        str(args.rounds),
        "--local-epochs",
        str(combo["local_epochs"]),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--momentum",
        str(args.momentum),
        "--weight-decay",
        str(args.weight_decay),
        "--seeds",
        *[str(s) for s in args.seeds],
        "--partition",
        str(args.partition),
        "--dirichlet-alpha",
        str(combo["dirichlet_alpha"]),
        "--projection-dim",
        str(args.projection_dim),
        "--compression-seed",
        str(args.compression_seed),
        "--ema-alpha",
        str(args.ema_alpha),
        "--tau-gain",
        str(args.tau_gain),
        "--tau-max",
        str(args.tau_max),
        "--conflict-mix",
        str(args.conflict_mix),
        "--warmup-rounds",
        str(args.warmup_rounds),
        "--knn-k",
        str(as_list(args.knn_ks)[0]),
        "--graph-source",
        str(args.graph_source),
        "--aggregation-target",
        str(args.aggregation_target),
        "--edge-threshold",
        str(args.edge_threshold),
        "--graph-scale-sigma",
        str(args.graph_scale_sigma),
        "--learned-graph-lambda",
        str(args.learned_graph_lambda),
        "--graph-layer-start",
        str(args.graph_layer_start),
        "--graph-layer-end",
        str(args.graph_layer_end),
        "--graph-seed",
        str(args.graph_seed),
        "--use-ema-graph",
        str(args.use_ema_graph),
        "--disable-adaptive-tau",
        str(args.disable_adaptive_tau),
        "--fixed-tau",
        str(args.fixed_tau),
        "--tau-source",
        str(args.tau_source),
        "--graph-filter-strength",
        str(args.graph_filter_strength),
        "--client-update-ema-alpha",
        str(args.client_update_ema_alpha),
        "--diagnostic-only",
        str(args.diagnostic_only),
        "--e-std-threshold",
        str(args.e_std_threshold),
        "--min-client-weight",
        str(args.min_client_weight),
        "--server-learning-rate",
        str(args.server_learning_rate),
        "--server-momentum",
        str(args.server_momentum),
        "--ours-server-learning-rate",
        str(args.ours_server_learning_rate),
        "--ours-server-momentum",
        str(args.ours_server_momentum),
        "--fedprox-mu",
        str(args.fedprox_mu),
        "--fedopt-eta",
        str(args.fedopt_eta),
        "--fedopt-eta-l",
        str(args.fedopt_eta_l),
        "--fedopt-beta1",
        str(args.fedopt_beta1),
        "--fedopt-beta2",
        str(args.fedopt_beta2),
        "--fedopt-tau",
        str(args.fedopt_tau),
        "--trimmed-beta",
        str(args.trimmed_beta),
        "--data-root",
        str(args.data_root),
        "--train-subset-size",
        str(combo["train_subset_size"]),
        "--test-subset-size",
        str(combo["test_subset_size"]),
        "--out-dir",
        str(suite_dir),
        "--suite-tag",
        suite_tag,
        "--reuse-existing-results",
        str(args.reuse_existing_results),
        "--variants",
        *[str(v) for v in variants],
    ]


def read_summary_rows(
    path: Path,
    combo: Dict[str, Any],
    knn_ks: Sequence[int],
    suite_dir: Path,
    suite_tag: str,
) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out = dict(row)
            out.update(combo)
            out["knn_ks"] = " ".join(str(k) for k in knn_ks)
            out["variant_knn_k"] = variant_k_from_label(str(row.get("variant", "")))
            out["suite_tag"] = suite_tag
            out["suite_dir"] = str(suite_dir)
            rows.append(out)
    return rows


def csv_float(row: Dict[str, Any], key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except (TypeError, ValueError):
        return float("nan")


def finite(value: float) -> bool:
    return value == value and value not in {float("inf"), float("-inf")}


def condition_key(row: Dict[str, Any]) -> tuple:
    return (
        float(row.get("dirichlet_alpha", 0.0)),
        int(row.get("local_epochs", 0)),
        int(row.get("num_clients", 0)),
        int(row.get("train_subset_size", 0)),
        int(row.get("test_subset_size", 0)),
        float(row.get("min_client_weight", 0.0)),
    )


def variant_family(variant: str) -> str:
    v = str(variant).lower()
    if v == "fedavg":
        return "fedavg"
    if v.startswith("ours_random_matched_"):
        return "random_matched"
    if v.startswith("ours_layerwise_tail_"):
        return "layerwise_tail_knn"
    if v.startswith("ours_tail_"):
        return "tail_knn"
    if v.startswith("ours_knn_"):
        return "knn"
    return "other"


def build_auto_review_rows(
    summary_rows: List[Dict[str, Any]],
    collapse_acc_threshold: float,
    meaningful_delta: float,
    random_margin: float,
) -> List[Dict[str, Any]]:
    by_condition: Dict[tuple, List[Dict[str, Any]]] = {}
    for row in summary_rows:
        by_condition.setdefault(condition_key(row), []).append(row)

    review_rows: List[Dict[str, Any]] = []
    for key, group in sorted(by_condition.items()):
        fed_row = next((r for r in group if r.get("variant") == "fedavg"), None)
        fedavg_acc = (
            csv_float(fed_row, "mean_acc")
            if fed_row is not None
            else csv_float(group[0], "mean_fedavg_acc")
        )
        fedavg_collapsed = bool(finite(fedavg_acc) and fedavg_acc <= collapse_acc_threshold)
        num_clients = int(group[0].get("num_clients", 0))
        min_client_weight = csv_float(group[0], "min_client_weight")
        weight_floor_total = (
            min_client_weight * float(num_clients)
            if finite(min_client_weight)
            else float("nan")
        )
        weight_floor_saturated = bool(
            finite(weight_floor_total) and weight_floor_total >= 1.0 - 1e-12
        )
        weight_floor_heavy = bool(
            finite(weight_floor_total) and weight_floor_total >= 0.5
        )
        random_by_k = {
            variant_k_from_label(str(r.get("variant", ""))): r
            for r in group
            if variant_family(str(r.get("variant", ""))) == "random_matched"
        }
        for row in group:
            variant = str(row.get("variant", ""))
            family = variant_family(variant)
            if family == "fedavg":
                continue
            k_value = variant_k_from_label(variant)
            random_row = random_by_k.get(k_value)
            mean_delta = csv_float(row, "mean_delta")
            min_delta = csv_float(row, "min_delta")
            win_rate = csv_float(row, "win_rate")
            random_delta = (
                csv_float(random_row, "mean_delta")
                if random_row is not None
                else float("nan")
            )
            delta_vs_random = (
                mean_delta - random_delta
                if finite(mean_delta) and finite(random_delta)
                else float("nan")
            )
            beats_fedavg = bool(
                finite(mean_delta)
                and mean_delta >= meaningful_delta
                and (not finite(win_rate) or win_rate >= 0.5)
            )
            beats_random = bool(
                finite(delta_vs_random) and delta_vs_random >= random_margin
            )
            if weight_floor_saturated:
                verdict = "weight_floor_saturated"
            elif not fedavg_collapsed:
                verdict = "fedavg_not_collapsed"
            elif family == "random_matched":
                verdict = "random_control"
            elif beats_fedavg and beats_random:
                verdict = "promising_rescue"
            elif beats_fedavg and random_row is None:
                verdict = "rescue_without_random_control"
            elif beats_fedavg:
                verdict = "sparsity_or_random_close"
            elif finite(mean_delta) and mean_delta < -meaningful_delta:
                verdict = "worse_than_fedavg"
            else:
                verdict = "no_clear_rescue"

            score = mean_delta
            if finite(delta_vs_random):
                score += delta_vs_random
            if finite(min_delta):
                score += 0.25 * min_delta
            if finite(win_rate):
                score += 0.01 * win_rate
            review_rows.append(
                {
                    "verdict": verdict,
                    "review_score": score,
                    "fedavg_collapsed": fedavg_collapsed,
                    "fedavg_acc": fedavg_acc,
                    "collapse_acc_threshold": collapse_acc_threshold,
                    "meaningful_delta": meaningful_delta,
                    "random_margin": random_margin,
                    "weight_floor_total": weight_floor_total,
                    "weight_floor_saturated": weight_floor_saturated,
                    "weight_floor_heavy": weight_floor_heavy,
                    "variant_family": family,
                    "random_variant": (
                        random_row.get("variant", "") if random_row is not None else ""
                    ),
                    "random_mean_delta": random_delta,
                    "delta_vs_random": delta_vs_random,
                    "k": "" if k_value is None else k_value,
                    **row,
                }
            )
    return review_rows


def write_auto_review(
    review_rows: List[Dict[str, Any]],
    root: Path,
    args: argparse.Namespace,
) -> None:
    preferred = [
        "verdict",
        "review_score",
        "fedavg_collapsed",
        "fedavg_acc",
        "dirichlet_alpha",
        "local_epochs",
        "num_clients",
        "train_subset_size",
        "test_subset_size",
        "min_client_weight",
        "weight_floor_total",
        "weight_floor_saturated",
        "weight_floor_heavy",
        "k",
        "variant",
        "variant_family",
        "mean_acc",
        "mean_delta",
        "min_delta",
        "win_rate",
        "random_variant",
        "random_mean_delta",
        "delta_vs_random",
        "mean_graph_density",
        "suite_dir",
    ]
    write_csv(review_rows, root / "stress_grid_auto_review.csv", preferred)
    write_json(root / "stress_grid_auto_review.json", review_rows)

    completed = [r for r in review_rows if r.get("variant")]
    promising = sorted(
        [r for r in completed if r.get("verdict") == "promising_rescue"],
        key=lambda r: csv_float(r, "review_score"),
        reverse=True,
    )
    collapsed = [r for r in completed if str(r.get("fedavg_collapsed")) == "True"]
    no_collapse = [r for r in completed if str(r.get("fedavg_collapsed")) != "True"]
    saturated_floor = [
        r for r in completed if str(r.get("weight_floor_saturated")) == "True"
    ]
    heavy_floor = [
        r
        for r in completed
        if str(r.get("weight_floor_heavy")) == "True"
        and str(r.get("weight_floor_saturated")) != "True"
    ]
    by_condition: Dict[tuple, List[Dict[str, Any]]] = {}
    for row in completed:
        by_condition.setdefault(condition_key(row), []).append(row)
    best_by_condition = []
    for group in by_condition.values():
        candidates = [r for r in group if r.get("variant_family") != "random_matched"]
        if candidates:
            best_by_condition.append(
                max(candidates, key=lambda r: csv_float(r, "review_score"))
            )

    lines = [
        "# Stress Grid Auto Review",
        "",
        "This is a triage report, not a statistical proof.",
        "",
        "## Thresholds",
        "",
        f"- FedAvg collapse accuracy threshold: `{args.fedavg_collapse_acc_threshold}`",
        f"- meaningful mean_delta: `{args.meaningful_delta}`",
        f"- kNN-vs-random margin: `{args.random_margin}`",
        "",
        "## Verdict Counts",
        "",
    ]
    counts: Dict[str, int] = {}
    for row in completed:
        verdict = str(row.get("verdict", ""))
        counts[verdict] = counts.get(verdict, 0) + 1
    for verdict, count in sorted(counts.items()):
        lines.append(f"- `{verdict}`: {count}")
    lines.extend(
        [
            "",
            f"- rows in collapsed FedAvg conditions: `{len(collapsed)}`",
            f"- rows in non-collapsed FedAvg conditions: `{len(no_collapse)}`",
            f"- rows with saturated weight floor: `{len(saturated_floor)}`",
            f"- rows with heavy but unsaturated weight floor: `{len(heavy_floor)}`",
            "",
            "## Top Promising Rescue Candidates",
            "",
            "| score | alpha | local epochs | clients | train | test | k | variant | FedAvg acc | mean delta | vs random | win rate |",
            "|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in promising[:20]:
        lines.append(
            "| "
            f"{csv_float(row, 'review_score'):.4f} | "
            f"{float(row.get('dirichlet_alpha', float('nan'))):.4g} | "
            f"{row.get('local_epochs')} | "
            f"{row.get('num_clients')} | "
            f"{row.get('train_subset_size')} | "
            f"{row.get('test_subset_size')} | "
            f"{row.get('k')} | "
            f"{row.get('variant')} | "
            f"{csv_float(row, 'fedavg_acc'):.4f} | "
            f"{csv_float(row, 'mean_delta'):+.4f} | "
            f"{csv_float(row, 'delta_vs_random'):+.4f} | "
            f"{csv_float(row, 'win_rate'):.2f} |"
        )
    if not promising:
        lines.append("|  |  |  |  |  |  |  | no promising rescue rows yet |  |  |  |  |")

    if saturated_floor or heavy_floor:
        lines.extend(
            [
                "",
                "## Weight Floor Diagnostics",
                "",
                "| verdict | clients | min client weight | floor total | variant | mean delta | note |",
                "|---|---:|---:|---:|---|---:|---|",
            ]
        )
        for row in sorted(
            saturated_floor + heavy_floor,
            key=lambda r: (
                -csv_float(r, "weight_floor_total"),
                str(r.get("variant", "")),
            ),
        )[:30]:
            note = (
                "floor forces uniform client weights"
                if str(row.get("weight_floor_saturated")) == "True"
                else "floor may dominate graph weights"
            )
            lines.append(
                "| "
                f"{row.get('verdict')} | "
                f"{row.get('num_clients')} | "
                f"{csv_float(row, 'min_client_weight'):.4f} | "
                f"{csv_float(row, 'weight_floor_total'):.4f} | "
                f"{row.get('variant')} | "
                f"{csv_float(row, 'mean_delta'):+.4f} | "
                f"{note} |"
            )

    lines.extend(
        [
            "",
            "## Best Candidate Per Condition",
            "",
            "| verdict | alpha | local epochs | clients | train | test | k | variant | FedAvg acc | mean delta | vs random |",
            "|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|",
        ]
    )
    for row in sorted(
        best_by_condition,
        key=lambda r: (
            str(r.get("verdict", "")),
            -csv_float(r, "review_score"),
        ),
    )[:50]:
        lines.append(
            "| "
            f"{row.get('verdict')} | "
            f"{float(row.get('dirichlet_alpha', float('nan'))):.4g} | "
            f"{row.get('local_epochs')} | "
            f"{row.get('num_clients')} | "
            f"{row.get('train_subset_size')} | "
            f"{row.get('test_subset_size')} | "
            f"{row.get('k')} | "
            f"{row.get('variant')} | "
            f"{csv_float(row, 'fedavg_acc'):.4f} | "
            f"{csv_float(row, 'mean_delta'):+.4f} | "
            f"{csv_float(row, 'delta_vs_random'):+.4f} |"
        )
    (root / "stress_grid_auto_review.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def write_csv(rows: List[Dict[str, Any]], path: Path, preferred: Sequence[str]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(preferred)
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: List[Dict[str, Any]], path: Path, args: argparse.Namespace) -> None:
    ordered = sorted(
        rows,
        key=lambda r: (
            float(r.get("dirichlet_alpha", 0.0)),
            int(r.get("local_epochs", 0)),
            int(r.get("num_clients", 0)),
            int(r.get("train_subset_size", 0)),
            int(r.get("test_subset_size", 0)),
            str(r.get("variant", "")),
        ),
    )
    lines = [
        "# Vision FL Stress Grid Summary",
        "",
        f"- clients: `{', '.join(str(x) for x in as_list(args.client_counts))}`",
        f"- train subsets: `{', '.join(str(x) for x in as_list(args.train_subset_sizes))}`",
        f"- test subsets: `{', '.join(str(x) for x in as_list(args.test_subset_sizes))}`",
        f"- k values: `{', '.join(str(x) for x in as_list(args.knn_ks))}`",
        f"- alphas: `{', '.join(str(x) for x in as_list(args.dirichlet_alphas))}`",
        f"- local epochs: `{', '.join(str(x) for x in as_list(args.local_epochs))}`",
        f"- rounds: `{args.rounds}`, warmup_rounds: `{args.warmup_rounds}`, fixed_tau: `{args.fixed_tau}`",
        "",
        "| alpha | local epochs | clients | train | test | k | variant | mean acc | mean delta | min delta | win rate | density |",
        "|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in ordered:
        k_value = row.get("variant_knn_k")
        k_text = "" if k_value in {None, ""} else str(k_value)
        lines.append(
            "| "
            f"{float(row.get('dirichlet_alpha', float('nan'))):.4g} | "
            f"{row.get('local_epochs')} | "
            f"{row.get('num_clients')} | "
            f"{row.get('train_subset_size')} | "
            f"{row.get('test_subset_size')} | "
            f"{k_text} | "
            f"{row.get('variant')} | "
            f"{csv_float(row, 'mean_acc'):.4f} | "
            f"{csv_float(row, 'mean_delta'):+.4f} | "
            f"{csv_float(row, 'min_delta'):+.4f} | "
            f"{csv_float(row, 'win_rate'):.2f} | "
            f"{csv_float(row, 'mean_graph_density'):.4f} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_manifest(rows: List[Dict[str, Any]], root: Path) -> None:
    preferred = [
        "status",
        "suite_tag",
        "suite_dir",
        "dirichlet_alpha",
        "local_epochs",
        "num_clients",
        "min_client_weight",
        "weight_floor_total",
        "train_subset_size",
        "test_subset_size",
        "knn_ks",
        "variants",
        "command",
    ]
    write_csv(rows, root / "stress_grid_manifest.csv", preferred)
    write_json(root / "stress_grid_manifest.json", rows)


def run(args) -> None:
    root = Path(args.out_dir)
    root.mkdir(parents=True, exist_ok=True)
    dry_run = str_bool(args.dry_run)
    skip_existing = str_bool(args.skip_existing)
    knn_ks = [int(k) for k in as_list(args.knn_ks)]
    variants = expand_variant_templates(as_list(args.variant_templates), knn_ks)
    max_suites = int(args.max_suites)

    manifest_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    for idx, combo in enumerate(iter_grid(args), start=1):
        if max_suites > 0 and idx > max_suites:
            break
        tag = suite_name(args, combo)
        suite_dir = root / tag
        suite_dir.mkdir(parents=True, exist_ok=True)
        cmd = suite_cmd(args, combo, suite_dir, tag, variants)
        summary_path = suite_dir / "vision_suite_summary.csv"
        compatibility_summary_path = suite_dir / "general_suite_summary.csv"
        status = "dry_run"
        if not dry_run:
            if skip_existing and (summary_path.is_file() or compatibility_summary_path.is_file()):
                status = "skipped_existing"
            else:
                print(f"=== Running stress-grid suite {idx}: {tag} ===", flush=True)
                run_cmd(cmd, cwd=PROJECT_ROOT)
                status = "completed"
            if not summary_path.is_file() and compatibility_summary_path.is_file():
                summary_path = compatibility_summary_path
            summary_rows.extend(
                read_summary_rows(
                    summary_path,
                    combo=combo,
                    knn_ks=knn_ks,
                    suite_dir=suite_dir,
                    suite_tag=tag,
                )
            )
        manifest_rows.append(
            {
                "status": status,
                "suite_tag": tag,
                "suite_dir": str(suite_dir),
                **combo,
                "knn_ks": " ".join(str(k) for k in knn_ks),
                "variants": " ".join(variants),
                "command": " ".join(cmd),
            }
        )
        write_manifest(manifest_rows, root)

    preferred_summary_fields = [
        "dirichlet_alpha",
        "local_epochs",
        "num_clients",
        "min_client_weight",
        "weight_floor_total",
        "train_subset_size",
        "test_subset_size",
        "knn_ks",
        "variant_knn_k",
        "variant",
        "n_runs",
        "mean_fedavg_acc",
        "mean_acc",
        "mean_delta",
        "min_delta",
        "max_delta",
        "std_delta",
        "win_rate",
        "mean_graph_density",
        "mean_H_spec",
        "mean_tau",
        "suite_tag",
        "suite_dir",
    ]
    write_csv(summary_rows, root / "stress_grid_summary.csv", preferred_summary_fields)
    write_json(root / "stress_grid_summary.json", summary_rows)
    write_markdown(summary_rows, root / "stress_grid_summary.md", args)
    review_rows = build_auto_review_rows(
        summary_rows,
        collapse_acc_threshold=float(args.fedavg_collapse_acc_threshold),
        meaningful_delta=float(args.meaningful_delta),
        random_margin=float(args.random_margin),
    )
    write_auto_review(review_rows, root, args)
    print(f"Saved stress-grid outputs under {root}")
