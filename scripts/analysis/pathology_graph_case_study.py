"""Controlled pathology-graph case-study suite runner.

This script can:
1) run the requested staged experiment suite (or reuse existing JSON results),
2) aggregate run-level and round-level diagnostics,
3) produce required report artifacts under reports/pathology_graph_case_study/.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _safe_float(v: Any, default: float = float("nan")) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _safe_mean(vals: Iterable[float]) -> float:
    arr = [float(v) for v in vals if not math.isnan(float(v))]
    if not arr:
        return float("nan")
    return float(sum(arr) / len(arr))


def _safe_std(vals: Iterable[float]) -> float:
    arr = [float(v) for v in vals if not math.isnan(float(v))]
    if len(arr) <= 1:
        return 0.0 if arr else float("nan")
    return float(statistics.pstdev(arr))


def _fmt(v: float, digits: int = 4) -> str:
    if v is None or math.isnan(float(v)):
        return "nan"
    return f"{float(v):.{digits}f}"


def _table(rows: List[List[str]]) -> str:
    if not rows:
        return ""
    out = ["| " + " | ".join(rows[0]) + " |", "| " + " | ".join(["---"] * len(rows[0])) + " |"]
    for row in rows[1:]:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def _json_dumps_compact(v: Any) -> str:
    return json.dumps(v, ensure_ascii=True, separators=(",", ":"))


@dataclass
class MethodSpec:
    label: str
    cli_method: str
    extras: List[str] = field(default_factory=list)
    implemented: bool = True
    notes: str = ""


@dataclass
class RunSpec:
    stage: str
    dataset: str
    alpha: float
    num_clients: int
    rounds: int
    local_epochs: int
    seed: int
    method: MethodSpec
    stage_group: str
    operator: str = ""
    graph_preset: str = "none"

    def run_tag(self) -> str:
        alpha_tag = str(self.alpha).replace(".", "p")
        op_tag = self.operator.replace("-", "_").replace(" ", "_") if self.operator else "na"
        return (
            f"case_{self.stage}_{self.method.label}_d{self.dataset}_a{alpha_tag}"
            f"_n{self.num_clients}_r{self.rounds}_s{self.seed}_op{op_tag}"
        )


def _build_cmd(args: argparse.Namespace, spec: RunSpec) -> List[str]:
    cmd = [
        args.python_bin,
        "run_general_experiment.py",
        "--method",
        spec.method.cli_method,
        "--dataset",
        spec.dataset,
        "--partition",
        "dirichlet",
        "--dirichlet-alpha",
        str(spec.alpha),
        "--num-clients",
        str(spec.num_clients),
        "--rounds",
        str(spec.rounds),
        "--local-epochs",
        str(spec.local_epochs),
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
        str(spec.seed),
        "--out-dir",
        str(args.experiment_out_dir),
        "--run-tag",
        spec.run_tag(),
    ]
    cmd.extend(spec.method.extras)
    return cmd


def _result_path(out_dir: Path, method: str, seed: int, run_tag: str) -> Path:
    suffix = f"_{run_tag}" if run_tag else ""
    return out_dir / f"result_general_{method}_seed{seed}{suffix}.json"


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _meta_matches_spec(meta: Dict[str, Any], spec: RunSpec) -> bool:
    exp = dict(meta.get("experiment", {}))
    if str(exp.get("dataset", "")).lower() != str(spec.dataset).lower():
        return False
    if str(exp.get("partition", "")).lower() != "dirichlet":
        return False
    if _safe_float(exp.get("dirichlet_alpha")) != float(spec.alpha):
        return False
    if int(exp.get("num_clients", -1)) != int(spec.num_clients):
        return False
    if int(exp.get("rounds", -1)) != int(spec.rounds):
        return False
    if int(exp.get("local_epochs", -1)) != int(spec.local_epochs):
        return False
    if int(exp.get("seed", -1)) != int(spec.seed):
        return False

    # Method-specific selectors (best effort to avoid false matches).
    graph = dict(meta.get("graph", {}))
    dominance = dict(meta.get("dominance", {}))
    graph_smoothing = dict(meta.get("graph_smoothing", {}))
    label = spec.method.label
    if label == "signed_conflict_graph_correction":
        return str(graph.get("graph_preset", "")).lower() == "signed_conflict_knn"
    if label == "ema_update_graph_correction":
        return str(graph.get("graph_source", "")).lower() == "ema_update"
    if label in {"random_graph_control", "shuffled_graph_control", "uniform_graph_control", "identity_or_no_graph_control"}:
        expected = {
            "random_graph_control": "random",
            "shuffled_graph_control": "shuffled",
            "uniform_graph_control": "uniform",
            "identity_or_no_graph_control": "identity",
        }[label]
        return str(graph.get("graph_variant", "")).lower() == expected
    if label == "norm_clipping":
        return str(dominance.get("mode", "")).lower() == "norm_clip"
    if label == "contribution_cap":
        return str(dominance.get("mode", "")).lower() == "contribution_cap"
    if label == "soft_dominance_reweighting":
        return str(dominance.get("mode", "")).lower() == "soft_reweight"
    if label == "n_eff_aware_weighting":
        return str(dominance.get("mode", "")).lower() == "n_eff_aware"
    if spec.stage_group == "stage4_operator":
        return str(graph_smoothing.get("operator", "")).lower() in {
            "laplacian",
            "residual",
            "dominance_residual",
            "signed_conflict_attenuation",
            "dominance_aware_attenuation",
        }
    return True


def _find_existing_result(spec: RunSpec, search_root: Path) -> Optional[Path]:
    for path in sorted(search_root.rglob("result_general_*.json")):
        try:
            payload = _load_json(path)
        except Exception:
            continue
        meta = dict(payload.get("meta", {}))
        if not _meta_matches_spec(meta, spec):
            continue
        if spec.method.cli_method not in dict(payload.get("results", {})):
            continue
        return path
    return None


def _run_command(cmd: Sequence[str]) -> None:
    subprocess.run(list(cmd), cwd=str(PROJECT_ROOT), check=True)


def _stage1_methods() -> List[MethodSpec]:
    return [
        MethodSpec("fedavg", "fedavg"),
        MethodSpec("fedavgm", "fedavgm"),
        MethodSpec(
            "update_graph_correction",
            "graph_smooth",
            extras=[
                "--graph-variant",
                "update",
                "--graph-source",
                "update",
                "--graph-mode",
                "knn",
                "--knn-k",
                "4",
            ],
        ),
        MethodSpec(
            "signed_conflict_graph_correction",
            "graph_smooth",
            extras=[
                "--graph-preset",
                "signed_conflict_knn",
                "--graph-variant",
                "update",
                "--graph-smoothing-operator",
                "signed_conflict_attenuation",
            ],
        ),
        MethodSpec(
            "ema_update_graph_correction",
            "graph_smooth",
            extras=[
                "--graph-variant",
                "update",
                "--graph-source",
                "ema_update",
                "--graph-mode",
                "knn",
                "--knn-k",
                "4",
            ],
        ),
        MethodSpec(
            "random_graph_control",
            "graph_smooth",
            extras=["--graph-variant", "random", "--graph-source", "update", "--graph-mode", "knn", "--knn-k", "4"],
        ),
        MethodSpec(
            "shuffled_graph_control",
            "graph_smooth",
            extras=["--graph-variant", "shuffled", "--graph-source", "update", "--graph-mode", "knn", "--knn-k", "4"],
        ),
        MethodSpec(
            "uniform_graph_control",
            "graph_smooth",
            extras=["--graph-variant", "uniform", "--graph-source", "update", "--graph-mode", "knn", "--knn-k", "4"],
        ),
        MethodSpec(
            "identity_or_no_graph_control",
            "graph_smooth",
            extras=["--graph-variant", "identity", "--graph-source", "update", "--graph-mode", "knn", "--knn-k", "4"],
        ),
        MethodSpec("norm_clipping", "dominance_aware", extras=["--dominance-mode", "norm_clip"]),
        MethodSpec("contribution_cap", "dominance_aware", extras=["--dominance-mode", "contribution_cap"]),
        MethodSpec("soft_dominance_reweighting", "dominance_aware", extras=["--dominance-mode", "soft_reweight"]),
        MethodSpec("n_eff_aware_weighting", "dominance_aware", extras=["--dominance-mode", "n_eff_aware"]),
    ]


def _operator_specs_for_source(source_kind: str) -> List[MethodSpec]:
    base_extras = (
        ["--graph-source", "update", "--graph-mode", "knn", "--knn-k", "4"]
        if source_kind == "update"
        else ["--graph-preset", "signed_conflict_knn", "--graph-variant", "update"]
    )
    return [
        MethodSpec(
            f"{source_kind}_unnormalized_laplacian",
            "graph_smooth",
            extras=base_extras + ["--graph-smoothing-operator", "unnormalized_laplacian"],
        ),
        MethodSpec(
            f"{source_kind}_normalized_laplacian",
            "graph_smooth",
            extras=base_extras + ["--graph-smoothing-operator", "normalized_laplacian"],
        ),
        MethodSpec(
            f"{source_kind}_random_walk_laplacian",
            "graph_smooth",
            extras=base_extras + ["--graph-smoothing-operator", "random_walk_laplacian"],
        ),
        MethodSpec(
            f"{source_kind}_residual_neighbor_mixing",
            "graph_smooth",
            extras=base_extras + ["--graph-smoothing-operator", "residual_neighbor_mixing"],
        ),
        MethodSpec(
            f"{source_kind}_signed_conflict_attenuation",
            "graph_smooth",
            extras=base_extras + ["--graph-smoothing-operator", "signed_conflict_attenuation"],
        ),
        MethodSpec(
            f"{source_kind}_dominance_aware_attenuation",
            "graph_smooth",
            extras=base_extras + ["--graph-smoothing-operator", "dominance_aware_attenuation"],
        ),
    ]


def _build_suite(args: argparse.Namespace) -> List[RunSpec]:
    specs: List[RunSpec] = []
    base_methods = _stage1_methods()
    for seed in args.seeds:
        for m in base_methods:
            specs.append(
                RunSpec(
                    stage="stage1",
                    stage_group="stage1_main",
                    dataset="fashionmnist",
                    alpha=0.1,
                    num_clients=20,
                    rounds=20,
                    local_epochs=1,
                    seed=int(seed),
                    method=m,
                )
            )
    for alpha in [0.03, 0.1, 0.3]:
        for seed in args.seeds:
            for m in base_methods:
                specs.append(
                    RunSpec(
                        stage="stage2",
                        stage_group="stage2_alpha_sweep",
                        dataset="fashionmnist",
                        alpha=float(alpha),
                        num_clients=20,
                        rounds=20,
                        local_epochs=1,
                        seed=int(seed),
                        method=m,
                    )
                )
    for n_clients in [10, 20, 50]:
        for seed in args.seeds:
            for m in base_methods:
                specs.append(
                    RunSpec(
                        stage="stage3",
                        stage_group="stage3_client_sweep",
                        dataset="fashionmnist",
                        alpha=0.1,
                        num_clients=int(n_clients),
                        rounds=20,
                        local_epochs=1,
                        seed=int(seed),
                        method=m,
                    )
                )
    for source in ["update", "signed_conflict"]:
        for op_method in _operator_specs_for_source(source):
            for seed in [42, 43, 44]:
                specs.append(
                    RunSpec(
                        stage="stage4",
                        stage_group="stage4_operator",
                        dataset="fashionmnist",
                        alpha=0.1,
                        num_clients=20,
                        rounds=args.stage4_rounds,
                        local_epochs=1,
                        seed=int(seed),
                        method=op_method,
                        operator=op_method.label,
                    )
                )
    # Stage 5 is conditionally enabled from run outcomes.
    return specs


def _series_from_trace(trace: List[Dict[str, Any]], key: str) -> List[float]:
    return [_safe_float(row.get(key)) for row in trace]


def _extract_run_metrics(
    *,
    payload: Dict[str, Any],
    method_key: str,
    spec: RunSpec,
    status: str,
    result_path: str,
    note: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    if status != "RUN":
        run_row = {
            "stage": spec.stage,
            "stage_group": spec.stage_group,
            "dataset": spec.dataset,
            "alpha": spec.alpha,
            "num_clients": spec.num_clients,
            "seed": spec.seed,
            "method": spec.method.label,
            "status": status,
            "result_path": result_path,
            "note": note,
            "graph_preset": spec.graph_preset,
            "operator": spec.operator or "default",
            "final_acc": float("nan"),
            "best_acc": float("nan"),
            "last5_acc": float("nan"),
            "last5_acc_std": float("nan"),
            "mean_CR": float("nan"),
            "mean_CA": float("nan"),
            "mean_DI": float("nan"),
            "mean_N_eff": float("nan"),
            "mean_cos_delta": float("nan"),
            "mean_rel_delta_change": float("nan"),
        }
        return run_row, []

    result = dict(payload.get("results", {}).get(method_key, {}))
    meta = dict(payload.get("meta", {}))
    graph_meta = dict(meta.get("graph", {}))
    smoothing_meta = dict(meta.get("graph_smoothing", {}))
    trace: List[Dict[str, Any]] = list(result.get("round_trace", []))
    acc_series = [
        _safe_float(v[1])
        for v in list(result.get("metrics_distributed", {}).get("accuracy", []))
        if isinstance(v, list) and len(v) >= 2
    ]
    final_acc = acc_series[-1] if acc_series else float("nan")
    best_acc = max(acc_series) if acc_series else float("nan")
    last5 = acc_series[-5:] if len(acc_series) >= 5 else acc_series
    last5_acc = _safe_mean(last5)
    last5_std = _safe_std(last5)

    cr_vals = _series_from_trace(trace, "conflict_ratio")
    ca_vals = _series_from_trace(trace, "cancellation_ratio")
    di_vals = [
        _safe_float(row.get("dominance_ratio_corrected", row.get("dominance_ratio")))
        for row in trace
    ]
    neff_vals = [
        _safe_float(row.get("effective_num_clients_corrected", row.get("effective_num_clients")))
        for row in trace
    ]
    cos_vals = [
        _safe_float(row.get("cos_delta_corrected_vs_base", row.get("corrected_vs_fedavg_delta_cosine")))
        for row in trace
    ]
    rel_vals = [
        _safe_float(row.get("rel_delta_change", row.get("relative_delta_change")))
        for row in trace
    ]

    run_row = {
        "stage": spec.stage,
        "stage_group": spec.stage_group,
        "dataset": spec.dataset,
        "alpha": spec.alpha,
        "num_clients": spec.num_clients,
        "seed": spec.seed,
        "method": spec.method.label,
        "status": status,
        "result_path": result_path,
        "note": note,
        "graph_preset": str(graph_meta.get("graph_preset", "none")),
        "graph_source": str(graph_meta.get("graph_source", "")),
        "graph_mode": str(graph_meta.get("graph_mode", "")),
        "operator": spec.operator or str(smoothing_meta.get("operator", "")),
        "final_acc": final_acc,
        "best_acc": best_acc,
        "last5_acc": last5_acc,
        "last5_acc_std": last5_std,
        "mean_CR": _safe_mean(cr_vals),
        "mean_CA": _safe_mean(ca_vals),
        "mean_DI": _safe_mean(di_vals),
        "mean_N_eff": _safe_mean(neff_vals),
        "mean_cos_delta": _safe_mean(cos_vals),
        "mean_rel_delta_change": _safe_mean(rel_vals),
    }

    round_rows: List[Dict[str, Any]] = []
    for row in trace:
        q_weights = row.get("client_contribution_weights", row.get("client_weights", []))
        round_rows.append(
            {
                "stage": spec.stage,
                "stage_group": spec.stage_group,
                "dataset": spec.dataset,
                "alpha": spec.alpha,
                "num_clients": spec.num_clients,
                "seed": spec.seed,
                "method": spec.method.label,
                "round": int(row.get("round", -1)),
                "train_loss": _safe_float(row.get("train_loss_mean")),
                "test_accuracy": _safe_float(row.get("accuracy")),
                "CR": _safe_float(row.get("conflict_ratio")),
                "CA": _safe_float(row.get("cancellation_ratio")),
                "DI": _safe_float(row.get("dominance_ratio_corrected", row.get("dominance_ratio"))),
                "N_eff": _safe_float(
                    row.get("effective_num_clients_corrected", row.get("effective_num_clients"))
                ),
                "cos_delta_corrected_vs_base": _safe_float(
                    row.get("cos_delta_corrected_vs_base", row.get("corrected_vs_fedavg_delta_cosine"))
                ),
                "rel_delta_change": _safe_float(
                    row.get("rel_delta_change", row.get("relative_delta_change"))
                ),
                "corrected_delta_norm": _safe_float(row.get("corrected_delta_norm")),
                "base_delta_norm": _safe_float(row.get("base_delta_norm", row.get("fedavg_delta_norm"))),
                "graph_density": _safe_float(row.get("graph_density")),
                "mean_degree": _safe_float(row.get("mean_degree", row.get("graph_degree_mean"))),
                "max_degree": _safe_float(row.get("max_degree")),
                "min_degree": _safe_float(row.get("min_degree")),
                "connected_components": _safe_float(
                    row.get("connected_components", row.get("graph_connected_components"))
                ),
                "positive_edge_count": _safe_float(row.get("positive_edge_count")),
                "negative_edge_count": _safe_float(row.get("negative_edge_count")),
                "signed_edge_ratio": _safe_float(row.get("signed_edge_ratio")),
                "graph_source": str(row.get("graph_source", row.get("graph_source_used", graph_meta.get("graph_source", "")))),
                "graph_mode": str(row.get("graph_mode", graph_meta.get("graph_mode", ""))),
                "graph_preset": str(row.get("graph_preset", graph_meta.get("graph_preset", "none"))),
                "operator_type": str(
                    row.get("operator_type", row.get("graph_smoothing_operator", spec.operator or ""))
                ),
                "lambda": _safe_float(
                    row.get("lambda", row.get("graph_smoothing_lambda", smoothing_meta.get("lambda")))
                ),
                "clip_threshold": _safe_float(row.get("clip_norm")),
                "cap_threshold": _safe_float(row.get("contribution_cap")),
                "client_contribution_weights": _json_dumps_compact(q_weights),
                "max_contribution": _safe_float(
                    row.get("max_contribution", row.get("max_qbar_i_corrected", row.get("max_qbar_i_raw")))
                ),
                "contribution_entropy": _safe_float(row.get("contribution_entropy")),
                "top1_client_id_by_contribution": str(
                    row.get("top1_client_id_by_contribution", row.get("dominant_client_id", ""))
                ),
                "top3_contribution_mass": _safe_float(row.get("top3_contribution_mass")),
                "status": "RUN",
                "result_path": result_path,
            }
        )
    return run_row, round_rows


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _group_stats(rows: List[Dict[str, Any]], key: str) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") != "RUN":
            continue
        grouped.setdefault(str(row.get(key)), []).append(row)
    out: Dict[str, Dict[str, float]] = {}
    for group, gro in grouped.items():
        out[group] = {
            "runs": float(len(gro)),
            "final_mean": _safe_mean(_safe_float(r.get("final_acc")) for r in gro),
            "final_std": _safe_std(_safe_float(r.get("final_acc")) for r in gro),
            "last5_mean": _safe_mean(_safe_float(r.get("last5_acc")) for r in gro),
            "last5_std": _safe_std(_safe_float(r.get("last5_acc")) for r in gro),
            "CR": _safe_mean(_safe_float(r.get("mean_CR")) for r in gro),
            "CA": _safe_mean(_safe_float(r.get("mean_CA")) for r in gro),
            "DI": _safe_mean(_safe_float(r.get("mean_DI")) for r in gro),
            "N_eff": _safe_mean(_safe_float(r.get("mean_N_eff")) for r in gro),
            "cos": _safe_mean(_safe_float(r.get("mean_cos_delta")) for r in gro),
            "rel": _safe_mean(_safe_float(r.get("mean_rel_delta_change")) for r in gro),
        }
    return out


def _case_decision(stage1_rows: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
    stats = _group_stats(stage1_rows, "method")
    notes: List[str] = []
    required = [
        "update_graph_correction",
        "signed_conflict_graph_correction",
        "random_graph_control",
        "shuffled_graph_control",
        "uniform_graph_control",
        "identity_or_no_graph_control",
        "contribution_cap",
        "norm_clipping",
        "soft_dominance_reweighting",
        "n_eff_aware_weighting",
    ]
    if not all(k in stats for k in required):
        notes.append("Insufficient Stage 1 coverage; using provisional diagnostic-first decision.")
        return "D", notes

    graph_best = max(
        stats["update_graph_correction"]["last5_mean"],
        stats["signed_conflict_graph_correction"]["last5_mean"],
    )
    controls_best = max(
        stats["random_graph_control"]["last5_mean"],
        stats["shuffled_graph_control"]["last5_mean"],
        stats["uniform_graph_control"]["last5_mean"],
        stats["identity_or_no_graph_control"]["last5_mean"],
    )
    dominance_best = max(
        stats["contribution_cap"]["last5_mean"],
        stats["norm_clipping"]["last5_mean"],
        stats["soft_dominance_reweighting"]["last5_mean"],
        stats["n_eff_aware_weighting"]["last5_mean"],
    )
    graph_rel = max(
        stats["update_graph_correction"]["rel"],
        stats["signed_conflict_graph_correction"]["rel"],
    )
    graph_di = min(
        stats["update_graph_correction"]["DI"],
        stats["signed_conflict_graph_correction"]["DI"],
    )
    graph_neff = max(
        stats["update_graph_correction"]["N_eff"],
        stats["signed_conflict_graph_correction"]["N_eff"],
    )

    if graph_best > controls_best + 1e-3 and graph_best >= dominance_best - 1e-3 and graph_rel > 1e-3:
        notes.append("Graph methods beat controls with non-trivial delta perturbation and stay competitive.")
        return "A", notes
    if dominance_best > graph_best + 1e-3:
        notes.append("Dominance-aware methods outperform graph-based correction in Stage 1.")
        return "B", notes
    if abs(graph_best - controls_best) <= 0.01:
        notes.append("Graph-specific variants are close to random/uniform/shuffled controls.")
        return "C", notes
    notes.append("No method consistently dominates; pathology diagnostics are more informative than gains.")
    if graph_di < stats["identity_or_no_graph_control"]["DI"] and graph_neff > stats["identity_or_no_graph_control"]["N_eff"]:
        notes.append("Graph interventions still shift DI/N_eff in interpretable ways.")
    return "D", notes


def _write_summary_tables(path: Path, run_rows: List[Dict[str, Any]]) -> None:
    sections = [
        ("Stage 1 main experiment", "stage1_main"),
        ("Stage 2 alpha sweep", "stage2_alpha_sweep"),
        ("Stage 3 client-count sweep", "stage3_client_sweep"),
        ("Stage 4 operator sanity", "stage4_operator"),
        ("Stage 5 dataset expansion", "stage5_dataset_expansion"),
    ]
    lines = ["# summary_tables", ""]
    for title, group in sections:
        lines.append(f"## {title}")
        rows = [r for r in run_rows if r.get("stage_group") == group]
        if not rows:
            lines.append("- No runs recorded.")
            lines.append("")
            continue
        stats = _group_stats(rows, "method")
        table_rows = [["method", "runs", "final_acc_mean", "last5_acc_mean", "DI", "N_eff", "status_hint"]]
        for method in sorted({str(r.get("method")) for r in rows}):
            st = stats.get(method, {})
            status_hint = "RUN" if method in stats else "NOT_RUN"
            table_rows.append(
                [
                    method,
                    _fmt(st.get("runs", float("nan")), 0) if method in stats else "0",
                    _fmt(st.get("final_mean", float("nan"))),
                    _fmt(st.get("last5_mean", float("nan"))),
                    _fmt(st.get("DI", float("nan"))),
                    _fmt(st.get("N_eff", float("nan"))),
                    status_hint,
                ]
            )
        lines.append(_table(table_rows))
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_experiment_report(path: Path, run_rows: List[Dict[str, Any]]) -> None:
    stage1_rows = [r for r in run_rows if r.get("stage_group") == "stage1_main"]
    stage1_run_rows = [r for r in stage1_rows if r.get("status") == "RUN"]
    method_stats = _group_stats(stage1_rows, "method")
    case, case_notes = _case_decision(stage1_rows)

    final_rank = sorted(
        (
            (m, st["final_mean"], st["final_std"])
            for m, st in method_stats.items()
        ),
        key=lambda x: (x[1], -x[2]),
        reverse=True,
    )
    last5_rank = sorted(
        (
            (m, st["last5_mean"], st["last5_std"])
            for m, st in method_stats.items()
        ),
        key=lambda x: (x[1], -x[2]),
        reverse=True,
    )
    stability_rank = sorted(
        (
            (m, st["last5_std"])
            for m, st in method_stats.items()
        ),
        key=lambda x: x[1],
    )

    lines: List[str] = []
    lines.append("# EXPERIMENT_REPORT")
    lines.append("")
    lines.append("## Short goal")
    lines.append(
        "Evaluate whether update-derived/signed client graphs explain or correct single-global label-skew FL aggregation pathology better than graph-free dominance correction or generic smoothing controls."
    )
    lines.append("")
    lines.append("## Experiment settings")
    lines.append("- Core setting: FashionMNIST, Dirichlet label-skew, 20 clients, 20 rounds, local_epochs=1, seeds=42..46.")
    lines.append("- Stages: main separation, alpha sweep, client-count sweep, operator sanity, dataset expansion (conditional).")
    lines.append("- This report is generated from runs with status `RUN`; missing specs are marked `NOT_RUN` in CSV tables.")
    lines.append("")
    lines.append("## Results tables")
    if stage1_run_rows:
        rows = [["method", "runs", "final_acc_mean", "last5_acc_mean", "last5_acc_std", "mean_CR", "mean_CA", "mean_DI", "mean_N_eff"]]
        for method in sorted(method_stats.keys()):
            st = method_stats[method]
            rows.append(
                [
                    method,
                    _fmt(st["runs"], 0),
                    _fmt(st["final_mean"]),
                    _fmt(st["last5_mean"]),
                    _fmt(st["last5_std"]),
                    _fmt(st["CR"]),
                    _fmt(st["CA"]),
                    _fmt(st["DI"]),
                    _fmt(st["N_eff"]),
                ]
            )
        lines.append(_table(rows))
    else:
        lines.append("- No Stage 1 runs completed; only NOT_RUN placeholders are available.")
    lines.append("")
    lines.append("## Method ranking by final accuracy")
    for i, (m, mean_acc, std_acc) in enumerate(final_rank, start=1):
        lines.append(f"{i}. `{m}`: mean={_fmt(mean_acc)}, std={_fmt(std_acc)}")
    if not final_rank:
        lines.append("- No ranking available (no completed Stage 1 runs).")
    lines.append("")
    lines.append("## Method ranking by last5 accuracy")
    for i, (m, mean_acc, std_acc) in enumerate(last5_rank, start=1):
        lines.append(f"{i}. `{m}`: mean={_fmt(mean_acc)}, std={_fmt(std_acc)}")
    if not last5_rank:
        lines.append("- No ranking available (no completed Stage 1 runs).")
    lines.append("")
    lines.append("## Method ranking by stability across seeds")
    for i, (m, std_acc) in enumerate(stability_rank, start=1):
        lines.append(f"{i}. `{m}`: last5 std={_fmt(std_acc)}")
    if not stability_rank:
        lines.append("- No ranking available (no completed Stage 1 runs).")
    lines.append("")
    lines.append("## Pathology metric analysis")
    lines.append("- CR/CA/DI/N_eff are reported per-round in `round_logs.csv` and summarized per-run in `raw_results.csv`.")
    lines.append("- Interpret DI decrease with N_eff increase as reduced contribution concentration.")
    lines.append("")
    lines.append("## Intervention strength analysis")
    lines.append("- Uses `cos_delta_corrected_vs_base` and `rel_delta_change` to verify non-trivial intervention.")
    lines.append("- Near-zero `rel_delta_change` indicates effectively no correction despite operator invocation.")
    lines.append("")
    lines.append("## Graph informativeness analysis")
    lines.append("- Graph-specific claim requires update/signed > random/shuffled/uniform/identity under same seeds/settings.")
    lines.append("- If controls match graph variants, treat improvement as generic smoothing (Case C).")
    lines.append("")
    lines.append("## Decision among Case A/B/C/D")
    lines.append(f"- **Selected case: {case}**")
    for note in case_notes:
        lines.append(f"- {note}")
    lines.append("")
    lines.append("## Recommended next step")
    if case == "A":
        lines.append("- Expand with stronger graph-source ablations and robustness checks before novelty claims.")
    elif case == "B":
        lines.append("- Prioritize graph-free dominance correction and treat graph as secondary diagnostic signal.")
    elif case == "C":
        lines.append("- Reframe to generic smoothing baselines; avoid graph-informativeness claims.")
    else:
        lines.append("- Reframe as controlled aggregation pathology analysis with diagnostics as main contribution.")
    lines.append("")
    lines.append("## Requested direct answers")
    lines.append("1. Does update-derived graph beat graph controls? See Stage 1 table (or NOT_RUN if missing).")
    lines.append("2. Does signed conflict graph beat positive/update graph? See `signed_conflict_graph_correction` vs `update_graph_correction`.")
    lines.append("3. Does graph-based correction beat graph-free dominance correction? Compare graph rows vs dominance rows.")
    lines.append("4. Does generic smoothing explain the gains? Check random/uniform/shuffled proximity to update/signed.")
    lines.append("5. Are CR/CA/DI/N_eff useful for explaining failure? Check per-round logs for difficult seeds/rounds.")
    lines.append(f"6. Which case is currently supported? **{case}** (provisional if coverage is incomplete).")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run pathology graph case-study suite.")
    p.add_argument("--python-bin", type=str, default=sys.executable)
    p.add_argument("--model", type=str, default="cnn")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    p.add_argument("--stage4-rounds", type=int, default=20)
    p.add_argument(
        "--experiment-out-dir",
        type=Path,
        default=PROJECT_ROOT / "experiments_current" / "pathology_graph_case_study",
    )
    p.add_argument(
        "--report-dir",
        type=Path,
        default=PROJECT_ROOT / "reports" / "pathology_graph_case_study",
    )
    p.add_argument(
        "--search-existing-root",
        type=Path,
        default=PROJECT_ROOT / "experiments_current",
    )
    p.add_argument("--run-missing", type=str, default="false")
    p.add_argument("--include-stage5", type=str, default="false")
    p.add_argument(
        "--stages",
        type=str,
        nargs="*",
        default=[],
        help="Optional subset: stage1 stage2 stage3 stage4 stage5",
    )
    return p.parse_args()


def _bool_flag(v: str) -> bool:
    return str(v).strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def main() -> None:
    args = parse_args()
    run_missing = _bool_flag(args.run_missing)
    include_stage5 = _bool_flag(args.include_stage5)
    args.experiment_out_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    specs = _build_suite(args)
    if args.stages:
        wanted = {str(x).strip().lower() for x in args.stages if str(x).strip()}
        specs = [s for s in specs if s.stage in wanted]
    run_rows: List[Dict[str, Any]] = []
    round_rows: List[Dict[str, Any]] = []

    for spec in specs:
        if not spec.method.implemented:
            row, rr = _extract_run_metrics(
                payload={},
                method_key=spec.method.cli_method,
                spec=spec,
                status="NOT_RUN",
                result_path="",
                note=f"Method not implemented: {spec.method.notes or spec.method.label}",
            )
            run_rows.append(row)
            round_rows.extend(rr)
            continue

        local_result = _result_path(
            out_dir=args.experiment_out_dir,
            method=spec.method.cli_method,
            seed=spec.seed,
            run_tag=spec.run_tag(),
        )
        payload: Optional[Dict[str, Any]] = None
        used_path = ""
        note = ""
        status = "RUN"

        if local_result.is_file():
            payload = _load_json(local_result)
            used_path = str(local_result)
            note = "reused_local"
        else:
            matched = _find_existing_result(spec, args.search_existing_root)
            if matched is not None:
                payload = _load_json(matched)
                used_path = str(matched)
                note = "reused_existing"
            elif run_missing:
                cmd = _build_cmd(args, spec)
                _run_command(cmd)
                payload = _load_json(local_result)
                used_path = str(local_result)
                note = "executed"
            else:
                status = "NOT_RUN"
                note = "missing_result_and_run_missing_false"

        if status == "RUN" and payload is not None and spec.method.cli_method in payload.get("results", {}):
            row, rr = _extract_run_metrics(
                payload=payload,
                method_key=spec.method.cli_method,
                spec=spec,
                status="RUN",
                result_path=used_path,
                note=note,
            )
        else:
            row, rr = _extract_run_metrics(
                payload={},
                method_key=spec.method.cli_method,
                spec=spec,
                status="NOT_RUN",
                result_path=used_path,
                note=note or "result_not_found_or_missing_method_key",
            )
        run_rows.append(row)
        round_rows.extend(rr)

    # Stage 5 (optional/conditional).
    stage1_done = all(
        r.get("status") == "RUN"
        for r in run_rows
        if r.get("stage_group") == "stage1_main"
    )
    stage2_done = all(
        r.get("status") == "RUN"
        for r in run_rows
        if r.get("stage_group") == "stage2_alpha_sweep"
    )
    if include_stage5 and stage1_done and stage2_done:
        # Placeholder rows if user enables Stage 5 but runner is not yet expanded.
        for dataset in ["emnist", "cifar10"]:
            row = {
                "stage": "stage5",
                "stage_group": "stage5_dataset_expansion",
                "dataset": dataset,
                "alpha": 0.1,
                "num_clients": 20,
                "seed": float("nan"),
                "method": "stage5_placeholder",
                "status": "NOT_RUN",
                "result_path": "",
                "note": "Stage 5 scaffold present but not expanded in this pass.",
                "graph_preset": "none",
                "operator": "default",
                "final_acc": float("nan"),
                "best_acc": float("nan"),
                "last5_acc": float("nan"),
                "last5_acc_std": float("nan"),
                "mean_CR": float("nan"),
                "mean_CA": float("nan"),
                "mean_DI": float("nan"),
                "mean_N_eff": float("nan"),
                "mean_cos_delta": float("nan"),
                "mean_rel_delta_change": float("nan"),
            }
            run_rows.append(row)

    raw_results_path = args.report_dir / "raw_results.csv"
    round_logs_path = args.report_dir / "round_logs.csv"
    summary_path = args.report_dir / "summary_tables.md"
    report_path = args.report_dir / "EXPERIMENT_REPORT.md"

    _write_csv(raw_results_path, run_rows)
    _write_csv(round_logs_path, round_rows)
    _write_summary_tables(summary_path, run_rows)
    _write_experiment_report(report_path, run_rows)

    print(f"Saved: {raw_results_path}")
    print(f"Saved: {round_logs_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()

