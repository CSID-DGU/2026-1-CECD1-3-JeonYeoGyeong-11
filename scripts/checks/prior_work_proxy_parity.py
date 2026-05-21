"""Lightweight parity check for prior-work-inspired GraphFLDesign proxies.

This is not an exact reproduction validator. It checks whether each runnable
prior-work proxy resolves to the intended lifecycle knobs, whether a smoke run
actually produced artifacts/traces for that proxy, and whether exact gaps remain
explicitly documented by support level and notes.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from graphfl_lab.designs import interface_target_designs, resolve_design

REQUIRED_ARTIFACTS = {
    "round_metrics.csv",
    "client_metrics.csv",
    "graph_stats.csv",
    "counterfactual_metrics.csv",
    "module_traces.jsonl",
}
REQUIRED_COUNTERFACTUALS = {
    "actual",
    "matched_random",
    "shuffled",
    "clustering_only",
    "graphfree_dominance_reweight",
}

METHODS = {
    "pFedGraph": {
        "design": "pfedgraph_proxy",
        "label": "pfedgraph_proxy",
        "expected_args": {
            "graph_method": "pfedgraph",
            "graph_source": "update",
            "graph_mode": "pfedgraph_qp",
            "aggregation_target": "spectral_filtered_update",
        },
        "expected_trace_names": {"qp_collaboration", "pfedgraph_qp:symmetric_diagnostic_projection"},
        "matched_mechanism": "sample-size-prior QP collaboration graph is represented as a diagnostic topology.",
        "remaining_gap": "No exact row-wise personalized model delivery; aggregation still returns one global model.",
    },
    "FedAMP": {
        "design": "fedamp_proxy",
        "label": "fedamp_proxy",
        "expected_args": {
            "graph_method": "fedamp",
            "graph_source": "weight",
            "graph_mode": "rbf",
            "aggregation_target": "spectral_filtered_weight",
        },
        "expected_trace_names": {"rbf", "dense"},
        "matched_mechanism": "model-weight distance relation is represented by an RBF dense graph.",
        "remaining_gap": "No exact personalized cloud model delivery or FedAMP proximal local objective.",
    },
    "SFL": {
        "design": "sfl_proxy",
        "label": "sfl_proxy",
        "expected_args": {
            "graph_method": "sfl",
            "graph_source": "weight",
            "graph_mode": "learned_smooth",
            "aggregation_target": "spectral_filtered_weight",
        },
        "expected_trace_names": {"euclidean_distance", "learned_smooth_proxy"},
        "matched_mechanism": "model-state graph smoothing is represented by learned_smooth topology.",
        "remaining_gap": "No exact server GCN aggregation or client-specific model generation.",
    },
    "FedAGA": {
        "design": "ema_magnitude_knn_filtered",
        "label": "fedaga_proxy",
        "expected_args": {
            "graph_method": "fedaga",
            "graph_source": "ema_update",
            "graph_mode": "magnitude_knn",
            "aggregation_target": "spectral_filtered_ema_update",
        },
        "expected_trace_names": {"cosine", "magnitude_aware_knn"},
        "matched_mechanism": "accumulated-gradient-style signal is represented by EMA update and magnitude-aware kNN.",
        "remaining_gap": "EMA update is a proxy, not the exact accumulated-gradient/convergence rule.",
    },
}


def _latest_summary() -> Path:
    candidates = sorted(
        (REPO_ROOT / "experiments_current" / "prior_work_proxy_smoke").glob(
            "*/prior_work_proxy_summary.json"
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("No prior_work_proxy_summary.json found")
    return candidates[0]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _read_module_trace_names(path: Path) -> set[str]:
    names: set[str] = set()
    if not path.exists():
        return names
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            name = str(record.get("name", ""))
            if name:
                names.add(name)
    return names


def _counterfactual_status(path: Path) -> dict[str, Any]:
    variants: set[str] = set()
    bad_rows = 0
    rows = 0
    if not path.exists():
        return {"rows": 0, "bad_rows": 0, "variants": []}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows += 1
            variants.add(str(row.get("counterfactual", "")))
            if str(row.get("status", "")) != "ok":
                bad_rows += 1
    return {"rows": rows, "bad_rows": bad_rows, "variants": sorted(variants)}


def _run_by_label(summary: dict[str, Any], label: str) -> dict[str, Any] | None:
    for run in summary.get("runs", []):
        if run.get("label") == label:
            return run
    return None


def _check_method(name: str, spec: dict[str, Any], smoke_summary: dict[str, Any]) -> dict[str, Any]:
    design = resolve_design(spec["design"])
    args = design.to_legacy_args()
    mapping_ok = all(args.get(key) == value for key, value in spec["expected_args"].items())
    support_ok = design.support_level == "proxy-supported"

    run = _run_by_label(smoke_summary, spec["label"])
    smoke_ok = False
    artifact_ok = False
    trace_ok = False
    counterfactual_ok = False
    trace_names: set[str] = set()
    cf_status: dict[str, Any] = {"rows": 0, "bad_rows": 0, "variants": []}
    if run is not None:
        artifact_ok = all(bool(run.get("artifact_exists", {}).get(name)) for name in REQUIRED_ARTIFACTS)
        diagnostics_dir = Path(str(run.get("diagnostics_dir", "")))
        trace_names = _read_module_trace_names(diagnostics_dir / "module_traces.jsonl")
        trace_ok = spec["expected_trace_names"].issubset(trace_names)
        cf_status = _counterfactual_status(diagnostics_dir / "counterfactual_metrics.csv")
        counterfactual_ok = (
            cf_status["bad_rows"] == 0
            and REQUIRED_COUNTERFACTUALS.issubset(set(cf_status["variants"]))
        )
        smoke_ok = artifact_ok and trace_ok and counterfactual_ok

    return {
        "method": name,
        "design": design.name,
        "support_level": design.support_level,
        "mapping_ok": mapping_ok,
        "support_ok": support_ok,
        "smoke_ok": smoke_ok,
        "artifact_ok": artifact_ok,
        "trace_ok": trace_ok,
        "counterfactual_ok": counterfactual_ok,
        "expected_args": spec["expected_args"],
        "actual_args": {key: args.get(key) for key in spec["expected_args"]},
        "expected_trace_names": sorted(spec["expected_trace_names"]),
        "observed_trace_names": sorted(trace_names),
        "counterfactual_status": cf_status,
        "matched_mechanism": spec["matched_mechanism"],
        "remaining_gap": spec["remaining_gap"],
        "verdict": (
            "proxy-parity-pass"
            if mapping_ok and support_ok and smoke_ok
            else "needs-review"
        ),
    }


def _check_interface_targets() -> dict[str, Any]:
    targets = interface_target_designs()
    fedpub = targets.get("FED-PUB", {})
    return {
        "method": "FED-PUB",
        "support_level": fedpub.get("support_level"),
        "interface_ok": fedpub.get("support_level") == "interface-target",
        "reason": fedpub.get("reason"),
        "verdict": (
            "interface-target-pass"
            if fedpub.get("support_level") == "interface-target"
            else "needs-review"
        ),
    }


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Prior-Work Parity Lite",
        "",
        f"- Created at: {report['created_at']}",
        f"- Smoke summary: `{report['smoke_summary']}`",
        "",
        "| method | design | support | mapping | smoke | verdict |",
        "|---|---|---|---|---|---|",
    ]
    for row in report["methods"]:
        lines.append(
            "| {method} | `{design}` | `{support}` | {mapping} | {smoke} | `{verdict}` |".format(
                method=row["method"],
                design=row["design"],
                support=row["support_level"],
                mapping="ok" if row["mapping_ok"] else "check",
                smoke="ok" if row["smoke_ok"] else "check",
                verdict=row["verdict"],
            )
        )
    fedpub = report["interface_targets"]["FED-PUB"]
    lines.extend(
        [
            "",
            "## Mechanism Notes",
            "",
        ]
    )
    for row in report["methods"]:
        lines.append(f"### {row['method']}")
        lines.append(f"- Matched: {row['matched_mechanism']}")
        lines.append(f"- Gap: {row['remaining_gap']}")
        lines.append(f"- Trace names checked: {', '.join(f'`{x}`' for x in row['expected_trace_names'])}")
        lines.append("")
    lines.extend(
        [
            "### FED-PUB",
            f"- Support: `{fedpub['support_level']}`",
            f"- Reason: {fedpub['reason']}",
            "",
            "## Bottom Line",
            "",
            "These checks validate executable proxy parity, not exact paper reproduction.",
            "The runnable prior-work designs are mechanism proxies; exact personalized delivery or method-specific local objectives remain explicit gaps.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", default="")
    parser.add_argument("--out-dir", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary) if args.summary else _latest_summary()
    if not summary_path.is_absolute():
        summary_path = (REPO_ROOT / summary_path).resolve()
    smoke_summary = _load_json(summary_path)
    out_dir = Path(args.out_dir) if args.out_dir else summary_path.parent
    if not out_dir.is_absolute():
        out_dir = (REPO_ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "created_at": datetime.now().isoformat(),
        "smoke_summary": str(summary_path),
        "methods": [
            _check_method(name, spec, smoke_summary) for name, spec in METHODS.items()
        ],
        "interface_targets": {"FED-PUB": _check_interface_targets()},
    }
    json_path = out_dir / "prior_work_parity_lite.json"
    md_path = out_dir / "prior_work_parity_lite.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_markdown(md_path, report)
    print(f"PARITY_JSON {json_path}")
    print(f"PARITY_MD {md_path}")
    for row in report["methods"]:
        print(row["method"], row["verdict"])
    print("FED-PUB", report["interface_targets"]["FED-PUB"]["verdict"])


if __name__ == "__main__":
    main()
