"""Run smoke-scale prior-work-inspired GraphFLDesign presets.

This script validates that runnable prior-work proxy presets resolve through
GraphFLDesign, execute through the general FL entrypoint, and emit the standard
diagnostic artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]

RUNS = [
    {
        "label": "core_head_knn",
        "reference": "framework_core",
        "graph_preset": "head_knn_filtered_update",
    },
    {
        "label": "pfedgraph_proxy",
        "reference": "pFedGraph",
        "graph_preset": "pfedgraph_proxy",
    },
    {
        "label": "fedamp_proxy",
        "reference": "FedAMP",
        "graph_preset": "fedamp_proxy",
    },
    {
        "label": "sfl_proxy",
        "reference": "SFL",
        "graph_preset": "sfl_proxy",
    },
    {
        "label": "fedaga_proxy",
        "reference": "FedAGA",
        "graph_preset": "ema_magnitude_knn_filtered",
    },
]

REQUIRED_ARTIFACTS = [
    "round_metrics.csv",
    "client_metrics.csv",
    "graph_stats.csv",
    "counterfactual_metrics.csv",
    "module_traces.jsonl",
]


def _last_pair(pairs: list[list[float]]) -> float | None:
    if not pairs:
        return None
    return float(pairs[-1][1])


def _load_result(path: Path, method: str) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)
    return {
        "meta": payload.get("meta", {}),
        "result": payload.get("results", {}).get(method, {}),
    }


def _counterfactual_variants(path: Path) -> list[str]:
    if not path.exists():
        return []
    variants: list[str] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = str(row.get("counterfactual", ""))
            if name and name not in variants:
                variants.append(name)
    return variants


def _module_trace_summary(path: Path) -> dict[str, Any]:
    phases: list[str] = []
    modules: list[str] = []
    names: list[str] = []
    count = 0
    if not path.exists():
        return {"count": 0, "phases": [], "modules": [], "names": []}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            count += 1
            record = json.loads(line)
            for key, target in [
                ("phase", phases),
                ("module", modules),
                ("name", names),
            ]:
                value = str(record.get(key, ""))
                if value and value not in target:
                    target.append(value)
    return {"count": count, "phases": phases, "modules": modules, "names": names}


def _summarize_run(run_dir: Path, run: dict[str, str], args: argparse.Namespace) -> dict[str, Any]:
    result_path = (
        run_dir
        / f"result_general_{args.method}_seed{args.seed}_{run['label']}.json"
    )
    loaded = _load_result(result_path, args.method)
    meta = loaded["meta"]
    result = loaded["result"]
    diagnostics_dir = run_dir / "diagnostics"
    artifacts = {name: (diagnostics_dir / name).exists() for name in REQUIRED_ARTIFACTS}
    trace_summary = _module_trace_summary(diagnostics_dir / "module_traces.jsonl")

    return {
        "label": run["label"],
        "reference": run["reference"],
        "graph_preset": run["graph_preset"],
        "result_path": str(result_path),
        "diagnostics_dir": str(diagnostics_dir),
        "final_accuracy": _last_pair(
            result.get("metrics_distributed", {}).get("accuracy", [])
        ),
        "final_loss": _last_pair(result.get("losses_distributed", [])),
        "graph_source": meta.get("graph", {}).get("graph_source"),
        "graph_mode": meta.get("graph", {}).get("graph_mode"),
        "graph_method": meta.get("graph", {}).get("graph_method"),
        "aggregation_target": meta.get("aggregation", {}).get("aggregation_target"),
        "artifact_exists": artifacts,
        "counterfactual_variants": _counterfactual_variants(
            diagnostics_dir / "counterfactual_metrics.csv"
        ),
        "module_traces": trace_summary,
    }


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Prior-Work Proxy Smoke Summary",
        "",
        f"- Created at: {summary['created_at']}",
        f"- Dataset: {summary['settings']['dataset']}",
        f"- Clients: {summary['settings']['num_clients']}",
        f"- Rounds: {summary['settings']['rounds']}",
        f"- Warmup rounds: {summary['settings']['warmup_rounds']}",
        f"- Train subset: {summary['settings']['train_subset_size']}",
        f"- Test subset: {summary['settings']['test_subset_size']}",
        "",
        "| label | reference | preset | source | mode | aggregation | acc | loss | artifacts | trace records |",
        "|---|---|---|---|---|---|---:|---:|---|---:|",
    ]
    for row in summary["runs"]:
        artifact_ok = all(row["artifact_exists"].values())
        lines.append(
            "| {label} | {reference} | `{graph_preset}` | `{graph_source}` | `{graph_mode}` | "
            "`{aggregation_target}` | {acc:.4f} | {loss:.4f} | {artifacts} | {traces} |".format(
                label=row["label"],
                reference=row["reference"],
                graph_preset=row["graph_preset"],
                graph_source=row["graph_source"],
                graph_mode=row["graph_mode"],
                aggregation_target=row["aggregation_target"],
                acc=float(row["final_accuracy"] or 0.0),
                loss=float(row["final_loss"] or 0.0),
                artifacts="ok" if artifact_ok else "missing",
                traces=int(row["module_traces"]["count"]),
            )
        )
    lines.extend(
        [
            "",
            "## Counterfactual Variants",
            "",
        ]
    )
    for row in summary["runs"]:
        variants = ", ".join(f"`{name}`" for name in row["counterfactual_variants"])
        lines.append(f"- `{row['label']}`: {variants}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--out-root", default="experiments_current/prior_work_proxy_smoke")
    parser.add_argument("--dataset", default="fashionmnist")
    parser.add_argument("--model", default="mlp")
    parser.add_argument("--method", default="ours")
    parser.add_argument("--num-clients", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--train-subset-size", type=int, default=200)
    parser.add_argument("--test-subset-size", type=int, default=100)
    parser.add_argument("--partition", default="dirichlet", choices=["iid", "dirichlet"])
    parser.add_argument("--dirichlet-alpha", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--graph-seed", type=int, default=0)
    parser.add_argument("--warmup-rounds", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = (REPO_ROOT / args.out_root / stamp).resolve()
    root.mkdir(parents=True, exist_ok=True)
    script = REPO_ROOT / "run_vision_experiment.py"
    summaries: list[dict[str, Any]] = []

    common = [
        str(script),
        "--method",
        args.method,
        "--dataset",
        args.dataset,
        "--model",
        args.model,
        "--num-clients",
        str(args.num_clients),
        "--rounds",
        str(args.rounds),
        "--local-epochs",
        str(args.local_epochs),
        "--train-subset-size",
        str(args.train_subset_size),
        "--test-subset-size",
        str(args.test_subset_size),
        "--partition",
        args.partition,
        "--dirichlet-alpha",
        str(args.dirichlet_alpha),
        "--seed",
        str(args.seed),
        "--graph-seed",
        str(args.graph_seed),
        "--warmup-rounds",
        str(args.warmup_rounds),
        "--diagnostics-enable",
        "true",
        "--loo-enabled",
        "true",
    ]

    for run in RUNS:
        run_dir = root / run["label"]
        cmd = [
            args.python_bin,
            *common,
            "--graph-preset",
            run["graph_preset"],
            "--out-dir",
            str(run_dir),
            "--run-tag",
            run["label"],
        ]
        print("RUN", run["label"], " ".join(cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, cwd=REPO_ROOT, check=True)
            summaries.append(_summarize_run(run_dir, run, args))

    summary = {
        "created_at": datetime.now().isoformat(),
        "root": str(root),
        "settings": {
            "dataset": args.dataset,
            "model": args.model,
            "method": args.method,
            "num_clients": args.num_clients,
            "rounds": args.rounds,
            "local_epochs": args.local_epochs,
            "train_subset_size": args.train_subset_size,
            "test_subset_size": args.test_subset_size,
            "partition": args.partition,
            "dirichlet_alpha": args.dirichlet_alpha,
            "seed": args.seed,
            "graph_seed": args.graph_seed,
            "warmup_rounds": args.warmup_rounds,
        },
        "runs": summaries,
    }
    if not args.dry_run:
        summary_path = root / "prior_work_proxy_summary.json"
        report_path = root / "prior_work_proxy_summary.md"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        _write_markdown(report_path, summary)
        print(f"SUMMARY_JSON {summary_path}", flush=True)
        print(f"SUMMARY_MD {report_path}", flush=True)


if __name__ == "__main__":
    main()
