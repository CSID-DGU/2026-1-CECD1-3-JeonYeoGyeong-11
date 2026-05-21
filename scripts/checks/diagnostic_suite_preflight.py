"""Preflight-check the original diagnostic experiment code without training.

The original framework purpose is to compare real graph paths against matched
controls, clustering-only controls, and graph-free corrections. This script
validates that the diagnostic suite configuration resolves to runnable commands
and expected variant knobs, but it does not launch any Flower training jobs.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from graphfl_lab.cli import vision_suite as vision_suite_cli
from graphfl_lab.config_io import load_config
from graphfl_lab.experiments.suites.vision.variants import parse_variant, variant_cmd


EXPECTED_VARIANTS: dict[str, dict[str, str]] = {
    "fedavg": {"method": "fedavg"},
    "fedavgm": {"method": "fedavgm"},
    "ours_real_graph_k2": {
        "--correction-family": "real_graph",
        "--graph-source": "classifier_head_update",
        "--aggregation-target": "spectral_filtered_update",
        "--graph-mode": "knn",
        "--knn-k": "2",
    },
    "ours_shuffled_control_k2": {
        "--correction-family": "control_graph",
        "--control-graph-mode": "shuffled",
        "--graph-source": "classifier_head_update",
        "--aggregation-target": "spectral_filtered_update",
        "--graph-mode": "knn",
        "--knn-k": "2",
    },
    "ours_identity_control_k2": {
        "--correction-family": "control_graph",
        "--control-graph-mode": "identity",
        "--graph-source": "classifier_head_update",
        "--aggregation-target": "spectral_filtered_update",
        "--graph-mode": "knn",
        "--knn-k": "2",
    },
    "ours_cluster_only_k2": {
        "--correction-family": "clustering_only",
        "--cluster-method": "kmeans",
        "--cluster-auto-k": "true",
        "--graph-source": "classifier_head_update",
        "--aggregation-target": "spectral_filtered_update",
        "--graph-mode": "knn",
        "--knn-k": "2",
    },
    "ours_graphfree_reweight": {
        "--correction-family": "graph_free",
        "--graph-free-mode": "dominance_reweight",
        "--aggregation-target": "update",
    },
}

STANDARD_ARTIFACTS = [
    "vision_suite_summary.json",
    "vision_suite_rows.json",
    "vision_suite_summary.csv",
    "vision_suite_summary.md",
    "general_suite_summary.json",
    "general_suite_rows.json",
    "general_suite_summary.csv",
    "general_suite_summary.md",
    "diagnostic_summary.csv",
    "knn_vs_random_matched.csv",
    "interpretation.md",
    "dashboard_mockup.md",
    "validation_verdict.json",
]


def _args_from_config(config_path: Path) -> argparse.Namespace:
    old_argv = list(sys.argv)
    try:
        sys.argv = ["run_vision_suite.py", "--config", str(config_path)]
        return vision_suite_cli.parse_args()
    finally:
        sys.argv = old_argv


def _pairs(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    index = 0
    while index < len(items):
        key = items[index]
        if key.startswith("--") and index + 1 < len(items):
            out[key] = items[index + 1]
            index += 2
        else:
            index += 1
    return out


def _is_subpath(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _variant_preflight(args: argparse.Namespace, variant: str, out_dir: Path) -> dict[str, Any]:
    method, label, extras = parse_variant(variant, args)
    command, command_method, result_path = variant_cmd(
        args,
        variant,
        int(args.seeds[0]),
        Path(str(args.out_dir)).name,
        out_dir,
    )
    extra_map = _pairs(extras)
    expected = EXPECTED_VARIANTS.get(variant, {})
    expected_method = expected.get("method", "ours")
    method_ok = method == expected_method and command_method == expected_method
    expected_args = {key: value for key, value in expected.items() if key.startswith("--")}
    arg_mismatches = {
        key: {"expected": value, "actual": extra_map.get(key)}
        for key, value in expected_args.items()
        if extra_map.get(key) != value
    }
    diagnostics_ok = (
        "--diagnostics-enable" in command
        and str(command[command.index("--diagnostics-enable") + 1]).lower() == "true"
        and "--loo-enabled" in command
        and str(command[command.index("--loo-enabled") + 1]).lower() == "true"
    )
    return {
        "variant": variant,
        "method": method,
        "label": label,
        "extras": extras,
        "expected_args": expected_args,
        "arg_mismatches": arg_mismatches,
        "method_ok": method_ok,
        "diagnostics_ok": diagnostics_ok,
        "result_path": str(result_path),
        "result_path_in_out_dir": _is_subpath(result_path, out_dir),
        "command_preview": command,
        "ok": method_ok
        and not arg_mismatches
        and diagnostics_ok
        and _is_subpath(result_path, out_dir),
    }


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Original Purpose Experiment Preflight",
        "",
        f"- Created at: {report['created_at']}",
        f"- Config: `{report['config_path']}`",
        f"- Out dir: `{report['out_dir']}`",
        f"- Training launched: `{report['training_launched']}`",
        "",
        "| variant | method | expected args | diagnostics | path | verdict |",
        "|---|---|---|---|---|---|",
    ]
    for row in report["variants"]:
        lines.append(
            "| {variant} | `{method}` | {args} | {diagnostics} | {path} | `{verdict}` |".format(
                variant=row["variant"],
                method=row["method"],
                args="ok" if not row["arg_mismatches"] else "check",
                diagnostics="ok" if row["diagnostics_ok"] else "check",
                path="ok" if row["result_path_in_out_dir"] else "check",
                verdict="pass" if row["ok"] else "needs-review",
            )
        )
    lines.extend(
        [
            "",
            "## Expected Suite Artifacts",
            "",
        ]
    )
    for artifact in report["expected_suite_artifacts"]:
        lines.append(f"- `{artifact}`")
    lines.extend(
        [
            "",
            "## Verdict",
            "",
            (
                "PASS: the diagnostic experiment code is ready to launch the original-purpose smoke suite."
                if report["ok"]
                else "NEEDS REVIEW: at least one preflight check failed."
            ),
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/vision/diagnostic/smoke/fashionmnist_n5_r3_seed42.json",
    )
    parser.add_argument("--out-dir", default="")
    return parser.parse_args()


def main() -> None:
    cli_args = parse_args()
    config_path = Path(cli_args.config)
    if not config_path.is_absolute():
        config_path = (REPO_ROOT / config_path).resolve()
    config_values = load_config(config_path)
    args = _args_from_config(config_path)
    out_dir = Path(cli_args.out_dir or str(args.out_dir))
    if not out_dir.is_absolute():
        out_dir = (REPO_ROOT / out_dir).resolve()
    variants = [str(v).strip().lower() for v in args.variants]
    required_missing = [name for name in EXPECTED_VARIANTS if name not in variants]
    variant_rows = [_variant_preflight(args, variant, out_dir) for variant in variants]
    report = {
        "created_at": datetime.now().isoformat(),
        "config_path": str(config_path),
        "out_dir": str(out_dir),
        "training_launched": False,
        "config_values": config_values,
        "required_missing": required_missing,
        "variants": variant_rows,
        "expected_suite_artifacts": STANDARD_ARTIFACTS,
        "ok": not required_missing and all(row["ok"] for row in variant_rows),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "original_purpose_preflight.json"
    md_path = out_dir / "original_purpose_preflight.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_markdown(md_path, report)
    print(f"PREFLIGHT_JSON {json_path}")
    print(f"PREFLIGHT_MD {md_path}")
    print("VERDICT", "PASS" if report["ok"] else "NEEDS_REVIEW")
    if required_missing:
        print("MISSING", ", ".join(required_missing))
    for row in variant_rows:
        print(row["variant"], "PASS" if row["ok"] else "NEEDS_REVIEW")


if __name__ == "__main__":
    main()
