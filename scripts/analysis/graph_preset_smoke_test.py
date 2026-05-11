"""Smoke-test graph presets with behavior checks.

This test verifies not only execution success but also that each preset
actually changes the graph construction path as intended.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from spectral_fl.graph.presets import resolve_graph_preset_spec


def _result_path(out_dir: Path, seed: int, run_tag: str) -> Path:
    suffix = f"_{run_tag}" if run_tag else ""
    return out_dir / f"result_general_graph_smooth_seed{seed}{suffix}.json"


def _expected_source_token(graph_source: str) -> str:
    src = str(graph_source).lower()
    if "classifier_head" in src:
        return "classifier_head"
    if "ema" in src:
        return "ema"
    if "weight" in src:
        return "weight"
    return "update"


def _validate_result(path: Path, preset: str, expected: Dict[str, object]) -> None:
    if not path.is_file():
        raise RuntimeError(f"Missing result file for preset={preset}: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    meta = payload.get("meta", {})
    graph_meta = dict(meta.get("graph", {}))

    preset_meta = str(graph_meta.get("graph_preset", meta.get("graph_preset", "none")))
    if preset_meta != preset:
        raise RuntimeError(
            f"[{preset}] graph_preset mismatch: meta={preset_meta!r}, expected={preset!r}"
        )

    exp_mode = str(expected.get("graph_mode", ""))
    if exp_mode and str(graph_meta.get("graph_mode")) != exp_mode:
        raise RuntimeError(
            f"[{preset}] graph_mode mismatch: "
            f"meta={graph_meta.get('graph_mode')!r}, expected={exp_mode!r}"
        )

    round_trace = payload.get("results", {}).get("graph_smooth", {}).get("round_trace", [])
    if not round_trace:
        raise RuntimeError(f"[{preset}] missing graph_smooth round_trace")

    first = dict(round_trace[0])
    rt_mode = str(first.get("graph_mode", ""))
    if exp_mode and rt_mode != exp_mode:
        raise RuntimeError(
            f"[{preset}] round_trace graph_mode mismatch: {rt_mode!r} != {exp_mode!r}"
        )

    exp_source = str(expected.get("graph_source", ""))
    if exp_source:
        source_used = str(first.get("graph_source_used", ""))
        token = _expected_source_token(exp_source)
        if token not in source_used.lower():
            raise RuntimeError(
                f"[{preset}] graph_source_used mismatch: "
                f"source_used={source_used!r}, expected token={token!r}"
            )

    density = float(first.get("graph_density", 0.0) or 0.0)
    if density <= 0.0:
        raise RuntimeError(
            f"[{preset}] degenerate graph detected in smoke test (graph_density={density})."
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Graph preset smoke test.")
    p.add_argument("--python-bin", type=str, default=sys.executable)
    p.add_argument("--dataset", type=str, default="fashionmnist")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rounds", type=int, default=1)
    p.add_argument("--num-clients", type=int, default=5)
    p.add_argument("--dirichlet-alpha", type=float, default=0.03)
    p.add_argument("--train-subset-size", type=int, default=1000)
    p.add_argument("--test-subset-size", type=int, default=500)
    p.add_argument(
        "--presets",
        nargs="+",
        default=["signed_conflict_knn", "pfedsim_like", "gfedfilt_like"],
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "experiments_current" / "graph_preset_smoke",
    )
    return p.parse_args()


def run() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Keep Ray startup stable across sequential tiny simulations.
    subprocess.run(
        [args.python_bin, "-m", "ray", "stop", "--force"],
        cwd=str(PROJECT_ROOT),
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    for preset in args.presets:
        expected = resolve_graph_preset_spec(preset)
        run_tag = f"preset_smoke_{preset}_seed{args.seed}"
        cmd: List[str] = [
            args.python_bin,
            "run_general_experiment.py",
            "--method",
            "graph_smooth",
            "--dataset",
            args.dataset,
            "--partition",
            "dirichlet",
            "--dirichlet-alpha",
            str(args.dirichlet_alpha),
            "--num-clients",
            str(args.num_clients),
            "--rounds",
            str(args.rounds),
            "--local-epochs",
            "1",
            "--batch-size",
            "64",
            "--model",
            "cnn",
            "--lr",
            "0.01",
            "--momentum",
            "0.9",
            "--weight-decay",
            "0.0005",
            "--seed",
            str(args.seed),
            "--graph-preset",
            preset,
            "--graph-variant",
            "update",
            "--train-subset-size",
            str(args.train_subset_size),
            "--test-subset-size",
            str(args.test_subset_size),
            "--out-dir",
            str(out_dir),
            "--run-tag",
            run_tag,
        ]
        print(f"[smoke] running preset={preset}")
        subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)
        result_path = _result_path(out_dir=out_dir, seed=int(args.seed), run_tag=run_tag)
        _validate_result(result_path, preset=preset, expected=expected)
        print(f"[smoke] preset={preset} validated")

    print(f"[smoke] all presets passed. output_dir={out_dir}")


if __name__ == "__main__":
    run()
