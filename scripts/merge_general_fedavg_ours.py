"""Merge separate FedAvg and Ours general-FL JSONs for scripts/deep_dive_seed.py."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fedavg", type=str, required=True)
    p.add_argument("--ours", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    args = p.parse_args()
    fa = json.loads(Path(args.fedavg).read_text(encoding="utf-8"))
    ou = json.loads(Path(args.ours).read_text(encoding="utf-8"))
    merged = {
        "meta": ou.get("meta", {}),
        "results": {
            "fedavg": fa["results"]["fedavg"],
            "ours": ou["results"]["ours"],
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
