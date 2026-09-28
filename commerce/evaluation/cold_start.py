"""A-0 (evaluation.md §3): saved FL models scored at sellers the shared training never saw.

    python -m commerce.evaluation.cold_start --instacart-dir fedcommerce/data/instacart \
        --runs commerce/evaluation/runs/fl_100

Each held-out seller takes a run's saved shared weights, its own item text and
its own past visits, and is scored with no local weight update. GCI keeps word
token tables seller-local, so an hx model at a new seller has only the table's
initial values: that is hx's A-0 state, not a fault here. Held-out sellers are a
second split over the customers the cohort did not take (HELD_OUT_TARGETS). A
run trained with C-new keeps the same items out of every history and relation
here too, and "cnew" then reports a new seller's answers among items that no
training example contained.

--local-variants adds a reference: a model each held-out seller trains alone on
its own train visits (local_only, harex_compare.fit), what the seller could do
without the platform. Records stay under the Git-ignored runs/.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import time

import torch

from commerce.evaluation.encoder_probe import peak_memory_mb
from commerce.evaluation.harex_compare import architecture, build, code_version, evaluate, fit, new_arms, parts
from commerce.packages.recommender.harex import HarexRecommender


def find_runs(roots):
    """Every folder under the roots that holds an FL record and its saved shared weights."""
    runs = []
    for root in roots:
        for weights in sorted(Path(root).rglob("shared_weights.pt")):
            record = json.loads((weights.parent / "record.json").read_text(encoding="utf-8"))
            runs.append((weights.parent, record))
    return runs


def data_key(settings):
    """Runs that share the cohort split and the C-new rule can share one held-out build."""
    return settings.get("split_seed", 0), settings.get("holdout_frac", 0.0), settings.get("holdout_seed", 0)


def summary(metrics):
    out = {}
    for part, result in metrics.items():
        if result.get("examples"):
            out[part] = {k: round(v, 4) for k, v in result["macro"].items() if k in ("ndcg@10", "recall@20")}
            out[part]["queries"] = result["examples"]
    return out


def score_run(folder, record, sellers, first, device):
    settings = record["settings"]
    variant, target, seed = settings["variant"], settings["target"], settings["seed"]
    config = architecture(variant)
    weights = torch.load(folder / "shared_weights.pt", map_location="cpu")
    if weights["architecture"] != config.architecture_version:
        raise ValueError("%s: saved %s, expected %s" % (folder, weights["architecture"], config.architecture_version))
    arm = "%s FL A-0" % variant
    result = {"folder": str(folder), "variant": variant, "target": target, "seed": seed,
              "holdout_frac": settings.get("holdout_frac", 0.0), "best_round": weights["best_round"]}
    for which in ("final_round", "best_round"):
        arms = new_arms((arm, "popularity", "P-TopFreq"))
        for sid, info in sellers.items():
            torch.manual_seed(seed)  # the same initial values fl_lab gives every seller
            model = HarexRecommender(config, vocab_size=info["vocab_size"] if config.text == "hx" else None)
            missing, unexpected = model.load_state_dict(weights["%s_shared" % which], strict=False)
            if unexpected or any(not k.startswith("local_") for k in missing):
                raise ValueError("%s: shared weights do not fit %s" % (folder, config.architecture_version))
            model.to(device)
            evaluate(model, info, parts(info, sid, "test", variant, target, first, device, seed), target, arms, arm)
        result[which] = {name: {p: a.result() for p, a in by_part.items()} for name, by_part in arms.items()}
    return result


def local_reference(args, sellers, first, variant, target, device):
    arm = "%s local_only" % variant
    arms = new_arms((arm,))
    training = {}
    for sid, info in sellers.items():
        train_parts = parts(info, sid, "train", variant, target, first, device, args.seed)
        val_parts = parts(info, sid, "validation", variant, target, first, device, args.seed)
        model, log = fit(args, info, train_parts, val_parts, variant, device)
        evaluate(model, info, parts(info, sid, "test", variant, target, first, device, args.seed), target, arms, arm)
        training[sid] = {k: v for k, v in log.items() if k != "history"}
    return {"variant": variant, "target": target, "training": training,
            "metrics": {p: a.result() for p, a in arms[arm].items()}}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--instacart-dir", type=Path, required=True)
    parser.add_argument("--encoder-cache", type=Path, default=Path("commerce/evaluation/cache/encoders"))
    parser.add_argument("--z-cache", type=Path, default=Path("commerce/evaluation/cache/z/instacart.sqlite"))
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--held-out-sellers", type=int, default=20)
    parser.add_argument("--local-variants", default="")
    parser.add_argument("--max-steps", type=int, default=20000)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=Path, default=Path("commerce/evaluation/runs/cold_start"))
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    local_variants = [v for v in args.local_variants.split(",") if v]

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    record = {"run": "A-0 cold start", "started_at": stamp, "code": code_version(),
              "settings": {k: (str(v) if isinstance(v, Path) else [str(x) for x in v] if isinstance(v, list) else v)
                           for k, v in vars(args).items()},
              "machine": {"os": platform.platform(), "torch": torch.__version__, "device": str(device),
                          "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None},
              "labels": ["pilot", "stand-in seller sizes", "A-0: no local weight update", "temporary scoring"],
              "groups": []}
    runs = find_runs(args.runs)
    if not runs:
        raise SystemExit("no shared_weights.pt under %s" % ", ".join(map(str, args.runs)))
    groups = {}
    for folder, run in runs:
        groups.setdefault(data_key(run["settings"]), []).append((folder, run))

    started = time.perf_counter()
    for (split_seed, frac, holdout_seed), members in sorted(groups.items()):
        variants = sorted({run["settings"]["variant"] for _, run in members} | set(local_variants))
        build_args = argparse.Namespace(
            instacart_dir=args.instacart_dir, encoder_cache=args.encoder_cache, z_cache=args.z_cache,
            sellers=members[0][1]["settings"]["sellers"], held_out_sellers=args.held_out_sellers,
            split_seed=split_seed, holdout_frac=frac, holdout_seed=holdout_seed, variants=variants)
        group = {"split_seed": split_seed, "holdout_frac": frac, "holdout_seed": holdout_seed, "runs": [], "local": []}
        sellers, first = build(build_args, group, held_out=True)
        for folder, run in members:
            result = score_run(folder, run, sellers, first, device)
            group["runs"].append(result)
            arm = "%s FL A-0" % result["variant"]
            print(json.dumps({"run": folder.name, "final_round": summary(result["final_round"][arm]),
                              "best_round": summary(result["best_round"][arm]),
                              "P-TopFreq": summary(result["final_round"]["P-TopFreq"])}, ensure_ascii=False), flush=True)
        for target in sorted({run["settings"]["target"] for _, run in members}):
            for variant in local_variants:
                result = local_reference(args, sellers, first, variant, target, device)
                group["local"].append(result)
                print(json.dumps({"local_only": variant, "target": target, "metrics": summary(result["metrics"])},
                                 ensure_ascii=False), flush=True)
        record["groups"].append(group)
    record["seconds"] = round(time.perf_counter() - started, 1)
    record["peak_memory_mb"] = peak_memory_mb()
    out = args.out_dir / ("%s_%d" % (stamp, os.getpid()))
    out.mkdir(parents=True, exist_ok=True)
    (out / "record.json").write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    print("-> %s" % out)


if __name__ == "__main__":
    main()
