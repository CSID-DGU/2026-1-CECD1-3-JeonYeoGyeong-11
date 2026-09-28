"""federated_lab_sim (model-lab.md §4, D0020) of the D0022 variants: an UNPROTECTED FL simulation.

    python -m commerce.evaluation.fl_lab --instacart-dir fedcommerce/data/instacart --variant T_hx

Each round every seller starts from the same global shared weights, trains
--local-epochs on its own data with a fresh AdamW (GCI: one epoch per round,
batch 128) and returns shared_delta = after - before. A seller's word-token
table (hx) stays with it and is never aggregated, as in GCI's glocalization.
The round is kept only if every seller completed; then the global model adds
the uniform mean of the deltas (D0017). aggregate_uniform is a TEMPORARY copy
of that rule until C's aggregation core lands (working-agreement §8, D3); swap
it in then and check the result is the same.

The run has a fixed number of rounds (evaluation.md §5) and its result is the
last round. Beside it, as an auxiliary, the round with the lowest validation
loss summed over all sellers and divided by the total example count is tested
too: an aggregate, never a per-seller value. The shared weights of both rounds
are saved next to the record (Git-ignored runs/) for later held-out scoring.
Results are labelled "비보호 FL 시뮬레이션" (unprotected FL simulation).
"""
import argparse
from contextlib import contextmanager
import copy
import dataclasses
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import time

import torch

from commerce.evaluation.encoder_probe import peak_memory_mb
from commerce.evaluation.harex_compare import (
    TARGETS, VARIANTS, architecture, build, code_version, evaluate, new_arms, parts,
)
from commerce.packages.recommender.harex import HarexRecommender
from commerce.packages.recommender.training import TrainConfig, seller_on, train, validation_loss


def aggregate_uniform(deltas: list[dict[str, torch.Tensor] | None]) -> dict[str, torch.Tensor] | None:
    """TEMPORARY stand-in for C's aggregation core: all complete or discard, then the uniform mean."""
    if not deltas or any(d is None for d in deltas):
        return None
    keys = set(deltas[0])
    if any(set(d) != keys for d in deltas):
        raise ValueError("sellers returned different tensor sets")
    return {k: torch.stack([d[k] for d in deltas]).mean(0) for k in keys}


def on_device(seller_parts, device):
    """Move one seller's data to the device only while it trains: 100 sellers' relation snapshots
    do not fit in GPU memory together."""
    return [(bucket, seller_on(p, device)) for bucket, p in seller_parts]


@contextmanager
def staged(model, device):
    """A seller's model waits on the CPU and visits the device only for its own turn: 100 sellers'
    models held on one shared GPU for the whole run take gigabytes per run."""
    model.to(device)
    try:
        yield model
    finally:
        model.to("cpu")


def local_round(model: HarexRecommender, global_shared: dict, train_parts, epochs: int, batch_size: int,
                lr: float, seed: int) -> dict[str, torch.Tensor] | None:
    model.load_state_dict(global_shared, strict=False)  # local_tokens keep the seller's own values
    before = {k: v.detach().clone() for k, v in model.shared_state().items()}
    n = sum(len(p.examples) for _, p in train_parts)
    if n == 0:
        return None  # nothing to learn from: the round cannot complete
    steps = max(1, math.ceil(n / batch_size) * epochs)
    train(model, [p for _, p in train_parts], TrainConfig(steps=steps, batch_size=batch_size, lr=lr, seed=seed))
    after = model.shared_state()
    return {k: (after[k] - before[k]) for k in before if before[k].is_floating_point()}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--instacart-dir", type=Path, required=True)
    parser.add_argument("--encoder-cache", type=Path, default=Path("commerce/evaluation/cache/encoders"))
    parser.add_argument("--z-cache", type=Path, default=Path("commerce/evaluation/cache/z/instacart.sqlite"))
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--target", default="basket", choices=TARGETS)
    parser.add_argument("--sellers", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=300)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--holdout-frac", type=float, default=0.0)  # C-new: 0.1 (evaluation.md §3)
    parser.add_argument("--holdout-seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=Path, default=Path("commerce/evaluation/runs/fl_lab"))
    args = parser.parse_args(argv)
    args.variants = [args.variant]
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    config = architecture(args.variant)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    record = {"run": "D0022 federated_lab_sim", "mode": "비보호 FL 시뮬레이션 (unprotected FL simulation, D0020)",
              "aggregation": "temporary uniform mean, all complete or discard (stand-in for C's core)",
              "started_at": stamp, "code": code_version(), "settings": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
              "machine": {"os": platform.platform(), "torch": torch.__version__, "device": str(device),
                          "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                          "cpu_count": os.cpu_count()},
              "labels": ["pilot", "single seed", "stand-in seller sizes", "unprotected FL simulation",
                         "temporary aggregation", "temporary scoring"]
              + (["C-new: items held out of training"] if args.holdout_frac > 0 else [])}
    sellers, first = build(args, record)

    # One model object per seller carries its local token table; all share the global weights.
    torch.manual_seed(args.seed)
    template = HarexRecommender(config, vocab_size=next(iter(sellers.values()))["vocab_size"]
                                if config.text == "hx" else None)
    global_shared = {k: v.detach().clone().to(device) for k, v in template.shared_state().items()}
    local = {}
    for seller, info in sellers.items():
        torch.manual_seed(args.seed)
        model = HarexRecommender(config, vocab_size=info["vocab_size"] if config.text == "hx" else None)
        model.load_state_dict(global_shared, strict=False)
        cpu = torch.device("cpu")
        local[seller] = {"model": model,
                         "train": parts(info, seller, "train", args.variant, args.target, first, cpu, args.seed),
                         "validation": parts(info, seller, "validation", args.variant, args.target, first, cpu, args.seed),
                         "test": parts(info, seller, "test", args.variant, args.target, first, cpu, args.seed)}

    best, history, discarded = None, [], 0
    started = time.perf_counter()
    for rnd in range(args.rounds):
        deltas = []
        for i, s in enumerate(local.values()):
            with staged(s["model"], device) as model:
                deltas.append(local_round(model, global_shared, on_device(s["train"], device), args.local_epochs,
                                          args.batch_size, args.lr, args.seed * 100000 + rnd * 1000 + i))
        mean = aggregate_uniform(deltas)
        if mean is None:
            discarded += 1
            continue
        global_shared = {k: (v + mean[k] if k in mean else v) for k, v in global_shared.items()}
        total, count = 0.0, 0
        for s in local.values():
            n = sum(len(p.examples) for _, p in s["validation"])
            with staged(s["model"], device) as model:
                model.load_state_dict(global_shared, strict=False)
                loss = validation_loss(model, [p for _, p in on_device(s["validation"], device)], seed=args.seed)
            if loss is not None:
                total, count = total + loss * n, count + n  # only the sum and count leave the loop
        val = total / count if count else None
        history.append([rnd + 1, val])
        if val is not None and (best is None or val < best["val"]):
            best = {"round": rnd + 1, "val": val, "shared": {k: v.clone() for k, v in global_shared.items()},
                    "local": {sid: {k: v.clone() for k, v in s["model"].state_dict().items() if k.startswith("local_")}
                              for sid, s in local.items()}}
        if (rnd + 1) % 10 == 0:
            print("round %d val %.4f best %d (%.4f) %.0fs" % (rnd + 1, val, best["round"], best["val"],
                                                            time.perf_counter() - started), flush=True)

    # The first run has a fixed round count (evaluation.md §5), so the last round is the result.
    # The best aggregate-validation round is reported beside it as an auxiliary.
    final = {"shared": global_shared,
             "local": {sid: {k: v.clone() for k, v in s["model"].state_dict().items() if k.startswith("local_")}
                       for sid, s in local.items()}}
    # Runs started in the same second must not share a folder.
    out = args.out_dir / ("%s_%s_%s_s%d_%d" % (stamp, args.variant, args.target, args.seed, os.getpid()))
    out.mkdir(parents=True, exist_ok=True)
    # The shared weights alone (no seller's token table), so sellers left out of training can
    # later be scored with the same models (evaluation.md §3, A-0).
    torch.save({"architecture": config.architecture_version, "variant": args.variant, "best_round": best["round"],
                "final_round_shared": {k: v.cpu() for k, v in final["shared"].items()},
                "best_round_shared": {k: v.cpu() for k, v in best["shared"].items()}}, out / "shared_weights.pt")
    record["weights"] = "shared_weights.pt"
    arm = "%s FL" % args.variant
    record["metrics"] = {}
    for which, state in (("final_round", final), ("best_round", best)):
        arms = new_arms((arm, "popularity", "P-TopFreq"))
        for sid, s in local.items():
            with staged(s["model"], device) as model:
                model.load_state_dict(state["shared"], strict=False)
                model.load_state_dict(state["local"][sid], strict=False)
                evaluate(model, sellers[sid], on_device(s["test"], device), args.target, arms, arm)
        record["metrics"][which] = {name: {p: a.result() for p, a in parts_.items()} for name, parts_ in arms.items()}
    record["metrics"]["primary"] = "final_round"
    record["training"] = {"rounds": args.rounds, "discarded_rounds": discarded, "best_round": best["round"],
                          "best_val_loss_aggregate": best["val"], "history": history,
                          # Best round in the last tenth: the curve was still falling, so run longer.
                          "plateaued": best["round"] <= 0.9 * args.rounds,
                          "seconds": round(time.perf_counter() - started, 1)}
    record["peak_memory_mb"] = peak_memory_mb()
    (out / "record.json").write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({which: {name: {k: round(v, 4) for k, v in m["all"]["macro"].items()
                                     if k in ("ndcg@10", "recall@20", "hr@10")} for name, m in record["metrics"][which].items()}
                      for which in ("final_round", "best_round")}, indent=2))
    print("best round %d of %d (plateaued=%s), discarded %d -> %s" % (
        best["round"], args.rounds, record["training"]["plateaued"], discarded, out))


if __name__ == "__main__":
    main()
