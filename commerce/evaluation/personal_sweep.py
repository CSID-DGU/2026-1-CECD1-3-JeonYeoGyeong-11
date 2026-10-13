"""Personalization settings chosen on validation (model.md §8.1), on held-out Instacart sellers.

    python -m commerce.evaluation.personal_sweep --instacart-dir fedcommerce/data/instacart \\
        --encoder-dir commerce/evaluation/cache/encoders/<model>/<revision> \\
        --releases commerce/evaluation/outputs/releases --sellers 10 --lr 3e-4 1e-3 --steps 10 20 40

Each held-out seller (A-0) gets every customer's visits but the last as its ledger and
the first releases as its bases. For each variant and setting, a copy of the base learns
query_proj and scorer on the train customers of personalize_local's split
(PERSONAL_SPLIT_SEED) and is judged on its validation customers, as personalize_local
judges it. The choice uses the validation loss only; each customer's last visit (HR@10)
only shows what a choice would serve. One setting for both variants (model.md §8.1).
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import tempfile

import torch

from commerce.evaluation.e_g0 import STAND_IN_TARGETS
from commerce.evaluation.harex_compare import HELD_OUT_TARGETS
from commerce.packages.data_adapters.assignment import assign_clients
from commerce.packages.data_adapters.instacart import load_instacart
from commerce.packages.recommender.seller_runtime import (
    CLIP_NORM, DEFAULT_PERSONAL, PERSONAL_GROUPS, PERSONAL_SPLIT_SEED, WEIGHT_DECAY, SellerRuntime,
)
from commerce.packages.recommender.serving import FrozenText, build_model, load_bundle, load_shared
from commerce.packages.recommender.training import TrainConfig, train, validation_loss

RELEASES = {"text_only": "ic100-T_lm-r500", "text_relation": "ic100-R_lm-r500"}


def split_last_visit(events):
    """(answers: customer -> items of the last visit, ledger: every earlier visit)."""
    by_customer = defaultdict(list)
    for event in events:
        by_customer[event["customer_id_local"]].append(event)
    answers, ledger = {}, []
    for customer, own in by_customer.items():
        own.sort(key=lambda e: e["order_rank"])
        if len(own) >= 2:
            answers[customer] = {i["item_id_local"] for i in own[-1]["items"]}
            ledger.extend(own[:-1])
    return answers, ledger


def hit_rate(runtime, arch, base, model, snap, answers) -> float | None:
    """HR@10 of each customer's last visit when the model ranks the seller's catalog."""
    hits = []
    for customer, answer in answers.items():
        scores = runtime._model_scores(arch, base, model, snap, list(snap.baskets), customer)
        if scores is None:
            continue
        top = [i for _, i in sorted(((s, i) for i, s in scores.items()), key=lambda p: (-p[0], p[1]))[:10]]
        hits.append(float(bool(answer & set(top))))
    return statistics.fmean(hits) if hits else None


def seller_rows(seller, events, catalog, text, releases, grid) -> list[dict]:
    answers, ledger = split_last_visit(events)
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        runtime = SellerRuntime(seller, Path(tmp) / "features.sqlite", Path(tmp) / "models", text=text)
        for seq, item in enumerate(catalog, start=1):
            runtime.upsert_catalog_item(item, seq)
        for event in ledger:
            runtime.ingest_purchase_event(event)
        snap = runtime.store.snapshot()
        for variant, folder in RELEASES.items():
            release, manifest, tensors = load_bundle(releases / folder)
            runtime.install_release(release, manifest, tensors, model_variant=variant)
            arch, base = runtime._arch(variant), runtime._handles[variant]
            train_parts, val_parts = runtime.training_parts(snap.epoch, variant, PERSONAL_SPLIT_SEED)
            n_neg = DEFAULT_PERSONAL["n_neg"]
            base_model = build_model(arch.config)
            load_shared(base_model, base.tensors)
            base_model.eval()
            base_loss = validation_loss(base_model, val_parts, n_negatives=n_neg, seed=PERSONAL_SPLIT_SEED)
            base_hr = hit_rate(runtime, arch, base, base_model, snap, answers)
            for lr, steps in grid:
                model = build_model(arch.config)
                load_shared(model, base.tensors)
                train(model, train_parts, TrainConfig(steps=steps, batch_size=DEFAULT_PERSONAL["batch_size"],
                                                      n_negatives=n_neg, lr=lr, weight_decay=WEIGHT_DECAY,
                                                      clip_norm=CLIP_NORM, seed=DEFAULT_PERSONAL["seed"]),
                      only=PERSONAL_GROUPS)
                model.eval()
                loss = validation_loss(model, val_parts, n_negatives=n_neg, seed=PERSONAL_SPLIT_SEED)
                rows.append({"seller": seller, "variant": variant, "lr": lr, "steps": steps,
                             "base_loss": base_loss, "loss": loss,
                             "accepted": base_loss is not None and loss is not None and loss < base_loss,
                             "base_hr10": base_hr, "hr10": hit_rate(runtime, arch, base, model, snap, answers)})
        runtime.close()
    return rows


def summarize(rows, grid) -> dict:
    out = {}
    for lr, steps in grid:
        setting = {}
        for variant in RELEASES:
            r = [x for x in rows if x["variant"] == variant and x["lr"] == lr and x["steps"] == steps]
            gains = [(x["base_loss"] - x["loss"]) / x["base_loss"] for x in r if x["base_loss"] and x["loss"] is not None]
            served = [(x["hr10"] if x["accepted"] else x["base_hr10"]) - x["base_hr10"]
                      for x in r if x["base_hr10"] is not None]
            setting[variant] = {"accepted": "%d/%d" % (sum(x["accepted"] for x in r), len(r)),
                                "val_loss_gain_mean": round(statistics.fmean(gains), 4),
                                "test_hr10_delta_served": round(statistics.fmean(served), 4)}
        setting["val_loss_gain_both"] = round(statistics.fmean(s["val_loss_gain_mean"] for s in setting.values()), 4)
        out["lr=%g steps=%d" % (lr, steps)] = setting
    return out


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--instacart-dir", type=Path, required=True)
    parser.add_argument("--encoder-dir", type=Path, required=True)
    parser.add_argument("--releases", type=Path, required=True, help="the folder holding the release folders")
    parser.add_argument("--sellers", type=int, default=10)
    parser.add_argument("--lr", type=float, nargs="+", default=[3e-4, 1e-3])
    parser.add_argument("--steps", type=int, nargs="+", default=[10, 20, 40])
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--rows", type=Path, help="write every (seller, variant, setting) row here")
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads)
    grid = [(lr, steps) for lr in args.lr for steps in args.steps]
    cohort = assign_clients(args.instacart_dir, STAND_IN_TARGETS, alpha=0.25, seed=0)
    held = assign_clients(args.instacart_dir, HELD_OUT_TARGETS, alpha=0.25, seed=0, exclude=set(cohort.clients))
    chosen = sorted(HELD_OUT_TARGETS)[:args.sellers]
    sample = load_instacart(args.instacart_dir, {u: c for u, c in held.clients.items() if c in chosen})
    text = FrozenText(args.encoder_dir)
    rows = []
    for client in chosen:
        seller = "ic-client-%d" % client
        rows += seller_rows(seller, [e for e in sample.events if e["seller_id"] == seller],
                            [i for i in sample.catalog_items if i["seller_id"] == seller], text, args.releases, grid)
    if args.rows is not None:
        args.rows.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    summary = summarize(rows, grid)
    best = max(summary, key=lambda k: summary[k]["val_loss_gain_both"])
    print(json.dumps({"sellers": len(chosen), "by_setting": summary, "chosen_by_validation": best},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
