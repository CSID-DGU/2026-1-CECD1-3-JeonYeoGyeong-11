"""D0022: four item representations on one HAREX (GCI) style backbone, trained to convergence.

    python -m commerce.evaluation.harex_compare --instacart-dir fedcommerce/data/instacart

Variants: T_hx (GCI word tokens, the HAREX baseline), R_hx (+ local purchase relations),
T_lm (pretrained encoder), R_lm. Targets: "basket" is the next visit's item set
(evaluation.md §1); "next_item" is the first item added to the next visit's cart,
predicted from earlier visits only (D0022, HAREX style).

Mode local_only (model-lab.md §4): for every variant, target and seller, a model
starts from the same seed, trains on that seller only with one AdamW, and is
checked on its fixed validation loss every --eval-every steps; training stops
after --patience checks without improvement (GCI used patience 20) or at
--max-steps, and the best check is kept. Test examples are ranked over the
seller's whole catalog next to local popularity and P-TopFreq. Sellers come from
the train-only split with stand-in target sizes until the Dunnhumby roster exists.
Scores come from scoring.py, a temporary copy of A's definitions. Records stay
under the Git-ignored commerce/evaluation/runs/.
"""
import argparse
import copy
import dataclasses
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import torch

from commerce.evaluation.e_g0 import STAND_IN_TARGETS
from commerce.evaluation.encoder_probe import CANDIDATES, peak_memory_mb
from commerce.evaluation.scoring import MacroAverager, expected_metrics, p_topfreq_scores, popularity_scores
from commerce.packages.data_adapters.assignment import assign_clients
from commerce.packages.data_adapters.baskets import basket_from_event, customer_visits
from commerce.packages.data_adapters.instacart import first_in_cart, item_id, load_instacart, split_role
from commerce.packages.data_adapters.text import catalog_item_text
from commerce.packages.recommender.examples import customer_examples
from commerce.packages.recommender.harex import HAREX_ARCHITECTURES, HarexRecommender, WordVocabulary
from commerce.packages.recommender.relations import replay_relations
from commerce.packages.recommender.replay import SellerReplay, progress_bucket
from commerce.packages.recommender.text_encoder import FrozenTextEncoder
from commerce.packages.recommender.training import (
    SellerData, TrainConfig, catalog_scores, seller_on, train, validation_loss,
)
from commerce.packages.recommender.z_cache import ZCache

VARIANTS = ("T_hx", "R_hx", "T_lm", "R_lm")
TARGETS = ("basket", "next_item")


def build(args, record):
    started = time.perf_counter()
    split = assign_clients(args.instacart_dir, STAND_IN_TARGETS, alpha=0.25, seed=args.seed)
    chosen = sorted(STAND_IN_TARGETS)[:args.sellers]
    users = {u: c for u, c in split.clients.items() if c in chosen}
    sample = load_instacart(args.instacart_dir, users)
    orders = [int(e["basket_id_local"].rsplit("-", 1)[1]) for e in sample.events]
    first = first_in_cart(args.instacart_dir, orders)
    record["data"] = {"assignment": split.record, "stand_in_targets": "100 x 1,040 train orders",
                      "chosen_clients": chosen, "adapter_report": sample.report}

    by_seller, catalogs = {}, {}
    for event in sample.events:
        b = basket_from_event(event)
        by_seller.setdefault(b.seller_id, {}).setdefault(b.customer_id_local, []).append(b)
    for item in sample.catalog_items:
        catalogs.setdefault(item["seller_id"], []).append(item)

    spec = CANDIDATES["minilm-l12"]
    model_dir = args.encoder_cache / spec.model_id.replace("/", "__") / spec.revision
    args.z_cache.parent.mkdir(parents=True, exist_ok=True)
    cache = ZCache(args.z_cache, FrozenTextEncoder(model_dir, spec), model_dir)
    need_relations = any(v.startswith("R_") for v in args.variants)

    sellers = {}
    counts = {"examples": {r: 0 for r in ("train", "validation", "test")}, "catalog_items": 0, "vocab": {}}
    for seller, customers in sorted(by_seller.items()):
        items = tuple(c["item_id_local"] for c in catalogs[seller])
        row_of = {item: row for row, item in enumerate(items)}
        texts = [catalog_item_text(c) for c in catalogs[seller]]
        z = torch.from_numpy(cache.vectors(texts).copy())
        vocab = WordVocabulary(texts)
        tokens = vocab.encode(texts, HAREX_ARCHITECTURES["harex.T_hx.v1"].max_tokens)
        visits = {cust: customer_visits(bs) for cust, bs in customers.items()}
        grouped = {}
        for vs in visits.values():
            for example in customer_examples(vs, range(1, len(vs) + 1)):
                key = (split_role(example.target_position, len(vs)), progress_bucket(example.target_position, len(vs)))
                grouped.setdefault(key, []).append(example)
        snapshots = {b: replay_relations(visits, b, row_of, len(items))
                     for b in sorted({b for _, b in grouped})} if need_relations else {}
        sellers[seller] = {"items": items, "z": z, "tokens": tokens, "vocab_size": len(vocab),
                           "visits": visits, "replay": SellerReplay(visits), "grouped": grouped,
                           "snapshots": snapshots}
        for (role, _), examples in grouped.items():
            counts["examples"][role] += len(examples)
        counts["catalog_items"] += len(items)
        counts["vocab"][seller] = len(vocab)
    cache.close()
    record["data"].update(counts)
    record["encoder"] = {"text_artifact_hash": cache.text_artifact_hash,
                         "preprocessing_version": cache.preprocessing_version}
    record["seconds"] = {"build": round(time.perf_counter() - started, 1)}
    return sellers, first


def with_target(examples, target, first):
    """next_item keeps only the first item added to the target visit's cart."""
    if target == "basket":
        return examples
    out = []
    for example in examples:
        order = int(example.target_basket_id.rsplit("-", 1)[1])
        out.append(dataclasses.replace(example, target_items=frozenset({item_id(first[order])})))
    return out


def parts(info, seller, role, variant, target, first, device):
    config = HAREX_ARCHITECTURES["harex.%s.v1" % variant]
    out = []
    for (r, bucket), examples in sorted(info["grouped"].items()):
        if r != role:
            continue
        data = SellerData(seller, info["items"], info["z"], with_target(examples, target, first),
                          info["snapshots"][bucket] if config.relation else None,
                          info["tokens"] if config.text == "hx" else None)
        out.append((bucket, seller_on(data, device)))
    return out


def fit(args, info, train_parts, val_parts, variant, device):
    config = HAREX_ARCHITECTURES["harex.%s.v1" % variant]
    torch.manual_seed(args.seed)
    model = HarexRecommender(config, vocab_size=info["vocab_size"] if config.text == "hx" else None).to(device)
    base = TrainConfig(steps=args.eval_every, batch_size=args.batch_size, lr=args.lr)
    optimizer = torch.optim.AdamW(model.parameters(), lr=base.lr, weight_decay=base.weight_decay)
    best, best_loss, best_step, since, history, step = None, None, 0, 0, [], 0
    started = time.perf_counter()
    while step < args.max_steps and since < args.patience:
        log = train(model, [p for _, p in train_parts], dataclasses.replace(base, seed=args.seed * 100000 + step),
                    optimizer)
        step += args.eval_every
        val = validation_loss(model, [p for _, p in val_parts], seed=args.seed)
        history.append([step, log["loss_mean"], val])
        if val is not None and (best_loss is None or val < best_loss - 1e-4):
            best, best_loss, best_step, since = copy.deepcopy(model.state_dict()), val, step, 0
        else:
            since += 1
    if best is not None:
        model.load_state_dict(best)
    return model, {"best_val_loss": best_loss, "best_step": best_step, "steps_run": step,
                   "converged": since >= args.patience, "seconds": round(time.perf_counter() - started, 1),
                   "history": history}


def evaluate(model, info, test_parts, target, arms, model_arm):
    for bucket, seller in test_parts:
        if not seller.examples:
            continue
        scores = catalog_scores(model, seller, seller.examples).numpy()
        seller_counts = info["replay"].counts(bucket)
        for row, example in enumerate(seller.examples):
            relevant = {seller.row_of[i] for i in example.target_items}
            candidates = {model_arm: scores[row],
                          "popularity": popularity_scores(seller.items, seller_counts),
                          "P-TopFreq": p_topfreq_scores(seller.items, example.prior_counts, seller_counts)}
            repeat = {r for r in relevant if seller.items[r] in example.prior_counts}
            split = {"all": relevant, "repeat": repeat, "explore": relevant - repeat}
            for name, ranking in candidates.items():
                for part, rel in split.items():
                    if rel:
                        arms[name][part].add(seller.seller_id, example.customer_id_local,
                                             expected_metrics(ranking, rel))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--instacart-dir", type=Path, required=True)
    parser.add_argument("--encoder-cache", type=Path, default=Path("commerce/evaluation/cache/encoders"))
    parser.add_argument("--z-cache", type=Path, default=Path("commerce/evaluation/cache/z/instacart.sqlite"))
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--targets", default=",".join(TARGETS))
    parser.add_argument("--sellers", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=20000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)  # GCI: BATCH_SIZE
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=Path, default=Path("commerce/evaluation/runs/harex_compare"))
    args = parser.parse_args(argv)
    args.variants = [v for v in args.variants.split(",") if v]
    args.targets = [t for t in args.targets.split(",") if t]
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    record = {"run": "D0022 harex_compare", "started_at": stamp, "seed": args.seed,
              "settings": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
              "machine": {"os": platform.platform(), "python": platform.python_version(), "torch": torch.__version__,
                          "device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                          "threads": torch.get_num_threads(), "cpu_count": os.cpu_count()},
              "labels": ["pilot", "single seed", "stand-in seller sizes", "local_only, not FL", "temporary scoring"],
              "results": {}}
    sellers, first = build(args, record)
    for target in args.targets:
        for variant in args.variants:
            arm = "%s %s" % (variant, "(HAREX baseline)" if variant == "T_hx" else "")
            arm = arm.strip()
            arms = {name: {p: MacroAverager() for p in ("all", "repeat", "explore")}
                    for name in (arm, "popularity", "P-TopFreq")}
            per_seller = {}
            for seller, info in sellers.items():
                train_parts = parts(info, seller, "train", variant, target, first, device)
                val_parts = parts(info, seller, "validation", variant, target, first, device)
                test_parts = parts(info, seller, "test", variant, target, first, device)
                model, log = fit(args, info, train_parts, val_parts, variant, device)
                evaluate(model, info, test_parts, target, arms, arm)
                per_seller[seller] = {k: v for k, v in log.items() if k != "history"} | {"history": log["history"]}
                print("%s %s %s best_val %.4f at %d/%d converged=%s %.0fs" % (
                    target, variant, seller, log["best_val_loss"] or float("nan"), log["best_step"], log["steps_run"],
                    log["converged"], log["seconds"]), flush=True)
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            record["results"].setdefault(target, {})[variant] = {
                "training": per_seller,
                "metrics": {name: {p: a.result() for p, a in parts_.items()} for name, parts_ in arms.items()}}
    record["peak_memory_mb"] = peak_memory_mb()
    if device.type == "cuda":
        record["peak_gpu_memory_mb"] = round(torch.cuda.max_memory_allocated() / 2**20)
    # Runs started in the same second must not share a folder.
    out = args.out_dir / ("%s_%s_s%d_%d" % (stamp, "-".join(args.targets), args.seed, os.getpid()))
    out.mkdir(parents=True, exist_ok=True)
    (out / "record.json").write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = {}
    for target, variants in record["results"].items():
        for variant, res in variants.items():
            m = res["metrics"]
            summary["%s/%s" % (target, variant)] = {name: {k: round(v, 4) for k, v in m[name]["all"]["macro"].items()
                                                           if k in ("ndcg@10", "recall@20", "hr@10")}
                                                    for name in m}
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("-> %s" % out)


if __name__ == "__main__":
    main()
