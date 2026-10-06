"""E-G0: the small first run of the text-only baseline (evaluation.md §1·§5).

    python -m commerce.evaluation.e_g0 --instacart-dir fedcommerce/data/instacart

Instacart only. Sellers come from the train-only assignment, but until the
Dunnhumby roster exists its target sizes are a stand-in (100 x 1,040 train
orders), recorded as such. The mode is local_only (model-lab.md §4): every
seller trains its own model on its own data from the same initial weights,
picks its own chunk by its own fixed validation loss and is scored on its own
test examples; the results are then averaged over sellers. Several sellers'
data are never pooled into one optimizer. FL runs (federated_lab_sim) come with
C's aggregation core. The training runs in chunks that each start a fresh
AdamW, as FL rounds do. Scores come from scoring.py, a temporary copy of A's
metric definitions.
Everything is written under commerce/evaluation/runs/, which Git ignores.
"""
import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import torch

from commerce.evaluation.encoder_probe import CANDIDATES, peak_memory_mb
from commerce.evaluation.scoring import MacroAverager, expected_metrics, p_topfreq_scores, popularity_scores
from commerce.packages.data_adapters.assignment import assign_clients
from commerce.packages.data_adapters.baskets import basket_from_event, customer_visits
from commerce.packages.data_adapters.instacart import load_instacart, split_role
from commerce.packages.data_adapters.text import catalog_item_text
from commerce.packages.recommender.examples import customer_examples
from commerce.packages.recommender.model import TextOnlyRecommender, architecture, config_record
from commerce.packages.recommender.replay import SellerReplay, progress_bucket
from commerce.packages.recommender.text_encoder import FrozenTextEncoder
from commerce.packages.recommender.training import (
    SellerData, TrainConfig, catalog_scores, train, validation_loss,
)
from commerce.packages.recommender.z_cache import ZCache

STAND_IN_TARGETS = {990001 + k: 1040 for k in range(100)}


def build(args, record):
    started = time.perf_counter()
    split = assign_clients(args.instacart_dir, STAND_IN_TARGETS, alpha=0.25, seed=args.seed)
    chosen = sorted(STAND_IN_TARGETS)[:args.sellers]
    users = {u: c for u, c in split.clients.items() if c in chosen}
    sample = load_instacart(args.instacart_dir, users)
    record["data"] = {"assignment": split.record, "stand_in_targets": "100 x 1,040 train orders",
                      "chosen_clients": chosen, "adapter_report": sample.report}

    by_seller = {}
    for event in sample.events:
        b = basket_from_event(event)
        by_seller.setdefault(b.seller_id, {}).setdefault(b.customer_id_local, []).append(b)
    catalogs = {}
    for item in sample.catalog_items:
        catalogs.setdefault(item["seller_id"], []).append(item)

    spec = CANDIDATES[args.encoder]
    model_dir = args.encoder_cache / spec.model_id.replace("/", "__") / spec.revision
    encoder = FrozenTextEncoder(model_dir, spec)
    args.z_cache.parent.mkdir(parents=True, exist_ok=True)
    cache = ZCache(args.z_cache, encoder, model_dir)
    encode_started = time.perf_counter()

    data = {"train": [], "validation": [], "test": []}
    context = {}  # seller -> (visits by customer, replay)
    counts = {"examples": {r: 0 for r in data}, "catalog_items": 0}
    for seller, customers in sorted(by_seller.items()):
        items = [c["item_id_local"] for c in catalogs[seller]]
        z = torch.from_numpy(cache.vectors([catalog_item_text(c) for c in catalogs[seller]]).copy())
        visits = {cust: customer_visits(baskets) for cust, baskets in customers.items()}
        context[seller] = (visits, SellerReplay(visits))
        split_examples = {r: [] for r in data}
        for vs in visits.values():
            for example in customer_examples(vs, range(1, len(vs) + 1)):
                split_examples[split_role(example.target_position, len(vs))].append(example)
        for role in data:
            data[role].append(SellerData(seller, tuple(items), z, split_examples[role]))
            counts["examples"][role] += len(split_examples[role])
        counts["catalog_items"] += len(items)
    record["data"].update(counts)
    record["encoder"] = {"spec": spec.__dict__, "text_artifact_hash": cache.text_artifact_hash,
                         "preprocessing_version": cache.preprocessing_version,
                         "texts_encoded_this_run": cache.encoded,
                         "encode_seconds": round(time.perf_counter() - encode_started, 1)}
    cache.close()
    record["seconds"] = {"build": round(time.perf_counter() - started, 1)}
    return data, context


def fit_seller(args, train_part, val_part):
    """local_only: one seller's model on that seller's data; its own validation picks the chunk."""
    torch.manual_seed(args.seed)  # every seller starts from the same weights
    model = TextOnlyRecommender(architecture("text_only.v1"))
    best, best_loss, best_step, history = None, None, None, []
    for chunk in range(args.steps // args.eval_every):
        log = train(model, train_part, TrainConfig(steps=args.eval_every, seed=args.seed * 1000 + chunk))
        val = validation_loss(model, val_part, seed=args.seed)
        history.append({"step": (chunk + 1) * args.eval_every, "train_loss_mean": log["loss_mean"],
                        "val_loss": val, "skipped_batches": log["skipped_batches"]})
        if val is not None and (best_loss is None or val < best_loss):
            best, best_loss, best_step = copy.deepcopy(model.state_dict()), val, (chunk + 1) * args.eval_every
    if best is not None:
        model.load_state_dict(best)
    return model, {"best_val_loss": best_loss, "best_step": best_step, "history": history}


def fit(args, data, record):
    models, per_seller = {}, {}
    started = time.perf_counter()
    for seller in sorted({s.seller_id for s in data["train"]}):
        train_part = [s for s in data["train"] if s.seller_id == seller and s.examples]
        val_part = [s for s in data["validation"] if s.seller_id == seller and s.examples]
        if not train_part:
            per_seller[seller] = {"skipped": "no training examples"}
            continue
        models[seller], per_seller[seller] = fit_seller(args, train_part, val_part)
        print("%s best val %s at step %s" % (seller, per_seller[seller]["best_val_loss"],
                                            per_seller[seller]["best_step"]), flush=True)
    train_seconds = time.perf_counter() - started
    any_model = next(iter(models.values()))
    record["model"] = {"config": config_record(any_model.config),
                       "parameters_per_seller_model": sum(p.numel() for p in any_model.parameters())}
    record["training"] = {"mode": "local_only (model-lab.md §4): one model per seller, trained on that seller only",
                          "steps_per_seller": args.steps, "eval_every": args.eval_every,
                          "per_seller": per_seller, "seconds": round(train_seconds, 1),
                          "seconds_per_step": round(train_seconds / (args.steps * len(models)), 4)}
    return models


def evaluate(models, data, context, record):
    started = time.perf_counter()
    arms = {name: {"all": MacroAverager(), "repeat": MacroAverager(), "explore": MacroAverager()}
            for name in ("T-G text_only", "popularity", "P-TopFreq")}
    skipped = 0
    for seller in data["test"]:
        if not seller.examples or seller.seller_id not in models:
            continue
        visits, replay = context[seller.seller_id]
        scores = catalog_scores(models[seller.seller_id], seller, seller.examples).numpy()
        for row, example in enumerate(seller.examples):
            relevant = {seller.row_of[i] for i in example.target_items if i in seller.row_of}
            if len(relevant) != len(example.target_items):
                skipped += 1  # a target outside the catalog; cannot happen with the observed catalog
                continue
            bucket = progress_bucket(example.target_position, len(visits[example.customer_id_local]))
            seller_counts = replay.counts(bucket)
            candidates = {
                "T-G text_only": scores[row],
                "popularity": popularity_scores(seller.items, seller_counts),
                "P-TopFreq": p_topfreq_scores(seller.items, example.prior_counts, seller_counts),
            }
            repeat = {r for r in relevant if seller.items[r] in example.prior_counts}
            parts = {"all": relevant, "repeat": repeat, "explore": relevant - repeat}
            for name, ranking in candidates.items():
                for part, rel in parts.items():
                    if rel:
                        arms[name][part].add(seller.seller_id, example.customer_id_local,
                                             expected_metrics(ranking, rel))
    record["evaluation"] = {
        "split": "test", "targets_outside_catalog": skipped,
        "scoring": "commerce/evaluation/scoring.py (temporary copy of A's definitions)",
        "results": {name: {part: avg.result() for part, avg in parts.items()} for name, parts in arms.items()},
        "seconds": round(time.perf_counter() - started, 1),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--instacart-dir", type=Path, required=True)
    parser.add_argument("--encoder", default="minilm-l12", choices=sorted(CANDIDATES))
    parser.add_argument("--encoder-cache", type=Path, default=Path("commerce/evaluation/cache/encoders"))
    parser.add_argument("--z-cache", type=Path, default=Path("commerce/evaluation/cache/z/instacart.sqlite"))
    parser.add_argument("--sellers", type=int, default=5)
    parser.add_argument("--steps", type=int, default=2000, help="per seller")
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=Path("commerce/evaluation/runs/e_g0"))
    args = parser.parse_args(argv)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    record = {"run": "E-G0", "started_at": stamp, "seed": args.seed,
              "machine": {"os": platform.platform(), "python": platform.python_version(),
                          "torch": torch.__version__, "threads": torch.get_num_threads(),
                          "cpu": platform.processor(), "cpu_count": os.cpu_count()},
              "labels": ["pilot", "single seed", "stand-in seller sizes", "local_only, not FL",
                         "temporary scoring"]}
    data, context = build(args, record)
    models = fit(args, data, record)
    evaluate(models, data, context, record)
    record["peak_memory_mb"] = peak_memory_mb()
    out = args.out_dir / stamp
    out.mkdir(parents=True, exist_ok=True)
    (out / "record.json").write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    torch.save({s: m.state_dict() for s, m in models.items()}, out / "models.pt")  # Git-ignored; derived from raw data
    summary = {name: {part: r.get("macro") for part, r in parts.items()}
               for name, parts in record["evaluation"]["results"].items()}
    print(json.dumps({"summary_macro": summary, "data": {k: record["data"][k] for k in ("examples", "catalog_items")},
                      "seconds": record["seconds"] | {"train": record["training"]["seconds"],
                                                      "evaluate": record["evaluation"]["seconds"]},
                      "peak_memory_mb": record["peak_memory_mb"]}, indent=2))
    print("-> %s" % out)


if __name__ == "__main__":
    main()
