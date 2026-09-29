"""D0022: four item representations on one HAREX (GCI) style backbone, trained to convergence.

    python -m commerce.evaluation.harex_compare --instacart-dir fedcommerce/data/instacart

Variants: T_hx (GCI word tokens, the HAREX baseline), R_hx (+ local purchase relations),
T_lm (pretrained encoder), R_lm, and R_hx_shuffled, the relation shuffle control
(evaluation.md §4): R_hx with the item-relation correspondence permuted inside
each seller. Targets: "basket" is the next visit's item set
(evaluation.md §1); "next_item" is the first item added to the next visit's cart,
predicted from earlier visits only (D0022, HAREX style).

--holdout-frac > 0 runs C-new (evaluation.md §3): items chosen in advance from
the seed and the item ID hash, the same set at every seller, leave every
training and validation example (answers, negatives, input history, relations)
and the hx vocabulary. At test they come back as candidates with no relation,
their earlier purchases stay hidden, and "cnew" reports the answers among them.

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
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import zlib

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
from commerce.packages.recommender.relations import RelationTensors, replay_relations
from commerce.packages.recommender.replay import SellerReplay, progress_bucket
from commerce.packages.recommender.text_encoder import FrozenTextEncoder
from commerce.packages.recommender.training import (
    SellerData, TrainConfig, catalog_scores, seller_on, train, validation_loss,
)
from commerce.packages.recommender.z_cache import ZCache

VARIANTS = ("T_hx", "R_hx", "T_lm", "R_lm", "R_hx_shuffled",
            "T_hx_rep", "R_hx_rep", "T_lm_rep", "R_lm_rep",  # _rep: D0023 service configuration
            "T_hx_rep2", "R_hx_rep2", "T_lm_rep2", "R_lm_rep2")  # _rep2: it also reads the item
TARGETS = ("basket", "next_item")
SHUFFLED = "_shuffled"
PARTS = ("all", "repeat", "explore", "cnew")
# A-0 sellers (evaluation.md §3), fixed in advance: a second split over the customers
# the cohort did not take, with the cohort's stand-in size.
HELD_OUT_TARGETS = {990201 + k: 1040 for k in range(20)}


def held_out_item(item: str, frac: float, seed: int) -> bool:
    """C-new membership from the seed and the item ID alone, never from purchase counts."""
    digest = hashlib.sha256(("%d\x00%s" % (seed, item)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") < frac * 2 ** 64


def without_items(visits, held):
    """The visits with the held-out items removed; a visit left empty keeps its place and time."""
    return [dataclasses.replace(v, basket=dataclasses.replace(
        v.basket, items=tuple(i for i in v.basket.items if i.item_id_local not in held))) for v in visits]


def new_arms(names):
    return {name: {p: MacroAverager() for p in PARTS} for name in names}


def architecture(variant: str):
    """R_hx_shuffled is R_hx's architecture; only its relation rows are permuted."""
    return HAREX_ARCHITECTURES["harex.%s.v1" % variant.removesuffix(SHUFFLED)]


def shuffled_relations(relations: RelationTensors, seller: str, seed: int) -> RelationTensors:
    """Each item takes another item's relation row from the same seller: input form and capacity
    stay, the item-relation correspondence goes. One permutation per seller and seed, the same in
    every snapshot."""
    generator = torch.Generator().manual_seed(zlib.crc32(("%s/%d" % (seller, seed)).encode()))
    perm = torch.randperm(len(relations.has_neighbor), generator=generator)
    return RelationTensors(**{f.name: getattr(relations, f.name)[perm] for f in dataclasses.fields(relations)})


def code_version() -> dict:
    """The commit a run used and whether tracked files differed from it."""
    def git(*args):
        return subprocess.run(["git", *args], capture_output=True, text=True).stdout.strip()
    return {"commit": git("rev-parse", "HEAD") or None,
            "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}


def build(args, record, held_out=False):
    """The cohort's sellers, or with held_out the A-0 sellers none of its customers belong to."""
    started = time.perf_counter()
    # The cohort is fixed in advance (evaluation.md §5): --seed varies the model, not the sellers.
    split = assign_clients(args.instacart_dir, STAND_IN_TARGETS, alpha=0.25, seed=args.split_seed)
    chosen = sorted(STAND_IN_TARGETS)[:args.sellers]
    record["data"] = {"assignment": split.record, "stand_in_targets": "100 x 1,040 train orders"}
    if held_out:
        split = assign_clients(args.instacart_dir, HELD_OUT_TARGETS, alpha=0.25, seed=args.split_seed,
                               exclude=set(split.clients))
        chosen = sorted(HELD_OUT_TARGETS)[:args.held_out_sellers]
        record["data"]["held_out_assignment"] = split.record
    users = {u: c for u, c in split.clients.items() if c in chosen}
    sample = load_instacart(args.instacart_dir, users)
    orders = [int(e["basket_id_local"].rsplit("-", 1)[1]) for e in sample.events]
    first = first_in_cart(args.instacart_dir, orders)
    record["data"].update(chosen_clients=chosen, adapter_report=sample.report)

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
    counts = {"examples": {r: 0 for r in ("train", "validation", "test")}, "catalog_items": 0, "vocab": {},
              "held_out_items": 0, "test_examples_with_held_out_answers": 0, "training_examples_only_held_out": 0,
              "examples_history_all_held_out": 0}
    for seller, customers in sorted(by_seller.items()):
        items = tuple(c["item_id_local"] for c in catalogs[seller])
        texts = [catalog_item_text(c) for c in catalogs[seller]]
        z = torch.from_numpy(cache.vectors(texts).copy())
        visits = {cust: customer_visits(bs) for cust, bs in customers.items()}
        sellers[seller] = seller_info(items, texts, z, visits, args.holdout_frac, args.holdout_seed,
                                      need_relations, counts)
        counts["vocab"][seller] = sellers[seller]["vocab_size"]
    cache.close()
    record["data"].update(counts)
    record["data"]["holdout"] = {"frac": args.holdout_frac, "seed": args.holdout_seed,
                                 "rule": "sha256(seed, item_id) < frac"}
    record["encoder"] = {"text_artifact_hash": cache.text_artifact_hash,
                         "preprocessing_version": cache.preprocessing_version}
    record["seconds"] = {"build": round(time.perf_counter() - started, 1)}
    return sellers, first


def seller_info(items, texts, z, full, frac, holdout_seed, need_relations, counts):
    """One seller's examples, vocabulary and relation snapshots; C-new items leave training here."""
    row_of = {item: row for row, item in enumerate(items)}
    held = {i for i in items if held_out_item(i, frac, holdout_seed)} if frac > 0 else set()
    keep = [row for row, item in enumerate(items) if item not in held]
    train_items = tuple(items[row] for row in keep)
    # The vocabulary comes from the items training sees; a held-out name's new words are UNKNOWN.
    vocab = WordVocabulary([texts[row] for row in keep])
    tokens = vocab.encode(texts, HAREX_ARCHITECTURES["harex.T_hx.v1"].max_tokens)
    visits = {cust: without_items(vs, held) for cust, vs in full.items()} if held else full
    grouped = {}
    for cust, vs in visits.items():
        answers = {v.basket.basket_id_local: v.basket.item_ids for v in full[cust]}
        for example in customer_examples(vs, range(1, len(vs) + 1)):
            role = split_role(example.target_position, len(vs))
            if held and not any(v.items for v in example.history):
                counts["examples_history_all_held_out"] += 1  # nothing left to read
                continue
            if held and role == "test":
                # Held-out items come back only as answers; their earlier purchases stay hidden.
                example = dataclasses.replace(example, target_items=answers[example.target_basket_id])
                counts["test_examples_with_held_out_answers"] += bool(example.target_items & held)
            elif not example.target_items:
                counts["training_examples_only_held_out"] += 1  # left out of training (evaluation.md §3)
                continue
            grouped.setdefault((role, progress_bucket(example.target_position, len(vs))), []).append(example)

    def snapshots(roles, catalog):
        if not need_relations:
            return {}
        rows = {item: row for row, item in enumerate(catalog)}
        return {b: replay_relations(visits, b, rows, len(catalog))
                for b in sorted({b for r, b in grouped if r in roles})}
    if held:
        train_snapshots = snapshots(("train", "validation"), train_items)
        test_snapshots = snapshots(("test",), items)
    else:
        train_snapshots = test_snapshots = snapshots(("train", "validation", "test"), items)
    for (role, _), examples in grouped.items():
        counts["examples"][role] += len(examples)
    counts["catalog_items"] += len(items)
    counts["held_out_items"] += len(held)
    return {"items": items, "train_items": train_items, "keep": torch.tensor(keep, dtype=torch.long),
            "held_rows": {row_of[i] for i in held}, "z": z, "tokens": tokens, "vocab_size": len(vocab),
            "visits": visits, "replay": SellerReplay(visits), "grouped": grouped,
            "train_snapshots": train_snapshots, "test_snapshots": test_snapshots}


def with_target(examples, target, first):
    """next_item keeps only the first item added to the target visit's cart."""
    if target == "basket":
        return examples
    out = []
    for example in examples:
        order = int(example.target_basket_id.rsplit("-", 1)[1])
        out.append(dataclasses.replace(example, target_items=frozenset({item_id(first[order])})))
    return out


def parts(info, seller, role, variant, target, first, device, seed=0):
    """Training and validation see the catalog without held-out items; test ranks the whole catalog.

    device None leaves every tensor where info keeps it."""
    config = architecture(variant)
    test = role == "test"
    whole = test or len(info["train_items"]) == len(info["items"])
    items = info["items"] if test else info["train_items"]
    if not whole and "train_z" not in info:
        # One reduced copy for the training and validation parts alike.
        info["train_z"], info["train_tokens"] = info["z"][info["keep"]], info["tokens"][info["keep"]]
    z = info["z"] if whole else info["train_z"]
    tokens = None if config.text != "hx" else info["tokens"] if whole else info["train_tokens"]
    snapshots = info["test_snapshots" if test else "train_snapshots"]
    out = []
    for (r, bucket), examples in sorted(info["grouped"].items()):
        if r != role:
            continue
        relations = snapshots[bucket] if config.relation else None
        if variant.endswith(SHUFFLED):
            relations = shuffled_relations(relations, seller, seed)
        data = SellerData(seller, items, z, with_target(examples, target, first), relations, tokens)
        out.append((bucket, data if device is None else seller_on(data, device)))
    return out


def fit(args, info, train_parts, val_parts, variant, device):
    config = architecture(variant)
    torch.manual_seed(args.seed)
    model = HarexRecommender(config, vocab_size=info["vocab_size"] if config.text == "hx" else None).to(device)
    base = TrainConfig(steps=args.eval_every, batch_size=args.batch_size, lr=args.lr, n_negatives=args.negatives)
    optimizer = torch.optim.AdamW(model.parameters(), lr=base.lr, weight_decay=base.weight_decay)
    best, best_loss, best_step, since, history, step = None, None, 0, 0, [], 0
    started = time.perf_counter()
    while step < args.max_steps and since < args.patience:
        log = train(model, [p for _, p in train_parts], dataclasses.replace(base, seed=args.seed * 100000 + step),
                    optimizer)
        step += args.eval_every
        val = validation_loss(model, [p for _, p in val_parts], n_negatives=args.negatives, seed=args.seed)
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
            split = {"all": relevant, "repeat": repeat, "explore": relevant - repeat,
                     "cnew": relevant & info["held_rows"]}
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
    parser.add_argument("--negatives", type=int, default=200)  # 0: the whole catalog
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--holdout-frac", type=float, default=0.0)  # C-new: 0.1 (evaluation.md §3)
    parser.add_argument("--holdout-seed", type=int, default=0)
    # gci: GCI's item-level units and random split (gci_protocol.py), an added-scope reproduction.
    parser.add_argument("--protocol", default="next_visit", choices=("next_visit", "gci"))
    parser.add_argument("--menu-size", type=int, default=0)  # with --protocol gci: a BBQ-like menu
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=Path, default=Path("commerce/evaluation/runs/harex_compare"))
    args = parser.parse_args(argv)
    args.variants = [v for v in args.variants.split(",") if v]
    args.targets = [t for t in args.targets.split(",") if t]
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    record = {"run": "D0022 harex_compare", "started_at": stamp, "seed": args.seed, "code": code_version(),
              "settings": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
              "machine": {"os": platform.platform(), "python": platform.python_version(), "torch": torch.__version__,
                          "device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                          "threads": torch.get_num_threads(), "cpu_count": os.cpu_count()},
              "labels": ["pilot", "single seed", "stand-in seller sizes", "local_only, not FL", "temporary scoring"]
              + (["C-new: items held out of training"] if args.holdout_frac > 0 else []),
              "results": {}}
    if args.protocol == "gci":
        if args.targets != ["basket"]:
            raise SystemExit("--protocol gci has one label per unit: use --targets basket")
        from commerce.evaluation.gci_protocol import build as gci_build
        record["labels"].append("HAREX conditions (added scope, evaluation.md §4): item-level units, random split")
        sellers, first = gci_build(args, record)
    else:
        sellers, first = build(args, record)
    for target in args.targets:
        for variant in args.variants:
            arm = "%s %s" % (variant, "(HAREX baseline)" if variant == "T_hx" else "")
            arm = arm.strip()
            arms = new_arms((arm, "popularity", "P-TopFreq"))
            per_seller = {}
            for seller, info in sellers.items():
                train_parts = parts(info, seller, "train", variant, target, first, device, args.seed)
                val_parts = parts(info, seller, "validation", variant, target, first, device, args.seed)
                test_parts = parts(info, seller, "test", variant, target, first, device, args.seed)
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
