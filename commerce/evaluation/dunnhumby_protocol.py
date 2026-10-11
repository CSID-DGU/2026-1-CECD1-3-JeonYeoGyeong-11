"""Dunnhumby as a separate, auxiliary cohort (evaluation.md §2, §4).

    python -m commerce.evaluation.fl_lab --protocol dunnhumby --dunnhumby-dir fedcommerce/data --variant R_lm ...
    python -m commerce.evaluation.harex_compare --protocol dunnhumby --dunnhumby-dir fedcommerce/data ...

The sellers are stores of the roster the adapter computes (at least MIN_STORE_BASKETS
train-week baskets, data.md §2); --sellers takes the first N by store id, a rule fixed
before any result. Weeks follow the adapter: train 2-39, validation 40-43, test 44-52,
and a target's role is its week's (week 1 is warm-up history only). History is the
customer's earlier visits in (timestamp, basket_id) order. Relations and popularity at a
target of week w read only visits of weeks before w, the events before the week starts
(evaluation.md §2), so the week is the bucket that harex_compare.parts groups by.

Dunnhumby has no product names: about two thirds of a store's items share their text
with another item of the same store (nlp-encoder.md, OQ04), and the models have no item
ID. So the results also split the answers into items whose text is unique in the store
and items that share it (evaluation.md §4); harex_compare.evaluate reads `shared_rows`.
"""
from collections import Counter
import time

import torch

from commerce.evaluation.encoder_probe import CANDIDATES
from commerce.packages.data_adapters.baskets import basket_from_event, customer_visits
from commerce.packages.data_adapters.dunnhumby import load_dunnhumby, seller_id, split_role, week_of
from commerce.packages.data_adapters.text import catalog_item_text
from commerce.packages.recommender.examples import customer_examples
from commerce.packages.recommender.harex import HAREX_ARCHITECTURES, WordVocabulary
from commerce.packages.recommender.relations import build_relations
from commerce.packages.recommender.text_encoder import FrozenTextEncoder
from commerce.packages.recommender.z_cache import ZCache

ROLES = ("train", "validation", "test")


def visit_week(visit) -> int:
    return week_of(visit.basket.time_value)


class WeekCounts:
    """Stands in for SellerReplay in evaluate(): purchases per item over the visits of weeks before w."""

    def __init__(self, visits_by_customer):
        self._items = {}
        for visits in visits_by_customer.values():
            for v in visits:
                self._items.setdefault(visit_week(v), Counter()).update(v.basket.item_ids)
        self._cache = {}

    def counts(self, week: int) -> Counter:
        if week not in self._cache:
            total = Counter()
            for w, items in self._items.items():
                if w < week:
                    total.update(items)
            self._cache[week] = total
        return self._cache[week]


def before_week(visits_by_customer, week: int):
    """Each customer's visits of weeks before `week`: what a relation snapshot at that week may read."""
    return {c: [v for v in vs if visit_week(v) < week] for c, vs in visits_by_customer.items()}


def week_examples(visits_by_customer, counts):
    """{(role, week): examples} for every target whose week has a role; warm-up and later weeks only feed history."""
    grouped = {}
    for customer in sorted(visits_by_customer):
        vs = visits_by_customer[customer]
        for example in customer_examples(vs, range(1, len(vs) + 1)):
            week = visit_week(vs[example.target_position - 1])
            role = split_role(week)
            if role not in ROLES:
                counts["targets_outside_weeks"] += 1
                continue
            grouped.setdefault((role, week), []).append(example)
    return grouped


def shared_text_rows(texts) -> set[int]:
    """Rows whose builder text another item of the same store also has."""
    seen = Counter(texts)
    return {row for row, text in enumerate(texts) if seen[text] > 1}


def build(args, record):
    """The seller layout harex_compare.build returns, with weeks as buckets."""
    if getattr(args, "holdout_frac", 0.0):
        raise ValueError("the Dunnhumby cohort takes no C-new holdout yet")
    if not getattr(args, "dunnhumby_dir", None):
        raise SystemExit("--protocol dunnhumby needs --dunnhumby-dir (transactions.rds, products.rda)")
    started = time.perf_counter()
    full = load_dunnhumby(args.dunnhumby_dir)
    chosen = sorted(full.roster)[:args.sellers]
    keep = {seller_id(str(s)) for s in chosen}
    events = [e for e in full.events if e["seller_id"] in keep]
    catalog_items = [i for i in full.catalog_items if i["seller_id"] in keep]
    record["data"] = {"protocol": "dunnhumby", "roster_size": len(full.roster), "chosen_stores": chosen,
                      "rule": "first --sellers stores of the computed roster by store id",
                      "weeks": {"train": [2, 39], "validation": [40, 43], "test": [44, 52]},
                      "adapter_report": full.report}

    by_seller, catalogs = {}, {}
    for event in events:
        b = basket_from_event(event)
        by_seller.setdefault(b.seller_id, {}).setdefault(b.customer_id_local, []).append(b)
    for item in catalog_items:
        catalogs.setdefault(item["seller_id"], []).append(item)
    if set(by_seller) != keep:
        raise ValueError("chosen stores without events: %s" % sorted(keep - set(by_seller)))

    spec = CANDIDATES["minilm-l12"]
    model_dir = args.encoder_cache / spec.model_id.replace("/", "__") / spec.revision
    args.z_cache.parent.mkdir(parents=True, exist_ok=True)
    cache = ZCache(args.z_cache, FrozenTextEncoder(model_dir, spec), model_dir)
    need_relations = any(v.startswith("R_") for v in args.variants)

    sellers = {}
    counts = {"examples": {r: 0 for r in ROLES}, "catalog_items": 0, "targets_outside_weeks": 0,
              "items_with_shared_text": 0}
    for seller, customers in sorted(by_seller.items()):
        catalog = catalogs[seller]
        items = tuple(c["item_id_local"] for c in catalog)
        row_of = {item: row for row, item in enumerate(items)}
        texts = [catalog_item_text(c) for c in catalog]
        z = torch.from_numpy(cache.vectors(texts).copy())
        vocab = WordVocabulary(texts)
        tokens = vocab.encode(texts, HAREX_ARCHITECTURES["harex.T_hx.v1"].max_tokens)
        visits = {cust: customer_visits(bs) for cust, bs in customers.items()}
        grouped = week_examples(visits, counts)
        snapshots = {}
        if need_relations:
            snapshots = {week: build_relations(before_week(visits, week), row_of, len(items))
                         for week in sorted({w for _, w in grouped})}
        shared = shared_text_rows(texts)
        sellers[seller] = {"items": items, "train_items": items, "keep": torch.arange(len(items)), "held_rows": set(),
                           "shared_rows": shared, "z": z, "tokens": tokens, "vocab_size": len(vocab),
                           "visits": visits, "replay": WeekCounts(visits), "grouped": grouped,
                           "train_snapshots": snapshots, "test_snapshots": snapshots}
        for (role, _), examples in grouped.items():
            counts["examples"][role] += len(examples)
        counts["catalog_items"] += len(items)
        counts["items_with_shared_text"] += len(shared)
    cache.close()
    record["data"].update(counts)
    record["encoder"] = {"text_artifact_hash": cache.text_artifact_hash,
                         "preprocessing_version": cache.preprocessing_version}
    record["seconds"] = {"build": round(time.perf_counter() - started, 1)}
    return sellers, None
