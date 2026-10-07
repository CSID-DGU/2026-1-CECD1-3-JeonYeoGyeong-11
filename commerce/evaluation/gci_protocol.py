"""HAREX (GCI) evaluation conditions on Instacart: an added-scope reproduction (evaluation.md §4).

    python -m commerce.evaluation.fl_lab --protocol gci --variant T_hx --sellers 20 ...
    python -m commerce.evaluation.harex_compare --protocol gci --sellers 20 ...

GCI (Lee et al., AAAI-24) joins each customer's receipt lines into one item
sequence, cuts it into units of five with the fifth item as the label, adds a
second set of units shifted by two, and reports HR@10 of the label. The paper
names a 9:1 train/validation split but neither a test share nor a time order.

Here, per customer, items follow order_number and then add_to_cart_order. Units
start at every position p with p % 5 in {0, 2}: four items in, the fifth is the
label. Units are split at random per seller (--split-seed): 10% test, which is
our assumption, then the rest 9:1 into training and validation as in GCI. A
label may sit in the same order as its inputs, which the main evaluation rules
out (evaluation.md §2), so these numbers only answer "what would the paper's
conditions give" and never replace the main results.

--menu-size N narrows everything to a BBQ-like menu: the N products bought most
often in the train period (each customer's first floor(0.7 n) orders) over all
chosen sellers, shared by every seller as BBQ's clients share one menu. Each
customer's sequence keeps only menu items before units are cut.

R variants read one relation snapshot built from the seller's orders that hold
no validation or test label, so no label reaches a model through a relation;
time relations are then counted between the orders that remain. Popularity
counts the training labels; P-TopFreq counts the customer's items before the
label in the sequence.
"""
from collections import Counter
import dataclasses
import random
import time

import torch

from commerce.evaluation.encoder_probe import CANDIDATES
from commerce.packages.data_adapters.assignment import assign_clients
from commerce.packages.data_adapters.baskets import basket_from_event, customer_visits
from commerce.packages.data_adapters.instacart import cart_orders, item_id, load_instacart, train_cutoff
from commerce.packages.data_adapters.text import catalog_item_text
from commerce.packages.recommender.examples import Example, HistoryVisit
from commerce.packages.recommender.harex import HAREX_ARCHITECTURES, WordVocabulary
from commerce.packages.recommender.relations import build_relations
from commerce.packages.recommender.text_encoder import FrozenTextEncoder
from commerce.packages.recommender.z_cache import ZCache

UNIT, SHIFT = 5, 2  # GCI: units of five, a second set shifted by two
TEST_SHARE = 0.1  # not in the paper: our assumption
VALIDATION_SHARE = 0.1  # of the rest, GCI's 9:1


def unit_starts(length: int) -> list[int]:
    return [p for p in range(0, length - UNIT + 1) if p % UNIT in (0, SHIFT)]


class LabelCounts:
    """Stands in for SellerReplay in evaluate(): one popularity count for every bucket."""

    def __init__(self, counts: Counter):
        self._counts = counts

    def counts(self, bucket: int) -> Counter:
        return self._counts


def menu_items(visits_by_seller, carts, size):
    """The `size` products bought most in the train period over all sellers (ties: item id)."""
    counts = Counter()
    for visits in visits_by_seller.values():
        for vs in visits.values():
            for v in vs[:train_cutoff(len(vs))]:
                counts.update(item_id(p) for p in carts[int(v.basket.basket_id_local.rsplit("-", 1)[1])])
    return {item for item, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:size]}


def keep_items(visits, menu):
    """The visits with only menu items; a visit left empty keeps its place and time."""
    return [dataclasses.replace(v, basket=dataclasses.replace(
        v.basket, items=tuple(i for i in v.basket.items if i.item_id_local in menu))) for v in visits]


def seller_units(seller, visits, carts, seed, menu=None):
    """Examples per role from one seller's customers, and the orders that hold a held-back label."""
    rng = random.Random("%d:%s" % (seed, seller))
    grouped = {("train", 0): [], ("validation", 0): [], ("test", 0): []}
    held_orders, label_counts = set(), Counter()
    for customer in sorted(visits):
        sequence = [(item_id(product), v.basket.basket_id_local) for v in visits[customer]
                    for product in carts[int(v.basket.basket_id_local.rsplit("-", 1)[1])]
                    if menu is None or item_id(product) in menu]
        for p in unit_starts(len(sequence)):
            inputs, (label, order) = sequence[p:p + UNIT - 1], sequence[p + UNIT - 1]
            draw = rng.random()
            role = ("test" if draw < TEST_SHARE else
                    "validation" if draw < TEST_SHARE + (1 - TEST_SHARE) * VALIDATION_SHARE else "train")
            grouped[(role, 0)].append(Example(
                seller_id=seller, customer_id_local=customer, target_basket_id=order, target_position=p + UNIT,
                history=tuple(HistoryVisit((item,), 0, None, False, False) for item, _ in inputs),
                target_items=frozenset({label}),
                prior_counts=dict(Counter(item for item, _ in sequence[:p + UNIT - 1]))))
            if role == "train":
                label_counts[label] += 1
            else:
                held_orders.add(order)
    return grouped, held_orders, label_counts


def relation_visits(visits, held_orders):
    """The visits relations may read: none that holds a validation or test label."""
    return {c: [v for v in vs if v.basket.basket_id_local not in held_orders] for c, vs in visits.items()}


def build(args, record):
    """The same seller data layout harex_compare.build returns, one bucket 0, under GCI's units."""
    if getattr(args, "holdout_frac", 0.0):
        raise ValueError("the GCI conditions take no C-new holdout")
    started = time.perf_counter()
    from commerce.evaluation.harex_compare import cohort_targets
    targets = cohort_targets(args)
    split = assign_clients(args.instacart_dir, targets, alpha=0.25, seed=args.split_seed)
    chosen = sorted(targets)[:args.sellers]
    users = {u: c for u, c in split.clients.items() if c in chosen}
    sample = load_instacart(args.instacart_dir, users)
    carts = cart_orders(args.instacart_dir, [int(e["basket_id_local"].rsplit("-", 1)[1]) for e in sample.events])
    record["data"] = {"protocol": "gci", "assignment": split.record, "stand_in_targets": "100 x 1,040 train orders",
                      "chosen_clients": chosen, "adapter_report": sample.report,
                      "units": {"length": UNIT, "shift": SHIFT, "test_share": TEST_SHARE,
                                "validation_share_of_rest": VALIDATION_SHARE, "split": "random per seller"}}

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
    all_visits = {seller: {cust: customer_visits(bs) for cust, bs in customers.items()}
                  for seller, customers in by_seller.items()}
    menu = menu_items(all_visits, carts, args.menu_size) if getattr(args, "menu_size", 0) else None
    record["data"]["menu_size"] = len(menu) if menu else None

    sellers = {}
    counts = {"examples": {r: 0 for r in ("train", "validation", "test")}, "catalog_items": 0,
              "orders_left_out_of_relations": 0}
    for seller, customers in sorted(by_seller.items()):
        catalog = [c for c in catalogs[seller] if menu is None or c["item_id_local"] in menu]
        items = tuple(c["item_id_local"] for c in catalog)
        row_of = {item: row for row, item in enumerate(items)}
        texts = [catalog_item_text(c) for c in catalog]
        z = torch.from_numpy(cache.vectors(texts).copy())
        vocab = WordVocabulary(texts)
        tokens = vocab.encode(texts, HAREX_ARCHITECTURES["harex.T_hx.v1"].max_tokens)
        visits = all_visits[seller]
        if menu is not None:
            visits = {cust: keep_items(vs, menu) for cust, vs in visits.items()}
        grouped, held_orders, label_counts = seller_units(seller, visits, carts, args.split_seed, menu)
        snapshots = {}
        if need_relations:
            snapshots = {0: build_relations(relation_visits(visits, held_orders), row_of, len(items))}
        sellers[seller] = {"items": items, "train_items": items, "keep": torch.arange(len(items)), "held_rows": set(),
                           "z": z, "tokens": tokens, "vocab_size": len(vocab), "visits": visits,
                           "replay": LabelCounts(label_counts), "grouped": grouped,
                           "train_snapshots": snapshots, "test_snapshots": snapshots}
        for (role, _), examples in grouped.items():
            counts["examples"][role] += len(examples)
        counts["catalog_items"] += len(items)
        counts["orders_left_out_of_relations"] += len(held_orders)
    cache.close()
    record["data"].update(counts)
    record["encoder"] = {"text_artifact_hash": cache.text_artifact_hash,
                         "preprocessing_version": cache.preprocessing_version}
    record["seconds"] = {"build": round(time.perf_counter() - started, 1)}
    return sellers, {}
