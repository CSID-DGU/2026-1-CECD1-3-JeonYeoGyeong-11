"""Instacart virtual sellers from each customer's train orders only (data.md §2).

Instacart has no retailer, so customers are split into virtual clients. The
previous split (fedcommerce/src/instacart_match.py) labelled each customer by
the aisle bought most over all prior orders and sized clients by all prior
orders, so validation/test visits shaped the sellers. Here the pick, the label
and the load use only train orders, order_number <= floor(0.7 n).

Apart from that cut the procedure is the previous one:
1. Shuffle eligible customers and take them until their train orders reach
   pick_factor x the total target size.
2. Label = the aisle with the most train order lines (ties: smallest aisle_id).
3. For each label, draw p ~ Dir(alpha K w) over the clients, w = target share,
   and give each customer (shuffled) a client drawn from p among clients whose
   load is below cap_factor x target. alpha = inf uses p = w (a random split).

Targets are client_id -> train-period size. They come from the Dunnhumby
store roster (dunnhumby.load_dunnhumby(...).roster: store -> baskets in weeks
2-39); the runs so far used a stand-in (e_g0.STAND_IN_TARGETS).
"""
from collections import Counter
import csv
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Mapping

import numpy as np

from commerce.packages.contracts.ids import canonical_json
from commerce.packages.data_adapters.instacart import read_table, train_cutoff

ALGORITHM = "instacart_dirichlet_label_split.train_only.v1"


@dataclass(frozen=True)
class Assignment:
    clients: dict[int, int]  # user_id -> client_id; customer-derived, never commit it
    labels: dict[int, int]  # user_id -> label aisle_id, same handling
    loads: dict[int, int]  # client_id -> assigned train orders
    record: dict  # parameters and counts for the Git-excluded run record


def assign_clients(data_dir: str | Path, targets: Mapping[int, int], *, alpha: float, seed: int,
                   pick_factor: float = 1.25, cap_factor: float = 1.15) -> Assignment:
    if not targets or min(targets.values()) <= 0:
        raise ValueError("targets need at least one client with a positive size")
    data_dir = Path(data_dir)
    rng = np.random.default_rng(seed)

    prior: dict[int, list[tuple[int, int]]] = {}  # user -> [(order_number, order_id)]
    for r in read_table(data_dir, "orders.csv"):
        if r["eval_set"] == "prior":
            prior.setdefault(int(r["user_id"]), []).append((int(r["order_number"]), int(r["order_id"])))
    train_orders = {user: [o for n, o in history if n <= train_cutoff(len(history))]
                    for user, history in prior.items()}
    eligible = sorted(u for u, orders in train_orders.items() if orders)

    need = sum(targets.values())
    picked, picked_orders = [], 0
    for i in rng.permutation(len(eligible)):
        if picked_orders >= pick_factor * need:
            break
        picked.append(eligible[i])
        picked_orders += len(train_orders[eligible[i]])

    owner = {order: user for user in picked for order in train_orders[user]}
    aisle_of = {int(r["product_id"]): int(r["aisle_id"]) for r in read_table(data_dir, "products.csv")}
    lines: dict[int, Counter[int]] = {user: Counter() for user in picked}
    seen: set[tuple[int, int]] = set()
    for r in read_table(data_dir, "order_products__prior.csv"):
        order, product = int(r["order_id"]), int(r["product_id"])
        if order in owner and (order, product) not in seen:  # a duplicate line counts once
            seen.add((order, product))
            lines[owner[order]][aisle_of[product]] += 1
    labels = {}
    for user, counts in lines.items():
        if not counts:
            raise ValueError("a picked customer has no product lines in the train orders")
        top = max(counts.values())
        labels[user] = min(aisle for aisle, n in counts.items() if n == top)

    clients = sorted(targets)
    size = np.array([targets[c] for c in clients], dtype=float)
    weights = size / size.sum()
    cap = cap_factor * size
    load = np.zeros(len(clients))
    by_label: dict[int, list[int]] = {}
    for user in sorted(labels):
        by_label.setdefault(labels[user], []).append(user)
    assigned = {}
    for label in sorted(by_label):
        members = by_label[label]
        p = weights if math.isinf(alpha) else rng.dirichlet(np.clip(alpha * len(clients) * weights, 1e-6, None))
        for i in rng.permutation(len(members)):
            user = members[i]
            open_clients = load < cap
            pp = np.where(open_clients, p, 0.0)
            if not np.isfinite(pp).all() or pp.sum() <= 0:
                pp = open_clients.astype(float)
            if pp.sum() <= 0:
                pp = np.ones(len(clients))  # every client is full: overflow evenly
            k = int(rng.choice(len(clients), p=pp / pp.sum()))
            assigned[user] = clients[k]
            load[k] += len(train_orders[user])

    record = {
        "algorithm": ALGORITHM, "alpha": "inf" if math.isinf(alpha) else alpha, "seed": seed,
        "pick_factor": pick_factor, "cap_factor": cap_factor, "train_cutoff": "floor(0.7 n)",
        "targets_sha256": hashlib.sha256(canonical_json(sorted(targets.items()))).hexdigest(),
        "users_with_prior": len(prior), "users_without_train_orders": len(prior) - len(eligible),
        "users_picked": len(picked), "train_orders_picked": picked_orders, "target_total": need,
        "labels": len(by_label),
    }
    loads = {client: int(value) for client, value in zip(clients, load)}
    return Assignment(assigned, labels, loads, record)


def write_assignment(path: str | Path, clients: Mapping[int, int]) -> None:
    """user_id,client_id CSV that instacart.load_assignment reads back."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(("user_id", "client_id"))
        writer.writerows(sorted(clients.items()))
