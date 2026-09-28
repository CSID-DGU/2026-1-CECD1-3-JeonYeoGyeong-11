"""TEMPORARY stand-in for A's commerce/evaluation/metrics (evaluation.md §4).

A owns the metrics and the non-model baselines so that the side building the
model does not grade itself. Until A's module lands, E-G0 uses this copy of the
same definitions; every number from it is re-scored with A's module before it
is reported. Delete this file then.

Ties are not broken by ID: Recall@K and NDCG@K take the expected value over
every order of a tied group.
"""
from collections import defaultdict
import math
from typing import Iterable, Mapping, Sequence

import numpy as np

KS = (10, 20)


def expected_metrics(scores: np.ndarray, relevant: Iterable[int], ks: Sequence[int] = KS) -> dict[str, float]:
    """Recall@K and NDCG@K of one ranking over all candidates, ties at their expected value."""
    relevant = set(relevant)
    if not relevant:
        raise ValueError("an example needs at least one relevant candidate")
    order = np.argsort(-scores, kind="stable")
    ranked = scores[order]
    out = {}
    for k in ks:
        hits = dcg = 0.0
        start = 0
        while start < len(order) and start < k:
            end = start
            while end < len(order) and ranked[end] == ranked[start]:
                end += 1
            group_relevant = sum(1 for i in order[start:end] if int(i) in relevant)
            share = group_relevant / (end - start)  # chance that any one slot of the group is relevant
            visible = range(start, min(end, k))
            hits += share * len(visible)
            dcg += share * sum(1.0 / math.log2(p + 2) for p in visible)
            start = end
        ideal = sum(1.0 / math.log2(p + 2) for p in range(min(len(relevant), k)))
        out["recall@%d" % k] = hits / len(relevant)
        out["ndcg@%d" % k] = dcg / ideal
    return out


def popularity_scores(items: Sequence[str], seller_counts: Mapping[str, int]) -> np.ndarray:
    """Local popularity: the seller's purchases per item visible at the cutoff."""
    return np.array([seller_counts.get(i, 0) for i in items], dtype=np.float64)


def p_topfreq_scores(items: Sequence[str], prior_counts: Mapping[str, int],
                     seller_counts: Mapping[str, int]) -> np.ndarray:
    """P-TopFreq: the customer's own purchases per item, ties broken by local popularity."""
    top = max(seller_counts.values(), default=0) + 1
    return np.array([prior_counts.get(i, 0) + seller_counts.get(i, 0) / top for i in items], dtype=np.float64)


class MacroAverager:
    """Mean per customer, then per seller, then over sellers; micro over examples too."""

    def __init__(self):
        self._rows = defaultdict(lambda: defaultdict(list))  # seller -> customer -> [metrics]

    def add(self, seller: str, customer: str, metrics: dict[str, float]) -> None:
        self._rows[seller][customer].append(metrics)

    def result(self) -> dict:
        if not self._rows:
            return {"examples": 0}
        names = sorted(next(iter(next(iter(self._rows.values())).values()))[0])
        per_seller, flat = [], []
        for customers in self._rows.values():
            per_customer = []
            for rows in customers.values():
                flat += rows
                per_customer.append({n: sum(r[n] for r in rows) / len(rows) for n in names})
            per_seller.append({n: sum(c[n] for c in per_customer) / len(per_customer) for n in names})
        return {
            "sellers": len(per_seller), "examples": len(flat),
            "macro": {n: sum(s[n] for s in per_seller) / len(per_seller) for n in names},
            "micro": {n: sum(r[n] for r in flat) / len(flat) for n in names},
            "per_seller": per_seller,
        }
