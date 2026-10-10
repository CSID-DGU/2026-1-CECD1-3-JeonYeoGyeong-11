"""Ranking metrics and non-model baselines (evaluation.md §4). A owns this module
(working-agreement.md §6: "B의 부담이 가장 크므로 평가 지표와 모델이 아닌 기준선은
처음부터 A가 맡는다"), so the side building the model does not grade itself.

Input is always a model's or baseline's own candidate scores plus the counts the
B runner hands in per example (A card: the customer's and the seller's purchase
counts before the cutoff, the relevant items). This module never reads raw
transactions or feature DBs.

Two equivalent entry points:
- Arrays, the shape B's runner uses (`expected_metrics`, `popularity_scores`,
  `p_topfreq_scores`, `MacroAverager`): scores over the candidate list, relevant
  given as candidate indices. These names match the temporary
  commerce/evaluation/scoring.py so B can switch by changing the import.
- Ranked lists of (item_id, score) (`recall_at_k`, `ndcg_at_k`, the *_ranking
  baselines), used by the merchant app and the hand-checkable examples.

Ties: a candidate's position inside a group of equal scores is never decided by
ID or sort stability. Recall@K and NDCG@K credit every item of a tied group with
the expected outcome over a uniformly random order of that group.

Averages (evaluation.md §4 "고객별, 다음 판매자별 평균 ... 판매자 macro ... 전체
예제 micro"): macro = mean per customer, then per seller, then over sellers;
micro = mean over all examples.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable, Mapping, Optional, Sequence

ScoredItem = tuple[str, float]
KS = (10, 20)


# --- core: one ranking --------------------------------------------------------

def _score_groups(ranked: Sequence[ScoredItem]) -> list[tuple[int, list[str]]]:
    """(1-indexed start rank, item_ids) for each run of equal-score items. `ranked` is sorted by score desc."""
    groups: list[tuple[int, list[str]]] = []
    rank = 1
    i, n = 0, len(ranked)
    while i < n:
        score = ranked[i][1]
        ids = []
        while i < n and ranked[i][1] == score:
            ids.append(ranked[i][0])
            i += 1
        groups.append((rank, ids))
        rank += len(ids)
    return groups


def _expected_hits_at_k(ranked: Sequence[ScoredItem], relevant: set[str], k: int) -> float:
    hits = 0.0
    for start, ids in _score_groups(ranked):
        if start > k:
            break
        end = start + len(ids) - 1
        relevant_in_group = sum(1 for item in ids if item in relevant)
        if relevant_in_group == 0:
            continue
        if end <= k:
            hits += relevant_in_group
        else:
            hits += relevant_in_group * ((k - start + 1) / len(ids))
    return hits


def _dcg_at_k(ranked: Sequence[ScoredItem], relevant: set[str], k: int) -> float:
    dcg = 0.0
    for start, ids in _score_groups(ranked):
        if start > k:
            break
        end = start + len(ids) - 1
        relevant_in_group = sum(1 for item in ids if item in relevant)
        if relevant_in_group == 0:
            continue
        discount_sum = sum(1.0 / math.log2(p + 1) for p in range(start, min(end, k) + 1))
        dcg += relevant_in_group * (discount_sum / len(ids))
    return dcg


def _ideal_dcg_at_k(num_relevant: int, k: int) -> float:
    return sum(1.0 / math.log2(p + 1) for p in range(1, min(num_relevant, k) + 1))


def recall_at_k(ranked: Sequence[ScoredItem], relevant: set[str], k: int) -> float:
    """|relevant| = 0 or k <= 0 has no defined recall; returns 0.0 by convention."""
    if not relevant or k <= 0:
        return 0.0
    return _expected_hits_at_k(ranked, relevant, k) / len(relevant)


def ndcg_at_k(ranked: Sequence[ScoredItem], relevant: set[str], k: int) -> float:
    if not relevant or k <= 0:
        return 0.0
    ideal = _ideal_dcg_at_k(len(relevant), k)
    return _dcg_at_k(ranked, relevant, k) / ideal if ideal > 0 else 0.0


def hit_rate_at_k(ranked: Sequence[ScoredItem], relevant: set[str], k: int) -> float:
    """Chance that at least one relevant item is in the top k, over every order of tied groups.

    A tied group wholly inside the top k contributes 1 if it holds a relevant
    item. If the cutoff falls inside a group of m items with r relevant and s
    slots left above the cutoff, the chance is 1 - C(m-r, s) / C(m, s). Either
    only counts when no earlier group held a relevant item (same definition as
    B's scoring._hit_chance)."""
    if not relevant or k <= 0:
        return 0.0
    for start, ids in _score_groups(ranked):
        if start > k:
            break
        r = sum(1 for item in ids if item in relevant)
        if r == 0:
            continue
        m, end = len(ids), start + len(ids) - 1
        if end <= k:
            return 1.0
        s = k - start + 1
        return 1.0 - math.comb(m - r, s) / math.comb(m, s)
    return 0.0


def _sorted_by_score(scored: Iterable[ScoredItem]) -> list[ScoredItem]:
    """Score desc; equal scores by item_id only so the list is reproducible -- metrics treat them as tied."""
    return sorted(scored, key=lambda pair: (-pair[1], pair[0]))


# --- arrays: the shape B's runner uses ----------------------------------------

def expected_metrics(scores: Sequence[float], relevant: Iterable[int], ks: Sequence[int] = KS) -> dict[str, float]:
    """Recall@K, NDCG@K and HR@K of one ranking over all candidates, ties at their expected value.

    `scores[i]` is candidate i's score; `relevant` are candidate indices. An
    example needs at least one relevant candidate (ValueError otherwise), so a
    subset with nothing in it -- e.g. no explore item -- is skipped by the caller.
    """
    relevant_ids = {str(int(i)) for i in relevant}
    if not relevant_ids:
        raise ValueError("an example needs at least one relevant candidate")
    ranked = _sorted_by_score((str(i), float(s)) for i, s in enumerate(scores))
    out: dict[str, float] = {}
    for k in ks:
        out["recall@%d" % k] = recall_at_k(ranked, relevant_ids, k)
        out["ndcg@%d" % k] = ndcg_at_k(ranked, relevant_ids, k)
        out["hr@%d" % k] = hit_rate_at_k(ranked, relevant_ids, k)
    return out


def popularity_scores(items: Sequence[str], seller_counts: Mapping[str, int]):
    """Local popularity: the seller's purchases per item before the cutoff."""
    import numpy as np  # only the array entry points need numpy
    return np.array([seller_counts.get(i, 0) for i in items], dtype=np.float64)


def p_topfreq_scores(items: Sequence[str], prior_counts: Mapping[str, int], seller_counts: Mapping[str, int]):
    """P-TopFreq: the customer's own purchases per item before the cutoff, ties
    broken by local popularity (seller counts scaled below 1, so they never
    outweigh a single own purchase)."""
    import numpy as np
    top = max(seller_counts.values(), default=0) + 1
    return np.array([prior_counts.get(i, 0) + seller_counts.get(i, 0) / top for i in items], dtype=np.float64)


def repeat_explore_indices(items: Sequence[str], relevant: Iterable[int],
                           prior_counts: Mapping[str, int]) -> tuple[list[int], list[int]]:
    """Split relevant candidate indices into repeat (the customer bought it before the cutoff) and explore."""
    repeat, explore = [], []
    for i in relevant:
        (repeat if prior_counts.get(items[i], 0) > 0 else explore).append(i)
    return repeat, explore


def new_item_indices(items: Sequence[str], relevant: Iterable[int], seller_counts: Mapping[str, int]) -> list[int]:
    """Relevant candidates the seller had never sold before the cutoff (the new-item cohort)."""
    return [i for i in relevant if seller_counts.get(items[i], 0) == 0]


class MacroAverager:
    """Mean per customer, then per seller, then over sellers; micro over examples too.

    One averager per reported slice: overall, repeat, explore, new-item cohort.
    `examples` in the result is the number of valid queries for that slice
    (evaluation.md §4 asks for it next to the cohort metrics).
    """

    def __init__(self):
        self._rows: dict[str, dict[str, list[dict[str, float]]]] = defaultdict(lambda: defaultdict(list))

    def add(self, seller: str, customer: str, metrics: Mapping[str, float]) -> None:
        self._rows[seller][customer].append(dict(metrics))

    def result(self) -> dict:
        if not self._rows:
            return {"sellers": 0, "examples": 0}
        names = sorted(next(iter(next(iter(self._rows.values())).values()))[0])
        per_seller: dict[str, dict[str, float]] = {}
        flat: list[dict[str, float]] = []
        for seller, customers in self._rows.items():
            per_customer = []
            for rows in customers.values():
                flat += rows
                per_customer.append({n: sum(r[n] for r in rows) / len(rows) for n in names})
            per_seller[seller] = {n: sum(c[n] for c in per_customer) / len(per_customer) for n in names}
        return {
            "sellers": len(per_seller), "examples": len(flat),
            "macro": {n: sum(s[n] for s in per_seller.values()) / len(per_seller) for n in names},
            "micro": {n: sum(r[n] for r in flat) / len(flat) for n in names},
            "per_seller": per_seller,
        }


# --- ranked lists: baselines for the merchant app -----------------------------

def local_popularity_ranking(seller_counts_before_cutoff: Mapping[str, int],
                             candidate_item_ids: Iterable[str]) -> list[ScoredItem]:
    """Non-model baseline: score = the seller's purchase count of the item before the cutoff."""
    return _sorted_by_score((i, float(seller_counts_before_cutoff.get(i, 0))) for i in candidate_item_ids)


def p_topfreq_ranking(customer_counts_before_cutoff: Mapping[str, int], seller_counts_before_cutoff: Mapping[str, int],
                      candidate_item_ids: Iterable[str]) -> list[ScoredItem]:
    """Non-model baseline P-TopFreq as a ranked list (same scores as p_topfreq_scores).
    A customer with no history gets the local popularity order."""
    items = list(candidate_item_ids)
    top = max(seller_counts_before_cutoff.values(), default=0) + 1
    return _sorted_by_score(
        (i, customer_counts_before_cutoff.get(i, 0) + seller_counts_before_cutoff.get(i, 0) / top) for i in items)


def classify_relevant_items(
    relevant: set[str],
    customer_counts_before_cutoff: Mapping[str, int],
    seller_counts_before_cutoff: Mapping[str, int],
) -> dict[str, dict[str, bool]]:
    """Ground-truth labels only, no ranking: per relevant item, whether the
    customer already bought it before cutoff (repeat vs. explore) and whether
    the seller had zero sales of it before cutoff (new_item_cohort)."""
    return {
        item: {
            "repeat": customer_counts_before_cutoff.get(item, 0) > 0,
            "new_item_cohort": seller_counts_before_cutoff.get(item, 0) == 0,
        }
        for item in relevant
    }


def split_recall_by_repeat_explore(
    ranked: Sequence[ScoredItem], relevant: set[str], k: int, customer_counts_before_cutoff: Mapping[str, int],
) -> dict[str, Optional[float]]:
    """Recall@k on the repeat subset and the explore subset of `relevant` separately (None when a subset is empty)."""
    repeat = {item for item in relevant if customer_counts_before_cutoff.get(item, 0) > 0}
    explore = relevant - repeat
    return {
        "repeat": recall_at_k(ranked, repeat, k) if repeat else None,
        "explore": recall_at_k(ranked, explore, k) if explore else None,
    }
