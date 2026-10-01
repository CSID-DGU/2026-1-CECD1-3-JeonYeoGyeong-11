"""Ranking metrics and non-model baselines. A owns this module (working-agreement.md §6:
"B의 부담이 가장 크므로 평가 지표와 모델이 아닌 기준선은 처음부터 A가 맡는다").

Scoring input is always a model's or baseline's own candidate scores; this module
never reads raw transactions or feature DBs (A card "입력은 B 실행기가 예제마다
넘기는 두 묶음이다").

Tie handling: `ranked` must already be sorted by score descending. When several
candidates share a score, which of them lands inside vs. outside a fixed cutoff k
is arbitrary. Rather than pick one tie-break, every metric here credits each item
in a tied group with the *expected* outcome over a uniformly random ordering of
that group -- so a metric value never depends on incidental sort stability.
"""
from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence

ScoredItem = tuple[str, float]
RankedRelevant = tuple[Sequence[ScoredItem], set[str]]


def _score_groups(ranked: Sequence[ScoredItem]) -> list[tuple[int, list[str]]]:
    """(1-indexed start rank, item_ids) for each run of equal-score items."""
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
            slots_in_k = k - start + 1
            hits += relevant_in_group * (slots_in_k / len(ids))
    return hits


def recall_at_k(ranked: Sequence[ScoredItem], relevant: set[str], k: int) -> float:
    """|relevant| = 0 or k <= 0 has no defined recall; returns 0.0 by convention."""
    if not relevant or k <= 0:
        return 0.0
    return _expected_hits_at_k(ranked, relevant, k) / len(relevant)


def _dcg_at_k(ranked: Sequence[ScoredItem], relevant: set[str], k: int) -> float:
    dcg = 0.0
    for start, ids in _score_groups(ranked):
        if start > k:
            break
        end = start + len(ids) - 1
        relevant_in_group = sum(1 for item in ids if item in relevant)
        if relevant_in_group == 0:
            continue
        last_rank_in_k = min(end, k)
        discount_sum = sum(1.0 / math.log2(p + 1) for p in range(start, last_rank_in_k + 1))
        dcg += relevant_in_group * (discount_sum / len(ids))
    return dcg


def _ideal_dcg_at_k(num_relevant: int, k: int) -> float:
    top = min(num_relevant, k)
    return sum(1.0 / math.log2(p + 1) for p in range(1, top + 1))


def ndcg_at_k(ranked: Sequence[ScoredItem], relevant: set[str], k: int) -> float:
    if not relevant or k <= 0:
        return 0.0
    ideal = _ideal_dcg_at_k(len(relevant), k)
    return _dcg_at_k(ranked, relevant, k) / ideal if ideal > 0 else 0.0


def macro_recall_at_k(per_customer: Iterable[RankedRelevant], k: int) -> Optional[float]:
    """Mean of each customer's own Recall@k. None when no customer has ground truth."""
    values = [recall_at_k(ranked, relevant, k) for ranked, relevant in per_customer if relevant]
    return sum(values) / len(values) if values else None


def micro_recall_at_k(per_customer: Iterable[RankedRelevant], k: int) -> Optional[float]:
    """Pooled hits / pooled relevant across all customers, not an average of ratios."""
    per_customer = list(per_customer)
    total_relevant = sum(len(relevant) for _, relevant in per_customer)
    if total_relevant == 0:
        return None
    total_hits = sum(_expected_hits_at_k(ranked, relevant, k) for ranked, relevant in per_customer)
    return total_hits / total_relevant


def macro_ndcg_at_k(per_customer: Iterable[RankedRelevant], k: int) -> Optional[float]:
    values = [ndcg_at_k(ranked, relevant, k) for ranked, relevant in per_customer if relevant]
    return sum(values) / len(values) if values else None


def micro_ndcg_at_k(per_customer: Iterable[RankedRelevant], k: int) -> Optional[float]:
    per_customer = list(per_customer)
    total_idcg = sum(_ideal_dcg_at_k(len(relevant), k) for _, relevant in per_customer)
    if total_idcg == 0:
        return None
    total_dcg = sum(_dcg_at_k(ranked, relevant, k) for ranked, relevant in per_customer)
    return total_dcg / total_idcg


def p_topfreq_ranking(seller_counts_before_cutoff: dict[str, int], candidate_item_ids: Iterable[str]) -> list[ScoredItem]:
    """Non-model baseline (A card): score = seller-wide purchase count before cutoff,
    computed straight from counts the B runner hands in -- never a model feature."""
    scored = [(item_id, float(seller_counts_before_cutoff.get(item_id, 0))) for item_id in candidate_item_ids]
    scored.sort(key=lambda pair: (-pair[1], pair[0]))
    return scored


def classify_relevant_items(
    relevant: set[str],
    customer_counts_before_cutoff: dict[str, int],
    seller_counts_before_cutoff: dict[str, int],
) -> dict[str, dict[str, bool]]:
    """Ground-truth labels only, no ranking: per relevant item, whether the
    customer already bought it before cutoff (repeat vs. explore) and whether
    the seller had zero sales of it before cutoff (new_item_cohort, A-0/C-new/C0
    candidates per evaluation.md §3)."""
    return {
        item: {
            "repeat": customer_counts_before_cutoff.get(item, 0) > 0,
            "new_item_cohort": seller_counts_before_cutoff.get(item, 0) == 0,
        }
        for item in relevant
    }


def split_recall_by_repeat_explore(
    ranked: Sequence[ScoredItem], relevant: set[str], k: int, customer_counts_before_cutoff: dict[str, int],
) -> dict[str, Optional[float]]:
    """Recall@k on the repeat subset and the explore subset of `relevant` separately."""
    repeat = {item for item in relevant if customer_counts_before_cutoff.get(item, 0) > 0}
    explore = relevant - repeat
    return {
        "repeat": recall_at_k(ranked, repeat, k) if repeat else None,
        "explore": recall_at_k(ranked, explore, k) if explore else None,
    }
