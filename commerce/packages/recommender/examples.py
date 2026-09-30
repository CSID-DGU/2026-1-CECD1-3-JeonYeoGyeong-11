"""Examples: a customer's past visits -> the next visit's item set (evaluation.md §2).

One example targets one visit of one customer at one seller. The cutoff is the
target's own order key: the input holds only earlier visits, at most the last
L_VISITS of them, each cut to at most K_ITEMS items. The target's item set is
never cut and never enters the input, so no part of the basket predicts the
rest of it. Training, evaluation and the service build examples here alike.
"""
from collections import Counter
from dataclasses import dataclass
import hashlib
from typing import Iterable, Sequence

from commerce.packages.data_adapters.baskets import LocalBasket, Visit

L_VISITS = 10  # model.md §6 starting value
K_ITEMS = 32
MIN_HISTORY = 3  # at least three earlier visits


@dataclass(frozen=True)
class HistoryVisit:
    items: tuple[str, ...]  # at most K_ITEMS, in the cut order below
    items_dropped: int  # items the cut removed; the truncation rate counts these
    gap_days: float | None  # since this visit's own previous visit; None on the customer's first
    gap_censored: bool
    time_lower_bound: bool


@dataclass(frozen=True)
class Example:
    seller_id: str
    customer_id_local: str
    target_basket_id: str
    target_position: int  # 1-based position of the target among the customer's visits here
    history: tuple[HistoryVisit, ...]  # oldest first
    target_items: frozenset[str]
    # Purchases per item over every visit before the target, not only the last L_VISITS:
    # the P-TopFreq baseline and the repeat/explore split read these.
    prior_counts: dict[str, int]


def cut_items(basket: LocalBasket, k: int = K_ITEMS) -> tuple[tuple[str, ...], int]:
    """Keep k items in a fixed pseudo-random order, so the cut does not favour low IDs."""
    def rank(item: str) -> str:
        return hashlib.sha256(("%s\x00%s" % (basket.basket_id_local, item)).encode("utf-8")).hexdigest()
    ordered = sorted((i.item_id_local for i in basket.items), key=rank)
    return tuple(ordered[:k]), max(0, len(ordered) - k)


def customer_examples(visits: Sequence[Visit], positions: Iterable[int]) -> list[Example]:
    """Examples for the 1-based target positions that have MIN_HISTORY earlier visits.

    visits is one customer's ordered history (baskets.customer_visits). Positions
    without enough history are skipped; callers count them from the difference.
    """
    examples = []
    for position in sorted(set(positions)):
        if not 1 <= position <= len(visits):
            raise ValueError("target position outside the customer's visits")
        before = visits[:position - 1]
        if len(before) < MIN_HISTORY:
            continue
        target = visits[position - 1].basket
        history = []
        for visit in before[-L_VISITS:]:
            items, dropped = cut_items(visit.basket)
            history.append(HistoryVisit(items, dropped, visit.gap_days, visit.gap_censored,
                                        visit.time_lower_bound))
        counts = Counter(item for visit in before for item in visit.basket.item_ids)
        examples.append(Example(
            seller_id=target.seller_id, customer_id_local=target.customer_id_local,
            target_basket_id=target.basket_id_local, target_position=position,
            history=tuple(history), target_items=target.item_ids, prior_counts=dict(counts),
        ))
    return examples
