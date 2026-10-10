"""What a seller may know at an Instacart target: the progress replay (evaluation.md §2).

Instacart customers share no calendar. For customer u's target visit k out of
n_u prior visits, the bucket is t = floor(10 (k - 1) / n_u), and every customer
v at the same seller contributes only their first floor(t n_v / 10) visits. This
is a virtual replay matched on progress, not a real simultaneous cut. Since
floor(t n_u / 10) <= k - 1, u's own target never counts.

Seller-wide counts under this rule feed the local popularity baseline now and
the relation snapshot later. Sources with absolute time cut by timestamp instead.
"""
from collections import Counter
from typing import Mapping, Sequence

from commerce.packages.data_adapters.baskets import Visit

BUCKETS = 10


def progress_bucket(position: int, n_visits: int) -> int:
    """t in 0..9 for the 1-based target position among n visits."""
    if not 1 <= position <= n_visits:
        raise ValueError("target position outside 1..n")
    return BUCKETS * (position - 1) // n_visits


def visible_prefix(bucket: int, n_visits: int) -> int:
    """How many of a customer's first visits are visible at bucket t."""
    return bucket * n_visits // BUCKETS


class SellerReplay:
    """Per-item purchase counts of one seller's customers at each progress bucket."""

    def __init__(self, visits_by_customer: Mapping[str, Sequence[Visit]]):
        sellers = {v.basket.seller_id for visits in visits_by_customer.values() for v in visits}
        if len(sellers) > 1:
            raise ValueError("a replay covers one seller")
        self._counts = [Counter() for _ in range(BUCKETS)]
        self.visible_visits = [0] * BUCKETS
        for visits in visits_by_customer.values():
            for bucket in range(BUCKETS):
                prefix = visible_prefix(bucket, len(visits))
                self.visible_visits[bucket] += prefix
                for visit in visits[:prefix]:
                    self._counts[bucket].update(visit.basket.item_ids)

    def counts(self, bucket: int) -> Counter:
        """Item -> number of visible visits containing it. Do not mutate the result."""
        return self._counts[bucket]
