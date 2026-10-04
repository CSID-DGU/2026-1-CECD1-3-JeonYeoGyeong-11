"""B's common local basket for live and historical purchases (data.md §4).

Every source reaches the same LocalBasket through purchase_event.v1, so the
offline adapters and a service replay share one conversion. The source
difference stays visible: absolute vs relative_day time, order_rank, and
quantity_observed=None where the source never observed quantity.
"""
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Literal

from commerce.packages.contracts.errors import ContractError
from commerce.packages.contracts.types import Payload
from commerce.packages.data_adapters.validation import check_payload

Source = Literal["live", "dunnhumby", "instacart"]
TimeKind = Literal["absolute", "relative_day"]

# interfaces.md §2. The validator accepts any combination; B consumes only these.
# source -> (seller_partition, time kind, order_rank present, quantity observed)
SOURCE_RULES: dict[str, tuple[str, str, bool, bool]] = {
    "live": ("platform_seller", "absolute", False, True),
    "dunnhumby": ("simulated_independent_store", "absolute", False, True),
    "instacart": ("synthetic_partition", "relative_day", True, False),
}

# Instacart caps days_since_prior_order at 30, so a recorded 30 means "30 or more".
INSTACART_GAP_CAP_DAYS = 30.0


@dataclass(frozen=True)
class LocalItem:
    item_id_local: str
    quantity_observed: int | None  # None: never observed. Not 0, and not filled with 1.


@dataclass(frozen=True)
class LocalBasket:
    source: Source
    seller_id: str
    seller_partition: str
    customer_id_local: str
    basket_id_local: str
    purchase_event_id: str
    time_kind: TimeKind
    time_value: datetime | float  # UTC datetime, or the customer's own cumulative days
    order_rank: int | None
    items: tuple[LocalItem, ...]  # sorted by item_id_local, one entry per item

    @property
    def item_ids(self) -> frozenset[str]:
        return frozenset(item.item_id_local for item in self.items)


@dataclass(frozen=True)
class Visit:
    basket: LocalBasket
    gap_days: float | None  # since this customer's previous visit; None on the first, not 0
    gap_censored: bool  # the recorded gap hit the source cap; the real gap may be longer
    time_lower_bound: bool  # relative time is only a lower bound after a censored gap


def basket_from_event(event: Payload) -> LocalBasket:
    """Validate one purchase_event.v1 and convert it; raises ContractError."""
    check_payload("purchase_event.v1", event)
    partition, kind, ranked, quantified = SOURCE_RULES[event["source"]]
    if event["seller_partition"] != partition:
        raise ContractError("SCHEMA_INVALID", "/seller_partition")
    if event["time"]["kind"] != kind:
        raise ContractError("SCHEMA_INVALID", "/time/kind")
    if (event["order_rank"] is not None) != ranked:
        raise ContractError("SCHEMA_INVALID", "/order_rank")
    for index, item in enumerate(event["items"]):
        if (item["quantity_observed"] is not None) != quantified:
            raise ContractError("SCHEMA_INVALID", "/items/%d/quantity_observed" % index)

    value = event["time"]["value"]
    if kind == "absolute":
        try:
            value = datetime.fromisoformat(value)  # the schema pattern leaves calendar checks to us
        except ValueError:
            raise ContractError("SCHEMA_INVALID", "/time/value") from None
        value = value.astimezone(timezone.utc)
    else:
        value = float(value)

    items = sorted((LocalItem(i["item_id_local"], i["quantity_observed"]) for i in event["items"]),
                   key=lambda item: item.item_id_local)
    return LocalBasket(
        source=event["source"], seller_id=event["seller_id"],
        seller_partition=event["seller_partition"],
        customer_id_local=event["customer_id_local"], basket_id_local=event["basket_id_local"],
        purchase_event_id=event["purchase_event_id"], time_kind=kind, time_value=value,
        order_rank=event["order_rank"], items=tuple(items),
    )


def customer_visits(baskets: Iterable[LocalBasket]) -> list[Visit]:
    """Order one customer's baskets and derive the gaps and censoring bits.

    Absolute time sorts by (time, basket_id_local). Relative time sorts by
    order_rank and needs the whole prefix 1..n, because a gap across a missing
    visit cannot be told apart from a censored one.
    """
    baskets = list(baskets)
    if not baskets:
        return []
    owner = {(b.source, b.seller_id, b.customer_id_local) for b in baskets}
    if len(owner) != 1:
        raise ValueError("customer_visits takes the baskets of one customer at one seller")
    if len({b.basket_id_local for b in baskets}) != len(baskets):
        raise ValueError("a basket appears twice in one customer's history")

    if baskets[0].time_kind == "absolute":
        baskets.sort(key=lambda b: (b.time_value, b.basket_id_local))
        visits, previous = [], None
        for basket in baskets:
            gap = None if previous is None else (basket.time_value - previous).total_seconds() / 86400
            visits.append(Visit(basket, gap, False, False))
            previous = basket.time_value
        return visits

    baskets.sort(key=lambda b: b.order_rank)
    if [b.order_rank for b in baskets] != list(range(1, len(baskets) + 1)):
        raise ValueError("relative-time history must hold order_rank 1..n without gaps")
    if baskets[0].time_value != 0:
        raise ValueError("relative-time history must start at day 0")
    visits, lower_bound = [Visit(baskets[0], None, False, False)], False
    for previous, basket in zip(baskets, baskets[1:]):
        gap = basket.time_value - previous.time_value
        if not 0 <= gap <= INSTACART_GAP_CAP_DAYS:
            raise ValueError("relative-time gap outside 0..%g days" % INSTACART_GAP_CAP_DAYS)
        censored = gap == INSTACART_GAP_CAP_DAYS
        lower_bound = lower_bound or censored
        visits.append(Visit(basket, gap, censored, lower_bound))
    return visits
