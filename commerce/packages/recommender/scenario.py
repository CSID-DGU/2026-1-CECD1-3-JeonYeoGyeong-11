"""B's scenario input for the integration gates (working-agreement §4: g2, g3; C writes the selfchecks).

Everything here is invented: product names, customers and baskets come from a
fixed seed, never from Dunnhumby or Instacart. Three cohort sellers sell the same
kind of grocery catalog; a fourth seller (NEW_SELLER) has its own catalog that
shares no product with them and takes no part in any round (C card: "미참여·
비중복 카탈로그의 네 번째 판매자"). Baskets repeat favourite items and fixed
pairs, so the relation tables are not empty, and most baskets hold the seller's
staples, a pattern every customer shares, so personalization has something to
learn that also holds for its validation customers.

For CI the runtime opens with TINY_ARCHITECTURES and FakeText (a hash vector per
text) instead of the 384-wide MiniLM; the API and file layout are the real ones.

    runtime = open_test_runtime("g3-seller-a", tmp_dir)
    load_seller(runtime, scenario().sellers["g3-seller-a"])
    manifest, tensors = random_init_release(runtime, "text_relation")   # for C's registry
"""
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
from pathlib import Path
import random

import numpy as np

from commerce.packages.contracts.ids import purchase_event_id
from commerce.packages.contracts.types import ModelVariant, Payload, TensorMap
from commerce.packages.recommender.harex import HarexConfig
from commerce.packages.recommender.serving import build_model, shared_tensors

COHORT = ("g3-seller-a", "g3-seller-b", "g3-seller-c")
NEW_SELLER = "g3-seller-new"
START = datetime(2026, 9, 1, 9, 0, tzinfo=timezone.utc)
AS_OF = "2026-10-15T00:00:00Z"  # after every scenario visit
GROCERY = ["유기농 우유 1L", "통밀 시리얼 500g", "바나나 1송이", "그릭 요거트 450g", "무염 버터 200g",
           "식빵 400g", "딸기잼 300g", "계란 10구", "두부 300g", "콩나물 300g", "대파 1단", "양파 1.5kg",
           "삼겹살 500g", "쌀 4kg", "생수 2L 6병", "탄산수 500ml", "커피 원두 200g", "녹차 티백 40개",
           "김 10봉", "참치캔 3개", "라면 5개", "고추장 500g", "올리브유 500ml", "사과 1.5kg"]
HOUSEHOLD = ["주방세제 1L", "키친타월 4롤", "수세미 3개", "지퍼백 중 50매", "물티슈 100매", "고무장갑 중",
             "섬유유연제 2L", "세탁세제 3L", "치약 120g", "칫솔 4개", "샴푸 500ml", "린스 500ml",
             "핸드크림 50ml", "건전지 AA 8개", "멀티탭 4구", "형광펜 5색"]
# Pairs bought together, and an item bought the visit after another (relation inputs).
PAIRS = [(0, 1), (3, 2), (5, 6), (8, 9), (12, 22)]
NEXT = [(13, 14), (16, 17)]
STAPLES = (0, 7)  # bought on most visits by every customer (milk and eggs, cleaner for the new seller)
NEW_ITEM = "g3-item-new"  # listed, never bought until new_item_purchase() (G3 신상품 검사)

TINY_ARCHITECTURES = {
    1: HarexConfig("test.T_lm", "lm", False, d_model=16, n_heads=2, d_ffn=32, dropout=0.0, max_items=12,
                   d_text=8, d_relation=4, mlp_hidden=16, d_time=4),
    2: HarexConfig("test.R_lm", "lm", True, d_model=16, n_heads=2, d_ffn=32, dropout=0.0, max_items=12,
                   d_text=8, d_relation=4, mlp_hidden=16, d_time=4),
}


class FakeText:
    """A frozen-encoder stand-in for CI: a fixed vector per text, no model files."""

    def __init__(self, dim: int = 8):
        self.dim = dim
        self.text_artifact_hash = hashlib.sha256(b"scenario.FakeText:%d" % dim).hexdigest()

    def encode(self, texts):
        rows = [np.frombuffer(hashlib.sha256(t.encode("utf-8")).digest()[:self.dim], dtype=np.uint8) for t in texts]
        return (np.stack(rows).astype(np.float32) / 255.0) if rows else np.zeros((0, self.dim), np.float32)


@dataclass(frozen=True)
class SellerInput:
    seller_id: str
    catalog: tuple[Payload, ...]  # catalog_item.v1 in source_seq order (1, 2, ...)
    events: tuple[Payload, ...]  # purchase_event.v1 in completion order
    customers: tuple[str, ...]


@dataclass(frozen=True)
class Scenario:
    sellers: dict[str, SellerInput]


def _iso(t: datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def catalog_item(seller: str, item_id: str, title: str, category: str, status: str = "active") -> Payload:
    return {"schema_version": "catalog_item.v1", "seller_id": seller, "item_id_local": item_id, "source": "live",
            "title_text": title, "description_text": None, "category_path": [category],
            "listing_status": status, "first_listed_at": "2026-08-01T00:00:00Z"}


def purchase_event(seller: str, customer: str, basket: str, when: datetime, items: list[str]) -> Payload:
    return {"schema_version": "purchase_event.v1", "seller_id": seller, "source": "live",
            "seller_partition": "platform_seller", "customer_id_local": customer, "basket_id_local": basket,
            "purchase_event_id": purchase_event_id(seller, "live", basket),
            "time": {"kind": "absolute", "value": _iso(when)}, "order_rank": None,
            "items": [{"item_id_local": i, "quantity_observed": 1} for i in sorted(set(items))]}


def _seller(seller: str, names: list[str], category: str, prefix: str, customers: int, visits: int,
            seed: int, *, with_new_item: bool) -> SellerInput:
    rng = random.Random(seed)
    ids = ["%s-%02d" % (prefix, i) for i in range(len(names))]
    catalog = [catalog_item(seller, i, name, category) for i, name in zip(ids, names)]
    if with_new_item:
        catalog.append(catalog_item(seller, NEW_ITEM, "[신상품] 귀리 음료 1L", category))
    events, people = [], []
    for c in range(customers):
        customer = "%s-cust-%02d" % (seller, c)
        people.append(customer)
        favourite = [c % len(ids), (c * 5 + 3) % len(ids)]
        previous = []
        for v in range(visits):
            items = set(favourite) | set(rng.sample(range(len(ids)), 2))
            items |= {i for i in STAPLES if i < len(ids) and rng.random() < 0.8}
            for a, b in PAIRS:
                if a < len(ids) and b < len(ids) and (a in items or rng.random() < 0.15):
                    items |= {a, b}
            for a, b in NEXT:
                if a in previous and b < len(ids):
                    items.add(b)
            when = START + timedelta(days=4 * v + c % 4, hours=c)
            events.append(purchase_event(seller, customer, "%s-b-%02d-%02d" % (seller, c, v), when,
                                         [ids[i] for i in sorted(items)]))
            previous = sorted(items)
    events.sort(key=lambda e: (e["time"]["value"], e["basket_id_local"]))
    return SellerInput(seller, tuple(catalog), tuple(events), tuple(people))


def scenario() -> Scenario:
    sellers = {s: _seller(s, GROCERY, "식품", "%s-item" % s.rsplit("-", 1)[1], customers=30, visits=7,
                          seed=10 + k, with_new_item=True) for k, s in enumerate(COHORT)}
    sellers[NEW_SELLER] = _seller(NEW_SELLER, HOUSEHOLD, "생활용품", "new-item", customers=6, visits=4, seed=99,
                                  with_new_item=False)
    return Scenario(sellers)


def new_item_purchase(seller: str, customer: str, partner_item: str) -> Payload:
    """One controlled basket that puts NEW_ITEM next to a bought item, so it gains a relation."""
    return purchase_event(seller, customer, "%s-b-new-item" % seller, START + timedelta(days=40),
                          [NEW_ITEM, partner_item])


def open_test_runtime(seller_id: str, root: str | Path, *, text=None, architectures=TINY_ARCHITECTURES):
    """A real SellerRuntime under root/<seller_id>/ with the CI encoder and tiny architectures."""
    from commerce.packages.recommender.seller_runtime import SellerRuntime
    folder = Path(root) / seller_id
    return SellerRuntime(seller_id, folder / "features.sqlite", folder / "models", text=text or FakeText(),
                         architectures=architectures)


def load_seller(runtime, seller: SellerInput) -> None:
    for seq, item in enumerate(seller.catalog, start=1):
        runtime.upsert_catalog_item(item, seq)
    for event in seller.events:
        runtime.ingest_purchase_event(event)


def random_init_release(runtime, variant: ModelVariant, *, seed: int = 0) -> tuple[Payload, TensorMap]:
    """(manifest, tensors) of a seeded random base for C's registry (--kind random_init)."""
    manifest = runtime.get_shared_manifest(model_variant=variant)
    config = runtime._arch(variant).config
    return manifest, shared_tensors(build_model(config, seed=seed))
