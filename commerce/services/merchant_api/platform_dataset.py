"""Synthetic platform dataset: several stores, their catalogs, customers and every
kind of interaction, built through A's own code so the screens, B's feature
ledger and an FL round all see the same history.

Why it exists: a marketplace only shows whether recommendations, the explore
feed and FL work once people have bought, liked and talked for a while. With
no real users yet, this generates that history -- deterministically (fixed
seeds, stable ids), so a re-run adds nothing and two machines get the same
stores.

What it makes per store (launcher layout: merchant-i in commerce/deploy/var/merchant_i):
- a seller account (owner-1 / demo-pass-1234) and a catalog of its business
  type with descriptions (B's text encoder reads title + description +
  categories); a few staples are sold by several stores; one item is listed
  late in the period (a new item with little history);
- 60-90 seller-local customers with an activity level, favourites they rebuy,
  store-specific bought-together pairs and next-visit sequences;
- 90 days of orders through orders_service (completed orders become
  purchase_events with their backdated completion time), a few still open or
  cancelled near the end;
- reviews from verified buyers, wishlists, carts, SNS posts and reels with
  generated media, views / likes / comments driven by each customer's taste,
  buyer-proposed group buys (succeeded, failed, open), DMs, price history.

The sixth store sells side dishes, shares no product with the others and has a
short history: the held-out / new-seller case of the FL evaluation.

For FL: export_seller_input() reads a store's catalog payloads and purchase
events back out of its orders.sqlite in the shape of B's scenario SellerInput
(catalog_item.v1 in source_seq order, purchase_event.v1 in completion order),
and --export writes them as JSON, so an FL runner can load the same history B
gets through A's outbox. Everything is invented (data.md §6): names, products
and baskets come from fixed seeds, not from any raw dataset.

Usage:
    python -m commerce.services.merchant_api.platform_dataset --stores 6 [--export DIR]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from commerce.services.merchant_api import (accounts_db, accounts_service, cart_db, chatbot, media, orders_db,
                                            orders_service, shop_db, sns_db, social_db)
from commerce.services.merchant_api.presentation import product_emoji, tone_for

DAYS = 90
PASSWORD = "demo-pass-1234"
OWNER = "owner-1"
VAR = Path(__file__).resolve().parents[2] / "deploy" / "var"

# (item id, title, price, category path, description). Staples share ids and text across stores.
STAPLES = {
    "staple-milk": ("유기농 우유 1L", 2900, ["식품", "유제품", "우유"], "무항생제 목장 원유로 만든 유기농 우유"),
    "staple-egg": ("동물복지 유정란 10구", 6900, ["식품", "축산", "계란"], "동물복지 인증 농장의 신선한 유정란"),
    "staple-rice": ("제주 쌀 4kg", 19800, ["농산", "쌀"], "올해 수확한 제주산 백미"),
}


@dataclass
class Store:
    key: str
    name: str
    reg_no: str
    intro: str
    items: list[tuple[str, str, int, list[str], str]]
    pairs: list[tuple[str, str]]          # bought together
    sequences: list[tuple[str, str]]      # bought on the visit after
    new_item: Optional[str] = None        # listed at day NEW_ITEM_DAY, so it has little history
    customers: int = 75
    hashtags: tuple[str, ...] = ()
    history_days: int = DAYS


def _s(*keys: str) -> list[tuple[str, str, int, list[str], str]]:
    return [(k, *STAPLES[k]) for k in keys]


STORES: list[Store] = [
    Store("farm", "제주 유기농 농장", "123-45-67890", "제주 동쪽 마을에서 기른 채소·과일과 목장 우유를 매일 아침 보내드려요.",
          [("sku-milk", "제주 목장 유기농 우유 1L", 2500, ["식품", "유제품", "우유"], "제주 목장에서 아침에 짠 무항생제 유기농 우유"),
           ("sku-egg", "동물복지 유정란 10입", 6900, ["식품", "축산", "계란"], "넓은 방사장에서 자란 닭의 유정란"),
           ("sku-bread", "우리밀 식빵", 3800, ["식품", "베이커리", "빵"], "국산 우리밀로 매일 굽는 담백한 식빵"),
           ("sku-coffee", "한라산 핸드드립 원두 200g", 12000, ["식품", "음료", "커피"], "고소한 견과류 향의 미디엄 로스트 원두"),
           ("sku-orange", "제주 노지 감귤 3kg", 18000, ["농산", "과일"], "햇볕을 듬뿍 받은 새콤달콤한 노지 감귤"),
           ("sku-fish", "제주 은갈치 (1마리)", 15000, ["수산", "생선"], "제주 근해에서 잡은 은빛 갈치"),
           ("farm-yogurt", "목장 플레인 요거트 450g", 3900, ["식품", "유제품"], "유산균이 살아있는 무가당 요거트"),
           ("farm-cheese", "제주 목장 슬라이스 치즈", 4500, ["식품", "유제품"], "샌드위치에 좋은 부드러운 치즈"),
           ("farm-hallabong", "한라봉 2kg", 24000, ["농산", "과일"], "과즙이 풍부한 제주 한라봉"),
           ("farm-tomato", "대저 토마토 1kg", 9900, ["농산", "채소"], "짭짤하고 단단한 대저 토마토"),
           ("farm-lettuce", "유기농 상추 200g", 3200, ["농산", "채소"], "쌈용 유기농 적상추"),
           ("farm-cucumber", "백오이 5입", 4800, ["농산", "채소"], "아삭한 백오이, 오이무침과 샐러드용"),
           ("farm-potato", "제주 감자 2kg", 7900, ["농산", "채소"], "포슬포슬한 제주 햇감자"),
           ("farm-honey", "제주 야생화 꿀 500g", 21000, ["식품", "꿀"], "제주 야생화에서 모은 진한 꿀"),
           ("farm-jam", "감귤 잼 300g", 6500, ["식품", "잼"], "노지 감귤로 만든 수제 잼"),
           ("farm-carrot", "구좌 당근 1kg", 5900, ["농산", "채소"], "단맛이 좋은 구좌 흙당근, 이번 달 첫 출하")]
          + _s("staple-rice"),
          pairs=[("sku-bread", "farm-jam"), ("sku-bread", "sku-milk"), ("farm-lettuce", "farm-tomato"),
                 ("farm-yogurt", "farm-honey"), ("farm-cheese", "sku-bread")],
          sequences=[("sku-coffee", "sku-milk"), ("sku-orange", "farm-jam")],
          new_item="farm-carrot", customers=80, hashtags=("제주", "유기농", "산지직송")),
    Store("seafood", "제주 바다 수산", "234-56-78901", "새벽 위판장에서 받은 수산물을 손질해 당일 보내요.",
          [("sea-galchi", "제주 은갈치 2마리", 28000, ["수산", "생선"], "두툼한 제주 은갈치, 구이·조림용 손질"),
           ("sea-mackerel", "손질 고등어 2손", 9800, ["수산", "생선"], "뼈를 발라 소금 간한 고등어"),
           ("sea-squid", "제주 한치 500g", 16000, ["수산"], "여름 제철 한치, 물회·숙회용"),
           ("sea-abalone", "완도 활전복 5미", 22000, ["수산"], "살아있는 활전복, 버터구이·죽용"),
           ("sea-seaweed", "기장 생미역 1kg", 6000, ["수산"], "부드러운 생미역, 미역국용"),
           ("sea-shrimp", "새우살 300g", 11000, ["수산"], "껍질 깐 탱글한 새우살"),
           ("sea-octopus", "자숙 문어 500g", 25000, ["수산"], "삶아서 바로 먹는 쫄깃한 문어"),
           ("sea-oyster", "통영 굴 1kg", 13000, ["수산"], "겨울 제철 생굴, 굴전·굴국용"),
           ("sea-flatfish", "제주 광어회 300g", 23000, ["수산", "생선"], "주문 당일 뜬 광어회"),
           ("sea-salmon", "노르웨이 연어 필렛 400g", 19000, ["수산", "생선"], "스테이크·회용 생연어"),
           ("sea-laver", "곱창김 10장", 8500, ["수산", "건어물"], "겨울 첫물 곱창김"),
           ("sea-anchovy", "볶음용 멸치 300g", 9000, ["수산", "건어물"], "반찬용 잔멸치"),
           ("sea-lemon", "레몬 3입", 3500, ["농산", "과일"], "생선 요리에 곁들이는 레몬"),
           ("sea-crab", "제주 꽃게 1kg", 26000, ["수산"], "알이 꽉 찬 꽃게, 신상품")]
          + _s("staple-rice", "staple-egg"),
          pairs=[("sea-salmon", "sea-lemon"), ("sea-flatfish", "sea-lemon"), ("sea-seaweed", "sea-abalone"),
                 ("sea-mackerel", "staple-rice")],
          sequences=[("sea-squid", "sea-octopus"), ("sea-oyster", "sea-laver")],
          new_item="sea-crab", customers=70, hashtags=("제주바다", "제철수산", "당일손질")),
    Store("bakery", "한라 베이커리 & 커피", "345-67-89012", "천연 발효종 빵과 매주 볶는 원두.",
          [("bak-bread", "천연발효 식빵", 5200, ["식품", "베이커리", "빵"], "르방으로 발효한 촉촉한 식빵"),
           ("bak-bagel", "통밀 베이글 4입", 6500, ["식품", "베이커리", "빵"], "쫄깃한 통밀 베이글"),
           ("bak-croissant", "버터 크루아상 3입", 7200, ["식품", "베이커리"], "프랑스산 버터로 겹겹이 구운 크루아상"),
           ("bak-scone", "얼그레이 스콘 4입", 8000, ["식품", "베이커리"], "얼그레이 향이 은은한 스콘"),
           ("bak-cake", "당근 케이크 1조각", 6800, ["식품", "베이커리", "케이크"], "크림치즈 프로스팅 당근 케이크"),
           ("bak-cookie", "초코칩 쿠키 5입", 7500, ["식품", "베이커리"], "바삭 쫀득한 초코칩 쿠키"),
           ("bak-granola", "수제 그래놀라 300g", 8900, ["식품", "베이커리"], "견과류 가득 오븐 그래놀라"),
           ("bak-bean-eth", "에티오피아 원두 200g", 13500, ["식품", "음료", "커피"], "꽃향과 산미가 산뜻한 원두"),
           ("bak-bean-col", "콜롬비아 원두 200g", 12500, ["식품", "음료", "커피"], "균형 잡힌 단맛의 원두"),
           ("bak-coldbrew", "콜드브루 원액 500ml", 9900, ["식품", "음료", "커피"], "물이나 우유에 타 마시는 콜드브루"),
           ("bak-latte", "말차 라떼 파우더 200g", 11000, ["식품", "음료"], "제주 말차로 만든 라떼 파우더"),
           ("bak-butter", "발효 버터 200g", 9500, ["식품", "유제품"], "빵에 바르는 고소한 발효 버터"),
           ("bak-sourdough", "무화과 깜빠뉴", 9800, ["식품", "베이커리", "빵"], "무화과가 쏙쏙 박힌 깜빠뉴, 신상품")]
          + _s("staple-milk"),
          pairs=[("bak-bread", "bak-butter"), ("bak-croissant", "bak-coldbrew"), ("bak-granola", "staple-milk"),
                 ("bak-scone", "bak-latte"), ("bak-cake", "bak-bean-eth")],
          sequences=[("bak-bean-eth", "bak-bean-col"), ("bak-bagel", "bak-butter")],
          new_item="bak-sourdough", customers=85, hashtags=("천연발효", "갓구운빵", "스페셜티")),
    Store("produce", "오이네 청과", "456-78-90123", "산지에서 바로 온 제철 과일과 채소.",
          [("pro-strawberry", "설향 딸기 500g", 12900, ["농산", "과일"], "달콤한 설향 딸기"),
           ("pro-apple", "부사 사과 3kg", 21000, ["농산", "과일"], "아삭하고 단 부사 사과"),
           ("pro-grape", "샤인머스캣 1송이", 15900, ["농산", "과일"], "씨 없이 달콤한 샤인머스캣"),
           ("pro-peach", "황도 복숭아 2kg", 23000, ["농산", "과일"], "말랑하고 달콤한 황도"),
           ("pro-watermelon", "고창 수박 1통", 19900, ["농산", "과일"], "여름 제철 고창 수박"),
           ("pro-blueberry", "국산 블루베리 300g", 9900, ["농산", "과일"], "새콤달콤 생 블루베리"),
           ("pro-kiwi", "골드키위 1kg", 11900, ["농산", "과일"], "비타민 가득 골드키위"),
           ("pro-banana", "유기농 바나나 1송이", 4900, ["농산", "과일"], "숙성이 잘된 유기농 바나나"),
           ("pro-cabbage", "해남 배추 2포기", 8900, ["농산", "채소"], "김장·겉절이용 해남 배추"),
           ("pro-onion", "무안 양파 3kg", 7900, ["농산", "채소"], "단단하고 단맛 좋은 양파"),
           ("pro-garlic", "의성 깐마늘 500g", 8500, ["농산", "채소"], "향이 진한 의성 마늘"),
           ("pro-mushroom", "표고버섯 300g", 6900, ["농산", "채소"], "향 좋은 원목 표고"),
           ("pro-sweetpotato", "꿀고구마 3kg", 14900, ["농산", "채소"], "구우면 꿀이 흐르는 고구마"),
           ("pro-corn", "찰옥수수 5입", 7900, ["농산", "채소"], "쫀득한 강원 찰옥수수, 신상품")]
          + _s("staple-rice"),
          pairs=[("pro-cabbage", "pro-garlic"), ("pro-onion", "pro-garlic"), ("pro-strawberry", "pro-blueberry"),
                 ("pro-banana", "pro-kiwi")],
          sequences=[("pro-apple", "pro-grape"), ("pro-peach", "pro-watermelon")],
          new_item="pro-corn", customers=75, hashtags=("제철과일", "산지직송", "오이네")),
    Store("butcher", "돌담 정육", "567-89-01234", "제주 흑돼지와 한우를 직접 손질해요.",
          [("but-porkbelly", "제주 흑돼지 오겹살 500g", 18900, ["식품", "축산", "정육"], "두툼하게 썬 흑돼지 오겹살"),
           ("but-neck", "흑돼지 목살 500g", 17900, ["식품", "축산", "정육"], "구이용 흑돼지 목살"),
           ("but-beefsoup", "한우 국거리 300g", 21900, ["식품", "축산", "정육"], "미역국·무국용 한우 양지"),
           ("but-sirloin", "한우 등심 300g", 39000, ["식품", "축산", "정육"], "마블링 좋은 한우 등심"),
           ("but-chicken", "무항생제 닭가슴살 1kg", 12900, ["식품", "축산"], "운동하는 분들을 위한 닭가슴살"),
           ("but-wings", "닭날개 1kg", 9900, ["식품", "축산"], "구이·튀김용 닭날개"),
           ("but-duck", "훈제 오리 500g", 14900, ["식품", "축산"], "기름 뺀 훈제 오리"),
           ("but-sausage", "수제 소시지 300g", 8900, ["식품", "축산"], "돼지고기 함량 90% 수제 소시지"),
           ("but-ribs", "흑돼지 등갈비 1kg", 24900, ["식품", "축산", "정육"], "찜·바베큐용 등갈비"),
           ("but-lettuce", "쌈 채소 모둠 300g", 4900, ["농산", "채소"], "고기와 먹기 좋은 쌈 채소"),
           ("but-ssamjang", "수제 쌈장 300g", 5500, ["식품", "장류"], "된장·고추장을 섞은 수제 쌈장"),
           ("but-garlic", "편마늘 200g", 3900, ["농산", "채소"], "구이용 편마늘"),
           ("but-tteokgalbi", "한우 떡갈비 4장", 16900, ["식품", "축산"], "한우로 빚은 떡갈비, 신상품")]
          + _s("staple-egg", "staple-rice"),
          pairs=[("but-porkbelly", "but-lettuce"), ("but-porkbelly", "but-ssamjang"), ("but-neck", "but-garlic"),
                 ("but-beefsoup", "staple-rice")],
          sequences=[("but-ribs", "but-sausage"), ("but-chicken", "but-chicken")],
          new_item="but-tteokgalbi", customers=70, hashtags=("흑돼지", "한우", "오늘고기")),
    Store("banchan", "할망 반찬가게", "678-90-12345", "제주 할망 손맛 그대로, 매일 만든 반찬.",
          [("ban-kimchi", "배추김치 1kg", 13000, ["식품", "반찬", "김치"], "제주 배추로 담근 포기김치"),
           ("ban-kkakdugi", "깍두기 500g", 7000, ["식품", "반찬", "김치"], "새콤하게 익은 깍두기"),
           ("ban-anchovy", "멸치볶음 200g", 6000, ["식품", "반찬"], "견과류 넣은 멸치볶음"),
           ("ban-jangjorim", "소고기 장조림 300g", 11000, ["식품", "반찬"], "메추리알 넣은 장조림"),
           ("ban-namul", "나물 3종 세트", 9000, ["식품", "반찬"], "시금치·콩나물·고사리 나물"),
           ("ban-japchae", "잡채 400g", 9500, ["식품", "반찬"], "당면이 쫄깃한 잡채"),
           ("ban-mandu", "고기만두 10입", 8500, ["식품", "만두"], "손으로 빚은 고기만두"),
           ("ban-tteok", "오메기떡 6입", 9000, ["식품", "떡"], "제주 차조로 만든 오메기떡"),
           ("ban-seaweedsoup", "미역국 1L", 7500, ["식품", "국"], "소고기 미역국, 데우기만 하면 끝")],
          pairs=[("ban-kimchi", "ban-mandu"), ("ban-namul", "ban-jangjorim")],
          sequences=[("ban-kimchi", "ban-kkakdugi")],
          customers=40, hashtags=("집반찬", "제주할망", "오늘반찬"), history_days=30),
]


def store_index(i: int) -> Store:
    """merchant-i -> store profile (repeats past six)."""
    return STORES[(i - 1) % len(STORES)]


_FAMILY = ["김", "이", "박", "최", "정", "강", "조", "윤", "장", "임", "한", "오", "서", "신", "권", "황", "안", "송", "류", "홍"]
_GIVEN = ["서준", "하윤", "도윤", "서연", "시우", "지우", "하준", "수아", "지호", "지유", "예준", "채원", "유준", "다은",
          "주원", "소율", "건우", "예린", "현우", "윤서", "민서", "지안", "태윤", "나은", "시윤", "하린"]
REVIEWS = {
    5: ["정말 신선해요! 또 살게요.", "가족들이 다 좋아해요.", "포장도 꼼꼼하고 맛있어요.", "이 가격에 이 품질이라니, 강추!", "벌써 세 번째 주문이에요."],
    4: ["맛있어요. 양이 조금만 더 많으면 좋겠어요.", "괜찮아요, 재구매 의사 있어요.", "신선한데 배송이 하루 늦었어요."],
    3: ["보통이에요.", "무난해요.", "생각했던 맛이랑 조금 달라요."],
    2: ["기대보다 별로였어요.", "크기가 작았어요."],
    1: ["상태가 좋지 않았어요. 판매자분께 쪽지 드렸어요."],
}
COMMENTS = ["와 맛있겠다!", "주문 완료했어요 😊", "언제 또 들어와요?", "공구 열어주세요!", "지난번에 산 거 최고였어요",
            "가격 좋네요", "사진만 봐도 신선해 보여요", "이거 선물용으로도 되나요?", "단골 될게요 👍"]
SELLER_REPLIES = ["감사합니다! 오늘도 신선하게 보내드릴게요.", "다음 주에 다시 들어와요 😊", "선물 포장도 가능해요, 쪽지 주세요!",
                  "늘 찾아주셔서 감사해요."]
DM_Q = ["오늘 들어온 물건 신선한가요?", "다음 주에도 재입고 되나요?", "직접 픽업도 가능한가요?", "선물 포장 되나요?", "대량 주문 할인 있나요?"]
DM_A = ["네, 오늘 아침에 들어왔어요!", "매주 화요일에 재입고돼요.", "픽업 가능해요, 오시기 전에 쪽지 주세요.", "네, 포장해 드릴게요.",
        "10개 이상은 공동구매로 열어 주시면 할인돼요."]
POST_LINES = {
    "article": ["{title} 들어왔어요! {desc}.", "오늘의 추천 {title}, {price}원이에요.", "{title} 재입고 완료! 서두르세요.",
                "요즘 제일 많이 찾는 {title} 소개해요. {desc}."],
    "short_video": ["{title} 손질부터 포장까지 보여드려요", "{title} 이렇게 드시면 더 맛있어요", "60초 만에 보는 {title} 이야기"],
}


def _iso(t: dt.datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


@dataclass
class Shopper:
    customer_id: str
    name: str
    visits: list[float]                     # days ago, oldest first
    favourites: list[str]
    taste: dict[str, float]                 # item -> weight
    likes_rate: float
    bought: dict[str, int] = field(default_factory=dict)


def _shoppers(store: Store, rng: random.Random, now_days: int, items: list[str]) -> list[Shopper]:
    out = []
    for n in range(store.customers):
        level = rng.choices(["weekly", "biweekly", "monthly", "once"], weights=[3, 4, 3, 2])[0]
        gap = {"weekly": 7, "biweekly": 14, "monthly": 28, "once": 999}[level]
        start = rng.uniform(2, store.history_days)
        visits, t = [], start
        while t > 0.5 and len(visits) < 20:
            visits.append(round(t, 3))
            t -= max(1.0, rng.gauss(gap, gap * 0.25)) if gap < 999 else 999
        favourites = rng.sample(items, k=min(3, len(items)))
        taste = {i: 0.4 for i in items}
        for f in favourites:
            taste[f] = 6.0
        out.append(Shopper("cust-%03d" % (n + 1), rng.choice(_FAMILY) + rng.choice(_GIVEN), visits, favourites, taste,
                           rng.uniform(0.15, 0.6)))
    return out


def seed_store(conn: sqlite3.Connection, seller_id: str, store: Store, now: Optional[dt.datetime] = None,
               media_folder: Optional[Path] = None) -> dict[str, int]:
    """Fill one store; idempotent (stable ids). Returns counts of what it added."""
    now = (now or dt.datetime.now(dt.timezone.utc)).replace(microsecond=0)
    rng = random.Random("platform-" + store.key)
    for ensure in (orders_db.ensure_schema, social_db.ensure_schema, sns_db.ensure_schema, accounts_db.ensure_schema,
                   cart_db.ensure_schema, shop_db.ensure_schema, chatbot.ensure_schema):
        ensure(conn)
    added = {"orders": 0, "reviews": 0, "posts": 0, "likes": 0, "comments": 0}
    days_ago = lambda d: now - dt.timedelta(days=d)  # noqa: E731

    if accounts_db.fetch_seller_account(conn, seller_id, OWNER) is None:
        accounts_service.signup_seller(conn, seller_id=seller_id, username=OWNER, display_name=store.name, password=PASSWORD,
                                       business_reg_no=store.reg_no, business_open_date="20190301", business_rep_name="데모")
    with conn:
        social_db.save_store_settings(conn, seller_id, 10, 3, store.intro)
    for item_id, title, price, path, desc in store.items:
        if orders_service.get_catalog_item_for_display(conn, seller_id=seller_id, item_id_local=item_id) is None:
            orders_service.register_catalog_item(conn, seller_id=seller_id, item_id_local=item_id, title_text=title,
                                                 description_text=desc, category_path=path, display_price_minor=price)
    catalog = {i[0]: i for i in store.items}
    new_item_day = store.history_days * 0.3  # listed when 30% of the period was left
    with conn:  # listing dates match the history (the new item really is new)
        for item_id in catalog:
            listed = days_ago(new_item_day if item_id == store.new_item else store.history_days + 1)
            _backdate_listing(conn, seller_id, item_id, _iso(listed))
    shoppers = _shoppers(store, rng, store.history_days, list(catalog))
    quality = {i: rng.uniform(3.7, 4.9) for i in catalog}

    # --- orders ------------------------------------------------------------
    for shopper in shoppers:
        if accounts_db.fetch_customer(conn, seller_id, shopper.customer_id) is None:
            accounts_service.signup_customer(conn, seller_id=seller_id, customer_id_local=shopper.customer_id,
                                             display_name=shopper.name, password=PASSWORD)
        previous: set[str] = set()
        for visit_no, ago in enumerate(shopper.visits):
            available = [i for i in catalog if not (i == store.new_item and ago > new_item_day)]
            weights = [shopper.taste[i] for i in available]
            basket = set(rng.choices(available, weights=weights, k=rng.randint(1, 3)))
            if rng.random() < 0.65:
                basket.add(rng.choice([f for f in shopper.favourites if f in available] or available))
            for a, b in store.pairs:
                if a in basket and b in available and rng.random() < 0.6:
                    basket.add(b)
            for a, b in store.sequences:
                if a in previous and b in available and rng.random() < 0.5:
                    basket.add(b)
            if store.new_item in available and rng.random() < 0.12:
                basket.add(store.new_item)  # the new item gathers a little history late in the period
            stage = "completed"
            if ago < 2.0:
                stage = rng.choices(["requested", "accepted", "completed"], weights=[2, 2, 3])[0]
            elif rng.random() < 0.03:
                stage = "cancelled"
            items = [{"item_id_local": i, "quantity": rng.choices([1, 2, 3], weights=[6, 3, 1])[0],
                      "unit_price_minor": catalog[i][2]} for i in sorted(basket)]
            key = "pd-%s-%s-%d" % (store.key, shopper.customer_id, visit_no)
            existed = orders_db.fetch_order_by_idempotency_key(conn, seller_id, key) is not None
            order = orders_service.place_order(conn, seller_id=seller_id, customer_id_local=shopper.customer_id,
                                               idempotency_key=key,
                                               items=items, currency="KRW")
            if stage == "completed":
                for i in sorted(basket):
                    shopper.bought[i] = shopper.bought.get(i, 0) + 1
            previous = basket
            delay = dt.timedelta(hours=rng.uniform(3, 30))  # drawn before the skip so a re-run stays in step
            if existed:
                continue
            added["orders"] += 1
            when = days_ago(ago)
            if stage in ("accepted", "completed"):
                orders_service.transition_order(conn, seller_id=seller_id, order_id=order["order_id"], action="accept",
                                                expected_status_version=1)
            if stage == "completed":
                orders_service.transition_order(conn, seller_id=seller_id, order_id=order["order_id"], action="complete",
                                                expected_status_version=2)
            if stage == "cancelled":
                orders_service.transition_order(conn, seller_id=seller_id, order_id=order["order_id"], action="cancel",
                                                expected_status_version=1)
            done = when + delay if stage == "completed" else None
            if done and done > now:
                done = now - dt.timedelta(minutes=5)
            with conn:
                conn.execute("UPDATE orders SET created_at = ?, completed_at = ? WHERE seller_id = ? AND order_id = ?",
                             (_iso(when), _iso(done) if done else None, seller_id, order["order_id"]))
                if done:
                    _backdate_event(conn, seller_id, order["order_id"], _iso(done))

    # --- reviews, wishlist, cart ---------------------------------------------
    with conn:
        for shopper in shoppers:
            for item, times in shopper.bought.items():
                if rng.random() < 0.35:
                    rating = max(1, min(5, round(rng.gauss(quality[item] + 0.15 * min(times, 3), 0.7))))
                    shop_db.upsert_review(conn, seller_id, item, shopper.customer_id, rating, rng.choice(REVIEWS[rating]),
                                          _iso(days_ago(rng.uniform(0, 20))))
                    added["reviews"] += 1
            for item in rng.sample(list(catalog), k=rng.randint(0, 3)):
                conn.execute("INSERT OR IGNORE INTO wishlist (seller_id, customer_id_local, item_id_local, created_at) VALUES (?, ?, ?, ?)",
                             (seller_id, shopper.customer_id, item, _iso(days_ago(rng.uniform(0, 30)))))
            if rng.random() < 0.15:
                for item in rng.sample(list(catalog), k=rng.randint(1, 2)):
                    conn.execute("INSERT OR IGNORE INTO cart_items (seller_id, customer_id_local, item_id_local, quantity, added_at) "
                                 "VALUES (?, ?, ?, 1, ?)", (seller_id, shopper.customer_id, item, _iso(days_ago(rng.uniform(0, 3)))))

    # --- SNS ---------------------------------------------------------------------
    folder = media_folder or media.media_dir(Path(conn.execute("PRAGMA database_list").fetchone()[2]))
    n_posts = max(8, int(store.history_days / 5))
    for n in range(n_posts):
        post_id = "pd-post-%s-%02d" % (store.key, n)
        ago = store.history_days * (1 - (n + 0.5) / n_posts)
        prng = random.Random(post_id)  # the post's own content, independent of what a re-run skips
        kind = "short_video" if prng.random() < 0.35 else "article"
        tagged = [i for i in prng.sample(list(catalog), k=prng.choice([1, 1, 2])) if not (i == store.new_item and ago > new_item_day)]
        if not tagged:
            tagged = [prng.choice([i for i in catalog if i != store.new_item])]
        item_id, title, price, path, desc = catalog[tagged[0]]
        line = prng.choice(POST_LINES[kind]).format(title=title, desc=desc, price=format(price, ","))
        tags = ["#" + path[-1].replace(" ", "")] + ["#" + h for h in prng.sample(list(store.hashtags), k=2)]
        caption = line + "\n" + " ".join(tags)
        if sns_db.fetch_post(conn, seller_id, post_id) is None:
            emoji = product_emoji(path, title) or "🥒"
            if kind == "short_video":
                svg = media.reel_svg(title, [line[:18], desc[:18], "%s원" % format(price, ",")], emoji, tone_for(post_id))
                stored = [media.write_generated(folder, svg, "motion")]
            else:
                badge = prng.choice(["오늘 입고", "제철", "인기", "", ""])
                stored = [media.write_generated(folder, media.photo_svg(title, "%s원 · %s" % (format(price, ","), " ".join(tags[:2])),
                                                                        emoji, tone_for(post_id), badge=badge), "image")]
                if prng.random() < 0.3 and len(tagged) > 1:
                    other = catalog[tagged[1]]
                    stored.append(media.write_generated(folder, media.photo_svg(other[1], "%s원" % format(other[2], ","),
                                                                                product_emoji(other[3], other[1]) or "🥒",
                                                                                tone_for(post_id + "b")), "image"))
            with conn:
                sns_db.insert_post(conn, seller_id, post_id, kind, line[:60], caption, stored, [t[1:] for t in tags],
                                   _iso(days_ago(ago)))
                sns_db.set_post_products(conn, seller_id, post_id, tagged)
            added["posts"] += 1
        # engagement from customers who were around after it went up, driven by taste
        with conn:
            for shopper in shoppers:
                if not shopper.visits or shopper.visits[0] < ago - 30:
                    continue
                affinity = max(shopper.taste.get(i, 0.4) for i in tagged) / 6.0
                if rng.random() < 0.25 + 0.6 * affinity:
                    seen = days_ago(max(0.0, ago - rng.uniform(0, min(ago, 10))))
                    conn.execute("INSERT OR IGNORE INTO post_views (seller_id, post_id, viewer, views, last_viewed_at) "
                                 "VALUES (?, ?, ?, ?, ?)", (seller_id, post_id, shopper.customer_id, rng.randint(1, 3), _iso(seen)))
                    if rng.random() < shopper.likes_rate * (0.4 + affinity) * (1.3 if kind == "short_video" else 1.0):
                        cur = conn.execute("INSERT OR IGNORE INTO post_likes (seller_id, post_id, customer_id_local, created_at) VALUES (?, ?, ?, ?)",
                                           (seller_id, post_id, shopper.customer_id, _iso(seen)))
                        added["likes"] += cur.rowcount
                    if rng.random() < 0.05 + 0.08 * affinity:
                        cid = "pd-c-%s-%s" % (post_id, shopper.customer_id)
                        cur = conn.execute("INSERT OR IGNORE INTO post_comments (seller_id, comment_id, post_id, author, customer_id_local, body, created_at) "
                                           "VALUES (?, ?, ?, 'customer', ?, ?, ?)",
                                           (seller_id, cid, post_id, shopper.customer_id, rng.choice(COMMENTS), _iso(seen)))
                        added["comments"] += cur.rowcount
            if rng.random() < 0.5:
                conn.execute("INSERT OR IGNORE INTO post_comments (seller_id, comment_id, post_id, author, customer_id_local, body, created_at) "
                             "VALUES (?, ?, ?, 'seller', NULL, ?, ?)",
                             (seller_id, "pd-r-" + post_id, post_id, rng.choice(SELLER_REPLIES), _iso(days_ago(max(0.0, ago - 1)))))
            guests = rng.randint(5, 40)
            conn.execute("INSERT OR IGNORE INTO post_views (seller_id, post_id, viewer, views, last_viewed_at) VALUES (?, ?, 'guest', ?, ?)",
                         (seller_id, post_id, guests, _iso(days_ago(max(0.0, ago - 2)))))

    # --- buyer-proposed group buys -------------------------------------------------
    _group_buys(conn, seller_id, store, shoppers, catalog, now)

    # --- DMs and price history -------------------------------------------------------
    rng = random.Random("dm-" + store.key)
    with conn:
        for shopper in rng.sample(shoppers, k=min(8, len(shoppers))):
            ago = rng.uniform(1, store.history_days)
            q = rng.randrange(len(DM_Q))
            conn.execute("INSERT OR IGNORE INTO messages (seller_id, message_id, customer_id_local, sender, body, created_at) VALUES (?, ?, ?, 'customer', ?, ?)",
                         (seller_id, "pd-m-%s-%s-q" % (store.key, shopper.customer_id), shopper.customer_id, DM_Q[q], _iso(days_ago(ago))))
            if rng.random() < 0.8:
                conn.execute("INSERT OR IGNORE INTO messages (seller_id, message_id, customer_id_local, sender, body, created_at) VALUES (?, ?, ?, 'seller', ?, ?)",
                             (seller_id, "pd-m-%s-%s-a" % (store.key, shopper.customer_id), shopper.customer_id, DM_A[q],
                              _iso(days_ago(max(0.0, ago - 0.2)))))
        for item_id, _title, price, _path, _desc in store.items[:4]:
            p = price
            for d in range(min(30, store.history_days), -1, -1):
                p = max(int(p * (1 + rng.uniform(-0.04, 0.04))), int(price * 0.75))
                social_db.upsert_price(conn, seller_id, item_id, days_ago(d).strftime("%Y-%m-%d"), p)
    return added


def _backdate_listing(conn, seller_id: str, item_id: str, listed_at: str) -> None:
    """first_listed_at of a catalog item this run registered and B has not received yet."""
    row = conn.execute("SELECT payload_json FROM catalog_items WHERE seller_id = ? AND item_id_local = ?",
                       (seller_id, item_id)).fetchone()
    payload = json.loads(row["payload_json"])
    pending = conn.execute("SELECT 1 FROM outbox WHERE seller_id = ? AND kind = 'catalog_item' AND ref_id = ? AND status = 'pending'",
                           (seller_id, item_id)).fetchone()
    if payload["first_listed_at"] <= listed_at or pending is None:
        return  # already as old, or B has this version: changing only A's copy would split the two
    payload["first_listed_at"] = listed_at
    body = json.dumps(payload, ensure_ascii=False)
    conn.execute("UPDATE catalog_items SET payload_json = ? WHERE seller_id = ? AND item_id_local = ?", (body, seller_id, item_id))
    conn.execute("UPDATE outbox SET payload_json = ? WHERE seller_id = ? AND kind = 'catalog_item' AND ref_id = ? AND status = 'pending'",
                 (body, seller_id, item_id))


def _backdate_event(conn, seller_id: str, order_id: str, completed_at: str) -> None:
    row = conn.execute("SELECT purchase_event_id, payload_json FROM purchase_events WHERE seller_id = ? AND order_id = ?",
                       (seller_id, order_id)).fetchone()
    if row is None:
        return
    payload = json.loads(row["payload_json"])
    payload["time"] = {"kind": "absolute", "value": completed_at}
    body = json.dumps(payload, ensure_ascii=False)
    conn.execute("UPDATE purchase_events SET payload_json = ? WHERE seller_id = ? AND purchase_event_id = ?",
                 (body, seller_id, row["purchase_event_id"]))
    conn.execute("UPDATE outbox SET payload_json = ? WHERE seller_id = ? AND kind = 'purchase_event' AND ref_id = ? AND status = 'pending'",
                 (body, seller_id, row["purchase_event_id"]))


def _group_buys(conn, seller_id: str, store: Store, shoppers: list[Shopper], catalog: dict,
                now: dt.datetime) -> None:
    plans = [("succeeded", 20, 6), ("failed", 12, 8), ("open", 2, 10), ("open", 1, 6)]
    for n, (outcome, started_ago, target) in enumerate(plans):
        gid = "pd-gb-%s-%d" % (store.key, n)
        if social_db.fetch_group_buy(conn, seller_id, gid) is not None:
            continue
        rng = random.Random(gid)  # its own draws, so skipping one on a re-run changes nothing else
        item = rng.choice([i for i in catalog if i != store.new_item])
        price = int(round(catalog[item][2] * 0.9, -1))
        created = now - dt.timedelta(days=started_ago)
        deadline = created + dt.timedelta(days=7)
        people = rng.sample(shoppers, k=min(len(shoppers), 6))
        with conn:
            social_db.insert_group_buy(conn, seller_id, gid, item, target, price, _iso(deadline), _iso(created),
                                       proposer_customer_id=people[0].customer_id,
                                       message=rng.choice(["우리 동네 같이 사요!", "회사 동료들이랑 나눠요", "주말 캠핑용으로 같이 사실 분", None]))
            joined = 0
            for k, p in enumerate(people):
                qty = rng.randint(1, 2)
                if outcome == "succeeded" and joined + qty > target:
                    qty = target - joined
                if outcome != "succeeded" and joined + qty >= target:
                    break
                if qty <= 0:
                    break
                social_db.insert_group_buy_participant(conn, seller_id, gid, p.customer_id, qty,
                                                       _iso(created + dt.timedelta(hours=6 * (k + 1))))
                joined += qty
        if outcome == "succeeded":
            # settle like the live path (one order per participant), then complete and backdate those orders
            from commerce.services.merchant_api import social_service
            gb = social_db.fetch_group_buy(conn, seller_id, gid)
            if social_db.sum_group_buy_quantity(conn, seller_id, gid) >= target:
                social_service._settle_succeeded(conn, seller_id=seller_id, group_buy=gb)
                for p in social_db.list_group_buy_participants(conn, seller_id, gid):
                    order = orders_db.fetch_order_by_idempotency_key(conn, seller_id, "group-buy-%s-%s" % (gid, p["customer_id_local"]))
                    if order and order["status"] == "requested":
                        orders_service.transition_order(conn, seller_id=seller_id, order_id=order["order_id"], action="accept",
                                                        expected_status_version=1)
                        orders_service.transition_order(conn, seller_id=seller_id, order_id=order["order_id"], action="complete",
                                                        expected_status_version=2)
                        done = _iso(created + dt.timedelta(days=2))
                        with conn:
                            conn.execute("UPDATE orders SET created_at = ?, completed_at = ? WHERE seller_id = ? AND order_id = ?",
                                         (_iso(created + dt.timedelta(days=1)), done, seller_id, order["order_id"]))
                            _backdate_event(conn, seller_id, order["order_id"], done)
        elif outcome == "failed":
            with conn:
                social_db.update_group_buy_status(conn, seller_id, gid, "failed")


# --- export for FL ---------------------------------------------------------------------

def export_seller_input(conn, seller_id: str) -> dict[str, Any]:
    """The store's history in the shape of B's scenario SellerInput (plain JSON types).

    catalog: each item's latest catalog_item.v1 body, ordered by its source_seq;
    events: purchase_event.v1 bodies of completed orders in completion order;
    customers: everyone with at least one purchase event. These are exactly the
    bodies A's outbox hands to B, so loading them directly gives B the same ledger.
    """
    catalog = [json.loads(r["payload_json"]) for r in conn.execute(
        "SELECT payload_json FROM catalog_items WHERE seller_id = ? ORDER BY source_seq, item_id_local", (seller_id,))]
    events = [json.loads(r["payload_json"]) for r in conn.execute(
        "SELECT payload_json FROM purchase_events WHERE seller_id = ?", (seller_id,))]
    events.sort(key=lambda e: (e["time"]["value"], e["basket_id_local"]))
    customers = sorted({e["customer_id_local"] for e in events})
    return {"seller_id": seller_id, "catalog": catalog, "events": events, "customers": customers}


def summary(conn, seller_id: str) -> dict[str, Any]:
    q = lambda sql: conn.execute(sql, (seller_id,)).fetchone()[0]  # noqa: E731
    events = export_seller_input(conn, seller_id)["events"]
    basket_sizes = [len(e["items"]) for e in events]
    per_customer: dict[str, int] = {}
    for e in events:
        per_customer[e["customer_id_local"]] = per_customer.get(e["customer_id_local"], 0) + 1
    repeat = 0
    seen: dict[str, set] = {}
    for e in events:
        prior = seen.setdefault(e["customer_id_local"], set())
        items = {i["item_id_local"] for i in e["items"]}
        repeat += len(items & prior)
        prior |= items
    total_lines = sum(basket_sizes)
    return {
        "products": q("SELECT COUNT(*) FROM catalog_items WHERE seller_id = ?"),
        "customers": q("SELECT COUNT(*) FROM customers WHERE seller_id = ?"),
        "orders": q("SELECT COUNT(*) FROM orders WHERE seller_id = ?"),
        "purchase_events": len(events),
        "avg_basket": round(sum(basket_sizes) / len(basket_sizes), 2) if basket_sizes else 0,
        "visits_per_buyer": round(len(events) / len(per_customer), 2) if per_customer else 0,
        "repeat_share": round(repeat / total_lines, 3) if total_lines else 0,
        "reviews": q("SELECT COUNT(*) FROM reviews WHERE seller_id = ?"),
        "posts": q("SELECT COUNT(*) FROM posts WHERE seller_id = ? AND deleted_at IS NULL"),
        "likes": q("SELECT COUNT(*) FROM post_likes WHERE seller_id = ?"),
        "comments": q("SELECT COUNT(*) FROM post_comments WHERE seller_id = ?"),
        "views": q("SELECT COALESCE(SUM(views), 0) FROM post_views WHERE seller_id = ?"),
        "group_buys": q("SELECT COUNT(*) FROM group_buys WHERE seller_id = ?"),
        "wishlist": q("SELECT COUNT(*) FROM wishlist WHERE seller_id = ?"),
        "messages": q("SELECT COUNT(*) FROM messages WHERE seller_id = ?"),
    }


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Generate the multi-store synthetic platform dataset.")
    parser.add_argument("--stores", type=int, default=6, help="how many launcher sellers to fill (merchant-1..N)")
    parser.add_argument("--var", default=str(VAR), help="launcher var folder (default commerce/deploy/var)")
    parser.add_argument("--export", help="write each store's SellerInput JSON and a summary here")
    args = parser.parse_args(argv)
    var = Path(args.var)
    totals = []
    for i in range(1, args.stores + 1):
        seller_id, store = "merchant-%d" % i, store_index(i)
        folder = var / ("merchant_%d" % i)
        db_path, feature_db = folder / "orders.sqlite", folder / "features.sqlite"
        if not db_path.exists() and feature_db.exists():
            print("%s exists without orders.sqlite; delete the pair together (DEMO.md step 2)." % feature_db, file=sys.stderr)
            raise SystemExit(1)
        conn = orders_db.connect(db_path)
        try:
            added = seed_store(conn, seller_id, store)
            stats = summary(conn, seller_id)
            if args.export:
                out = Path(args.export)
                out.mkdir(parents=True, exist_ok=True)
                (out / ("%s.json" % seller_id)).write_text(json.dumps(export_seller_input(conn, seller_id), ensure_ascii=False),
                                                         encoding="utf-8")
        finally:
            conn.close()
        totals.append({"seller_id": seller_id, "store": store.name, "key": store.key, **stats})
        print("%s %s: +%d orders, %d products, %d customers, %d events, %d posts, %d reviews"
              % (seller_id, store.name, added["orders"], stats["products"], stats["customers"], stats["purchase_events"],
                 stats["posts"], stats["reviews"]))
    if args.export:
        Path(args.export, "summary.json").write_text(json.dumps(totals, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
