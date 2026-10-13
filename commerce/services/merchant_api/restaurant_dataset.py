"""Synthetic restaurants for the POS version: six stores, their menus, point-card members, walk-in
guests and months of table checks, all written through pos_service (open a table, take the
order, send it to the kitchen, settle the bill), so every paid check is an ordinary completed A
order whose purchase_event reaches B through the outbox, exactly as a live sale would.

What makes the history learnable: members come back on a rhythm (weekly lunch regulars,
biweekly, monthly), each has a favourite section and two or three favourite dishes; parties
order a main per guest plus shared sides and drinks; every store has dishes ordered together
(김치찌개 + 공기밥, 짜장면 + 탕수육, 떡볶이 + 김밥, ...); one dish is added late in the period
(a new menu with little history: the case where the shared text model helps most). Walk-ins
(about 40% of checks) have no member id; each walk-in check is its own customer for B.

Everything is invented (data.md §6). The end time is fixed (DEFAULT_NOW) so two machines make
the same data; `--now today` ends the history at the current minute for a live demo (tables still eating if it
is meal time). Phone numbers are generated, and stored only as salted hashes (the salt is
fixed per store here so member ids repeat; a real store gets a random one).

    python -m commerce.services.merchant_api.restaurant_dataset --stores 6 [--now today] [--export DIR]
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from commerce.services.merchant_api import (accounts_db, accounts_service, fulfillment, orders_db, orders_service, pos_db,
                                            pos_service as pos, social_db)
from commerce.services.merchant_api.platform_dataset import export_seller_input

KST = pos.KST
DEFAULT_NOW = dt.datetime(2026, 10, 10, 10, 30, tzinfo=dt.timezone.utc)  # 19:30 KST, a busy dinner
PASSWORD = "demo-pass-1234"
OWNER = "owner-1"
VAR = Path(__file__).resolve().parents[2] / "deploy" / "var"

SHARED = {  # sold by several stores with the same text (like real staples)
    "staple-rice": ("공기밥", 1000, "사이드", "갓 지은 흰쌀밥 한 공기"),
    "staple-cola": ("콜라 355ml", 2000, "음료", "시원한 캔 콜라"),
    "staple-cider": ("사이다 355ml", 2000, "음료", "시원한 캔 사이다"),
    "staple-soju": ("소주", 5000, "주류", "참이슬 프레시 360ml"),
    "staple-beer": ("맥주", 5000, "주류", "카스 500ml 병맥주"),
}


@dataclass
class Restaurant:
    key: str
    name: str
    reg_no: str
    intro: str
    hours: str
    tables: int
    menu: list[tuple[str, str, int, str, str]]                 # id, title, price, section, description
    mains: tuple[str, ...]                                      # sections a guest picks one of
    pairs: list[tuple[str, str]] = field(default_factory=list)  # ordered together
    new_item: Optional[str] = None
    members: int = 70
    per_day: tuple[int, int] = (18, 26)                         # checks per weekday (min, max)
    lunch_share: float = 0.6
    takeout_share: float = 0.2
    history_days: int = 90
    slots: tuple[tuple[float, float], ...] = ((11.5, 13.8), (17.5, 20.8))  # KST hours: lunch, dinner


def _s(*keys: str) -> list[tuple[str, str, int, str, str]]:
    return [(k, *SHARED[k]) for k in keys]


STORES: list[Restaurant] = [
    Restaurant("hansik", "오이네 한식당", "123-45-67890", "제주 엄마 손맛 그대로, 매일 새로 끓이는 찌개와 정식.", "11:00~21:00 (브레이크 15:00~17:00)", 12,
               [("hs-kimchi", "김치찌개", 9000, "찌개", "1년 묵은지와 돼지고기를 넣고 푹 끓인 얼큰한 김치찌개"),
                ("hs-doenjang", "된장찌개", 9000, "찌개", "제주 재래된장에 두부·애호박·바지락을 넣은 구수한 된장찌개"),
                ("hs-sundubu", "순두부찌개", 9500, "찌개", "몽글몽글 순두부에 해물을 넣은 매콤한 찌개, 날계란 포함"),
                ("hs-budae", "부대찌개", 10000, "찌개", "햄·소시지·라면사리가 들어간 진한 부대찌개"),
                ("hs-jeyuk", "제육볶음 정식", 11000, "정식", "고추장 양념 돼지고기 볶음과 밥·국·반찬"),
                ("hs-mackerel", "고등어구이 정식", 12000, "정식", "제주 고등어를 노릇하게 구운 생선구이 정식"),
                ("hs-bulgogi", "불고기 정식", 12000, "정식", "달콤한 간장 양념 소불고기와 밥·국·반찬"),
                ("hs-tteokgalbi", "떡갈비 정식", 13000, "정식", "한우 섞은 수제 떡갈비 두 장과 반찬"),
                ("hs-galchi", "갈치조림", 15000, "일품", "제주 은갈치와 무를 매콤하게 졸인 조림"),
                ("hs-bibim", "비빔밥", 9000, "식사", "제철 나물과 고추장, 계란후라이를 올린 비빔밥"),
                ("hs-dolsot", "돌솥비빔밥", 10000, "식사", "뜨거운 돌솥에 누룽지까지 즐기는 비빔밥"),
                ("hs-yukgaejang", "육개장", 10000, "식사", "소고기와 대파를 듬뿍 넣은 얼큰한 육개장"),
                ("hs-galbitang", "갈비탕", 13000, "식사", "소갈비를 오래 고아 낸 맑고 진한 갈비탕"),
                ("hs-kimchijeon", "김치전", 12000, "일품", "바삭하게 부친 묵은지 김치전"),
                ("hs-pajeon", "해물파전", 15000, "일품", "쪽파와 오징어·새우를 듬뿍 넣은 해물파전"),
                ("hs-gyeranmari", "계란말이", 8000, "사이드", "채소를 넣어 도톰하게 만 계란말이"),
                ("hs-egg", "계란후라이", 1500, "사이드", "반숙 계란후라이 하나"),
                ("hs-sikhye", "식혜", 3000, "음료", "직접 만든 살얼음 식혜"),
                ("hs-makgeolli", "막걸리", 5000, "주류", "제주 생막걸리 한 병"),
                ("hs-makguksu", "들기름 막국수", 9000, "식사", "강원 메밀면에 들기름과 김가루를 듬뿍, 새로 나온 여름 메뉴")]
               + _s("staple-rice", "staple-cola", "staple-cider", "staple-soju", "staple-beer"),
               mains=("찌개", "정식", "식사"),
               pairs=[("hs-kimchi", "staple-rice"), ("hs-doenjang", "staple-rice"), ("hs-sundubu", "staple-rice"),
                      ("hs-budae", "staple-rice"), ("hs-jeyuk", "hs-egg"), ("hs-pajeon", "hs-makgeolli"),
                      ("hs-kimchijeon", "hs-makgeolli"), ("hs-galchi", "staple-soju")],
               new_item="hs-makguksu", members=80, per_day=(20, 30)),
    Restaurant("bbq", "돌담 흑돼지", "234-56-78901", "제주 흑돼지를 직접 숙성해 숯불에 구워 드려요.", "16:00~23:00", 14,
               [("bb-ogyeop", "흑돼지 오겹살 (200g)", 18000, "고기", "껍질째 두툼하게 썬 제주 흑돼지 오겹살"),
                ("bb-moksal", "흑돼지 목살 (200g)", 17000, "고기", "육즙 가득한 흑돼지 목살"),
                ("bb-hangjeong", "항정살 (150g)", 16000, "고기", "꼬들꼬들한 식감의 흑돼지 항정살"),
                ("bb-gabri", "가브리살 (150g)", 16000, "고기", "마블링 좋은 흑돼지 가브리살"),
                ("bb-yangnyeom", "양념갈비 (300g)", 16000, "고기", "과일로 재운 달콤한 돼지 양념갈비"),
                ("bb-skin", "돼지껍데기", 9000, "고기", "쫄깃하게 구워 콩가루에 찍어 먹는 껍데기"),
                ("bb-doenjang", "고깃집 된장찌개", 5000, "식사", "차돌을 넣은 진한 된장찌개"),
                ("bb-kimchi", "묵은지 김치찌개", 6000, "식사", "고기 굽고 남은 기름에 끓인 김치찌개"),
                ("bb-naengmyeon", "물냉면", 8000, "식사", "새콤한 육수의 시원한 물냉면"),
                ("bb-bibimnaeng", "비빔냉면", 8000, "식사", "매콤달콤 양념의 비빔냉면"),
                ("bb-fried-rice", "볶음밥", 3000, "식사", "불판에 김치와 볶아 먹는 볶음밥"),
                ("bb-gyeranjjim", "계란찜", 4000, "사이드", "뚝배기에 폭신하게 부풀린 계란찜"),
                ("bb-ssam", "쌈 추가", 2000, "사이드", "상추·깻잎·명이나물 추가"),
                ("bb-highball", "하이볼", 7000, "주류", "레몬을 넣은 위스키 하이볼"),
                ("bb-shoulder", "흑돼지 숄더랙 (300g)", 22000, "고기", "뼈째 굽는 흑돼지 어깨 갈비, 이달의 신메뉴")]
               + _s("staple-rice", "staple-cola", "staple-cider", "staple-soju", "staple-beer"),
               mains=("고기",),
               pairs=[("bb-ogyeop", "bb-doenjang"), ("bb-moksal", "bb-fried-rice"), ("bb-ogyeop", "staple-soju"),
                      ("bb-moksal", "staple-beer"), ("bb-hangjeong", "bb-naengmyeon"), ("bb-yangnyeom", "staple-rice"),
                      ("bb-skin", "staple-soju")],
               new_item="bb-shoulder", members=65, per_day=(14, 22), lunch_share=0.0, takeout_share=0.05,
               slots=((17.0, 18.5), (18.5, 21.8))),
    Restaurant("bunsik", "한라 분식", "345-67-89012", "학교 앞 그 맛, 김밥은 주문 즉시 말아요.", "10:00~20:00", 8,
               [("bs-gimbap", "기본 김밥", 4000, "김밥", "햄·계란·단무지·시금치를 넣은 기본 김밥"),
                ("bs-tuna", "참치 김밥", 5000, "김밥", "참치마요가 듬뿍 들어간 김밥"),
                ("bs-cheese", "치즈 김밥", 5000, "김밥", "슬라이스 치즈를 두 장 넣은 고소한 김밥"),
                ("bs-tteokbokki", "떡볶이", 5000, "떡볶이", "쌀떡에 어묵을 넣은 매콤달콤 떡볶이"),
                ("bs-rose", "로제 떡볶이", 7000, "떡볶이", "크림과 고추장을 섞은 부드러운 로제 떡볶이"),
                ("bs-sundae", "순대", 5000, "사이드", "찹쌀순대와 간·허파 모둠"),
                ("bs-twigim", "튀김 모둠", 5000, "사이드", "김말이·오징어·고구마 튀김 모둠"),
                ("bs-ramyeon", "라면", 5000, "면", "계란과 파를 넣은 얼큰한 라면"),
                ("bs-cheese-ramyeon", "치즈 라면", 6000, "면", "치즈를 올린 라면"),
                ("bs-jjolmyeon", "쫄면", 7000, "면", "채소를 듬뿍 올린 새콤달콤 쫄면"),
                ("bs-udon", "우동", 6000, "면", "가쓰오 국물의 따끈한 우동"),
                ("bs-donkatsu", "돈까스", 9000, "식사", "바삭하게 튀긴 수제 등심 돈까스"),
                ("bs-omurice", "오므라이스", 8500, "식사", "케첩볶음밥을 부드러운 계란으로 감싼 오므라이스"),
                ("bs-kimchi-rice", "김치볶음밥", 7500, "식사", "계란후라이를 올린 김치볶음밥"),
                ("bs-eomuktang", "어묵탕", 6000, "사이드", "꼬치 어묵과 무를 넣은 어묵탕"),
                ("bs-coolpis", "쿨피스", 1500, "음료", "떡볶이와 찰떡궁합 쿨피스"),
                ("bs-mala", "마라 떡볶이", 7500, "떡볶이", "얼얼한 마라 소스로 볶은 떡볶이, 신메뉴")]
               + _s("staple-cola", "staple-cider"),
               mains=("김밥", "떡볶이", "면", "식사"),
               pairs=[("bs-tteokbokki", "bs-gimbap"), ("bs-tteokbokki", "bs-twigim"), ("bs-tteokbokki", "bs-sundae"),
                      ("bs-ramyeon", "bs-gimbap"), ("bs-rose", "bs-cheese"), ("bs-tteokbokki", "bs-coolpis")],
               new_item="bs-mala", members=70, per_day=(22, 34), lunch_share=0.55, takeout_share=0.45,
               slots=((11.3, 14.0), (15.5, 19.5))),
    Restaurant("cafe", "오름 카페 & 브런치", "456-78-90123", "오름이 보이는 창가에서 브런치와 스페셜티 커피.", "09:00~19:00", 10,
               [("cf-americano", "아메리카노", 4500, "커피", "산미가 산뜻한 오늘의 원두 아메리카노"),
                ("cf-latte", "카페라떼", 5000, "커피", "고소한 우유와 에스프레소"),
                ("cf-vanilla", "바닐라 라떼", 5500, "커피", "바닐라빈 시럽을 넣은 달콤한 라떼"),
                ("cf-coldbrew", "콜드브루", 5500, "커피", "18시간 우려낸 부드러운 콜드브루"),
                ("cf-matcha", "제주 말차 라떼", 6000, "음료", "제주 유기농 말차로 만든 라떼"),
                ("cf-gyul-ade", "감귤 에이드", 6500, "음료", "제주 감귤청에 탄산을 더한 에이드"),
                ("cf-hallabong-tea", "한라봉 티", 6000, "음료", "직접 담근 한라봉청 차"),
                ("cf-croffle", "크로플", 6000, "디저트", "버터 크루아상 생지를 와플로 구워 아이스크림 한 스쿱"),
                ("cf-croissant", "버터 크루아상", 4000, "베이커리", "프랑스 버터로 겹겹이 구운 크루아상"),
                ("cf-saltbread", "소금빵", 3500, "베이커리", "버터 풍미 가득한 소금빵"),
                ("cf-carrot-cake", "당근 케이크", 6500, "디저트", "크림치즈 프로스팅 당근 케이크"),
                ("cf-basque", "바스크 치즈케이크", 7000, "디저트", "겉을 그을린 진한 치즈케이크"),
                ("cf-tiramisu", "티라미수", 6500, "디저트", "마스카포네 크림 티라미수"),
                ("cf-ricotta", "리코타 샐러드", 11000, "브런치", "수제 리코타 치즈와 제철 채소 샐러드"),
                ("cf-benedict", "에그 베네딕트", 14000, "브런치", "수란과 홀랜다이즈 소스를 올린 잉글리시 머핀"),
                ("cf-french-toast", "프렌치토스트", 12000, "브런치", "브리오슈로 만든 촉촉한 프렌치토스트와 베리"),
                ("cf-avocado", "아보카도 샌드위치", 12000, "브런치", "아보카도와 수란을 올린 오픈 샌드위치"),
                ("cf-fig-bagel", "무화과 크림치즈 베이글", 6500, "베이커리", "무화과 잼과 크림치즈를 듬뿍 바른 베이글, 신메뉴")],
               mains=("커피", "음료", "브런치"),
               pairs=[("cf-americano", "cf-croffle"), ("cf-latte", "cf-saltbread"), ("cf-americano", "cf-basque"),
                      ("cf-benedict", "cf-americano"), ("cf-french-toast", "cf-latte"), ("cf-vanilla", "cf-carrot-cake")],
               new_item="cf-fig-bagel", members=85, per_day=(26, 38), lunch_share=0.5, takeout_share=0.35,
               slots=((9.5, 12.5), (13.0, 17.5))),
    Restaurant("chinese", "바다향 중화요리", "567-89-01234", "새벽에 받은 해산물로 끓이는 짬뽕이 자랑이에요.", "11:00~21:30", 12,
               [("cn-jjajang", "짜장면", 7000, "면", "춘장을 볶아 만든 정통 짜장면"),
                ("cn-ganjjajang", "간짜장", 8000, "면", "주문 즉시 볶는 고소한 간짜장"),
                ("cn-jjamppong", "짬뽕", 8500, "면", "홍합·오징어를 넣은 얼큰한 해물 짬뽕"),
                ("cn-samseon", "삼선짬뽕", 11000, "면", "새우·관자·전복까지 넣은 삼선짬뽕"),
                ("cn-fried-rice", "볶음밥", 8000, "밥", "고슬고슬 볶은 계란 볶음밥과 짜장 소스"),
                ("cn-japchae-rice", "잡채밥", 9000, "밥", "고추잡채를 올린 덮밥"),
                ("cn-mapa", "마파두부밥", 9000, "밥", "얼얼한 산초 향의 마파두부 덮밥"),
                ("cn-tangsuyuk-s", "탕수육 (소)", 16000, "요리", "바삭한 찹쌀 탕수육과 새콤달콤 소스"),
                ("cn-tangsuyuk-m", "탕수육 (중)", 22000, "요리", "여럿이 나눠 먹는 찹쌀 탕수육"),
                ("cn-kkanpung", "깐풍기", 24000, "요리", "매콤달콤 마늘 소스의 닭튀김"),
                ("cn-shrimp", "깐쇼새우", 26000, "요리", "칠리 소스에 볶은 큼직한 새우"),
                ("cn-yuringi", "유린기", 22000, "요리", "간장 소스를 끼얹은 바삭한 닭튀김"),
                ("cn-gunmandu", "군만두", 6000, "요리", "직접 빚어 바삭하게 구운 군만두"),
                ("cn-goguma", "고추잡채", 20000, "요리", "꽃빵과 함께 먹는 고추잡채"),
                ("cn-kaoliang", "고량주", 12000, "주류", "향이 진한 연태 고량주"),
                ("cn-tsingtao", "칭따오", 6000, "주류", "중식과 어울리는 칭따오 맥주"),
                ("cn-chadol", "차돌 짬뽕", 10000, "면", "구운 차돌박이를 듬뿍 올린 짬뽕, 신메뉴")]
               + _s("staple-cola", "staple-cider"),
               mains=("면", "밥"),
               pairs=[("cn-jjajang", "cn-tangsuyuk-s"), ("cn-jjamppong", "cn-gunmandu"), ("cn-jjajang", "cn-jjamppong"),
                      ("cn-tangsuyuk-m", "cn-kaoliang"), ("cn-kkanpung", "cn-tsingtao"), ("cn-fried-rice", "cn-jjamppong")],
               new_item="cn-chadol", members=70, per_day=(20, 30), takeout_share=0.25),
    Restaurant("salad", "그린볼 샐러드", "678-90-12345", "제주 채소로 만드는 포케와 샐러드, 이번 달에 문 열었어요.", "10:30~20:00", 8,
               [("gb-caesar", "시저 샐러드", 9900, "샐러드", "로메인과 파르메산, 수제 크루통의 시저 샐러드"),
                ("gb-salmon-poke", "연어 포케", 12900, "포케", "생연어와 아보카도, 현미밥 포케"),
                ("gb-chicken-poke", "닭가슴살 포케", 11900, "포케", "수비드 닭가슴살과 퀴노아 포케"),
                ("gb-tofu-poke", "두부 포케", 11500, "포케", "구운 두부와 병아리콩의 비건 포케"),
                ("gb-ricotta", "리코타 볼", 10900, "샐러드", "리코타 치즈와 견과류, 베리 샐러드"),
                ("gb-greek", "그릭요거트 볼", 7900, "샐러드", "꾸덕한 그릭요거트에 그래놀라와 꿀"),
                ("gb-pasta", "콜드 파스타", 10900, "샐러드", "바질 페스토로 버무린 차가운 파스타 샐러드"),
                ("gb-wrap", "비건 랩", 8900, "랩", "후무스와 구운 채소를 말은 통밀 랩"),
                ("gb-soup", "오늘의 수프", 5500, "사이드", "매일 바뀌는 채소 수프"),
                ("gb-kombucha", "콤부차", 4900, "음료", "유기농 콤부차"),
                ("gb-smoothie", "그린 스무디", 6500, "음료", "케일·바나나·사과 스무디"),
                ("gb-icetea", "아이스티", 3900, "음료", "직접 우린 복숭아 아이스티"),
                ("gb-sparkling", "탄산수", 2500, "음료", "레몬 탄산수")],
               mains=("샐러드", "포케", "랩"),
               pairs=[("gb-salmon-poke", "gb-kombucha"), ("gb-caesar", "gb-soup"), ("gb-chicken-poke", "gb-smoothie")],
               members=40, per_day=(10, 18), lunch_share=0.7, takeout_share=0.5, history_days=30),
]


def store_index(i: int) -> Restaurant:
    return STORES[(i - 1) % len(STORES)]


def _iso(t: dt.datetime) -> str:
    return t.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


@dataclass
class Member:
    phone: str
    gap: float
    next_visit: float           # days ago
    section: str
    favourites: list[str]
    party: int
    dinner: bool


def _members(store: Restaurant, rng: random.Random) -> list[Member]:
    by_section: dict[str, list[str]] = {}
    for item_id, _t, _p, section, _d in store.menu:
        if item_id != store.new_item:
            by_section.setdefault(section, []).append(item_id)
    mains = [s for s in store.mains if s in by_section]
    out = []
    for n in range(store.members):
        gap = rng.choice([2, 3, 4, 5, 7, 7, 10, 14])
        section = rng.choice(mains)
        favs = rng.sample(by_section[section], k=min(2, len(by_section[section])))
        other = rng.choice(mains)
        favs += rng.sample(by_section[other], k=1)
        out.append(Member(phone="010%04d%04d" % (rng.randint(1000, 9999), rng.randint(1000, 9999)), gap=gap,
                          next_visit=rng.uniform(1, store.history_days), section=section, favourites=favs,
                          party=rng.choice([1, 1, 2, 2, 2, 3, 4]), dinner=rng.random() > store.lunch_share))
    return out


def _basket(store: Restaurant, rng: random.Random, guests: int, member: Optional[Member], available: set[str],
            popularity: dict[str, float]) -> list[tuple[str, int]]:
    menu = {i[0]: i for i in store.menu if i[0] in available}
    mains = [i for i, row in menu.items() if row[3] in store.mains]
    sides = [i for i, row in menu.items() if row[3] not in store.mains]
    lines: dict[str, int] = {}
    for g in range(guests):
        if member and g == 0 and rng.random() < 0.8:
            pick = rng.choice([f for f in member.favourites if f in menu] or mains)
        elif member and rng.random() < 0.35:
            pick = rng.choice([f for f in member.favourites if f in menu] or mains)
        else:
            pick = rng.choices(mains, weights=[popularity.get(i, 1.0) for i in mains])[0]
        lines[pick] = lines.get(pick, 0) + 1
    for a, b in store.pairs:
        if a in lines and b in menu and rng.random() < 0.65:
            lines[b] = lines.get(b, 0) + (lines[a] if menu[b][3] in ("사이드", "식사") and menu[b][2] <= 1500 else 1)
    if sides and rng.random() < 0.45:
        extra = rng.choices(sides, weights=[popularity.get(i, 1.0) for i in sides])[0]
        lines[extra] = lines.get(extra, 0) + 1
    if store.new_item in menu and rng.random() < 0.1:
        lines[store.new_item] = lines.get(store.new_item, 0) + 1
    return sorted(lines.items())


def seed_store(conn, seller_id: str, store: Restaurant, now: Optional[dt.datetime] = None) -> dict[str, int]:
    now = (now or DEFAULT_NOW).replace(microsecond=0)
    for ensure in (orders_db.ensure_schema, social_db.ensure_schema, accounts_db.ensure_schema, fulfillment.ensure_schema,
                   pos_db.ensure_schema):
        ensure(conn)
    added = {"tickets": 0, "open": 0}
    if accounts_db.fetch_seller_account(conn, seller_id, OWNER) is None:
        accounts_service.signup_seller(conn, seller_id=seller_id, username=OWNER, display_name=store.name, password=PASSWORD,
                                       business_reg_no=store.reg_no, business_open_date="20190301", business_rep_name="데모")
    pos.settings(conn, seller_id)
    with conn:  # a fixed salt so the generated member ids repeat (a real store's is random)
        conn.execute("UPDATE pos_settings SET tables = ?, point_rate = 5, open_hours = ?, member_salt = ? WHERE seller_id = ?",
                     (store.tables, store.hours, hashlib.sha256(("salt/" + store.key).encode()).hexdigest()[:32], seller_id))
        social_db.save_store_settings(conn, seller_id, 10, 3, store.intro)
    start = now - dt.timedelta(days=store.history_days)
    new_from = now - dt.timedelta(days=store.history_days * 0.3)
    for item_id, title, price, section, desc in store.menu:
        if orders_service.get_catalog_item_for_display(conn, seller_id=seller_id, item_id_local=item_id) is None:
            orders_service.register_catalog_item(conn, seller_id=seller_id, item_id_local=item_id, title_text=title,
                                                 description_text=desc, category_path=["식당", section], display_price_minor=price)
        listed = _iso(new_from if item_id == store.new_item else start - dt.timedelta(days=1))
        _backdate_listing(conn, seller_id, item_id, listed)

    rng = random.Random("restaurant-" + store.key)
    members = _members(store, rng)
    popularity = {i[0]: rng.uniform(0.4, 2.0) for i in store.menu}
    for a, _b in store.pairs:
        popularity[a] = popularity.get(a, 1.0) * 1.6
    existing = {r[0] for r in conn.execute("SELECT ticket_id FROM pos_tickets WHERE seller_id = ?", (seller_id,))}

    today_kst = now.astimezone(KST).date()
    for day_ago in range(store.history_days, -1, -1):
        day = today_kst - dt.timedelta(days=day_ago)
        drng = random.Random("%s/%s" % (store.key, day.isoformat()))
        weekend = day.weekday() >= 5
        n = drng.randint(*store.per_day) + (drng.randint(2, 6) if weekend else 0)
        visits: list[Optional[Member]] = [None] * n
        for m in members:  # members whose turn has come take a slot
            due = (now - dt.datetime.combine(day, dt.time(12), KST)).total_seconds() / 86400
            if m.next_visit >= due > m.next_visit - 1:
                visits[drng.randrange(n)] = m
                m.next_visit -= max(1.0, drng.gauss(m.gap, m.gap * 0.3))
        plan = []
        for k, member in enumerate(visits):
            dinner = member.dinner if member else drng.random() > store.lunch_share
            lo, hi = store.slots[1 if dinner else 0]
            plan.append((dt.datetime.combine(day, dt.time(0), KST) + dt.timedelta(hours=drng.uniform(lo, hi)), k, member))
        plan.sort(key=lambda p: (p[0], p[1]))  # the counter sees them in time order
        for opened, k, member in plan:
            ticket_id = pos._id(seller_id, "ticket", day.isoformat(), str(k))
            takeout = drng.random() < store.takeout_share
            guests = 1 if takeout and drng.random() < 0.6 else (member.party if member else drng.choice([1, 2, 2, 2, 3, 4, 5]))
            stay = dt.timedelta(minutes=drng.uniform(8, 15) if takeout else drng.uniform(30, 75))
            if opened >= now:
                continue  # later today
            open_now = opened + stay > now
            if ticket_id in existing:
                continue
            available = {i[0] for i in store.menu if not (i[0] == store.new_item and opened < new_from)}
            lines = _basket(store, drng, guests, member, available, popularity)
            table = None if takeout else (k % store.tables) + 1
            if table is not None:
                busy = {r[0] for r in conn.execute("SELECT table_no FROM pos_tickets WHERE seller_id = ? AND status = 'open' "
                                                   "AND table_no IS NOT NULL", (seller_id,))}
                free = [t for t in range(1, store.tables + 1) if t not in busy]
                table = table if table not in busy else (free[k % len(free)] if free else None)
            pos.open_ticket(conn, seller_id, table_no=table, guests=guests, at=_iso(opened), ticket_id=ticket_id)
            for j, (item, qty) in enumerate(lines):
                memo = drng.choice(pos.MEMOS) if drng.random() < 0.08 else None
                pos.add_line(conn, seller_id, ticket_id, item, qty, memo, at=_iso(opened + dt.timedelta(minutes=1)),
                             line_id=pos._id(ticket_id, "line", str(j)))
            if member:
                pos.attach_member(conn, seller_id, ticket_id, member.phone, at=_iso(opened))
            sent = opened + dt.timedelta(minutes=2)
            if sent < now:
                pos.send_to_kitchen(conn, seller_id, ticket_id, at=_iso(sent))
            served = opened + dt.timedelta(minutes=drng.uniform(8, 18))
            if open_now:
                if served < now:
                    with conn:
                        conn.execute("UPDATE pos_lines SET served_at = ? WHERE seller_id = ? AND ticket_id = ? AND sent_at IS NOT NULL",
                                     (_iso(served), seller_id, ticket_id))
                added["open"] += 1
                continue
            with conn:
                conn.execute("UPDATE pos_lines SET served_at = ? WHERE seller_id = ? AND ticket_id = ? AND sent_at IS NOT NULL",
                             (_iso(served), seller_id, ticket_id))
            paid_at = opened + stay
            v = pos.view(conn, seller_id, ticket_id)
            if drng.random() < 0.03:
                pos.set_discount(conn, seller_id, ticket_id, percent=10)
                v = pos.view(conn, seller_id, ticket_id)
            pay_n = 0
            if v["member"] and v["member"]["points"] >= 3000 and drng.random() < 0.25:
                use = min(v["member"]["points"] // 1000 * 1000, v["due"])
                if use:
                    pos.pay(conn, seller_id, ticket_id, method="points", amount=use, at=_iso(paid_at),
                            payment_id=pos._id(ticket_id, "pay", str(pay_n)))
                    pay_n += 1
            v = pos.view(conn, seller_id, ticket_id)
            if v["due"]:
                if guests >= 3 and drng.random() < 0.2:  # split the bill
                    part = (v["due"] // guests // 10) * 10
                    for g in range(guests - 1):
                        pos.pay(conn, seller_id, ticket_id, method="card", amount=part, at=_iso(paid_at),
                                payment_id=pos._id(ticket_id, "pay", str(pay_n)))
                        pay_n += 1
                    v = pos.view(conn, seller_id, ticket_id)
                method = drng.choices(["card", "easy", "cash", "transfer"], weights=[68, 20, 10, 2])[0]
                received = None
                if method == "cash":
                    received = -(-v["due"] // 10000) * 10000
                pos.pay(conn, seller_id, ticket_id, method=method, received=received, at=_iso(paid_at),
                        payment_id=pos._id(ticket_id, "pay", str(pay_n)))
            if drng.random() < 0.006:
                pos.refund(conn, seller_id, ticket_id, at=_iso(paid_at + dt.timedelta(minutes=10)))
            added["tickets"] += 1
    return added


def _backdate_listing(conn, seller_id: str, item_id: str, listed_at: str) -> None:
    row = conn.execute("SELECT payload_json FROM catalog_items WHERE seller_id = ? AND item_id_local = ?", (seller_id, item_id)).fetchone()
    pending = conn.execute("SELECT 1 FROM outbox WHERE seller_id = ? AND kind = 'catalog_item' AND ref_id = ? AND status = 'pending'",
                           (seller_id, item_id)).fetchone()
    payload = json.loads(row["payload_json"])
    if payload["first_listed_at"] <= listed_at or pending is None:
        return
    payload["first_listed_at"] = listed_at
    body = json.dumps(payload, ensure_ascii=False)
    with conn:
        conn.execute("UPDATE catalog_items SET payload_json = ? WHERE seller_id = ? AND item_id_local = ?", (body, seller_id, item_id))
        conn.execute("UPDATE outbox SET payload_json = ? WHERE seller_id = ? AND kind = 'catalog_item' AND ref_id = ? AND status = 'pending'",
                     (body, seller_id, item_id))


def summary(conn, seller_id: str) -> dict[str, Any]:
    q = lambda sql: conn.execute(sql, (seller_id,)).fetchone()[0]  # noqa: E731
    events = export_seller_input(conn, seller_id)["events"]
    member_events = sum(1 for e in events if e["customer_id_local"].startswith("m-"))
    return {
        "menu": q("SELECT COUNT(*) FROM catalog_items WHERE seller_id = ?"),
        "members": q("SELECT COUNT(*) FROM pos_members WHERE seller_id = ?"),
        "tickets_paid": q("SELECT COUNT(*) FROM pos_tickets WHERE seller_id = ? AND status = 'paid'"),
        "tickets_open": q("SELECT COUNT(*) FROM pos_tickets WHERE seller_id = ? AND status = 'open'"),
        "refunds": q("SELECT COUNT(*) FROM pos_tickets WHERE seller_id = ? AND status = 'refunded'"),
        "revenue": q("SELECT COALESCE(SUM(amount), 0) FROM pos_payments WHERE seller_id = ? AND refunded_at IS NULL"),
        "purchase_events": len(events), "member_events": member_events,
        "avg_lines": round(sum(len(e["items"]) for e in events) / len(events), 2) if events else 0,
    }


def _resolve_now(text: Optional[str]) -> dt.datetime:
    if not text:
        return DEFAULT_NOW
    if text in ("today", "now"):  # up to this minute: nothing in the future, tables still eating if it is meal time
        return dt.datetime.now(dt.timezone.utc).replace(second=0, microsecond=0)
    t = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    return t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc)


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Generate the synthetic restaurant POS dataset.")
    parser.add_argument("--stores", type=int, default=6, help="how many launcher sellers to fill (merchant-1..N)")
    parser.add_argument("--var", default=str(VAR), help="launcher var folder (default commerce/deploy/var)")
    parser.add_argument("--export", help="write each store's SellerInput JSON and a summary here")
    parser.add_argument("--now", help="end of the history: 'today' (now) or an ISO time (default %s)" % DEFAULT_NOW.isoformat())
    args = parser.parse_args(argv)
    now = _resolve_now(args.now)
    totals = []
    for i in range(1, args.stores + 1):
        seller_id, store = "merchant-%d" % i, store_index(i)
        folder = Path(args.var) / ("merchant_%d" % i)
        db_path, feature_db = folder / "orders.sqlite", folder / "features.sqlite"
        if not db_path.exists() and feature_db.exists():
            print("%s exists without orders.sqlite; delete the pair together." % feature_db, file=sys.stderr)
            raise SystemExit(1)
        conn = orders_db.connect(db_path)
        try:
            added = seed_store(conn, seller_id, store, now)
            stats = summary(conn, seller_id)
            if args.export:
                out = Path(args.export)
                out.mkdir(parents=True, exist_ok=True)
                (out / ("%s.json" % seller_id)).write_text(json.dumps(export_seller_input(conn, seller_id), ensure_ascii=False), encoding="utf-8")
        finally:
            conn.close()
        totals.append({"seller_id": seller_id, "store": store.name, "key": store.key, **stats})
        print("%s %s: +%d checks (open %d) · menu %d · members %d · paid %d · revenue %s원"
              % (seller_id, store.name, added["tickets"], added["open"], stats["menu"], stats["members"], stats["tickets_paid"],
                 format(stats["revenue"], ",")))
    if args.export:
        Path(args.export, "summary.json").write_text(json.dumps(totals, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
