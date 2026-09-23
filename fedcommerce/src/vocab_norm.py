"""공유 토큰 어휘 정규화 — 약어 전개 · 오타 · 동의어 · 상위어 추가.

목적: 신규 가맹점이 합류하자마자 'milk', 'bread' 를 아는 모델을 받게 하는 것 (cold start).
GCI 는 클라이언트마다 어휘가 독립(368/372/19113/18590)이라 이걸 못 했다. 우리는 공유 토큰으로 간다.

원칙
  같은 것은 같게  : 진짜 대응만 잇는다 (SHELF STABLE = canned, HISPANIC = latino)
  다른 것은 그대로: 억지로 맞추지 않는다 (MAGAZINE, CIGARETTES 등은 매칭 없음이 정답)
  대체가 아니라 추가: BEEF -> {beef, meat}. 회사 안에서 beef/pork 구분은 살리고 회사 간엔 meat 로 잇는다.

사전은 양쪽 라벨 전체(Instacart aisle 134 + dunnhumby category 302)를 직접 보고 작성했다.
"""
import re, json, os
from collections import Counter
import pandas as pd
import pyreadr

# ── 1. 약어 · 오타 → 정식 표기 (대체) ────────────────────────────────────────
ABBREV = {
    "frzn": "frozen", "frz": "frozen", "refrgratd": "refrigerated", "rfrgrtd": "refrigerated",
    "refrig": "refrigerated", "veg": "vegetable", "fd": "food", "fds": "food",
    "drnks": "drink", "drnk": "drink", "bkd": "baked", "bkry": "bakery",
    "sherbts": "sherbet", "dsh": "dish", "whlsm": "wholesome", "brkfst": "breakfast",
    "bn": "bean", "sndwch": "sandwich", "sald": "salad", "sprd": "spread",
    "drsng": "dressing", "flvrd": "flavored", "flvr": "flavor", "pwdr": "powder",
    "crystl": "crystal", "mx": "mix", "mxs": "mix", "pkg": "packaged", "pckgd": "packaged",
    "cnd": "canned", "chkn": "chicken", "bev": "beverage", "bvrg": "beverage",
    "juc": "juice", "mw": "microwave", "asstd": "assorted", "sgl": "single",
    "prep": "prepared", "cond": "condiment", "lunchmeat": "lunch meat",
    "pnt": "peanut", "btr": "butter", "trsh": "trash", "sply": "supply",
    "sply": "supply", "wtr": "water", "cleang": "cleaning", "sweetners": "sweetener",
    "hsehld": "household", "hshld": "household", "clng": "cleaning", "ntrl": "natural",
    "sup": "supply", "cont": "container", "svc": "service", "gro": "grocery",
}

# ── 2. 동의어 → 표준어 (대체). 양쪽 taxonomy 가 같은 것을 다르게 부르는 경우만 ──
SYNONYM = {
    "hispanic": "latino", "isotonic": "sports", "novelty": "dessert",
    "novelties": "dessert", "margarine": "butter", "tissue": "paper",
    "towel": "paper", "noodle": "pasta", "pop": "soft drink", "soda": "soft drink",
    "sherbet": "ice cream", "poultry": "chicken", "toiletries": "personal care",
    "hbc": "health", "detergent": "cleaning", "housewares": "kitchen",
}

# ── 3. 구(phrase) 치환 — 토큰 분해 전에 먼저 적용 ──────────────────────────────
PHRASE = {
    "shelf stable": "canned",      # VEGETABLES - SHELF STABLE = 통조림 채소
    "by products": "product",
    "greeting cards": "card",
}

# ── 4. 상위어 추가 (원 토큰 유지 + 부모 추가) ─────────────────────────────────
#     dunnhumby 가 세분화된 곳을 Instacart 의 거친 분류에 잇는다
HYPERNYM = {
    # 과일 → fruit  (Instacart: fresh fruits / packaged vegetables fruits)
    "apple": "fruit", "grape": "fruit", "berry": "fruit", "berries": "fruit",
    "citrus": "fruit", "banana": "fruit", "melon": "fruit", "peach": "fruit",
    "pear": "fruit", "tropical": "fruit", "stone": "fruit",
    # 채소 → vegetable  (Instacart: fresh vegetables)
    "onion": "vegetable", "tomato": "vegetable", "potato": "vegetable",
    "pepper": "vegetable", "lettuce": "vegetable", "carrot": "vegetable",
    "corn": "vegetable", "mushroom": "vegetable", "broccoli": "vegetable",
    "celery": "vegetable", "cucumber": "vegetable", "squash": "vegetable",
    # 육류 → meat  (Instacart: meat counter / packaged meat)
    "beef": "meat", "pork": "meat", "bacon": "meat", "sausage": "meat",
    "ham": "meat", "turkey": "meat", "veal": "meat", "lamb": "meat",
    # 식사 → meal  (Instacart: frozen meals / prepared meals)
    "dinner": "meal", "entree": "meal", "lunch": "meal",
    # 제과 → bakery / dessert
    "baked": "bakery", "cake": "dessert", "cookie": "dessert", "pastry": "dessert",
    "pie": "dessert", "sweet": "dessert", "donut": "dessert",
    # 기타
    "ale": "beer", "wine": "alcohol", "beer": "alcohol", "spirit": "alcohol",
    "cereal": "breakfast", "diaper": "baby", "formula": "baby",
}

# ── 5. 의미 없는 수식어 · 운영 용어 (제거) ────────────────────────────────────
DROP = {
    "and", "the", "of", "with", "in", "other", "all", "by", "or", "for", "to", "no", "non", "n",
    "misc", "miscellaneous", "fluid", "checklane", "checkout", "coupon", "item", "items",
    "wholesome", "convenient", "assorted", "needs", "need", "type", "size", "regular",
    "bag", "bags", "box", "jar", "glass", "single", "serve", "value", "family",
}


def _sing(t, vocab):
    """단순 복수형: 단수형이 어휘에 있으면 단수로."""
    if len(t) > 3 and t.endswith("ies") and t[:-3] + "y" in vocab:
        return t[:-3] + "y"
    if len(t) > 3 and t.endswith("es") and t[:-2] in vocab:
        return t[:-2]
    if len(t) > 2 and t.endswith("s") and not t.endswith("ss") and t[:-1] in vocab:
        return t[:-1]
    return t


def normalize(label, vocab=None):
    s = re.sub(r"[^a-z0-9 ]", " ", str(label).lower())
    s = re.sub(r"\s+", " ", s).strip()
    for ph, rep in PHRASE.items():                      # 구 치환 먼저
        s = s.replace(ph, rep)
    out = set()
    for w in s.split():
        if w.isdigit() or len(w) < 2:
            continue
        for t in ABBREV.get(w, w).split():              # 약어 전개 (다중어 가능)
            t = SYNONYM.get(t, t)
            for u in t.split():
                if vocab:
                    u = _sing(u, vocab)
                u = SYNONYM.get(u, u)
                if u in DROP:
                    continue
                out.add(u)
                if u in HYPERNYM:                       # 상위어는 '추가'
                    out.add(HYPERNYM[u])
    return out


if __name__ == "__main__":
    os.makedirs("out", exist_ok=True)
    pr = pyreadr.read_r("data/products.rda")["products"]; pr.columns = [c.lower() for c in pr.columns]
    tx = pyreadr.read_r("data/transactions.rds")[None]; tx.columns = [c.lower() for c in tx.columns]
    tx = tx.merge(pr[["product_id", "product_category"]], on="product_id", how="left")
    freq = tx[tx.product_category.notna()].product_category.value_counts()
    freq = freq[freq.index != "COUPON/MISC ITEMS"]
    ic_labels = list(pd.read_csv("data/instacart/aisles.csv").aisle) + \
                list(pd.read_csv("data/instacart/departments.csv").department)

    raw = lambda s, v=None: {t for t in re.sub(r"[^a-z0-9 ]", " ", str(s).lower()).split()
                             if len(t) > 1 and not t.isdigit()}

    def cov(fn, vocab=None):
        ic = set().union(*[fn(s, vocab) for s in ic_labels])
        w = Counter()
        for c, n in freq.items():
            for t in fn(c, vocab):
                w[t] += n
        dh = set(w)
        tokcov = sum(n for t, n in w.items() if t in ic) / sum(w.values())
        labcov = sum(1 for c in freq.index if fn(c, vocab) & ic) / len(freq)
        return tokcov, labcov, ic, dh, w

    v0, l0, *_ = cov(raw)
    _, _, ic1, dh1, _ = cov(normalize)                   # 1차로 어휘 수집
    vocab = ic1 | dh1
    v1, l1, ic2, dh2, w2 = cov(normalize, vocab)

    print("=" * 72)
    print(f"{'':30} {'토큰 커버리지':>13} {'라벨 커버리지':>13}")
    print(f"{'정규화 없음':<28} {v0*100:>12.1f}% {l0*100:>12.1f}%")
    print(f"{'사전 적용':<29} {v1*100:>12.1f}% {l1*100:>12.1f}%")
    print(f"{'개선':<30} {(v1-v0)*100:>+12.1f}p {(l1-l0)*100:>+12.1f}p")
    print("=" * 72)
    shared = ic2 & dh2
    print(f"공유 토큰 {len(shared)}   (dunnhumby {len(dh2)} / Instacart {len(ic2)})")
    print(f"\n공유 토큰 (거래량 상위 30):")
    print("  " + ", ".join([t for t, _ in w2.most_common() if t in shared][:30]))
    print(f"\n남은 미매칭 상위 15 — 대응이 없는 게 정답인 것들:")
    for t, n in [(t, n) for t, n in w2.most_common() if t not in ic2][:15]:
        print(f"  {t:<16} {n:>9,}")

    json.dump({"abbrev": ABBREV, "synonym": SYNONYM, "phrase": PHRASE,
               "hypernym": HYPERNYM, "drop": sorted(DROP),
               "coverage_token": v1, "coverage_label": l1,
               "shared_tokens": sorted(shared),
               "dunnhumby_tokens": sorted(dh2), "instacart_tokens": sorted(ic2)},
              open("out/vocab_norm.json", "w"), ensure_ascii=False, indent=1)
    print("\n-> out/vocab_norm.json")
