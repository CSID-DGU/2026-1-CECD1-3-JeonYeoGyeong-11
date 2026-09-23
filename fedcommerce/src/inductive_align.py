"""Option A (canonical relational stats) 의 cross-merchant 정렬 검증.

가맹점 A 의 품목 c 를 관계 통계만으로 벡터화하고, 가맹점 B 의 품목들 중 가장 가까운 것을 찾는다.
dunnhumby 는 숨은 정답(product_category)이 있으므로 "같은 카테고리를 찾았는가" 로 채점할 수 있다.
비교: popularity 만(PrepRec 급) vs 관계 통계 전체.
"""
import json, itertools
import numpy as np
import pandas as pd
import pyreadr
from collections import defaultdict

rng = np.random.default_rng(0)
tx = pyreadr.read_r("data/transactions.rds")[None]; tx.columns = [c.lower() for c in tx.columns]
pr = pyreadr.read_r("data/products.rda")["products"]; pr.columns = [c.lower() for c in pr.columns]
tx = tx.merge(pr[["product_id", "product_category"]], on="product_id", how="left")
tx = tx[tx.product_category.notna() & (tx.quantity > 0) & (tx.product_category != "COUPON/MISC ITEMS")]
tx["ts"] = pd.to_datetime(tx.transaction_timestamp)
hs = tx.groupby(["household_id", "store_id"]).basket_id.nunique().rename("b").reset_index()
prim = (hs.sort_values("b", ascending=False).groupby("household_id").head(1)
          [["household_id", "store_id"]].rename(columns={"store_id": "ps"}))
dh = tx.merge(prim, on="household_id"); dh = dh[dh.store_id == dh.ps]
bk = dh.groupby("store_id").basket_id.nunique()
clients = list(bk.nlargest(30).index)                 # 큰 매장 30개 (통계 안정)
dh = dh[dh.store_id.isin(clients)]


def relational_features(d):
    """한 가맹점 내부 데이터만으로 품목별 canonical 통계. 이름·전역 정보 없음."""
    cats = sorted(d.product_category.unique()); idx = {c: i for i, c in enumerate(cats)}; n = len(cats)
    nb = d.basket_id.nunique(); nh = d.household_id.nunique()
    g = d.groupby("product_category")
    f = pd.DataFrame(index=cats)
    f["penet"] = g.basket_id.nunique() / nb
    f["reach"] = g.household_id.nunique() / nh
    f["repeat"] = g.basket_id.nunique() / g.household_id.nunique()
    f["qty"] = g.quantity.mean()
    # 바스켓 동시구매 행렬  B^T B
    bsets = d.groupby("basket_id").product_category.apply(lambda s: sorted(set(s)))
    Cb = np.zeros((n, n))
    for s in bsets:
        ii = [idx[c] for c in s]
        for a, b in itertools.combinations(ii, 2):
            Cb[a, b] += 1; Cb[b, a] += 1
    cnt = np.array([f.penet[c] * nb for c in cats])
    Pb = Cb / nb; pi = cnt / nb
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi_b = np.log(Pb / np.outer(pi, pi)); pmi_b[~np.isfinite(pmi_b)] = 0
    f["bdeg"] = (Cb > 0).sum(1) / n
    f["bpmi_mean"] = np.where((Cb > 0).sum(1) > 0, (pmi_b * (Cb > 0)).sum(1) / np.maximum((Cb > 0).sum(1), 1), 0)
    f["bpmi_max"] = pmi_b.max(1)
    # 고객 동시구매  X^T X
    hsets = d.groupby("household_id").product_category.apply(lambda s: sorted(set(s)))
    Cx = np.zeros((n, n))
    for s in hsets:
        ii = [idx[c] for c in s]
        for a, b in itertools.combinations(ii, 2):
            Cx[a, b] += 1; Cx[b, a] += 1
    Px = Cx / nh; ph = np.array([f.reach[c] for c in cats])
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi_x = np.log(Px / np.outer(ph, ph)); pmi_x[~np.isfinite(pmi_x)] = 0
    f["xdeg"] = (Cx > 0).sum(1) / n
    f["xpmi_mean"] = np.where((Cx > 0).sum(1) > 0, (pmi_x * (Cx > 0)).sum(1) / np.maximum((Cx > 0).sum(1), 1), 0)
    f["subst"] = f["xpmi_mean"] - f["bpmi_mean"]          # 대체재 신호: 고객 공유 高, 바스켓 공유 低
    # 재구매 주기 (시간 관계)
    gaps = defaultdict(list)
    v = d.groupby(["household_id", "basket_id"]).agg(ts=("ts", "min"),
                                                     cs=("product_category", lambda s: set(s))).reset_index()
    v = v.sort_values(["household_id", "ts"])
    for _, gh in v.groupby("household_id"):
        last = {}
        for t, cs in zip(gh.ts.values, gh.cs):
            for c in cs:
                if c in last: gaps[c].append((t - last[c]) / np.timedelta64(1, "D"))
                last[c] = t
    f["gap_med"] = [np.median(gaps[c]) if gaps[c] else np.nan for c in cats]
    f["gap_med"] = f.gap_med.fillna(f.gap_med.max())
    # 바스켓 크기 기여: 그 품목이 있는 바스켓의 평균 품목 수
    bsize = d.groupby("basket_id").product_category.nunique()
    d2 = d[["basket_id", "product_category"]].drop_duplicates().merge(bsize.rename("bs"), left_on="basket_id", right_index=True)
    f["bsize"] = d2.groupby("product_category").bs.mean()
    # 가맹점 내 분위수로 canonical 화
    return f.rank(pct=True)


POP = ["penet", "reach", "repeat"]                                   # PrepRec 급
REL = POP + ["bdeg", "bpmi_mean", "bpmi_max", "xdeg", "xpmi_mean", "subst", "gap_med", "bsize", "qty"]

F = {s: relational_features(dh[dh.store_id == s]) for s in clients}
print(f"client {len(clients)}  품목/가맹점 중앙 {np.median([len(f) for f in F.values()]):.0f}  특징 {len(REL)}개\n")


def retrieval(cols, pairs=200):
    """A 의 각 품목 → B 에서 최근접 → 같은 카테고리인가"""
    hit1, hit5, rand, n = 0, 0, 0, 0
    for a, b in rng.choice(len(clients), (pairs, 2)):
        if a == b: continue
        fa, fb = F[clients[a]][cols], F[clients[b]][cols]
        common = fa.index.intersection(fb.index)
        if len(common) < 50: continue
        A = fa.loc[common].values; B = fb.values; bidx = list(fb.index)
        D = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
        order = np.argsort(D, 1)
        for i, c in enumerate(common):
            top = [bidx[j] for j in order[i, :5]]
            hit1 += top[0] == c; hit5 += c in top; n += 1
        rand += len(common) * (1 / len(bidx))
    return hit1 / n, hit5 / n, rand / n


print(f"{'특징 집합':<26} {'top-1':>7} {'top-5':>7} {'random':>8} {'top-1 배율':>10}")
for name, cols in [("popularity 만 (PrepRec급)", POP), ("관계 통계 전체", REL),
                   ("바스켓 관계만", ["bdeg", "bpmi_mean", "bpmi_max"]),
                   ("시간 관계만", ["gap_med", "repeat"])]:
    h1, h5, r = retrieval(cols)
    print(f"{name:<26} {h1*100:>6.1f}% {h5*100:>6.1f}% {r*100:>7.2f}% {h1/r:>9.0f}x")

# 어떤 품목이 잘 정렬되나 — 상위/하위
h = defaultdict(list)
for a, b in rng.choice(len(clients), (150, 2)):
    if a == b: continue
    fa, fb = F[clients[a]][REL], F[clients[b]][REL]
    common = fa.index.intersection(fb.index)
    A = fa.loc[common].values; B = fb.values; bidx = list(fb.index)
    nn = np.argmin(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1), 1)
    for i, c in enumerate(common): h[c].append(bidx[nn[i]] == c)
acc = pd.Series({c: np.mean(v) for c, v in h.items() if len(v) >= 20}).sort_values(ascending=False)
print("\n잘 정렬되는 품목 (top-1):"); print("  " + " | ".join(f"{c} {v*100:.0f}%" for c, v in acc.head(8).items()))
print("안 되는 품목:"); print("  " + " | ".join(f"{c} {v*100:.0f}%" for c, v in acc.tail(6).items()))
json.dump({"n_clients": len(clients), "features": REL}, open("out/inductive_align.json", "w"), indent=1)
