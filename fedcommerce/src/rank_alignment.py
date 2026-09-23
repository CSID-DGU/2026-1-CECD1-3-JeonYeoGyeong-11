"""'몇 번 물건' 표현이 클라이언트 간에 정렬되는가?

가설: 품목을 임의 ID 가 아니라 (인기순위, 재구매율, 구매주기, 바스켓기여) 같은 행동 특징으로
      표현하면, 어휘를 공유하지 않아도 클라이언트 간에 의미가 정렬된다.
검증: (1) 같은 카테고리의 순위가 클라이언트마다 일치하는가 (Spearman)
      (2) 행동 특징이 클라이언트마다 일치하는가
      (3) 순위만으로 카테고리를 식별할 수 있는가 (순위 -> 카테고리 정확도)
"""
import json
import numpy as np
import pandas as pd
import pyreadr
from scipy.stats import spearmanr

# ------------------------------------------------------------------ dunnhumby
tx = pyreadr.read_r("data/transactions.rds")[None]; tx.columns = [c.lower() for c in tx.columns]
pr = pyreadr.read_r("data/products.rda")["products"]; pr.columns = [c.lower() for c in pr.columns]
tx = tx.merge(pr[["product_id", "product_category"]], on="product_id", how="left")
tx = tx[tx.product_category.notna() & (tx.quantity > 0) & (tx.product_category != "COUPON/MISC ITEMS")]
tx["ts"] = pd.to_datetime(tx.transaction_timestamp)
hs = tx.groupby(["household_id", "store_id"]).basket_id.nunique().rename("b").reset_index()
prim = (hs.sort_values("b", ascending=False).groupby("household_id").head(1)
          [["household_id", "store_id"]].rename(columns={"store_id": "ps"}))
dh = tx.merge(prim, on="household_id"); dh = dh[dh.store_id == dh.ps].copy()
bk = dh.groupby("store_id").basket_id.nunique()
dh = dh[dh.store_id.isin(bk[bk >= 300].index)].copy()
print(f"dunnhumby client {dh.store_id.nunique()}  카테고리 {dh.product_category.nunique()}")


def behav(df, ccol, icol, bcol, hcol):
    """클라이언트 x 품목 행동 특징."""
    nb = df.groupby(ccol)[bcol].nunique()
    g = df.groupby([ccol, icol])
    f = pd.DataFrame({
        "baskets": g[bcol].nunique(),
        "households": g[hcol].nunique(),
        "lines": g.size(),
    }).reset_index()
    f["penet"] = f.baskets / f[ccol].map(nb)                      # 바스켓 침투율
    nh = df.groupby(ccol)[hcol].nunique()
    f["reach"] = f.households / f[ccol].map(nh)                   # 고객 도달률
    f["repeat"] = f.baskets / f.households                        # 고객당 재구매
    f["rank"] = f.groupby(ccol).penet.rank(ascending=False, method="first")
    return f


F = behav(dh, "store_id", "product_category", "basket_id", "household_id")
piv = F.pivot(index="product_category", columns="store_id", values="rank")
cl = list(piv.columns)

print("\n" + "=" * 76)
print("(1) 카테고리 순위가 클라이언트 간에 일치하는가")
print("=" * 76)
rs = []
for i, a in enumerate(cl):
    for b in cl[i + 1:]:
        s = piv[[a, b]].dropna()
        if len(s) > 30:
            rs.append(spearmanr(s[a], s[b]).statistic)
rs = np.array(rs)
print(f"  클라이언트쌍 Spearman 순위상관: 중앙 {np.median(rs):.3f}  "
      f"Q1 {np.percentile(rs,25):.3f}  Q3 {np.percentile(rs,75):.3f}  최소 {rs.min():.3f}")

print("\n" + "=" * 76)
print("(2) 행동 특징이 클라이언트 간에 일치하는가")
print("=" * 76)
for col in ["penet", "reach", "repeat"]:
    p = F.pivot(index="product_category", columns="store_id", values=col)
    v = []
    for i, a in enumerate(cl):
        for b in cl[i + 1:]:
            s = p[[a, b]].dropna()
            if len(s) > 30:
                v.append(spearmanr(s[a], s[b]).statistic)
    print(f"  {col:<8} 순위상관 중앙 {np.median(v):.3f}")

print("\n" + "=" * 76)
print("(3) 순위만으로 카테고리를 식별할 수 있는가")
print("=" * 76)
# 전역 순위표를 만들고, 각 클라이언트의 k위 품목이 전역 k위와 같은 카테고리인지
glob = (dh.groupby("product_category").basket_id.nunique()
        .rank(ascending=False, method="first").sort_values())
gmap = {int(r): c for c, r in glob.items()}
for K in [1, 3, 5, 10, 20, 50]:
    hit = []
    for c in cl:
        s = piv[c].dropna().sort_values()
        for k in range(1, K + 1):
            if k in gmap and len(s) >= k:
                hit.append(s.index[k - 1] == gmap[k])
    print(f"  상위 {K:>2}위: 전역 순위와 카테고리 일치 {np.mean(hit)*100:5.1f}%")
# top-K 집합 일치 (순서 무시)
for K in [5, 10, 20]:
    gs = set(list(glob.index[:K]))
    ov = [len(set(piv[c].dropna().sort_values().index[:K]) & gs) / K for c in cl]
    print(f"  상위 {K:>2} 집합 겹침(순서무시): 중앙 {np.median(ov)*100:.0f}%")

print("\n" + "=" * 76)
print("(4) 전역 상위 12 카테고리의 클라이언트별 순위 분포")
print("=" * 76)
for c in glob.index[:12]:
    r = piv.loc[c].dropna()
    print(f"  {c:<32} 중앙 {r.median():>5.0f}위  범위 {r.min():>3.0f}~{r.max():>4.0f}")

json.dump({"rank_spearman_median": float(np.median(rs)),
           "rank_spearman_q1": float(np.percentile(rs, 25)),
           "rank_spearman_min": float(rs.min())},
          open("out/rank_alignment.json", "w"), indent=1)
print("\n-> out/rank_alignment.json")
