"""집계 단위 x 귀무가설 분해.

앞서 '거래행 5.0x' -> '방문집합 1.12x' 로 떨어진 것은 단위와 귀무가설을 동시에 바꾼 탓이라
원인을 알 수 없었다. 여기서는 데이터/클라이언트를 고정하고 두 축만 분리한다.

집계 단위 (미확정 — 이 표가 결정 근거):
  line : 거래 행 수            (구매 '건'  — 장바구니에 담긴 횟수)
  qty  : quantity 합           (구매 '개수' — 수요 예측이 쓰는 단위)
  set  : 방문당 unique 카테고리 (모델 입력이 집합일 경우)

귀무가설 (크기 동일):
  basket_random : 바스켓을 섞음      -> "매장이 누구를 받든 상관없이" 기준
  hh_random     : 가구를 통째로 섞음 -> "고객 구성 차이는 인정하고" 기준
"""
import json, os
import numpy as np
import pandas as pd
import pyreadr

rng = np.random.default_rng(0)
os.makedirs("out/het", exist_ok=True)
MIN_BASKETS = 300

tx = pyreadr.read_r("data/transactions.rds")[None]
tx.columns = [c.lower() for c in tx.columns]
pr = pyreadr.read_r("data/products.rda")["products"]
pr.columns = [c.lower() for c in pr.columns]
tx = tx.merge(pr[["product_id", "department", "product_category"]], on="product_id", how="left")
tx = tx[tx.product_category.notna() & tx.department.notna() & (tx.quantity > 0)
        & (tx.product_category != "COUPON/MISC ITEMS")]
tx["ts"] = pd.to_datetime(tx.transaction_timestamp)

# 배타 할당 (학습 데이터와 동일 기준)
hs = tx.groupby(["household_id", "store_id"]).basket_id.nunique().rename("b").reset_index()
prim = (hs.sort_values("b", ascending=False).groupby("household_id").head(1)
          [["household_id", "store_id"]].rename(columns={"store_id": "ps"}))
ex = tx.merge(prim, on="household_id")
ex = ex[ex.store_id == ex.ps].copy()
bk = ex.groupby("store_id").basket_id.nunique()
clients = bk[bk >= MIN_BASKETS].index
ex = ex[ex.store_id.isin(clients)].copy()
print(f"client {len(clients)}개   행 {len(ex):,}   바스켓 {ex.basket_id.nunique():,}   가구 {ex.household_id.nunique():,}")

# quantity 상위 절단 (p99) — 소수 대량구매가 qty 집계를 지배하지 않도록
q99 = ex.quantity.quantile(.99)
ex["qty_c"] = ex.quantity.clip(upper=q99)
print(f"quantity p99 = {q99:.0f} 로 clip   (clip 대상 {(ex.quantity>q99).mean()*100:.2f}%)")


def greedy(unit_sizes, targets):
    kk = np.array(list(targets.keys())); vv = np.array([targets[k] for k in kk], float)
    a = {}
    for u in rng.permutation(list(unit_sizes.index)):
        i = int(np.argmax(vv)); a[u] = kk[i]; vv[i] -= unit_sizes[u]
    return a


tgt = ex.store_id.value_counts().to_dict()
ex["s_bkrand"] = ex.basket_id.map(greedy(ex.groupby("basket_id").size(), tgt))
ex["s_hhrand"] = ex.household_id.map(greedy(ex.groupby("household_id").size(), tgt))


def js_rows(P, g):
    M = (P + g) / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(P > 0, P * np.log2(P / np.where(M > 0, M, 1)), 0.0).sum(1)
        G = np.broadcast_to(g, P.shape)
        b = np.where(G > 0, G * np.log2(G / np.where(M > 0, M, 1)), 0.0).sum(1)
    return 0.5 * a + 0.5 * b


def dist(df, col, level, unit):
    if unit == "line":
        M = df.groupby([col, level]).size().unstack(fill_value=0)
    elif unit == "qty":
        M = df.groupby([col, level]).qty_c.sum().unstack(fill_value=0)
    else:  # set : 방문당 unique
        d = df.drop_duplicates(subset=["basket_id", level])
        M = d.groupby([col, level]).size().unstack(fill_value=0)
    Cm = M.values.astype(float)
    P = Cm / Cm.sum(1, keepdims=True)
    return P, Cm.sum(0) / Cm.sum(), M.index


rep = {}
for level in ["department", "product_category"]:
    print("\n" + "=" * 90)
    print(f"LEVEL = {level}   —  JS 중앙값 / 귀무 대비 R")
    print("=" * 90)
    print(f"{'unit':<6} {'real JS':>9} | {'bk_rand':>9} {'R_bk':>6} | {'hh_rand':>9} {'R_hh':>6}")
    for unit in ["line", "qty", "set"]:
        Pr, gr, _ = dist(ex, "store_id", level, unit)
        jr = float(np.median(js_rows(Pr, gr)))
        row = {"real": jr}
        for col, tag in [("s_bkrand", "bk"), ("s_hhrand", "hh")]:
            Pn, gn, _ = dist(ex, col, level, unit)
            jn = float(np.median(js_rows(Pn, gn)))
            row[tag] = jn; row[f"R_{tag}"] = jr / jn
        rep.setdefault(level, {})[unit] = row
        print(f"{unit:<6} {row['real']:>9.4f} | {row['bk']:>9.4f} {row['R_bk']:>6.2f}x | "
              f"{row['hh']:>9.4f} {row['R_hh']:>6.2f}x")

# ---- quantity 자체가 store 마다 다른가 (수요 예측 서비스에 직결)
print("\n" + "=" * 90)
print("카테고리별 평균 구매개수 (line -> qty 변환계수) 가 store 마다 다른가")
print("=" * 90)
mq = ex.groupby(["store_id", "product_category"]).qty_c.mean().unstack()
top = ex.product_category.value_counts().head(15).index
sub = mq[top].dropna(axis=0, thresh=10)
cvs = (sub.std() / sub.mean()).sort_values(ascending=False)
print(f"  상위 15개 카테고리의 store 간 평균개수 CV: 중앙 {cvs.median():.3f}  최대 {cvs.max():.3f} ({cvs.idxmax()})")
print(f"  전체 평균 구매개수: {ex.qty_c.mean():.2f}   store 간 CV {ex.groupby('store_id').qty_c.mean().std()/ex.groupby('store_id').qty_c.mean().mean():.3f}")
rep["qty_per_category_cv"] = dict(median=float(cvs.median()), max=float(cvs.max()), argmax=str(cvs.idxmax()))

json.dump(rep, open("out/het/unit_ablation.json", "w"), indent=1)
print("\n-> out/het/unit_ablation.json")
