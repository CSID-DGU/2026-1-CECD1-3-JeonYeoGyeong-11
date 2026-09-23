"""store 10개 선정(크기 스펙트럼) + Non-IID 계산 + 데이터 품질 카운트. DESIGN.md §3 의 '필요한 계산'."""
import json, numpy as np, pandas as pd, pyreadr
from itertools import combinations

tx = pyreadr.read_r("data/transactions.rds")[None]; tx.columns = [c.lower() for c in tx.columns]
pr = pyreadr.read_r("data/products.rda")["products"]; pr.columns = [c.lower() for c in pr.columns]
tx["ts"] = pd.to_datetime(tx.transaction_timestamp)

# ---- 데이터 품질 (§3.0 근거)
n0 = len(tx)
tx = tx.merge(pr[["product_id", "department", "product_category", "product_type"]], on="product_id", how="left")
q = {
    "rows_total": n0,
    "rows_category_missing": int(tx.product_category.isna().sum()),
    "rows_quantity_le0": int((tx.quantity <= 0).sum()),
    "rows_sales_lt0": int((tx.sales_value < 0).sum()),
}
tx = tx[tx.product_category.notna() & (tx.quantity > 0) & (tx.sales_value >= 0)]
q["rows_after_clean"] = len(tx); q["clean_keep_%"] = round(len(tx) / n0 * 100, 2)

# ---- 배타 할당
hs = tx.groupby(["household_id", "store_id"]).basket_id.nunique().rename("b").reset_index()
prim = hs.sort_values("b", ascending=False).groupby("household_id").head(1)[["household_id", "store_id"]].rename(columns={"store_id": "primary_store"})
ex = tx.merge(prim, on="household_id"); ex = ex[ex.store_id == ex.primary_store].copy()
q["baskets_total"] = int(tx.basket_id.nunique()); q["baskets_after_exclusive"] = int(ex.basket_id.nunique())
q["exclusive_keep_%"] = round(q["baskets_after_exclusive"] / q["baskets_total"] * 100, 1)

# ---- store 프로파일 + 후보
g = ex.groupby("store_id")
S = pd.DataFrame({"households": g.household_id.nunique(), "baskets": g.basket_id.nunique(), "rows": g.size()})
cand = S[(S.households >= 20) & (S.baskets >= 500)].sort_values("baskets").reset_index()
n = len(cand)
mid_idx = [round(qq * (n - 1)) for qq in (0.35, 0.45, 0.55, 0.65)]
pick = pd.concat([cand.iloc[:3].assign(tier="small"), cand.iloc[mid_idx].assign(tier="mid"), cand.iloc[-3:].assign(tier="large")])
sel = pick.store_id.tolist()
print(f"후보 {n}개 (전속가구>=20 & 바스켓>=500) → 선정 10개\n")

# ---- 선정 store 상세 (데이터량 축)
E = ex[ex.store_id.isin(sel)].copy()
bask = E.groupby(["store_id", "household_id", "basket_id"]).agg(ts=("ts", "min"), ncat=("product_category", "nunique")).reset_index()
bask = bask.sort_values(["store_id", "household_id", "ts"])
bask["gap"] = bask.groupby(["store_id", "household_id"]).ts.diff().dt.total_seconds() / 86400
rows = []
for sid in sel:
    b = bask[bask.store_id == sid]; hh = b.household_id.nunique()
    visits_per_hh = b.groupby("household_id").size()
    n_seq = int((visits_per_hh - 3).clip(lower=0).sum())          # L=10 슬라이딩, 최소 history 3, stride 1
    rows.append(dict(store_id=sid, tier=pick.set_index("store_id").tier[sid], households=hh, baskets=len(b),
                     rows=int(E[E.store_id == sid].shape[0]), visits_per_hh=round(visits_per_hh.mean(), 1),
                     median_gap_days=round(b.gap.median(), 1), mean_basket_cats=round(b.ncat.mean(), 1),
                     est_train_seq=n_seq, est_test_seq=int(n_seq * 0.18)))
T = pd.DataFrame(rows).sort_values("baskets", ascending=False)
print(T.to_string(index=False))
print(f"\n데이터량 불균형: 바스켓 최대/최소 = {T.baskets.max()/T.baskets.min():.1f}x   "
      f"가구 최대/최소 = {T.households.max()/T.households.min():.1f}x")
print(f"총 예상 train 시퀀스 ≈ {T.est_train_seq.sum():,}  (FL 1라운드 비용의 기준)")

# ---- 분포 이질성 (category 분포, JS divergence base 2)
cats = sorted(E.product_category.unique())
P = E.groupby(["store_id", "product_category"]).size().unstack(fill_value=0).reindex(columns=cats, fill_value=0).loc[sel]
P = P.div(P.sum(1), axis=0)
glob = E.product_category.value_counts(normalize=True).reindex(cats, fill_value=0)
def js(p, q):
    p, q = np.asarray(p, float), np.asarray(q, float); m = (p + q) / 2
    kl = lambda a, b: np.sum(np.where(a > 0, a * np.log2(a / np.where(b > 0, b, 1)), 0))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)
T["js_vs_global"] = [round(js(P.loc[s], glob), 4) for s in T.store_id]
M = pd.DataFrame([[js(P.loc[a], P.loc[b]) for b in sel] for a in sel], index=sel, columns=sel).round(4)
off = M.values[~np.eye(len(sel), dtype=bool)]
print(f"\nJS(store ∥ global): 평균 {T.js_vs_global.mean():.4f}  범위 {T.js_vs_global.min():.4f}~{T.js_vs_global.max():.4f}")
print(f"pairwise JS: 평균 {off.mean():.4f}  최대 {off.max():.4f}   (참고: 동일분포=0, 완전분리=1)")
print("\n" + T[["store_id", "tier", "households", "baskets", "est_train_seq", "js_vs_global"]].to_string(index=False))

# ---- 상위 카테고리 비교 (large vs small 한 쌍) — 보고서 예시용
big, small = T.iloc[0].store_id, T.iloc[-1].store_id
cmp = pd.DataFrame({big: P.loc[big], small: P.loc[small]}).nlargest(8, big).round(3)
print(f"\n카테고리 점유율 상위 8 (store {big} 기준) vs store {small}:"); print((cmp * 100).round(1).to_string())

# ---- 저장
import os; os.makedirs("out/noniid", exist_ok=True)
T.to_csv("out/noniid/store_table.csv", index=False); M.to_csv("out/noniid/js_matrix.csv")
json.dump({"selected_stores": [str(s) for s in sel], "tiers": {str(r.store_id): r.tier for r in pick.itertuples()},
           "criteria": "exclusive households>=20 & baskets>=500; small=3 smallest, mid=quantiles .35/.45/.55/.65, large=3 largest",
           "n_candidates": n, "quality": q}, open("out/selected_stores.json", "w"), indent=1)
print("\n품질:", json.dumps(q, ensure_ascii=False))
print("-> out/selected_stores.json, out/noniid/store_table.csv, out/noniid/js_matrix.csv")
