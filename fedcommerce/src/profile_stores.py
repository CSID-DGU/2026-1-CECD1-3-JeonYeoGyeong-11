"""완전판 store 분포 프로파일 — FL 클라이언트가 몇 개나 성립하는지가 핵심."""
import numpy as np, pandas as pd, pyreadr

tx = pyreadr.read_r("data/transactions.rds")[None]
pr = pyreadr.read_r("data/products.rda")["products"]
tx.columns = [c.lower() for c in tx.columns]
pr.columns = [c.lower() for c in pr.columns]

print("=" * 78)
print(f"rows={len(tx):,}  households={tx.household_id.nunique():,}  "
      f"stores={tx.store_id.nunique():,}  products={tx.product_id.nunique():,}")
ts = pd.to_datetime(tx.transaction_timestamp)
print(f"기간: {ts.min().date()} ~ {ts.max().date()}  ({(ts.max()-ts.min()).days}일)")
print(f"바스켓: {tx.basket_id.nunique():,}")
print(f"상품계층: department={pr.department.nunique()}  "
      f"product_category={pr.product_category.nunique()}  product_type={pr.product_type.nunique()}")

# ---- store별 프로파일
g = tx.groupby("store_id")
s = pd.DataFrame({
    "rows": g.size(),
    "baskets": g.basket_id.nunique(),
    "households": g.household_id.nunique(),
    "products": g.product_id.nunique(),
}).sort_values("baskets", ascending=False)
s["basket_share_%"] = (s.baskets / s.baskets.sum() * 100).round(2)

print("\n" + "=" * 78)
print(f"STORE 총 {len(s):,}개 — 바스켓 수 분위수")
q = s.baskets.quantile([.5, .75, .9, .95, .99, 1.0])
for k, v in q.items(): print(f"  p{int(k*100):<3} {int(v):>8,}")
print(f"  상위 10개 store 가 전체 바스켓의 {s.baskets.head(10).sum()/s.baskets.sum()*100:.1f}%")
print(f"  상위 50개 store 가 전체 바스켓의 {s.baskets.head(50).sum()/s.baskets.sum()*100:.1f}%")

print("\n--- 임계값별 '유효 클라이언트' 수 ---")
for th in [200, 500, 1000, 2000, 5000, 10000]:
    sub = s[s.baskets >= th]
    print(f"  baskets>={th:>6,}: {len(sub):>4}개 store  |  가구수 중앙값 {int(sub.households.median()) if len(sub) else 0:>5,}")

print("\n--- 상위 25개 store ---")
print(s.head(25).to_string())

# ---- 고객이 몇 개 store 를 쓰는가 (언러닝 경계 문제)
print("\n" + "=" * 78)
hh_store = tx.groupby(["household_id", "store_id"]).basket_id.nunique().rename("b").reset_index()
n_store_per_hh = hh_store.groupby("household_id").size()
print("가구당 이용 store 수 분포:")
print(f"  평균 {n_store_per_hh.mean():.1f}  중앙값 {n_store_per_hh.median():.0f}  "
      f"최대 {n_store_per_hh.max()}")
for k in [1, 2, 3, 5, 10]:
    print(f"  store {k}개 이하 이용 가구: {(n_store_per_hh<=k).mean()*100:5.1f}%")

top = hh_store.sort_values("b", ascending=False).groupby("household_id").head(1)
tot = hh_store.groupby("household_id").b.sum()
conc = (top.set_index("household_id").b / tot).dropna()
print(f"\n주이용 store 집중도(최다 store 바스켓 비율): 평균 {conc.mean()*100:.1f}%  중앙값 {conc.median()*100:.1f}%")
print(f"  배타 할당 시 유지되는 바스켓 비율 = {conc.mean()*100:.1f}%  (나머지는 버려짐)")

s.to_csv("out/store_profile.csv")
top.rename(columns={"store_id": "primary_store", "b": "baskets_at_primary"}).to_csv("out/hh_primary_store.csv", index=False)
print("\n-> out/store_profile.csv, out/hh_primary_store.csv")
