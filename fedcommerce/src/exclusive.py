"""고객 배타 할당 후 각 store 가 실제로 갖는 데이터량 — 유효 클라이언트 수의 진짜 기준."""
import numpy as np, pandas as pd, pyreadr

tx = pyreadr.read_r("data/transactions.rds")[None]
tx.columns = [c.lower() for c in tx.columns]

hs = tx.groupby(["household_id", "store_id"]).basket_id.nunique().rename("b").reset_index()
primary = hs.sort_values("b", ascending=False).groupby("household_id").head(1)[["household_id", "store_id"]]
primary.columns = ["household_id", "primary_store"]

ex = tx.merge(primary, on="household_id")
ex = ex[ex.store_id == ex.primary_store]          # 주이용 store 에서의 거래만 유지

print(f"배타 할당 후: rows {len(ex):,} ({len(ex)/len(tx)*100:.1f}%)  "
      f"baskets {ex.basket_id.nunique():,} ({ex.basket_id.nunique()/tx.basket_id.nunique()*100:.1f}%)")

g = ex.groupby("store_id")
s = pd.DataFrame({"baskets": g.basket_id.nunique(), "households": g.household_id.nunique(),
                  "rows": g.size()}).sort_values("baskets", ascending=False)
print(f"전속 고객이 1명 이상인 store: {len(s)}개\n")

print("--- 배타 할당 후 유효 클라이언트 (두 조건 동시 충족) ---")
print(f"{'최소가구':>8} {'최소바스켓':>10} {'store수':>8} {'바스켓 중앙값':>12} {'최대/최소비':>10}")
for hh_th, b_th in [(10,200),(20,300),(20,500),(30,500),(30,1000),(50,1000),(50,1500)]:
    sub = s[(s.households >= hh_th) & (s.baskets >= b_th)]
    ratio = f"{sub.baskets.max()/sub.baskets.min():.1f}x" if len(sub) else "-"
    print(f"{hh_th:>8} {b_th:>10,} {len(sub):>8} {int(sub.baskets.median()) if len(sub) else 0:>12,} {ratio:>10}")

print("\n--- 배타 할당 후 상위 30개 ---")
print(s.head(30).to_string())

sel = s[(s.households >= 20) & (s.baskets >= 300)]
print(f"\n[기준: 전속가구>=20 & 바스켓>=300] -> {len(sel)}개 store")
if len(sel):
    print(f"  바스켓 범위 {int(sel.baskets.min()):,} ~ {int(sel.baskets.max()):,}  "
          f"(동적범위 {sel.baskets.max()/sel.baskets.min():.1f}배)")
    print(f"  전속가구 범위 {int(sel.households.min())} ~ {int(sel.households.max())}")
    print(f"  합계 바스켓 {int(sel.baskets.sum()):,} = 전체의 {sel.baskets.sum()/tx.basket_id.nunique()*100:.1f}%")
    qs = sel.baskets.quantile([0,.25,.5,.75,1.0])
    print("  크기 분위수:", {f"p{int(k*100)}": int(v) for k, v in qs.items()})

s.to_csv("out/store_profile_exclusive.csv")
primary.to_csv("out/hh_primary_assignment.csv", index=False)
print("\n-> out/store_profile_exclusive.csv, out/hh_primary_assignment.csv")
