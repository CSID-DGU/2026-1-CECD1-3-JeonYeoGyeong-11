"""Instacart 미러 무결성 검증 + dunnhumby 와 나란히 볼 학습 단위 실측 (DESIGN.md §6.1).

비공식 재업로드이므로 원본 공표값과 대조한 뒤에야 나머지 분석으로 넘어간다.
"""
import json, os
import numpy as np
import pandas as pd

D = "data/instacart"
os.makedirs("out/instacart", exist_ok=True)
rep = {}

ai = pd.read_csv(f"{D}/aisles.csv")
dp = pd.read_csv(f"{D}/departments.csv")
pnames = pd.read_csv(f"{D}/products.csv")
orders = pd.read_csv(f"{D}/orders.csv")
prior = pd.read_csv(f"{D}/order_products__prior.csv")

print("=" * 88)
print("1. 무결성 검증 (원본 공표값 대조)")
print("=" * 88)
checks = [
    ("aisles", len(ai), 134),
    ("departments", len(dp), 21),
    ("products", len(pnames), 49688),
    ("users (orders.user_id)", orders.user_id.nunique(), 206209),
    ("orders (전체)", len(orders), 3421083),
    ("order_products__prior 행", len(prior), 32434489),
]
ok = True
for name, got, exp in checks:
    good = got == exp
    ok &= good
    print(f"  {'OK ' if good else 'MISMATCH'}  {name:<26} {got:>12,}   (원본 {exp:,})")
rep["integrity_ok"] = bool(ok)
rep["counts"] = {n: int(g) for n, g, _ in checks}

print(f"\n  eval_set 분포: {orders.eval_set.value_counts().to_dict()}")
print(f"  결측: orders.days_since_prior_order {orders.days_since_prior_order.isna().sum():,} "
      f"(= 각 고객 첫 주문 {orders.user_id.nunique():,} 와 일치: "
      f"{orders.days_since_prior_order.isna().sum() == orders.user_id.nunique()})")
print(f"  products 결측 aisle/department: {pnames.aisle_id.isna().sum()} / {pnames.department_id.isna().sum()}")

# prior 만 사용 (train/test 는 대회용 홀드아웃)
op = orders[orders.eval_set == "prior"]
print(f"\n  prior 주문 {len(op):,}   고객 {op.user_id.nunique():,}")

print("\n" + "=" * 88)
print("2. 학습 단위 실측 (dunnhumby 와 나란히)")
print("=" * 88)
# 주문당 상품 수
n_per_order = prior.groupby("order_id").size()
print(f"  [주문 하나]  상품 {n_per_order.mean():.1f}개 (중앙 {n_per_order.median():.0f}, "
      f"p90 {n_per_order.quantile(.9):.0f})")
# 고객당 주문 수
n_per_user = op.groupby("user_id").size()
print(f"  [고객 하나]  주문 {n_per_user.mean():.1f}회 (중앙 {n_per_user.median():.0f}, "
      f"min {n_per_user.min()}, p90 {n_per_user.quantile(.9):.0f}, max {n_per_user.max()})")
print(f"    3회 이상 {(n_per_user>=3).mean()*100:.1f}%   10회 이상 {(n_per_user>=10).mean()*100:.1f}%")
for H in [1, 3, 5]:
    print(f"    최소 history {H} -> 학습 예제 {(n_per_user-H).clip(lower=0).sum():,}")

# 상대 시간 복원
op = op.sort_values(["user_id", "order_number"]).copy()
op["gap"] = op.days_since_prior_order.fillna(0)
op["t"] = op.groupby("user_id").gap.cumsum()
span = op.groupby("user_id").t.max()
print(f"\n  [고객 활동 기간]  평균 {span.mean():.0f}일 (중앙 {span.median():.0f}, "
      f"p90 {span.quantile(.9):.0f}, max {span.max():.0f})")
cap = (op.gap == 30).mean()
print(f"  days_since_prior_order == 30 (상한 절단) 비율: {cap*100:.2f}%")
print(f"  gap 분포: 중앙 {op[op.gap>0].gap.median():.0f}일  평균 {op[op.gap>0].gap.mean():.1f}일")

# 7일 창 주문 횟수 (dunnhumby 2.47회와 비교)
g = op.groupby("user_id")
cnt = []
for _, gg in op.groupby("user_id", sort=False):
    t = gg.t.values
    for i in range(len(t) - 1):
        cnt.append(int(((t > t[i]) & (t <= t[i] + 7)).sum()))
cnt = np.array(cnt)
print(f"\n  [향후 7일 윈도우]  0회 {100*(cnt==0).mean():.1f}%  1회 {100*(cnt==1).mean():.1f}%  "
      f"2회 {100*(cnt==2).mean():.1f}%  3회+ {100*(cnt>=3).mean():.1f}%   평균 {cnt.mean():.2f}회")
print(f"  7일 재방문 양성률(전체) {100*(cnt>0).mean():.1f}%    3일 기준은 gap<=3 비율로 별도 산출")
rep["unit"] = dict(items_per_order=float(n_per_order.mean()), orders_per_user=float(n_per_user.mean()),
                   span_days_median=float(span.median()), gap_cap30_share=float(cap),
                   win7_mean=float(cnt.mean()), win7_positive=float((cnt > 0).mean()))

print("\n" + "=" * 88)
print("3. 예측 단위 후보")
print("=" * 88)
pj = prior.merge(pnames[["product_id", "aisle_id", "department_id"]], on="product_id", how="left")
for col, name in [("product_id", "product_id"), ("aisle_id", "aisle"), ("department_id", "department")]:
    print(f"  {name:<12} {pj[col].nunique():>7,} 종")
# 주문당 unique aisle
ua = pj.groupby("order_id").aisle_id.nunique()
print(f"  주문당 unique aisle: 평균 {ua.mean():.1f} (중앙 {ua.median():.0f}, p90 {ua.quantile(.9):.0f})")
rep["vocab"] = dict(product=int(pj.product_id.nunique()), aisle=int(pj.aisle_id.nunique()),
                    department=int(pj.department_id.nunique()), aisles_per_order=float(ua.mean()))

print("\n" + "=" * 88)
print("4. 텍스트 (Phase 2 토큰 공유용)")
print("=" * 88)
print("  product_name 예시:", " | ".join(pnames.product_name.head(3)))
print("  aisle 예시      :", " | ".join(ai.aisle.head(5)))
print("  department 전체 :", ", ".join(dp.department))

json.dump(rep, open("out/instacart/verify.json", "w"), indent=1)
print(f"\n무결성: {'통과' if ok else '불일치 있음 — 확인 필요'}")
print("-> out/instacart/verify.json")
