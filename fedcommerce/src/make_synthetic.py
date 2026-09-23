"""
Dunnhumby Complete Journey 와 동일한 스키마의 합성 데이터 생성기.

실데이터가 도착하기 전에 eda.py / preprocess.py 를 검증하기 위한 용도이며,
--alpha 로 store 간 이질성을 직접 통제할 수 있으므로
"우리 분석 코드가 알려진 alpha 를 실제로 복원하는가" 를 확인하는 데도 쓴다.
"""
import argparse, os
import numpy as np
import pandas as pd


def build(n_household=2500, n_store=40, n_product=9000, n_commodity=300,
          n_days=711, alpha=0.5, home_bias=0.75, seed=0):
    rng = np.random.default_rng(seed)

    # ---- product 마스터: 상품 -> commodity -> department
    commodity_of = rng.integers(0, n_commodity, size=n_product)
    n_dept = 25
    dept_of_commodity = rng.integers(0, n_dept, size=n_commodity)
    product = pd.DataFrame({
        "PRODUCT_ID": np.arange(1, n_product + 1),
        "MANUFACTURER": rng.integers(1, 500, size=n_product),
        "DEPARTMENT": [f"DEPT_{d:02d}" for d in dept_of_commodity[commodity_of]],
        "BRAND": rng.choice(["National", "Private"], size=n_product, p=[0.7, 0.3]),
        "COMMODITY_DESC": [f"COMMODITY_{c:03d}" for c in commodity_of],
        "SUB_COMMODITY_DESC": [f"SUB_{c:03d}_{p % 8:d}" for p, c in enumerate(commodity_of)],
        "CURR_SIZE_OF_PRODUCT": "",
    })

    # ---- 전역 commodity 인기도 (long tail)
    pi = rng.dirichlet(np.full(n_commodity, 0.7))

    # ---- store 규모: 실제 Dunnhumby 처럼 극단적으로 치우치게
    store_scale = rng.pareto(1.1, size=n_store) + 0.05
    store_scale = store_scale / store_scale.sum()
    store_ids = rng.choice(np.arange(200, 200 + n_store * 6), size=n_store, replace=False)

    # ---- store별 commodity 선호: Dirichlet(alpha * pi)  <- 이게 ground-truth 이질성
    conc = np.clip(alpha * pi * n_commodity, 1e-3, None)
    store_pref = rng.dirichlet(conc, size=n_store)

    # ---- household -> home store (규모 비례), 일부는 타 store 도 방문
    home = rng.choice(n_store, size=n_household, p=store_scale)

    # commodity -> 소속 product 목록
    prods_by_comm = [np.where(commodity_of == c)[0] + 1 for c in range(n_commodity)]

    rows = []
    basket_seq = 26_000_000_000
    for h in range(n_household):
        n_visit = max(1, int(rng.gamma(2.0, 12.0)))
        days = np.sort(rng.choice(np.arange(1, n_days + 1), size=min(n_visit, n_days), replace=False))
        for d in days:
            s = home[h] if rng.random() < home_bias else rng.choice(n_store, p=store_scale)
            basket_seq += 1
            n_item = max(1, int(rng.gamma(2.2, 2.4)))
            comms = rng.choice(n_commodity, size=n_item, p=store_pref[s])
            tt = int(rng.integers(700, 2200))
            for c in comms:
                pool = prods_by_comm[c]
                if len(pool) == 0:
                    continue
                rows.append((
                    h + 1, basket_seq, int(d), int(rng.choice(pool)),
                    int(rng.integers(1, 4)), round(float(rng.gamma(2, 1.6)), 2),
                    int(store_ids[s]), 0.0, tt, int((d - 1) // 7) + 1, 0.0, 0.0,
                ))

    tx = pd.DataFrame(rows, columns=[
        "household_key", "BASKET_ID", "DAY", "PRODUCT_ID", "QUANTITY", "SALES_VALUE",
        "STORE_ID", "RETAIL_DISC", "TRANS_TIME", "WEEK_NO", "COUPON_DISC", "COUPON_MATCH_DISC"])

    truth = pd.DataFrame({"STORE_ID": store_ids, "_synthetic_scale": store_scale})
    return tx, product, truth


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--alpha", type=float, default=0.5, help="ground-truth Dirichlet concentration")
    ap.add_argument("--households", type=int, default=2500)
    ap.add_argument("--stores", type=int, default=40)
    ap.add_argument("--home-bias", type=float, default=0.75)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    tx, product, truth = build(n_household=a.households, n_store=a.stores,
                               alpha=a.alpha, home_bias=a.home_bias, seed=a.seed)
    tx.to_csv(os.path.join(a.out, "transaction_data.csv"), index=False)
    product.to_csv(os.path.join(a.out, "product.csv"), index=False)
    truth.to_csv(os.path.join(a.out, "_ground_truth.csv"), index=False)
    print(f"alpha(ground truth) = {a.alpha}   home_bias = {a.home_bias}")
    print(f"transaction rows    = {len(tx):,}")
    print(f"stores / households = {tx.STORE_ID.nunique()} / {tx.household_key.nunique()}")
    print(f"-> {a.out}")
