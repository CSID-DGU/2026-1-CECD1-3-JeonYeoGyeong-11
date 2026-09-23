"""Instacart 를 dunnhumby client 크기에 맞춰 100개 클라이언트로 분할하고 이질성을 측정.

3자 비교:
  (A) dunnhumby 실제 store      — 자연 분할
  (B) dunnhumby 랜덤 (크기 동일) — 같은 데이터, 고객만 섞음
  (C) Instacart 랜덤 (크기 동일) — 다른 회사, 같은 구성 방식

(A) vs (B) = "실제 매장 구조가 랜덤 분할보다 이질적인가"
(B) vs (C) = "회사가 다르면 같은 방식으로 나눠도 이질성이 다른가"

시간 정렬: Instacart 는 절대 시각이 없으므로 각 고객의 첫 주문을 t=0 으로 두고
누적 days_since_prior_order 로 자기 타임라인을 만든다 ("가입 후 n일").
"""
import json, os
import numpy as np
import pandas as pd
import pyreadr

rng = np.random.default_rng(0)
os.makedirs("out/instacart", exist_ok=True)
MIN_BASKETS = 300

# =================================================================== dunnhumby
tx = pyreadr.read_r("data/transactions.rds")[None]
tx.columns = [c.lower() for c in tx.columns]
pr = pyreadr.read_r("data/products.rda")["products"]
pr.columns = [c.lower() for c in pr.columns]
tx = tx.merge(pr[["product_id", "department", "product_category"]], on="product_id", how="left")
tx = tx[tx.product_category.notna() & tx.department.notna() & (tx.quantity > 0)
        & (tx.product_category != "COUPON/MISC ITEMS")]
hs = tx.groupby(["household_id", "store_id"]).basket_id.nunique().rename("b").reset_index()
prim = (hs.sort_values("b", ascending=False).groupby("household_id").head(1)
          [["household_id", "store_id"]].rename(columns={"store_id": "ps"}))
dh = tx.merge(prim, on="household_id")
dh = dh[dh.store_id == dh.ps].copy()
bk = dh.groupby("store_id").basket_id.nunique()
targets = bk[bk >= MIN_BASKETS].sort_values(ascending=False)
dh = dh[dh.store_id.isin(targets.index)].copy()
print(f"dunnhumby client {len(targets)}개  목표 바스켓 {targets.min()}~{targets.max()}  합계 {targets.sum():,}")


def greedy_assign(unit_sizes, target_series, rng):
    """단위를 무작위 순서로, 남은 결손이 가장 큰 client 에 배정 (크기 매칭)."""
    keys = np.array(target_series.index)
    vals = target_series.values.astype(float).copy()
    out = {}
    for u in rng.permutation(list(unit_sizes.index)):
        i = int(np.argmax(vals))
        out[u] = keys[i]
        vals[i] -= unit_sizes[u]
    return out


# (B) dunnhumby 랜덤: 가구를 섞어 같은 크기로
hh_bk = dh.groupby("household_id").basket_id.nunique()
dh["client_rand"] = dh.household_id.map(greedy_assign(hh_bk, targets, rng))

# =================================================================== Instacart
D = "data/instacart"
orders = pd.read_csv(f"{D}/orders.csv")
orders = orders[orders.eval_set == "prior"].sort_values(["user_id", "order_number"])
# 시간 정렬: 각 고객 첫 주문 t=0
orders["gap"] = orders.days_since_prior_order.fillna(0)
orders["t"] = orders.groupby("user_id").gap.cumsum()

# 고객을 dunnhumby client 크기에 맞춰 100개로 (주문 수 기준)
n_ord = orders.groupby("user_id").size()
ic_targets = targets.iloc[:100] if len(targets) > 100 else targets
need = ic_targets.sum()
pick, acc = [], 0
for u in rng.permutation(list(n_ord.index)):
    pick.append(u); acc += n_ord[u]
    if acc >= need * 1.05:
        break
n_sel = n_ord.loc[pick]
print(f"Instacart 선택 고객 {len(pick):,}명  주문 {n_sel.sum():,}  (목표 {need:,})")
assign = greedy_assign(n_sel, ic_targets, rng)
orders = orders[orders.user_id.isin(pick)].copy()
orders["client"] = orders.user_id.map(assign)

prior = pd.read_csv(f"{D}/order_products__prior.csv", usecols=["order_id", "product_id"])
prod = pd.read_csv(f"{D}/products.csv", usecols=["product_id", "aisle_id", "department_id"])
ic = prior[prior.order_id.isin(set(orders.order_id))].merge(prod, on="product_id", how="left")
ic = ic.merge(orders[["order_id", "user_id", "client", "t"]], on="order_id", how="left")
print(f"Instacart 사용 행 {len(ic):,}  client {ic.client.nunique()}  "
      f"client당 주문 중앙 {orders.groupby('client').size().median():.0f}")

# 귀무: 주문(=바스켓) 랜덤 / 고객 랜덤(다른 시드)
ord_sz = ic.groupby("order_id").size()
ic["client_bkrand"] = ic.order_id.map(greedy_assign(ord_sz, ic_targets, np.random.default_rng(7)))
usr_sz = ic.groupby("user_id").size()
ic["client_usrrand"] = ic.user_id.map(greedy_assign(usr_sz, ic_targets, np.random.default_rng(7)))

# dunnhumby 귀무도 동일하게
dh_bk = dh.groupby("basket_id").size()
dh["client_bkrand"] = dh.basket_id.map(greedy_assign(dh_bk, targets, np.random.default_rng(7)))


# =================================================================== metrics
def js_rows(P, g):
    M = (P + g) / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(P > 0, P * np.log2(P / np.where(M > 0, M, 1)), 0.0).sum(1)
        G = np.broadcast_to(g, P.shape)
        b = np.where(G > 0, G * np.log2(G / np.where(M > 0, M, 1)), 0.0).sum(1)
    return 0.5 * a + 0.5 * b


def pairwise_js(P, cap=100, rng=rng):
    K = len(P)
    idx = rng.choice(K, min(K, cap), replace=False) if K > cap else np.arange(K)
    Q = P[idx]; out = []
    for i in range(len(Q) - 1):
        M = (Q[i] + Q[i + 1:]) / 2
        with np.errstate(divide="ignore", invalid="ignore"):
            a = np.where(Q[i] > 0, Q[i] * np.log2(Q[i] / np.where(M > 0, M, 1)), 0.0).sum(1)
            b = np.where(Q[i + 1:] > 0, Q[i + 1:] * np.log2(Q[i + 1:] / np.where(M > 0, M, 1)), 0.0).sum(1)
        out.append(0.5 * a + 0.5 * b)
    return np.concatenate(out)


def gini(x):
    x = np.sort(np.asarray(x, float)); n = len(x)
    return (2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum())


def metrics(df, ccol, lcol):
    M = df.groupby([ccol, lcol]).size().unstack(fill_value=0)
    Cm = M.values.astype(float)
    P = Cm / Cm.sum(1, keepdims=True)
    g = Cm.sum(0) / Cm.sum()
    JS = js_rows(P, g)
    TV = 0.5 * np.abs(P - g).sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        H = -np.where(P > 0, P * np.log(P), 0.0).sum(1) / np.log(P.shape[1])
    return dict(n_clients=len(M), js=float(np.median(JS)), js_p90=float(np.percentile(JS, 90)),
                pair_js=float(np.median(pairwise_js(P))), tv=float(np.median(TV)),
                entropy=float(np.median(H)))


SETS = [
    ("A. dunnhumby 실제 store", dh, "store_id", {"dept": "department", "fine": "product_category"}),
    ("B. dunnhumby 랜덤(고객)", dh, "client_rand", {"dept": "department", "fine": "product_category"}),
    ("   dunnhumby 귀무(바스켓)", dh, "client_bkrand", {"dept": "department", "fine": "product_category"}),
    ("C. Instacart 랜덤(고객)", ic, "client", {"dept": "department_id", "fine": "aisle_id"}),
    ("   Instacart 귀무(주문)", ic, "client_bkrand", {"dept": "department_id", "fine": "aisle_id"}),
    ("   Instacart 귀무(고객,다른시드)", ic, "client_usrrand", {"dept": "department_id", "fine": "aisle_id"}),
]

rep = {}
for lvl in ["dept", "fine"]:
    print("\n" + "=" * 100)
    print(f"LEVEL = {lvl}   (dunnhumby: department 26 / category 301   |   Instacart: department 21 / aisle 134)")
    print("=" * 100)
    print(f"{'분할':<32} {'clients':>8} {'JS':>9} {'JS p90':>9} {'pairJS':>9} {'TV':>9} {'H_norm':>8}")
    for name, df, ccol, lv in SETS:
        m = metrics(df, ccol, lv[lvl])
        rep[f"{lvl}|{name.strip()}"] = m
        print(f"{name:<32} {m['n_clients']:>8} {m['js']:>9.4f} {m['js_p90']:>9.4f} "
              f"{m['pair_js']:>9.4f} {m['tv']:>9.4f} {m['entropy']:>8.3f}")
    a = rep[f"{lvl}|A. dunnhumby 실제 store"]["js"]
    b = rep[f"{lvl}|B. dunnhumby 랜덤(고객)"]["js"]
    bn = rep[f"{lvl}|dunnhumby 귀무(바스켓)"]["js"]
    c = rep[f"{lvl}|C. Instacart 랜덤(고객)"]["js"]
    cn = rep[f"{lvl}|Instacart 귀무(주문)"]["js"]
    print(f"\n  R(A/바스켓귀무) = {a/bn:.2f}x    R(B/바스켓귀무) = {b/bn:.2f}x    "
          f"R(A/B) = {a/b:.2f}x   <- 실제 매장 구조가 랜덤 대비 얼마나 더 이질적인가")
    print(f"  R(C/주문귀무)  = {c/cn:.2f}x    <- Instacart 에서 고객 묶음이 만드는 이질성")
    rep[f"{lvl}|R"] = dict(A_over_bknull=a/bn, B_over_bknull=b/bn, A_over_B=a/b, C_over_bknull=c/cn)

# 크기 불균형 확인
print("\n" + "=" * 100)
print("클라이언트 크기 (매칭 확인)")
print("=" * 100)
for name, df, ccol in [("dunnhumby 실제", dh, "store_id"), ("dunnhumby 랜덤", dh, "client_rand"),
                       ("Instacart 랜덤", ic, "client")]:
    s = df.groupby(ccol).size()
    bsz = (df.groupby(ccol).basket_id.nunique() if "basket_id" in df else
           df.groupby(ccol).order_id.nunique())
    print(f"  {name:<16} clients {len(s):>4}  바스켓 {bsz.min():>5}~{bsz.max():>5}  "
          f"중앙 {bsz.median():>6.0f}  Gini {gini(bsz):.3f}  max/min {bsz.max()/bsz.min():.1f}x")

json.dump(rep, open("out/instacart/three_way.json", "w"), indent=1)
print("\n-> out/instacart/three_way.json")
