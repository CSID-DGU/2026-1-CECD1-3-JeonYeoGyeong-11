"""Instacart 클라이언트를 dunnhumby 와 '전반적으로' 비슷하게 맞춘다.

랜덤 배정은 크기·분포는 맞았으나 update divergence 가 절반(1.18x vs 2.36x)이었다.
원인은 고객을 무작위로 섞어 클라이언트마다 고객 취향이 고르게 평균화된 것.
-> dunnhumby 에서 매장이 하는 일(지역마다 다른 고객이 온다)을 인위적으로 재현한다.

방법: 클라이언트마다 목표 프로필 q_k ~ Dir(alpha * pi) 를 뽑고,
      크기 결손이 큰 클라이언트부터 자기 프로필과 가장 잘 맞는 미배정 고객을 가져간다.
      -> 크기는 정확히 통제되고 이질성은 alpha 로 조절된다.

맞출 목표 (dunnhumby 실측):
  JS R (세부)          5.00x
  update divergence R  2.36x
  데이터량 Gini        0.244
  클라이언트 수        101 -> 100
"""
import json, os, sys
import numpy as np
import pandas as pd
import pyreadr
import torch
import torch.nn as nn

rng = np.random.default_rng(0)
torch.manual_seed(0)
os.makedirs("out/instacart", exist_ok=True)
D = "data/instacart"
TARGET = dict(js_R=5.00, upd_R=2.36, gini=0.244, pair_js=0.0723, tv=0.1568)

# --------------------------------------------------------- dunnhumby 목표 크기
tx = pyreadr.read_r("data/transactions.rds")[None]
tx.columns = [c.lower() for c in tx.columns]
pr = pyreadr.read_r("data/products.rda")["products"]
pr.columns = [c.lower() for c in pr.columns]
tx = tx.merge(pr[["product_id", "product_category"]], on="product_id", how="left")
tx = tx[tx.product_category.notna() & (tx.quantity > 0) & (tx.product_category != "COUPON/MISC ITEMS")]
hs = tx.groupby(["household_id", "store_id"]).basket_id.nunique().rename("b").reset_index()
prim = (hs.sort_values("b", ascending=False).groupby("household_id").head(1)
          [["household_id", "store_id"]].rename(columns={"store_id": "ps"}))
dh = tx.merge(prim, on="household_id"); dh = dh[dh.store_id == dh.ps]
targets = dh.groupby("store_id").basket_id.nunique()
targets = targets[targets >= 300].sort_values(ascending=False).iloc[:100]
del tx, dh, hs, prim
print(f"목표 크기 (dunnhumby 100 client): {targets.min()}~{targets.max()}, 합 {targets.sum():,}")

# --------------------------------------------------------- Instacart 로드
orders = pd.read_csv(f"{D}/orders.csv")
orders = orders[orders.eval_set == "prior"].sort_values(["user_id", "order_number"])
orders["gap"] = orders.days_since_prior_order.fillna(0)
orders["t"] = orders.groupby("user_id").gap.cumsum()
n_ord = orders.groupby("user_id").size()

need = targets.sum()
pick, acc = [], 0
for u in rng.permutation(list(n_ord.index)):
    pick.append(u); acc += n_ord[u]
    if acc >= need * 1.25:            # 프로필 매칭 여유분
        break
pick = set(pick)
orders = orders[orders.user_id.isin(pick)].copy()

prior = pd.read_csv(f"{D}/order_products__prior.csv", usecols=["order_id", "product_id"])
prod = pd.read_csv(f"{D}/products.csv", usecols=["product_id", "aisle_id"])
ic = (prior[prior.order_id.isin(set(orders.order_id))].merge(prod, on="product_id", how="left")
      .merge(orders[["order_id", "user_id", "t", "order_dow", "order_hour_of_day"]], on="order_id"))
del prior
A = int(ic.aisle_id.max()) + 1
print(f"후보 고객 {ic.user_id.nunique():,}  주문 {ic.order_id.nunique():,}  행 {len(ic):,}")

# 고객 프로필 (aisle 분포) + 주문 수
U = ic.groupby(["user_id", "aisle_id"]).size().unstack(fill_value=0)
uid = np.array(U.index)
Up = U.values.astype(float); Up = Up / Up.sum(1, keepdims=True)
u_ord = orders.groupby("user_id").size().reindex(uid).values.astype(float)
pi = ic.aisle_id.value_counts(normalize=True).reindex(U.columns, fill_value=0).values
acols = np.array(U.columns)


u_label = np.asarray(acols)[Up.argmax(1)]          # 고객의 주력 aisle (Dirichlet 분할의 라벨)


def assign_dirichlet(alpha, rng):
    """표준 Dirichlet 분할 (NIID-Bench 관례): 라벨 c 마다 클라이언트 간 배분 p_c ~ Dir(alpha*K*w).

    w 는 dunnhumby 목표 크기 비율이라 기대 크기가 목표에 맞고, alpha 가 이질성을 조절한다.
    alpha -> inf 이면 모든 p_c = w 가 되어 순수 랜덤 배정이 된다.
    """
    K = len(targets)
    keys = np.array(targets.index)
    w = targets.values.astype(float) / targets.values.sum()
    cap = targets.values.astype(float) * 1.15      # 과대 배정 방지 상한 (주문 수 기준)
    load = np.zeros(K)
    out = np.empty(len(uid), dtype=object)
    for c in np.unique(u_label):
        idx = np.where(u_label == c)[0]
        rng.shuffle(idx)
        p = w.copy() if np.isinf(alpha) else rng.dirichlet(np.clip(alpha * K * w, 1e-6, None))
        for i in idx:
            pp = np.where(load < cap, p, 0.0)
            if pp.sum() <= 0:
                pp = np.where(load < cap, 1.0, 0.0)
                if pp.sum() <= 0:
                    pp = np.ones(K)
            k = rng.choice(K, p=pp / pp.sum())
            out[i] = keys[k]
            load[k] += u_ord[i]
    return pd.Series(out, index=uid)


def js_rows(P, g):
    M = (P + g) / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(P > 0, P * np.log2(P / np.where(M > 0, M, 1)), 0.0).sum(1)
        G = np.broadcast_to(g, P.shape)
        b = np.where(G > 0, G * np.log2(G / np.where(M > 0, M, 1)), 0.0).sum(1)
    return 0.5 * a + 0.5 * b


def pair_js(P):
    out = []
    for i in range(len(P) - 1):
        M = (P[i] + P[i + 1:]) / 2
        with np.errstate(divide="ignore", invalid="ignore"):
            a = np.where(P[i] > 0, P[i] * np.log2(P[i] / np.where(M > 0, M, 1)), 0.0).sum(1)
            b = np.where(P[i + 1:] > 0, P[i + 1:] * np.log2(P[i + 1:] / np.where(M > 0, M, 1)), 0.0).sum(1)
        out.append(0.5 * a + 0.5 * b)
    return np.concatenate(out)


def gini(x):
    x = np.sort(np.asarray(x, float)); n = len(x)
    return (2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum())


def dist_stats(df, ccol):
    M = df.groupby([ccol, "aisle_id"]).size().unstack(fill_value=0)
    Cm = M.values.astype(float); P = Cm / Cm.sum(1, keepdims=True)
    g = Cm.sum(0) / Cm.sum()
    return dict(js=float(np.median(js_rows(P, g))), pair=float(np.median(pair_js(P))),
                tv=float(np.median(0.5 * np.abs(P - g).sum(1))))


def update_div(df, ccol):
    ob = (df.groupby([ccol, "user_id", "order_id"])
            .agg(t=("t", "first"), ai=("aisle_id", lambda s: sorted(set(s)))).reset_index()
            .sort_values([ccol, "user_id", "t"]))
    w0, b0 = torch.zeros(A, A), torch.zeros(A)
    Ds = []
    for _, gs in ob.groupby(ccol):
        X, Y = [], []
        for _, gh in gs.groupby("user_id"):
            cl = list(gh.ai)
            for i in range(len(cl) - 1):
                X.append(cl[i]); Y.append(cl[i + 1])
        if len(X) < 50: continue
        Xt, Yt = torch.zeros(len(X), A), torch.zeros(len(Y), A)
        for r, (a, b) in enumerate(zip(X, Y)):
            Xt[r, a] = 1.0; Yt[r, b] = 1.0
        m = nn.Linear(A, A)
        with torch.no_grad(): m.weight.copy_(w0); m.bias.copy_(b0)
        opt = torch.optim.SGD(m.parameters(), lr=0.05); lf = nn.BCEWithLogitsLoss()
        for _ in range(30):
            i = torch.randint(0, len(Xt), (min(128, len(Xt)),))
            opt.zero_grad(); lf(m(Xt[i]), Yt[i]).backward(); opt.step()
        with torch.no_grad():
            Ds.append(torch.cat([(m.weight - w0).flatten(), (m.bias - b0).flatten()]).numpy())
    Dm = np.stack(Ds); Dn = Dm / np.linalg.norm(Dm, axis=1, keepdims=True)
    C = Dn @ Dn.T; iu = np.triu_indices(len(Dm), 1)
    return float(np.median(C[iu]))


# --------------------------------------------------------- 귀무 (주문 랜덤) — 한 번만
ord_sz = ic.groupby("order_id").size()
keys = np.array(targets.index); vals = targets.values.astype(float).copy()
nul = {}
for o in np.random.default_rng(7).permutation(list(ord_sz.index)):
    i = int(np.argmax(vals)); nul[o] = keys[i]; vals[i] -= ord_sz[o]
ic["c_null"] = ic.order_id.map(nul)
null_d = dist_stats(ic.dropna(subset=["c_null"]), "c_null")
null_u = update_div(ic.dropna(subset=["c_null"]), "c_null")
print(f"귀무(주문랜덤): JS {null_d['js']:.4f}  pair {null_d['pair']:.4f}  "
      f"TV {null_d['tv']:.4f}  cos {null_u:.4f}")

# --------------------------------------------------------- alpha sweep
print("\n" + "=" * 100)
print(f"{'alpha':>8} {'clients':>8} {'Gini':>7} {'JS':>8} {'JS R':>7} {'pairJS':>8} {'TV':>8} "
      f"{'1-cos':>8} {'upd R':>7}")
print("=" * 100)
rows = []
for alpha in [0.12, 0.16, 0.20, 0.25, 0.35, np.inf]:
    a = assign_dirichlet(alpha, np.random.default_rng(1))
    ic["c"] = ic.user_id.map(a)
    sub = ic[ic.c.notna()]
    d = dist_stats(sub, "c")
    cosv = update_div(sub, "c")
    g = gini(sub.groupby("c").order_id.nunique())
    r = dict(alpha=float(alpha), n=int(sub.c.nunique()), gini=float(g), **d,
             js_R=d["js"] / null_d["js"], cos=cosv,
             upd_R=(1 - cosv) / max(1 - null_u, 1e-9))
    rows.append(r)
    print(f"{str(alpha):>8} {r['n']:>8} {g:>7.3f} {d['js']:>8.4f} {r['js_R']:>7.2f}x "
          f"{d['pair']:>8.4f} {d['tv']:>8.4f} {1-cosv:>8.4f} {r['upd_R']:>7.2f}x")

print("\n목표 (dunnhumby 실제 매장):")
print(f"{'':>8} {'':>8} {TARGET['gini']:>7.3f} {'':>8} {TARGET['js_R']:>7.2f}x "
      f"{TARGET['pair_js']:>8.4f} {TARGET['tv']:>8.4f} {'':>8} {TARGET['upd_R']:>7.2f}x")

# JS R 은 귀무 바닥값이 데이터셋마다 달라 교차 비교 불가 -> 절대량(pairJS, TV)과 upd_R 로 맞춘다
best = min(rows, key=lambda r: 2*abs(np.log(r["upd_R"]/TARGET["upd_R"])) + abs(np.log(r["pair"]/TARGET["pair_js"])) + abs(np.log(r["tv"]/TARGET["tv"])) + abs(np.log(r["gini"]/TARGET["gini"])))
print(f"\n>> 가장 가까운 alpha = {best['alpha']}  (JS R {best['js_R']:.2f}x, upd R {best['upd_R']:.2f}x)")

json.dump(dict(sweep=rows, null=dict(**null_d, cos=null_u), target=TARGET, best_alpha=best["alpha"]),
          open("out/instacart/alpha_sweep.json", "w"), indent=1)
a = assign_dirichlet(best["alpha"], np.random.default_rng(1))
a.rename("client").rename_axis("user_id").reset_index().dropna().to_csv(
    "out/instacart/client_assignment_matched.csv", index=False)
print("-> out/instacart/alpha_sweep.json, client_assignment_matched.csv")
