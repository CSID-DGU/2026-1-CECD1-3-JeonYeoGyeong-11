"""Instacart 100 클라이언트 이질성 다각도 측정 — dunnhumby 와 동일 지표.

클라이언트 구성: 고객을 dunnhumby client 크기에 맞춰 100개로 랜덤 배정.
시간 정렬: 각 고객 첫 주문 t=0 ("가입 후 n일").
귀무가설: 주문(바스켓) 랜덤 — 크기 동일.
"""
import json, os
import numpy as np
import pandas as pd
import pyreadr
import torch
import torch.nn as nn
from scipy.special import psi

rng = np.random.default_rng(0)
torch.manual_seed(0)
os.makedirs("out/instacart", exist_ok=True)
D = "data/instacart"

# ---------------------------------------------------------------- dunnhumby 목표 크기
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
bk = dh.groupby("store_id").basket_id.nunique()
targets = bk[bk >= 300].sort_values(ascending=False).iloc[:100]
del tx, dh

# ---------------------------------------------------------------- Instacart 클라이언트
orders = pd.read_csv(f"{D}/orders.csv")
orders = orders[orders.eval_set == "prior"].sort_values(["user_id", "order_number"])
orders["gap"] = orders.days_since_prior_order.fillna(0)
orders["t"] = orders.groupby("user_id").gap.cumsum()          # 가입 후 경과일


def greedy(unit_sizes, tgt, rng):
    keys = np.array(tgt.index); vals = tgt.values.astype(float).copy()
    out = {}
    for u in rng.permutation(list(unit_sizes.index)):
        i = int(np.argmax(vals)); out[u] = keys[i]; vals[i] -= unit_sizes[u]
    return out


n_ord = orders.groupby("user_id").size()
need = targets.sum()
pick, acc = [], 0
for u in rng.permutation(list(n_ord.index)):
    pick.append(u); acc += n_ord[u]
    if acc >= need * 1.05:
        break
orders = orders[orders.user_id.isin(pick)].copy()
orders["client"] = orders.user_id.map(greedy(n_ord.loc[pick], targets, rng))
orders[["user_id", "client"]].drop_duplicates().to_csv("out/instacart/client_assignment.csv", index=False)

prior = pd.read_csv(f"{D}/order_products__prior.csv", usecols=["order_id", "product_id"])
prod = pd.read_csv(f"{D}/products.csv", usecols=["product_id", "aisle_id", "department_id"])
ic = (prior[prior.order_id.isin(set(orders.order_id))]
      .merge(prod, on="product_id", how="left")
      .merge(orders[["order_id", "user_id", "client", "t", "order_dow", "order_hour_of_day"]],
             on="order_id", how="left"))
ord_sz = ic.groupby("order_id").size()
ic["c_null"] = ic.order_id.map(greedy(ord_sz, targets, np.random.default_rng(7)))
print(f"client {ic.client.nunique()}  주문 {ic.order_id.nunique():,}  고객 {ic.user_id.nunique():,}  행 {len(ic):,}")

# ---------------------------------------------------------------- helpers
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


def dm_kappa(C, mu, iters=500):
    N = C.sum(1); k = 100.0
    for _ in range(iters):
        num = (mu * (psi(C + k * mu) - psi(k * mu))).sum()
        den = (psi(N + k) - psi(k)).sum()
        if den <= 0 or not np.isfinite(num / den): break
        kn = k * num / den
        if not np.isfinite(kn) or kn <= 0: break
        conv = abs(kn - k) / k < 1e-9; k = kn
        if conv: break
    return k


rep = {}
print("\n" + "=" * 96); print("1. 분포 이질성 (aisle 134 / department 21)"); print("=" * 96)
for lvl, col in [("department", "department_id"), ("aisle", "aisle_id")]:
    blk = {}
    for tag, cc in [("real", "client"), ("null", "c_null")]:
        M = ic.groupby([cc, col]).size().unstack(fill_value=0)
        Cm = M.values.astype(float); P = Cm / Cm.sum(1, keepdims=True)
        g = Cm.sum(0) / Cm.sum(); JS = js_rows(P, g)
        with np.errstate(divide="ignore", invalid="ignore"):
            H = -np.where(P > 0, P * np.log(P), 0.0).sum(1) / np.log(P.shape[1])
        blk[tag] = dict(js_med=float(np.median(JS)), q1=float(np.percentile(JS, 25)),
                        q3=float(np.percentile(JS, 75)), p90=float(np.percentile(JS, 90)),
                        pair=float(np.median(pair_js(P))), tv=float(np.median(0.5*np.abs(P-g).sum(1))),
                        H=float(np.median(H)), kappa=float(dm_kappa(Cm, g)))
        b = blk[tag]
        print(f"  [{lvl:<10}|{tag:<4}] JS med {b['js_med']:.4f}  Q1 {b['q1']:.4f}  Q3 {b['q3']:.4f}  "
              f"P90 {b['p90']:.4f} | pair {b['pair']:.4f} | TV {b['tv']:.4f} | H {b['H']:.3f} | kappa {b['kappa']:,.0f}")
    blk["R_js"] = blk["real"]["js_med"] / blk["null"]["js_med"]
    blk["R_kappa"] = blk["null"]["kappa"] / blk["real"]["kappa"]
    print(f"  >> R(JS) {blk['R_js']:.2f}x   R(kappa) {blk['R_kappa']:.2f}x")
    rep[lvl] = blk

# 크기 계층별 R
print("\n  크기 계층별 (주문 수):")
M = ic.groupby(["client", "aisle_id"]).size().unstack(fill_value=0)
Mn = ic.groupby(["c_null", "aisle_id"]).size().unstack(fill_value=0)
oc = ic.groupby("client").order_id.nunique().reindex(M.index).values
ocn = ic.groupby("c_null").order_id.nunique().reindex(Mn.index).values
P = M.values / M.values.sum(1, keepdims=True); JS = js_rows(P, M.values.sum(0)/M.values.sum())
Pn = Mn.values / Mn.values.sum(1, keepdims=True); JSn = js_rows(Pn, Mn.values.sum(0)/Mn.values.sum())
strata = {}
for lo, hi, lab in [(0, 700, "<700"), (700, 1200, "700-1.2k"), (1200, 99999, ">1.2k")]:
    m, mn = (oc >= lo) & (oc < hi), (ocn >= lo) & (ocn < hi)
    if m.sum() < 3 or mn.sum() < 3: continue
    jr, jn = float(np.median(JS[m])), float(np.median(JSn[mn]))
    strata[lab] = dict(n=int(m.sum()), real=jr, null=jn, R=jr/jn)
    print(f"    {lab:>9}: clients {m.sum():>3}   JS real {jr:.4f}   null {jn:.4f}   R {jr/jn:.2f}x")
rep["by_size"] = strata

# ---------------------------------------------------------------- 데이터량
print("\n" + "=" * 96); print("2. 데이터량 불균형"); print("=" * 96)
g = ic.groupby("client")
q = {}
for k, v in [("orders", g.order_id.nunique()), ("items", g.size()), ("users", g.user_id.nunique())]:
    q[k] = dict(cv=float(v.std(ddof=1)/v.mean()), gini=float(gini(v)), median=float(v.median()),
                min=int(v.min()), max=int(v.max()), ratio=float(v.max()/v.min()))
    r = q[k]
    print(f"  {k:<8} CV {r['cv']:.2f}  Gini {r['gini']:.3f}  중앙 {r['median']:,.0f}  "
          f"{r['min']:,}~{r['max']:,}  max/min {r['ratio']:.1f}x")
rep["quantity_skew"] = q

# ---------------------------------------------------------------- 시간
print("\n" + "=" * 96); print("3. 시간 이질성"); print("=" * 96)
ic["wk"] = (ic.t // 7).astype(int).clip(upper=52)
t = {}
for f in ["order_dow", "order_hour_of_day", "wk"]:
    o = {}
    for tag, cc in [("real", "client"), ("null", "c_null")]:
        M = ic.groupby([cc, f]).size().unstack(fill_value=0).values.astype(float)
        P = M / M.sum(1, keepdims=True)
        o[tag] = float(np.median(js_rows(P, M.sum(0)/M.sum())))
    o["R"] = o["real"] / o["null"]; t[f] = o
    print(f"  {f:<18} JS real {o['real']:.4f}   null {o['null']:.4f}   R {o['R']:.2f}x")
rep["temporal"] = t

# ---------------------------------------------------------------- 태스크 정렬
print("\n" + "=" * 96); print("4. 태스크 정렬 (재방문 / 라벨 / 서비스)"); print("=" * 96)
ov = orders.sort_values(["user_id", "t"])
ov["nxt"] = ov.groupby("user_id").t.shift(-1) - ov.t
gg = ov.dropna(subset=["nxt"])
p7 = gg.groupby("client").nxt.apply(lambda x: (x <= 7).mean())
p14 = gg.groupby("client").nxt.apply(lambda x: (x <= 14).mean())
print(f"  7일 재방문 양성률  중앙 {p7.median()*100:.1f}%  범위 {p7.min()*100:.0f}~{p7.max()*100:.0f}%")
print(f"  14일 재방문 양성률 중앙 {p14.median()*100:.1f}%  범위 {p14.min()*100:.0f}~{p14.max()*100:.0f}%")
tops = {c: set(gp.aisle_id.value_counts().head(10).index) for c, gp in ic.groupby("client")}
ks = list(tops); ovl = [len(tops[a] & tops[b])/10 for i, a in enumerate(ks) for b in ks[i+1:]]
print(f"  client쌍 top-10 aisle 겹침 중앙 {np.median(ovl)*100:.0f}%")
rep["task"] = dict(p7_med=float(p7.median()), p7_min=float(p7.min()), p7_max=float(p7.max()),
                   p14_med=float(p14.median()), top10_overlap=float(np.median(ovl)))

# ---------------------------------------------------------------- 언러닝 영향력
print("\n" + "=" * 96); print("5. 언러닝 영향력 (client 제거 시 집계 변화)"); print("=" * 96)
M = ic.groupby(["client", "aisle_id"]).size().unstack(fill_value=0)
Cm = M.values.astype(float); P = Cm / Cm.sum(1, keepdims=True)
n = ic.groupby("client").size().reindex(M.index).values.astype(float)
w = n / n.sum(); G = (w[:, None] * P).sum(0)
infl = np.array([np.linalg.norm((w[np.arange(len(w)) != i][:, None] * P[np.arange(len(w)) != i]).sum(0)
                                / w[np.arange(len(w)) != i].sum() - G, 1) / np.linalg.norm(G, 1)
                 for i in range(len(w))])
print(f"  영향력 중앙 {np.median(infl)*100:.3f}%   최대 {infl.max()*100:.3f}%   "
      f"(dunnhumby: 중앙 0.274%, 최대 0.850%)")
rep["unlearn_influence"] = dict(median=float(np.median(infl)), max=float(infl.max()))

# ---------------------------------------------------------------- update cosine
print("\n" + "=" * 96); print("6. Update divergence (선형 프록시, 직전 주문 -> 다음 주문 aisle)"); print("=" * 96)
A = int(ic.aisle_id.max()) + 1
ob = ic.groupby(["client", "user_id", "order_id"]).agg(t=("t", "first"),
                                                       ai=("aisle_id", lambda s: sorted(set(s)))).reset_index()
ob = ob.sort_values(["client", "user_id", "t"])
res = {}
for tag, cc in [("real", "client"), ("null", "c_null")]:
    if tag == "null":
        omap = ic.drop_duplicates("order_id").set_index("order_id").c_null
        ob["c2"] = ob.order_id.map(omap); key = "c2"
    else:
        key = "client"
    w0, b0 = torch.zeros(A, A), torch.zeros(A)
    Ds = []
    for c, gs in ob.groupby(key):
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
    Cs = Dn @ Dn.T; iu = np.triu_indices(len(Dm), 1)
    res[tag] = dict(n=len(Dm), cos_med=float(np.median(Cs[iu])), cos_min=float(Cs[iu].min()))
    print(f"  [{tag:<4}] clients {len(Dm)}  cos 중앙 {res[tag]['cos_med']:.4f}  최소 {res[tag]['cos_min']:.4f}")
res["R_1mcos"] = (1 - res["real"]["cos_med"]) / max(1 - res["null"]["cos_med"], 1e-9)
print(f"  >> 1-cos: real {1-res['real']['cos_med']:.4f}  null {1-res['null']['cos_med']:.4f}  "
      f"=> {res['R_1mcos']:.2f}x   (dunnhumby: 2.36x)")
rep["update_divergence"] = res

json.dump(rep, open("out/instacart/heterogeneity.json", "w"), indent=1)
print("\n-> out/instacart/heterogeneity.json, out/instacart/client_assignment.csv")
