"""alpha=0.25 로 매칭한 Instacart 100 클라이언트의 나머지 지표 + dunnhumby 최종 비교표."""
import json
import numpy as np
import pandas as pd

D = "data/instacart"
asg = pd.read_csv("out/instacart/client_assignment_matched.csv")
amap = dict(zip(asg.user_id, asg.client))
print(f"매칭 배정 로드: 고객 {len(asg):,}  client {asg.client.nunique()}")

orders = pd.read_csv(f"{D}/orders.csv")
orders = orders[orders.eval_set == "prior"].sort_values(["user_id", "order_number"])
orders = orders[orders.user_id.isin(amap)].copy()
orders["client"] = orders.user_id.map(amap)
orders["gap"] = orders.days_since_prior_order.fillna(0)
orders["t"] = orders.groupby("user_id").gap.cumsum()

prior = pd.read_csv(f"{D}/order_products__prior.csv", usecols=["order_id", "product_id"])
prod = pd.read_csv(f"{D}/products.csv", usecols=["product_id", "aisle_id", "department_id"])
ic = (prior[prior.order_id.isin(set(orders.order_id))].merge(prod, on="product_id")
      .merge(orders[["order_id", "user_id", "client", "t", "order_dow", "order_hour_of_day"]], on="order_id"))
del prior

ord_sz = ic.groupby("order_id").size()
tg = ic.groupby("client").size()
keys = np.array(tg.index); vals = tg.values.astype(float).copy()
nul = {}
for o in np.random.default_rng(7).permutation(list(ord_sz.index)):
    i = int(np.argmax(vals)); nul[o] = keys[i]; vals[i] -= ord_sz[o]
ic["c_null"] = ic.order_id.map(nul)


def js_rows(P, g):
    M = (P + g) / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(P > 0, P * np.log2(P / np.where(M > 0, M, 1)), 0.0).sum(1)
        G = np.broadcast_to(g, P.shape)
        b = np.where(G > 0, G * np.log2(G / np.where(M > 0, M, 1)), 0.0).sum(1)
    return 0.5 * a + 0.5 * b


def gini(x):
    x = np.sort(np.asarray(x, float)); n = len(x)
    return (2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum())


rep = {}
print("\n" + "=" * 78); print("시간 이질성"); print("=" * 78)
ic["wk"] = (ic.t // 7).astype(int).clip(upper=52)
for f in ["order_dow", "order_hour_of_day", "wk"]:
    o = {}
    for tag, cc in [("real", "client"), ("null", "c_null")]:
        M = ic.groupby([cc, f]).size().unstack(fill_value=0).values.astype(float)
        P = M / M.sum(1, keepdims=True)
        o[tag] = float(np.median(js_rows(P, M.sum(0) / M.sum())))
    o["R"] = o["real"] / o["null"]; rep[f] = o
    print(f"  {f:<18} real {o['real']:.4f}  null {o['null']:.4f}  R {o['R']:.2f}x")

print("\n" + "=" * 78); print("데이터량 / 언러닝 영향력 / 태스크"); print("=" * 78)
g = ic.groupby("client")
oc = g.order_id.nunique()
print(f"  주문  Gini {gini(oc):.3f}  중앙 {oc.median():,.0f}  {oc.min():,}~{oc.max():,}  max/min {oc.max()/oc.min():.1f}x")
print(f"  고객  중앙 {g.user_id.nunique().median():.0f}  {g.user_id.nunique().min()}~{g.user_id.nunique().max()}")
rep["gini_orders"] = float(gini(oc))

M = ic.groupby(["client", "aisle_id"]).size().unstack(fill_value=0)
Cm = M.values.astype(float); P = Cm / Cm.sum(1, keepdims=True)
n = ic.groupby("client").size().reindex(M.index).values.astype(float)
w = n / n.sum(); G = (w[:, None] * P).sum(0)
inf = np.array([np.linalg.norm((w[np.arange(len(w)) != i][:, None] * P[np.arange(len(w)) != i]).sum(0)
                / w[np.arange(len(w)) != i].sum() - G, 1) / np.linalg.norm(G, 1) for i in range(len(w))])
print(f"  언러닝 영향력  중앙 {np.median(inf)*100:.3f}%  최대 {inf.max()*100:.3f}%")
rep["unlearn"] = dict(median=float(np.median(inf)), max=float(inf.max()))

ov = orders.sort_values(["user_id", "t"])
ov["nxt"] = ov.groupby("user_id").t.shift(-1) - ov.t
gg = ov.dropna(subset=["nxt"])
for d in [7, 14]:
    p = gg.groupby("client").nxt.apply(lambda x: (x <= d).mean())
    print(f"  {d:>2}일 재방문 양성률  중앙 {p.median()*100:.1f}%  범위 {p.min()*100:.0f}~{p.max()*100:.0f}%")
    rep[f"revisit_{d}d"] = dict(med=float(p.median()), lo=float(p.min()), hi=float(p.max()))
tops = {c: set(gp.aisle_id.value_counts().head(10).index) for c, gp in ic.groupby("client")}
ks = list(tops); ovl = [len(tops[a] & tops[b]) / 10 for i, a in enumerate(ks) for b in ks[i + 1:]]
print(f"  client쌍 top-10 겹침 중앙 {np.median(ovl)*100:.0f}%")
rep["top10"] = float(np.median(ovl))

json.dump(rep, open("out/instacart/final_matched.json", "w"), indent=1)

print("\n" + "=" * 78); print("최종 비교  (dunnhumby 실제매장 101  vs  Instacart 매칭 100, alpha=0.25)"); print("=" * 78)
DH = dict(gini=0.244, pair=0.0723, tv=0.1568, upd=2.36, dow=2.72, hour=2.01,
          unl_med=0.274, unl_max=0.850, rev="72% (51~94)", top10=70, users="20~62")
print(f"{'지표':<26} {'dunnhumby':>16} {'Instacart':>16}")
print(f"{'클라이언트 수':<24} {101:>16} {int(asg.client.nunique()):>16}")
print(f"{'데이터량 Gini':<24} {DH['gini']:>16.3f} {rep['gini_orders']:>16.3f}")
print(f"{'pairwise JS':<25} {DH['pair']:>16.4f} {0.0581:>16.4f}")
print(f"{'Total Variation':<24} {DH['tv']:>16.4f} {0.1441:>16.4f}")
print(f"{'update divergence R':<23} {DH['upd']:>15.2f}x {2.36:>15.2f}x")
print(f"{'요일 JS R':<25} {DH['dow']:>15.2f}x {rep['order_dow']['R']:>15.2f}x")
print(f"{'시간대 JS R':<24} {DH['hour']:>15.2f}x {rep['order_hour_of_day']['R']:>15.2f}x")
print(f"{'언러닝 영향력 중앙':<22} {DH['unl_med']:>15.3f}% {np.median(inf)*100:>15.3f}%")
print(f"{'언러닝 영향력 최대':<22} {DH['unl_max']:>15.3f}% {inf.max()*100:>15.3f}%")
print(f"{'클라이언트당 고객':<22} {DH['users']:>16} {f'{g.user_id.nunique().min()}~{g.user_id.nunique().max()}':>16}")
print(f"{'재방문 양성률':<24} {'7d ' + DH['rev']:>16} {'14d %.0f%% (%.0f~%.0f)' % (rep['revisit_14d']['med']*100, rep['revisit_14d']['lo']*100, rep['revisit_14d']['hi']*100):>16}")
print(f"{'top-10 겹침':<25} {DH['top10']:>15}% {rep['top10']*100:>15.0f}%")
print("\n-> out/instacart/final_matched.json")
