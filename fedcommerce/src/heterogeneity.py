"""dunnhumby store-level heterogeneity 종합 측정.

핵심 원칙: 모든 분포 지표를 (1) 실제 store 분할, (2) store 크기를 동일하게 유지한 랜덤 분할
과 나란히 계산하고 비율 R = median(real) / median(random) 으로 보고한다.
절대값만으로는 "JS 0.035 가 큰 건가" 에 답할 수 없기 때문.
"""
import json, os
import numpy as np
import pandas as pd
import pyreadr
from scipy.special import psi

rng = np.random.default_rng(0)
os.makedirs("out/het", exist_ok=True)

# ------------------------------------------------------------------ load & clean
tx = pyreadr.read_r("data/transactions.rds")[None]
tx.columns = [c.lower() for c in tx.columns]
pr = pyreadr.read_r("data/products.rda")["products"]
pr.columns = [c.lower() for c in pr.columns]
tx = tx.merge(pr[["product_id", "department", "product_category"]], on="product_id", how="left")
n0 = len(tx)
tx = tx[tx.product_category.notna() & tx.department.notna() & (tx.quantity > 0)
        & (tx.product_category != "COUPON/MISC ITEMS")]
tx["ts"] = pd.to_datetime(tx.transaction_timestamp)
print(f"rows {n0:,} -> {len(tx):,} ({len(tx)/n0*100:.1f}%)   "
      f"stores {tx.store_id.nunique()}   households {tx.household_id.nunique()}")


# ------------------------------------------------------------------ metrics
def js_to_ref(P, g):
    """P (K,C) 행별 분포, g (C,) 기준분포 -> JS (K,), base 2"""
    M = (P + g) / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(P > 0, P * np.log2(P / np.where(M > 0, M, 1)), 0.0).sum(1)
        G = np.broadcast_to(g, P.shape)
        b = np.where(G > 0, G * np.log2(G / np.where(M > 0, M, 1)), 0.0).sum(1)
    return 0.5 * a + 0.5 * b


def pairwise_js(P, cap=200):
    """행이 많으면 cap 개 무작위 표본으로 추정"""
    K = len(P)
    idx = rng.choice(K, min(K, cap), replace=False) if K > cap else np.arange(K)
    Q = P[idx]
    out = []
    for i in range(len(Q) - 1):
        M = (Q[i] + Q[i + 1:]) / 2
        with np.errstate(divide="ignore", invalid="ignore"):
            a = np.where(Q[i] > 0, Q[i] * np.log2(Q[i] / np.where(M > 0, M, 1)), 0.0).sum(1)
            b = np.where(Q[i + 1:] > 0, Q[i + 1:] * np.log2(Q[i + 1:] / np.where(M > 0, M, 1)), 0.0).sum(1)
        out.append(0.5 * a + 0.5 * b)
    return np.concatenate(out) if out else np.array([])


def tv_to_ref(P, g):
    return 0.5 * np.abs(P - g).sum(1)


def norm_entropy(P):
    with np.errstate(divide="ignore", invalid="ignore"):
        H = -np.where(P > 0, P * np.log(P), 0.0).sum(1)
    return H / np.log(P.shape[1])


def gini(x):
    x = np.sort(np.asarray(x, float))
    n = len(x)
    return (2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum())


def cv(x):
    x = np.asarray(x, float)
    return x.std(ddof=1) / x.mean()


def dm_kappa(C, mu, iters=500):
    """n_s ~ DirichletMultinomial(N_s, kappa*mu), mu 고정. Minka fixed-point.

    kappa -> inf 는 '각 store 가 global 분포에서 뽑은 순수 다항 표본' (= 랜덤 분할) 을 뜻한다.
    표본 크기를 모델이 직접 다루므로, 작은 store 의 비율 변동이 이질성으로 오인되지 않는다.
    """
    N = C.sum(1)
    k = 100.0
    for _ in range(iters):
        num = (mu * (psi(C + k * mu) - psi(k * mu))).sum()
        den = (psi(N + k) - psi(k)).sum()
        if den <= 0 or not np.isfinite(num) or not np.isfinite(den):
            break
        k_new = k * num / den
        if not np.isfinite(k_new) or k_new <= 0:
            break
        converged = abs(k_new - k) / k < 1e-9
        k = k_new
        if converged:
            break
    return k


def counts(df, level, store_col="store_id"):
    return df.groupby([store_col, level]).size().unstack(fill_value=0)


# ------------------------------------------------------------------ random partitions
store_sizes = tx.store_id.value_counts()
targets = store_sizes.to_dict()


def greedy_fill(unit_sizes, targets, rng):
    """단위(바스켓/가구)를 무작위 순서로, 남은 결손이 가장 큰 store 에 배정.
    단위 크기가 store 목표보다 크면 정확한 일치는 불가능하다 (가구 단위에서 실제로 발생)."""
    order = rng.permutation(list(unit_sizes.index))
    deficit = dict(targets)
    keys = np.array(list(deficit.keys()))
    vals = np.array([deficit[k] for k in keys], float)
    assign = {}
    for u in order:
        i = int(np.argmax(vals))
        assign[u] = keys[i]
        vals[i] -= unit_sizes[u]
    return assign


# (a) 랜덤-거래: store 라벨을 행 단위로 섞음 -> 크기 정확히 보존. 순수 다항 표본추출 노이즈.
tx["store_rand_tx"] = rng.permutation(tx.store_id.values)

# (b) 랜덤-바스켓: 바스켓을 통째로 섞음 -> 바스켓 내 상품 상관 보존. 주 귀무가설.
bk_rows = tx.groupby("basket_id").size()
tx["store_rand_bk"] = tx.basket_id.map(greedy_fill(bk_rows, targets, rng))

# (c) 랜덤-가구: 가구를 통째로 섞음 -> 가구 취향까지 보존. 가장 엄격.
hh_rows = tx.groupby("household_id").size()
tx["store_rand_hh"] = tx.household_id.map(greedy_fill(hh_rows, targets, rng))

for c in ["store_rand_tx", "store_rand_bk", "store_rand_hh"]:
    s = tx[c].value_counts()
    print(f"  {c:<15} stores {len(s):>4}  size corr(real) "
          f"{np.corrcoef(s.reindex(store_sizes.index).fillna(0), store_sizes)[0,1]:.3f}")

# ------------------------------------------------------------------ distribution skew
report = {}
for level in ["department", "product_category"]:
    print("\n" + "=" * 96)
    print(f"LEVEL = {level}  ({tx[level].nunique()} labels)")
    print("=" * 96)
    block = {}
    for tag, col in [("real", "store_id"), ("random_tx", "store_rand_tx"), ("random_bk", "store_rand_bk"), ("random_hh", "store_rand_hh")]:
        M = counts(tx, level, col)
        Cm = M.values.astype(float)
        N = Cm.sum(1)
        P = Cm / N[:, None]
        g = Cm.sum(0) / Cm.sum()
        JS, TV, HN = js_to_ref(P, g), tv_to_ref(P, g), norm_entropy(P)
        PJ = pairwise_js(P)
        kap = dm_kappa(Cm, g)
        block[tag] = dict(
            stores=int(len(M)),
            js=dict(median=float(np.median(JS)), q1=float(np.percentile(JS, 25)),
                    q3=float(np.percentile(JS, 75)), p90=float(np.percentile(JS, 90))),
            pairwise_js_median=float(np.median(PJ)) if len(PJ) else None,
            tv=dict(median=float(np.median(TV)), q1=float(np.percentile(TV, 25)),
                    q3=float(np.percentile(TV, 75))),
            entropy_norm_median=float(np.median(HN)),
            kappa=float(kap))
        b = block[tag]
        print(f"  [{tag:<9}] JS med {b['js']['median']:.4f}  Q1 {b['js']['q1']:.4f}  "
              f"Q3 {b['js']['q3']:.4f}  P90 {b['js']['p90']:.4f} | pairJS {b['pairwise_js_median']:.4f} "
              f"| TV med {b['tv']['median']:.4f} | H_norm {b['entropy_norm_median']:.3f} | kappa {kap:,.0f}")

    for nl in ["random_tx","random_bk","random_hh"]:
        block[f"R_JS_vs_{nl}"] = block["real"]["js"]["median"] / block[nl]["js"]["median"]
    print(f"  >> R_JS  vs random_tx {block['R_JS_vs_random_tx']:.2f}x   "
          f"vs random_bk {block['R_JS_vs_random_bk']:.2f}x   vs random_hh {block['R_JS_vs_random_hh']:.2f}x")

    # 크기 계층별 R
    Cm = counts(tx, level).values.astype(float)
    P = Cm / Cm.sum(1, keepdims=True)
    JSr = js_to_ref(P, Cm.sum(0) / Cm.sum())
    Cr = counts(tx, level, "store_rand_bk").values.astype(float)
    Pr = Cr / Cr.sum(1, keepdims=True)
    JSn = js_to_ref(Pr, Cr.sum(0) / Cr.sum())
    sz, szr = Cm.sum(1), Cr.sum(1)
    strata = {}
    print("  크기 계층별 (거래 행 수, 귀무=random_bk):")
    for lo, hi, lab in [(0, 500, "<500"), (500, 2000, "500-2k"),
                        (2000, 10000, "2k-10k"), (10000, np.inf, ">10k")]:
        m, mr = (sz >= lo) & (sz < hi), (szr >= lo) & (szr < hi)
        if m.sum() < 3 or mr.sum() < 3:
            continue
        jr, jn = float(np.median(JSr[m])), float(np.median(JSn[mr]))
        strata[lab] = dict(n_stores=int(m.sum()), js_real=jr, js_random=jn, R=jr / jn)
        print(f"    {lab:>7}: stores {m.sum():>3}   JS real {jr:.4f}   random {jn:.4f}   R {jr/jn:.2f}x")
    block["by_size"] = strata
    report[level] = block

# ------------------------------------------------------------------ quantity skew
print("\n" + "=" * 96)
print("QUANTITY SKEW (전체 store)")
print("=" * 96)
g = tx.groupby("store_id")
qrep = {}
for k, v in [("transactions", g.size()), ("baskets", g.basket_id.nunique()),
             ("households", g.household_id.nunique())]:
    qrep[k] = dict(cv=float(cv(v)), gini=float(gini(v)), median=float(v.median()),
                   p90=float(v.quantile(.9)), max=int(v.max()), min=int(v.min()),
                   ratio_p90_p10=float(v.quantile(.9) / max(v.quantile(.1), 1)))
    r = qrep[k]
    print(f"  {k:<13} CV {r['cv']:.2f}  Gini {r['gini']:.3f}  median {r['median']:,.0f}  "
          f"P90 {r['p90']:,.0f}  max {r['max']:,}  P90/P10 {r['ratio_p90_p10']:.1f}x")
report["quantity_skew"] = qrep

# ------------------------------------------------------------------ household overlap
print("\n" + "=" * 96)
print("HOUSEHOLD OVERLAP (client 독립성)")
print("=" * 96)
hs = tx.groupby(["household_id", "store_id"]).size().rename("n").reset_index()
nst = hs.groupby("household_id").size()
tot = hs.groupby("household_id").n.sum()
top = hs.sort_values("n", ascending=False).groupby("household_id").head(1)
conc = (top.set_index("household_id").n / tot).dropna()
big = list(store_sizes[store_sizes >= 2000].index)
sets = {s: set(hs[hs.store_id == s].household_id) for s in big}
J = [len(sets[a] & sets[b]) / len(sets[a] | sets[b])
     for i, a in enumerate(big) for b in big[i + 1:]]
horep = dict(mean_stores_per_hh=float(nst.mean()), median_stores_per_hh=float(nst.median()),
             p_uses_ge2=float((nst >= 2).mean()), p_uses_ge5=float((nst >= 5).mean()),
             primary_share_mean=float(conc.mean()), primary_share_median=float(conc.median()),
             jaccard_median=float(np.median(J)), jaccard_p90=float(np.percentile(J, 90)),
             n_stores_jaccard=len(big))
print(f"  가구당 store 수: 평균 {horep['mean_stores_per_hh']:.1f}  중앙 {horep['median_stores_per_hh']:.0f}")
print(f"  2개 이상 이용 {horep['p_uses_ge2']*100:.1f}%    5개 이상 {horep['p_uses_ge5']*100:.1f}%")
print(f"  주이용 store 집중도: 평균 {horep['primary_share_mean']*100:.1f}%  "
      f"중앙 {horep['primary_share_median']*100:.1f}%")
print(f"  store쌍 household Jaccard (>=2000행 store {len(big)}개): "
      f"중앙 {horep['jaccard_median']:.4f}  P90 {horep['jaccard_p90']:.4f}")
report["household_overlap"] = horep

# ------------------------------------------------------------------ temporal skew
print("\n" + "=" * 96)
print("TEMPORAL SKEW")
print("=" * 96)
tx["hour"] = tx.ts.dt.hour
tx["dow"] = tx.ts.dt.dayofweek
tx["month"] = tx.ts.dt.month
trep = {}
for f in ["hour", "dow", "month"]:
    out = {}
    for tag, col in [("real", "store_id"), ("random_bk", "store_rand_bk")]:
        M = counts(tx, f, col).values.astype(float)
        P = M / M.sum(1, keepdims=True)
        out[tag] = float(np.median(js_to_ref(P, M.sum(0) / M.sum())))
    out["R"] = out["real"] / out["random_bk"]
    trep[f] = out
    print(f"  {f:<6} JS real {out['real']:.4f}   random_bk {out['random_bk']:.4f}   R {out['R']:.2f}x")
report["temporal"] = trep

json.dump(report, open("out/het/heterogeneity_report.json", "w"), indent=1)
print("\n-> out/het/heterogeneity_report.json")
