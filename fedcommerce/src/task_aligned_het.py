"""학습/언러닝/서비스 단위에 정렬된 이질성 측정.

앞선 heterogeneity.py 는 '거래 행 × 전체 store' 기준이었으나, 실제로는
  - 모델이 보는 단위 = 방문(바스켓) 안의 '중복 제거된 카테고리 집합'
  - 학습 데이터 = 배타 할당 후
  - 태스크 = (a) 7일 내 재방문  (b) 다음 방문 카테고리 multi-label
  - FL 이 섞는 것 = shared 부분만 (local head 는 client 내부)
  - 언러닝 = store 하나를 빼면 집계가 얼마나 바뀌나
이므로 그 기준으로 다시 잰다. 모든 지표는 크기를 맞춘 랜덤 분할과 나란히.
"""
import json, os
import numpy as np
import pandas as pd
import pyreadr

rng = np.random.default_rng(0)
os.makedirs("out/het", exist_ok=True)
MIN_BASKETS = 300      # 배타 할당 후 client 후보 기준

# ------------------------------------------------------------------ load / clean
tx = pyreadr.read_r("data/transactions.rds")[None]
tx.columns = [c.lower() for c in tx.columns]
pr = pyreadr.read_r("data/products.rda")["products"]
pr.columns = [c.lower() for c in pr.columns]
tx = tx.merge(pr[["product_id", "department", "product_category"]], on="product_id", how="left")
tx = tx[tx.product_category.notna() & tx.department.notna() & (tx.quantity > 0)
        & (tx.product_category != "COUPON/MISC ITEMS")]
tx["ts"] = pd.to_datetime(tx.transaction_timestamp)

# ------------------------------------------------------------------ 배타 할당 (학습 데이터와 동일)
hs = tx.groupby(["household_id", "store_id"]).basket_id.nunique().rename("b").reset_index()
prim = (hs.sort_values("b", ascending=False).groupby("household_id").head(1)
          [["household_id", "store_id"]].rename(columns={"store_id": "ps"}))
raw_bk = tx.groupby("store_id").basket_id.nunique()
ex = tx.merge(prim, on="household_id")
ex = ex[ex.store_id == ex.ps].copy()

# 배타 할당 손실이 store 마다 균등한가 (편향 점검)
kept_bk = ex.groupby("store_id").basket_id.nunique()
loss = (1 - kept_bk / raw_bk.reindex(kept_bk.index)).dropna()

# ------------------------------------------------------------------ 방문(visit) 테이블 = 학습 단위
vis = (ex.groupby(["store_id", "household_id", "basket_id"])
         .agg(ts=("ts", "min"), cats=("product_category", lambda s: frozenset(s)))
         .reset_index().sort_values(["store_id", "household_id", "ts"]))
vis["ncat"] = vis.cats.map(len)
vis["gap"] = vis.groupby(["store_id", "household_id"]).ts.diff().dt.total_seconds() / 86400

clients = kept_bk[kept_bk >= MIN_BASKETS].index
vis = vis[vis.store_id.isin(clients)].copy()
print(f"배타 할당 후 client 후보(바스켓>={MIN_BASKETS}): {len(clients)}개   "
      f"방문 {len(vis):,}   가구 {vis.household_id.nunique():,}")
print(f"배타 할당 손실률 store 간 편차: 중앙 {loss.reindex(clients).median()*100:.1f}%  "
      f"범위 {loss.reindex(clients).min()*100:.1f}~{loss.reindex(clients).max()*100:.1f}%")

# 크기 맞춘 랜덤 분할 (가구 단위 — 배타 할당이 이미 가구를 묶었으므로 동일 단위)
hh_bk = vis.groupby("household_id").size()
tgt = vis.store_id.value_counts().to_dict()
kk = np.array(list(tgt.keys())); vv = np.array([tgt[k] for k in kk], float)
assign = {}
for h in rng.permutation(list(hh_bk.index)):
    i = int(np.argmax(vv)); assign[h] = kk[i]; vv[i] -= hh_bk[h]
vis["store_rand"] = vis.household_id.map(assign)


def js_rows(P, g):
    M = (P + g) / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(P > 0, P * np.log2(P / np.where(M > 0, M, 1)), 0.0).sum(1)
        G = np.broadcast_to(g, P.shape)
        b = np.where(G > 0, G * np.log2(G / np.where(M > 0, M, 1)), 0.0).sum(1)
    return 0.5 * a + 0.5 * b


rep = {}

# ---------------- (1) 라벨 분포: 다음 방문 카테고리 (바스켓 단위, 중복 제거) -------------
print("\n" + "=" * 92)
print("(1) 다음-바스켓 라벨 분포  [단위: 방문당 unique 카테고리 — 모델이 보는 그대로]")
print("=" * 92)
rows = []
for col in ["store_id", "store_rand"]:
    rec = {}
    for s, g in vis.groupby(col):
        c = pd.Series([x for cs in g.cats for x in cs]).value_counts()
        rec[s] = c
    M = pd.DataFrame(rec).fillna(0).T
    Cm = M.values.astype(float)
    P = Cm / Cm.sum(1, keepdims=True)
    JS = js_rows(P, Cm.sum(0) / Cm.sum())
    rows.append((col, M, P, JS))
    print(f"  [{'real' if col=='store_id' else 'random':<7}] JS 중앙 {np.median(JS):.4f}  "
          f"Q1 {np.percentile(JS,25):.4f}  Q3 {np.percentile(JS,75):.4f}  P90 {np.percentile(JS,90):.4f}")
R_lab = np.median(rows[0][3]) / np.median(rows[1][3])
print(f"  >> R = {R_lab:.2f}x   (참고: 거래행 기준으로 쟀을 때는 5.0x)")
rep["label_dist"] = dict(js_real=float(np.median(rows[0][3])), js_random=float(np.median(rows[1][3])), R=float(R_lab))

# ---------------- (2) 재방문 라벨: 방문 간격 분포 -----------------------------------
print("\n" + "=" * 92)
print("(2) 재방문 라벨  [태스크 (a) — 7일/3일 양성률과 간격 분포]")
print("=" * 92)
BINS = [0, 1, 2, 3, 5, 8, 15, 31, np.inf]
LBL = ["0", "1", "2", "3-4", "5-7", "8-14", "15-30", "31+"]
out = {}
for col, tag in [("store_id", "real"), ("store_rand", "random")]:
    g = vis.dropna(subset=["gap"])
    pos7 = g.groupby(col).gap.apply(lambda x: (x <= 7).mean())
    pos3 = g.groupby(col).gap.apply(lambda x: (x <= 3).mean())
    H = (g.assign(b=pd.cut(g.gap, BINS, labels=LBL, right=False))
           .groupby([col, "b"], observed=False).size().unstack(fill_value=0))
    Cm = H.values.astype(float); P = Cm / Cm.sum(1, keepdims=True)
    JS = js_rows(P, Cm.sum(0) / Cm.sum())
    out[tag] = dict(js=float(np.median(JS)), pos7_med=float(pos7.median()),
                    pos7_min=float(pos7.min()), pos7_max=float(pos7.max()),
                    pos3_med=float(pos3.median()), pos3_min=float(pos3.min()), pos3_max=float(pos3.max()))
    o = out[tag]
    print(f"  [{tag:<7}] 간격분포 JS 중앙 {o['js']:.4f}   "
          f"7일 양성률 중앙 {o['pos7_med']*100:.1f}% (범위 {o['pos7_min']*100:.0f}~{o['pos7_max']*100:.0f}%)   "
          f"3일 {o['pos3_med']*100:.1f}% ({o['pos3_min']*100:.0f}~{o['pos3_max']*100:.0f}%)")
out["R"] = out["real"]["js"] / out["random"]["js"]
print(f"  >> R = {out['R']:.2f}x")
rep["revisit"] = out

# ---------------- (3) 서비스: 상위 카테고리 겹침 ------------------------------------
print("\n" + "=" * 92)
print("(3) 서비스 관점  [store 별 상위 카테고리가 실제로 다른가 — 대시보드/재고]")
print("=" * 92)
for col, tag in [("store_id", "real"), ("store_rand", "random")]:
    tops = {}
    for s, g in vis.groupby(col):
        c = pd.Series([x for cs in g.cats for x in cs]).value_counts()
        tops[s] = list(c.head(10).index)
    ks = list(tops)
    ov = [len(set(tops[a]) & set(tops[b])) / 10 for i, a in enumerate(ks) for b in ks[i+1:]]
    glob = pd.Series([x for cs in vis.cats for x in cs]).value_counts().head(10).index
    og = [len(set(tops[s]) & set(glob)) / 10 for s in ks]
    print(f"  [{tag:<7}] store쌍 top-10 겹침 중앙 {np.median(ov)*100:.0f}%   "
          f"전체 top-10 과 겹침 중앙 {np.median(og)*100:.0f}%")
    rep.setdefault("service_top10", {})[tag] = dict(pair_overlap=float(np.median(ov)),
                                                    global_overlap=float(np.median(og)))

# ---------------- (4) 언러닝: leave-one-out 영향력 ---------------------------------
print("\n" + "=" * 92)
print("(4) 언러닝 영향력  [store 제거 시 FedAvg 집계가 얼마나 바뀌나]")
print("=" * 92)
M = rows[0][1]                      # real 라벨 분포 카운트
Cm = M.values.astype(float)
P = Cm / Cm.sum(1, keepdims=True)
n = vis.groupby("store_id").size().reindex(M.index).values.astype(float)
w = n / n.sum()
G = (w[:, None] * P).sum(0)
infl = {}
for i, s in enumerate(M.index):
    m = np.ones(len(w), bool); m[i] = False
    Gm = (w[m][:, None] * P[m]).sum(0) / w[m].sum()
    infl[s] = float(np.linalg.norm(Gm - G, 1) / np.linalg.norm(G, 1))
inf = pd.Series(infl).sort_values(ascending=False)
dfi = pd.DataFrame({"weight_%": (w * 100).round(2), "influence_%": (pd.Series(infl) * 100).round(3),
                    "baskets": n.astype(int)}, index=M.index)
dfi["infl_per_weight"] = (dfi["influence_%"] / dfi["weight_%"]).round(3)
print("  영향력 상위 5:"); print(dfi.sort_values("influence_%", ascending=False).head(5).to_string())
print("  가중치 대비 영향력(고유성) 상위 5:")
print(dfi.sort_values("infl_per_weight", ascending=False).head(5).to_string())
print(f"\n  영향력 중앙 {dfi['influence_%'].median():.3f}%   최대 {dfi['influence_%'].max():.3f}%")
rep["unlearn_influence"] = dfi.to_dict("index")
dfi.to_csv("out/het/unlearn_influence.csv")

json.dump(rep, open("out/het/task_aligned.json", "w"), indent=1)
print("\n-> out/het/task_aligned.json, out/het/unlearn_influence.csv")
