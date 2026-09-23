"""Update divergence: 동일 global init 에서 각 client 를 한 번 local training 하고
Delta_s = w_s - w0 의 cosine 을 본다.

"분포가 달라 보이는가" 가 아니라 "그 차이가 실제로 서로 다른 model update 를 만드는가" 를 재는
FL 에 가장 직접적인 이질성 지표. 실제 store 분할과 크기를 맞춘 랜덤 분할을 나란히 계산한다.

모델은 최종 Transformer 가 아니라 동일 태스크의 선형 프록시:
  직전 방문의 category multi-hot -> 다음 방문의 category multi-hot (BCE).
전처리 전에도 돌릴 수 있고, update 방향의 이질성을 보는 목적에는 충분하다.
"""
import json, os
import numpy as np
import pandas as pd
import pyreadr
import torch
import torch.nn as nn

rng = np.random.default_rng(0)
torch.manual_seed(0)
os.makedirs("out/het", exist_ok=True)

MIN_ROWS = 2000      # FL client 가 될 만한 store
LOCAL_STEPS = 30
BATCH = 128
LR = 0.05

# ------------------------------------------------------------------ data
tx = pyreadr.read_r("data/transactions.rds")[None]
tx.columns = [c.lower() for c in tx.columns]
pr = pyreadr.read_r("data/products.rda")["products"]
pr.columns = [c.lower() for c in pr.columns]
tx = tx.merge(pr[["product_id", "product_category"]], on="product_id", how="left")
tx = tx[tx.product_category.notna() & (tx.quantity > 0)
        & (tx.product_category != "COUPON/MISC ITEMS")]
tx["ts"] = pd.to_datetime(tx.transaction_timestamp)

cats = sorted(tx.product_category.unique())
cidx = {c: i for i, c in enumerate(cats)}
C = len(cats)
tx["ci"] = tx.product_category.map(cidx)

store_sizes = tx.store_id.value_counts()
keep = store_sizes[store_sizes >= MIN_ROWS].index
print(f"categories {C}   stores(>= {MIN_ROWS} rows) {len(keep)}")

# 크기를 맞춘 랜덤-바스켓 분할
targets = store_sizes.to_dict()
bk_rows = tx.groupby("basket_id").size()
order = rng.permutation(list(bk_rows.index))
kk = np.array(list(targets.keys()))
vv = np.array([targets[k] for k in kk], float)
assign = {}
for b in order:
    i = int(np.argmax(vv))
    assign[b] = kk[i]
    vv[i] -= bk_rows[b]
tx["store_rand_bk"] = tx.basket_id.map(assign)


def build_pairs(df, store_col):
    """(store -> (X, Y)) : 직전 방문 multi-hot -> 다음 방문 multi-hot"""
    b = (df.groupby([store_col, "household_id", "basket_id"])
           .agg(ts=("ts", "min"), cis=("ci", lambda s: sorted(set(s)))).reset_index())
    b = b.sort_values([store_col, "household_id", "ts"])
    out = {}
    for s, gs in b.groupby(store_col):
        Xi, Yi = [], []
        for _, gh in gs.groupby("household_id"):
            cl = list(gh.cis)
            for i in range(len(cl) - 1):
                Xi.append(cl[i]); Yi.append(cl[i + 1])
        if len(Xi) < 50:
            continue
        X = torch.zeros(len(Xi), C); Y = torch.zeros(len(Yi), C)
        for r, (a, c) in enumerate(zip(Xi, Yi)):
            X[r, a] = 1.0; Y[r, c] = 1.0
        out[s] = (X, Y)
    return out


def local_updates(pairs, w0, b0):
    """각 client 를 공통 init 에서 LOCAL_STEPS 만큼 학습 -> flatten 된 Delta 반환"""
    deltas, names = [], []
    for s, (X, Y) in pairs.items():
        m = nn.Linear(C, C)
        with torch.no_grad():
            m.weight.copy_(w0); m.bias.copy_(b0)
        opt = torch.optim.SGD(m.parameters(), lr=LR)
        lossf = nn.BCEWithLogitsLoss()
        n = len(X)
        for t in range(LOCAL_STEPS):
            i = torch.randint(0, n, (min(BATCH, n),))
            opt.zero_grad()
            lossf(m(X[i]), Y[i]).backward()
            opt.step()
        with torch.no_grad():
            d = torch.cat([(m.weight - w0).flatten(), (m.bias - b0).flatten()])
        deltas.append(d.numpy()); names.append(s)
    return np.stack(deltas), names


def cos_stats(D):
    Dn = D / np.linalg.norm(D, axis=1, keepdims=True)
    Cm = Dn @ Dn.T
    iu = np.triu_indices(len(D), 1)
    pair = Cm[iu]
    g = D.mean(0); g = g / np.linalg.norm(g)
    to_g = Dn @ g
    return pair, to_g


w0 = torch.zeros(C, C); b0 = torch.zeros(C)
report = {}
for tag, col in [("real", "store_id"), ("random_bk", "store_rand_bk")]:
    sub = tx[tx[col].isin(keep)]
    pairs = build_pairs(sub, col)
    D, names = local_updates(pairs, w0, b0)
    pair, to_g = cos_stats(D)
    norms = np.linalg.norm(D, axis=1)
    report[tag] = dict(
        n_clients=len(names),
        cos_pairwise=dict(median=float(np.median(pair)), q1=float(np.percentile(pair, 25)),
                          q3=float(np.percentile(pair, 75)), min=float(pair.min())),
        cos_to_global=dict(median=float(np.median(to_g)), q1=float(np.percentile(to_g, 25)),
                           q3=float(np.percentile(to_g, 75)), min=float(to_g.min())),
        update_norm=dict(median=float(np.median(norms)),
                         ratio_max_min=float(norms.max() / norms.min())))
    r = report[tag]
    print(f"\n[{tag}]  clients {r['n_clients']}")
    print(f"  cos(Delta_i, Delta_j)  median {r['cos_pairwise']['median']:.4f}   "
          f"Q1 {r['cos_pairwise']['q1']:.4f}  Q3 {r['cos_pairwise']['q3']:.4f}  min {r['cos_pairwise']['min']:.4f}")
    print(f"  cos(Delta_s, Delta_G)  median {r['cos_to_global']['median']:.4f}   "
          f"Q1 {r['cos_to_global']['q1']:.4f}  min {r['cos_to_global']['min']:.4f}")
    print(f"  ||Delta|| median {r['update_norm']['median']:.4f}   max/min {r['update_norm']['ratio_max_min']:.1f}x")

rp, rr = report["real"]["cos_pairwise"]["median"], report["random_bk"]["cos_pairwise"]["median"]
report["divergence_gap"] = dict(
    pairwise_real=rp, pairwise_random=rr, diff=float(rr - rp),
    ratio_1mcos=float((1 - rp) / max(1 - rr, 1e-9)))
print(f"\n>> 1-cos (update 이질성): real {1-rp:.4f}  vs  random {1-rr:.4f}  "
      f"=>  {(1-rp)/max(1-rr,1e-9):.2f}x")

json.dump(report, open("out/het/update_divergence.json", "w"), indent=1)
print("-> out/het/update_divergence.json")
