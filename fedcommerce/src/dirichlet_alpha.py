"""선정 10 store 의 category 분포로 Dirichlet 집중도 alpha 추정 (FL 관례: p_k ~ Dir(alpha * pi))."""
import json, numpy as np, pandas as pd, pyreadr
from scipy.special import digamma, polygamma

sel = json.load(open("out/selected_stores.json"))["selected_stores"]
tx = pyreadr.read_r("data/transactions.rds")[None]; tx.columns=[c.lower() for c in tx.columns]
pr = pyreadr.read_r("data/products.rda")["products"]; pr.columns=[c.lower() for c in pr.columns]
tx = tx.merge(pr[["product_id","product_category"]], on="product_id", how="left")
tx = tx[tx.product_category.notna() & (tx.quantity>0)]
hs = tx.groupby(["household_id","store_id"]).basket_id.nunique().rename("b").reset_index()
prim = hs.sort_values("b",ascending=False).groupby("household_id").head(1)[["household_id","store_id"]].rename(columns={"store_id":"ps"})
ex = tx.merge(prim,on="household_id"); ex = ex[(ex.store_id==ex.ps)&ex.store_id.isin(sel)]

C = ex.groupby(["store_id","product_category"]).size().unstack(fill_value=0).loc[sel]   # K x C counts
N = C.sum(1).values; K, Cn = C.shape
pi = C.sum(0).values / C.values.sum()
P = C.values / N[:,None]
def js(p,q):
    m=(p+q)/2; f=lambda a,b: np.sum(a[a>0]*np.log2(a[a>0]/m[a>0])) ; return 0.5*f(p,m)+0.5*f(q,m)
obs = np.mean([js(P[k],pi) for k in range(K)])
print(f"stores={K}  categories={Cn}  rows/store: {N.min():,}~{N.max():,}")
print(f"관측 mean JS(store‖global) = {obs:.4f}\n")

# (1) Minka fixed-point MLE, Dir(a); alpha = sum(a)  (smoothing 1e-3 for zeros)
Ps = (C.values + 1e-3); Ps = Ps/Ps.sum(1,keepdims=True)
logp = np.log(Ps).mean(0)
a = np.full(Cn, 1.0)
for _ in range(2000):
    a_new = np.exp(digamma(a.sum()) + logp)  # crude init step
    # proper Minka: solve digamma(a_c) = digamma(sum a) + logp_c via Newton on each c
    s = digamma(a.sum()); y = s + logp
    x = a.copy()
    for _ in range(20):
        x = x - (digamma(x)-y)/polygamma(1,x); x = np.clip(x,1e-8,None)
    if np.max(np.abs(x-a)) < 1e-9: a = x; break
    a = x
alpha_mle = a.sum()
print(f"(1) Dirichlet MLE (Minka):  alpha_hat = {alpha_mle:.2f}   [zero-smoothing 에 민감 — 참고용]")

# (2) simulation matching with finite-sample noise
rng = np.random.default_rng(0)
def sim_js(alpha, Ns, B=150):
    out=[]
    for _ in range(B):
        pk = rng.dirichlet(np.clip(alpha*pi,1e-6,None), size=len(Ns))
        cnt = np.stack([rng.multinomial(n,p) for n,p in zip(Ns,pk)])
        ph = cnt/cnt.sum(1,keepdims=True); g = cnt.sum(0)/cnt.sum()
        out.append(np.mean([js(ph[k],g) for k in range(len(Ns))]))
    return np.mean(out)
grid = np.logspace(-1, 3.5, 46)
curve = np.array([sim_js(al, N) for al in grid])
floor = sim_js(1e7, N)          # alpha→∞: 순수 샘플링 노이즈
def match(target):
    if target <= curve.min(): return np.inf
    i = np.where(curve <= target)[0][0]          # curve 는 alpha 증가에 따라 감소
    if i==0: return grid[0]
    x0,x1,y0,y1 = np.log10(grid[i-1]),np.log10(grid[i]),curve[i-1],curve[i]
    return 10**(x0 + (target-y0)*(x1-x0)/(y1-y0))
alpha_sim = match(obs)
print(f"(2) 시뮬레이션 매칭:        alpha_hat = {alpha_sim:.1f}")
print(f"    샘플링 노이즈 하한 (alpha→∞ 일 때 mean JS) = {floor:.4f}  → 관측치는 노이즈의 {obs/floor:.1f}배")

# bootstrap over stores (10개 재표집) → CI
bs=[]
for _ in range(300):
    idx = rng.integers(0,K,K)
    o = np.mean([js(P[k], pi) for k in idx]); bs.append(match(o))
bs=np.array(bs); bs=bs[np.isfinite(bs)]
lo,hi = np.percentile(bs,[2.5,97.5])
print(f"    bootstrap 95% CI (store 재표집): [{lo:.1f}, {hi:.1f}]")

print("\n참고 스케일 (Hsu et al. 2019 관례, Dir(alpha·pi)):")
for al in [0.1,0.5,1,10,100,1000]:
    print(f"  alpha={al:>6}: 예상 mean JS ≈ {sim_js(al,N,B=60):.4f}")
json.dump({"mean_js_obs":obs,"alpha_mle_minka":alpha_mle,"alpha_sim_match":alpha_sim,
           "alpha_ci95":[lo,hi],"sampling_noise_floor_js":floor}, open("out/noniid/dirichlet_alpha.json","w"), indent=1)
