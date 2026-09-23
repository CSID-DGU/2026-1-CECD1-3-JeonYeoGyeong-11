"""NIID-Bench / MOON / FedDyn 코드 관례: 카테고리 c 마다 클라이언트 간 분배 q_c ~ Dir(alpha * 1_K).
store 크기가 다르므로 IID 극한이 store 점유율 s 가 되도록 Dir(alpha * K * s) 로 두고 alpha(성분값) 를 시뮬레이션 매칭."""
import json, numpy as np, pandas as pd, pyreadr
sel = json.load(open("out/selected_stores.json"))["selected_stores"]
tx = pyreadr.read_r("data/transactions.rds")[None]; tx.columns=[c.lower() for c in tx.columns]
pr = pyreadr.read_r("data/products.rda")["products"]; pr.columns=[c.lower() for c in pr.columns]
tx = tx.merge(pr[["product_id","product_category"]],on="product_id",how="left")
tx = tx[tx.product_category.notna()&(tx.quantity>0)&(tx.product_category!="COUPON/MISC ITEMS")]
hs = tx.groupby(["household_id","store_id"]).basket_id.nunique().rename("b").reset_index()
prim = hs.sort_values("b",ascending=False).groupby("household_id").head(1)[["household_id","store_id"]].rename(columns={"store_id":"ps"})
ex = tx.merge(prim,on="household_id"); ex = ex[(ex.store_id==ex.ps)&ex.store_id.isin(sel)]

C = ex.groupby(["product_category","store_id"]).size().unstack(fill_value=0).reindex(columns=sel,fill_value=0)
C = C[C.sum(1)>=200]                      # 너무 희귀한 카테고리는 노이즈만 → 제외
K = len(sel); s = C.sum(0).values/C.values.sum()           # store 점유율 (IID 극한)
Q = C.values/C.values.sum(1,keepdims=True); n_c = C.sum(1).values
def js(p,q):
    m=(p+q)/2; f=lambda a: np.sum(a[a>0]*np.log2(a[a>0]/m[a>0])); return .5*f(p)+.5*f(q)
obs = np.mean([js(Q[i],s) for i in range(len(Q))])
rng=np.random.default_rng(0)
def sim(alpha,B=60):
    out=[]
    for _ in range(B):
        qs = rng.dirichlet(np.clip(alpha*K*s,1e-6,None), size=len(n_c))
        cnt = np.stack([rng.multinomial(n,q) for n,q in zip(n_c,qs)])
        qh = cnt/cnt.sum(1,keepdims=True); sh = cnt.sum(0)/cnt.sum()
        out.append(np.mean([js(qh[i],sh) for i in range(len(qh))]))
    return np.mean(out)
grid = np.logspace(-1.5,2.5,41); curve=np.array([sim(a) for a in grid])
def match(t):
    i=np.where(curve<=t)[0]; 
    if len(i)==0: return np.inf
    i=i[0]
    if i==0: return grid[0]
    x0,x1,y0,y1=np.log10(grid[i-1]),np.log10(grid[i]),curve[i-1],curve[i]
    return 10**(x0+(t-y0)*(x1-x0)/(y1-y0))
a_hat = match(obs)
bs=[]
for _ in range(200):
    idx=rng.integers(0,len(Q),len(Q)); bs.append(match(np.mean([js(Q[i],s) for i in idx])))
bs=np.array(bs); bs=bs[np.isfinite(bs)]; lo,hi=np.percentile(bs,[2.5,97.5])
print(f"카테고리 {len(Q)}개 (행>=200), store {K}개")
print(f"관측 mean JS(q_c ‖ s) = {obs:.4f}   IID 노이즈 하한 = {sim(1e6):.4f}")
print(f"\n성분 기준 alpha (Dir(alpha·1_K), NIID-Bench/MOON 관례) = {a_hat:.2f}   95% CI [{lo:.2f}, {hi:.2f}]")
print("\n참고: 같은 관례에서 alpha 별 기대 JS")
for a in [0.1,0.5,1,5,10,100]: print(f"  alpha={a:>5}: {sim(a,40):.4f}")
json.dump({"alpha_perclass":a_hat,"ci95":[lo,hi],"n_categories":int(len(Q)),"mean_js":obs},
          open("out/noniid/dirichlet_alpha_perclass.json","w"),indent=1)
