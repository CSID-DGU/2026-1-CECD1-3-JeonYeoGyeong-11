"""신규 가게가 바스켓 몇 건을 모아야 행동 특징이 쓸만해지나.
큰 매장을 N 바스켓으로 서브샘플 -> 전체 데이터 기준 특징과의 순위상관."""
import numpy as np, pandas as pd, pyreadr
from scipy.stats import spearmanr
rng=np.random.default_rng(0)
tx = pyreadr.read_r("data/transactions.rds")[None]; tx.columns=[c.lower() for c in tx.columns]
pr = pyreadr.read_r("data/products.rda")["products"]; pr.columns=[c.lower() for c in pr.columns]
tx = tx.merge(pr[["product_id","product_category"]],on="product_id",how="left")
tx = tx[tx.product_category.notna()&(tx.quantity>0)&(tx.product_category!="COUPON/MISC ITEMS")]
hs=tx.groupby(["household_id","store_id"]).basket_id.nunique().rename("b").reset_index()
prim=hs.sort_values("b",ascending=False).groupby("household_id").head(1)[["household_id","store_id"]].rename(columns={"store_id":"ps"})
dh=tx.merge(prim,on="household_id"); dh=dh[dh.store_id==dh.ps]
bk=dh.groupby("store_id").basket_id.nunique()
big=bk.nlargest(6).index                     # 큰 매장 6개로 실험
print("대상 매장:", list(big), " 바스켓:", list(bk[big]))

def feats(d):
    nb=d.basket_id.nunique(); nh=d.household_id.nunique()
    g=d.groupby("product_category")
    return pd.DataFrame({"penet":g.basket_id.nunique()/nb,
                         "reach":g.household_id.nunique()/max(nh,1),
                         "repeat":g.basket_id.nunique()/g.household_id.nunique().clip(lower=1)})

print(f"\n{'바스켓':>7} {'penet ρ':>9} {'reach ρ':>9} {'repeat ρ':>10} {'카테고리 커버':>13}")
print("-"*54)
for N in [20,50,100,200,400,800,1600]:
    ps,rs,rp,cv=[],[],[],[]
    for s in big:
        d=dh[dh.store_id==s]; bl=d.basket_id.unique()
        if len(bl)<N*1.5: continue
        full=feats(d)
        for _ in range(3):
            sel=rng.choice(bl,N,replace=False)
            sub=feats(d[d.basket_id.isin(sel)])
            j=full.join(sub,rsuffix="_s",how="inner")
            if len(j)<15: continue
            ps.append(spearmanr(j.penet,j.penet_s).statistic)
            rs.append(spearmanr(j.reach,j.reach_s).statistic)
            rp.append(spearmanr(j["repeat"],j["repeat_s"]).statistic)
            cv.append(len(sub)/len(full))
    if ps: print(f"{N:>7} {np.mean(ps):>9.3f} {np.mean(rs):>9.3f} {np.mean(rp):>10.3f} {np.mean(cv)*100:>12.0f}%")
