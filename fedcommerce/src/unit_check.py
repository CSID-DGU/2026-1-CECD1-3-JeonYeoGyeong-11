import numpy as np, pandas as pd, pyreadr, json
tx = pyreadr.read_r("data/transactions.rds")[None]; tx.columns=[c.lower() for c in tx.columns]
pr = pyreadr.read_r("data/products.rda")["products"]; pr.columns=[c.lower() for c in pr.columns]
tx = tx.merge(pr[["product_id","product_category"]],on="product_id",how="left")
tx = tx[tx.product_category.notna()&(tx.quantity>0)&(tx.product_category!="COUPON/MISC ITEMS")]
tx["ts"]=pd.to_datetime(tx.transaction_timestamp)
hs=tx.groupby(["household_id","store_id"]).basket_id.nunique().rename("b").reset_index()
prim=hs.sort_values("b",ascending=False).groupby("household_id").head(1)[["household_id","store_id"]].rename(columns={"store_id":"ps"})
ex=tx.merge(prim,on="household_id"); ex=ex[ex.store_id==ex.ps]
bk=ex.groupby("store_id").basket_id.nunique(); cl=bk[bk>=300].index; ex=ex[ex.store_id.isin(cl)]

v=(ex.groupby(["store_id","household_id","basket_id"])
     .agg(ts=("ts","min"),ncat=("product_category","nunique"),nline=("product_category","size"),nqty=("quantity","sum"))
     .reset_index().sort_values(["store_id","household_id","ts"]))
print(f"client {len(cl)}  방문 {len(v):,}  가구 {v.household_id.nunique():,}")
print(f"\n[방문 하나]  카테고리 {v.ncat.mean():.1f}개(중앙 {v.ncat.median():.0f}, p90 {v.ncat.quantile(.9):.0f})  "
      f"라인 {v.nline.mean():.1f}  수량합 {v.nqty.mean():.1f}")
vp=v.groupby(["store_id","household_id"]).size()
print(f"[가구 하나]  방문 {vp.mean():.1f}회 (중앙 {vp.median():.0f}, p10 {vp.quantile(.1):.0f}, p90 {vp.quantile(.9):.0f})")
print(f"  방문 3회 이상 가구 {(vp>=3).mean()*100:.1f}%   10회 이상 {(vp>=10).mean()*100:.1f}%")

# stride=1, 최소 history H 일 때 시퀀스 수
for H in [1,3,5]:
    n=(vp-H).clip(lower=0).sum()
    print(f"  최소 history {H} -> 학습 예제 {n:,.0f}개  (client 당 중앙 {(vp.groupby(level=0).apply(lambda s:(s-H).clip(lower=0).sum())).median():,.0f})")

# 7일 윈도우 안에 방문이 몇 번인가  (타깃을 '다음 방문' vs '향후 7일' 중 무엇으로?)
v["g"]=v.groupby(["store_id","household_id"]).ts.diff().dt.total_seconds()/86400
res=[]
for (s,h),g in v.groupby(["store_id","household_id"]):
    t=g.ts.values
    for i in range(len(t)-1):
        w=((t>t[i])&(t<=t[i]+np.timedelta64(7,'D'))).sum()
        res.append(w)
res=np.array(res)
print(f"\n[향후 7일 윈도우]  방문 0회 {100*(res==0).mean():.1f}%  1회 {100*(res==1).mean():.1f}%  "
      f"2회 {100*(res==2).mean():.1f}%  3회+ {100*(res>=3).mean():.1f}%   평균 {res.mean():.2f}회")
print(f"  -> '다음 방문' 타깃은 7일 수요의 {100/max(res[res>0].mean(),1e-9):.0f}% 만 설명")
