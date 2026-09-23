"""다음 바스켓의 몇 %가 '그 고객이 이미 사던 것'인가.
높을수록 예측이 품목 특정적 = 공유(identity-free) 영역이 기여할 여지가 작다."""
import numpy as np, pandas as pd, pyreadr
tx = pyreadr.read_r("data/transactions.rds")[None]; tx.columns=[c.lower() for c in tx.columns]
pr = pyreadr.read_r("data/products.rda")["products"]; pr.columns=[c.lower() for c in pr.columns]
tx = tx.merge(pr[["product_id","product_category"]],on="product_id",how="left")
tx = tx[tx.product_category.notna()&(tx.quantity>0)&(tx.product_category!="COUPON/MISC ITEMS")]
tx["ts"]=pd.to_datetime(tx.transaction_timestamp)
hs=tx.groupby(["household_id","store_id"]).basket_id.nunique().rename("b").reset_index()
prim=hs.sort_values("b",ascending=False).groupby("household_id").head(1)[["household_id","store_id"]].rename(columns={"store_id":"ps"})
dh=tx.merge(prim,on="household_id"); dh=dh[dh.store_id==dh.ps]
bk=dh.groupby("store_id").basket_id.nunique(); dh=dh[dh.store_id.isin(bk[bk>=300].index)]

v=(dh.groupby(["store_id","household_id","basket_id"])
     .agg(ts=("ts","min"), cats=("product_category", lambda s:frozenset(s))).reset_index()
     .sort_values(["store_id","household_id","ts"]))
print(f"client {v.store_id.nunique()}  방문 {len(v):,}")

rows=[]
for (s,h),g in v.groupby(["store_id","household_id"]):
    cl=list(g.cats)
    if len(cl)<4: continue
    for i in range(3,len(cl)):
        tgt=cl[i]
        hist_all=set().union(*cl[:i])                 # 전체 이력
        hist_l10=set().union(*cl[max(0,i-10):i])      # 최근 10방문
        prev=cl[i-1]                                  # 직전 방문
        rows.append((len(tgt & hist_all)/len(tgt), len(tgt & hist_l10)/len(tgt),
                     len(tgt & prev)/len(tgt), len(tgt)))
R=pd.DataFrame(rows, columns=["all","l10","prev","n"])
print(f"\n다음 방문 카테고리 중 '이미 사던 것' 비율 (n={len(R):,})")
print(f"  전체 이력에 있던 것   평균 {R['all'].mean()*100:.1f}%   중앙 {R['all'].median()*100:.1f}%")
print(f"  최근 10방문에 있던 것 평균 {R['l10'].mean()*100:.1f}%   중앙 {R['l10'].median()*100:.1f}%")
print(f"  직전 방문에 있던 것   평균 {R['prev'].mean()*100:.1f}%   중앙 {R['prev'].median()*100:.1f}%")
print(f"\n=> 완전 신규 카테고리 비율: 평균 {(1-R['all'].mean())*100:.1f}%")

# Instacart 의 reordered 플래그와 비교
op = pd.read_csv("data/instacart/order_products__prior.csv", usecols=["reordered"])
print(f"\n[대조] Instacart reordered 플래그: {op.reordered.mean()*100:.1f}% 가 재주문 (상품 단위)")
