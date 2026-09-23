"""4주 horizon 설계의 전제 3가지 검증."""
import numpy as np, pandas as pd, pyreadr
tx = pyreadr.read_r("data/transactions.rds")[None]; tx.columns=[c.lower() for c in tx.columns]
pr = pyreadr.read_r("data/products.rda")["products"]; pr.columns=[c.lower() for c in pr.columns]
tx = tx.merge(pr[["product_id","product_category"]],on="product_id",how="left")
tx = tx[tx.product_category.notna()&(tx.quantity>0)&(tx.product_category!="COUPON/MISC ITEMS")]
tx["ts"]=pd.to_datetime(tx.transaction_timestamp)
hs=tx.groupby(["household_id","store_id"]).basket_id.nunique().rename("b").reset_index()
prim=hs.sort_values("b",ascending=False).groupby("household_id").head(1)[["household_id","store_id"]].rename(columns={"store_id":"ps"})
dh=tx.merge(prim,on="household_id"); dh=dh[dh.store_id==dh.ps]
bk=dh.groupby("store_id").basket_id.nunique(); cl=bk[bk>=300].index; dh=dh[dh.store_id.isin(cl)]

print("=== (2) client별 실제 vocabulary 크기 ===")
nc=dh.groupby("store_id").product_category.nunique()
print(f"  카테고리 수: 중앙 {nc.median():.0f}  범위 {nc.min()}~{nc.max()}  max/min {nc.max()/nc.min():.2f}x")
print(f"  분위: p10 {nc.quantile(.1):.0f}  p25 {nc.quantile(.25):.0f}  p75 {nc.quantile(.75):.0f}  p90 {nc.quantile(.9):.0f}")
b=bk.reindex(nc.index)
print(f"  바스켓 수와의 상관: {np.corrcoef(np.log(b),nc)[0,1]:.3f}  (클수록 품목도 많음)")

v=(dh.groupby(["store_id","household_id","basket_id"])
     .agg(ts=("ts","min"), cats=("product_category", lambda s:frozenset(s))).reset_index()
     .sort_values(["store_id","household_id","ts"]))
END=v.ts.max()
print(f"\n=== (3) 4주 타깃의 우측 절단 ===")
for W,name in [(7,"1주"),(28,"4주")]:
    ok=(v.ts <= END-pd.Timedelta(days=W)).mean()
    print(f"  {name} 창: 타깃 생성 가능한 방문 {ok*100:.1f}%")

print(f"\n=== (1) 주차별 품목 양성률 (카테고리 단위) ===")
rows=[]
for (s,h),g in v.groupby(["store_id","household_id"]):
    t=g.ts.values; cs=list(g.cats)
    for i in range(3,len(t)):
        if t[i] > (END-pd.Timedelta(days=28)).to_datetime64(): break
        base=t[i]
        hist=set().union(*cs[max(0,i-10):i])
        for w in range(4):
            lo=base+np.timedelta64(7*w,'D'); hi=base+np.timedelta64(7*(w+1),'D')
            m=(t>lo)&(t<=hi)
            tgt=set().union(*[cs[j] for j in np.where(m)[0]]) if m.any() else set()
            rows.append((w+1, len(tgt), int(m.sum()), len(tgt&hist)/max(len(tgt),1) if tgt else np.nan))
R=pd.DataFrame(rows,columns=["week","n_cat","n_visit","in_hist"])
print(f"  (시퀀스 {len(R)//4:,}개 기준)")
print(f"{'주차':>4} {'구매 카테고리 수':>16} {'양성률(301중)':>14} {'방문횟수':>9} {'무방문%':>8} {'이력내%':>8}")
for w in [1,2,3,4]:
    a=R[R.week==w]
    print(f"{w:>4} {a.n_cat.mean():>16.1f} {a.n_cat.mean()/301*100:>13.1f}% {a.n_visit.mean():>9.2f} "
          f"{(a.n_visit==0).mean()*100:>7.1f}% {a.in_hist.mean()*100:>7.1f}%")
