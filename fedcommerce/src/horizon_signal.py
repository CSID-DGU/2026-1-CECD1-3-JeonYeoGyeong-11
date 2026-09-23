"""4주 horizon 이 실제로 주차별 신호를 갖는가 — 품목 재구매 주기별로 본다."""
import numpy as np, pandas as pd, pyreadr
tx = pyreadr.read_r("data/transactions.rds")[None]; tx.columns=[c.lower() for c in tx.columns]
pr = pyreadr.read_r("data/products.rda")["products"]; pr.columns=[c.lower() for c in pr.columns]
tx = tx.merge(pr[["product_id","product_category"]],on="product_id",how="left")
tx = tx[tx.product_category.notna()&(tx.quantity>0)&(tx.product_category!="COUPON/MISC ITEMS")]
tx["ts"]=pd.to_datetime(tx.transaction_timestamp)
hs=tx.groupby(["household_id","store_id"]).basket_id.nunique().rename("b").reset_index()
prim=hs.sort_values("b",ascending=False).groupby("household_id").head(1)[["household_id","store_id"]].rename(columns={"store_id":"ps"})
dh=tx.merge(prim,on="household_id"); dh=dh[dh.store_id==dh.ps]
bk=dh.groupby("store_id").basket_id.nunique()
dh=dh[dh.store_id.isin(bk.nlargest(25).index)]     # 큰 매장 25개로 (충분한 표본)

v=(dh.groupby(["store_id","household_id","basket_id"])
     .agg(ts=("ts","min"), cats=("product_category", lambda s:frozenset(s))).reset_index()
     .sort_values(["store_id","household_id","ts"]))
END=v.ts.max()

# 카테고리별 전역 재구매 주기 (같은 고객이 다시 살 때까지)
gap=[]
for (s,h),g in v.groupby(["store_id","household_id"]):
    t=g.ts.values; cs=list(g.cats)
    last={}
    for i,(ti,ci) in enumerate(zip(t,cs)):
        for c in ci:
            if c in last: gap.append((c,(ti-last[c])/np.timedelta64(1,'D')))
            last[c]=ti
G=pd.DataFrame(gap,columns=["cat","d"]).groupby("cat").d.median()
band=pd.cut(G,[0,7,21,1e9],labels=["짧음(<7일)","중간(7-21일)","김(>21일)"])
print("카테고리 재구매 주기 분포:", band.value_counts().to_dict())

# (고객,품목) 단위로 주차별 구매 여부
rows=[]
cut=(END-pd.Timedelta(days=28)).to_datetime64()
for (s,h),g in v.groupby(["store_id","household_id"]):
    t=g.ts.values; cs=list(g.cats)
    for i in range(3,len(t)):
        if t[i] > cut: break
        hist=set().union(*cs[max(0,i-10):i])
        if not hist: continue
        wk=[]
        for w in range(4):
            m=(t>t[i]+np.timedelta64(7*w,'D'))&(t<=t[i]+np.timedelta64(7*(w+1),'D'))
            wk.append(set().union(*[cs[j] for j in np.where(m)[0]]) if m.any() else set())
        for c in hist:
            rows.append((c,*[int(c in x) for x in wk]))
        if len(rows)>3_000_000: break
    if len(rows)>3_000_000: break
R=pd.DataFrame(rows,columns=["cat","w1","w2","w3","w4"])
R["band"]=R.cat.map(band)
print(f"\n(고객,품목) 관측 {len(R):,}개 — 이력에 있던 품목만\n")
print(f"{'재구매주기':<14} {'n':>10} {'week1':>8} {'week2':>8} {'week3':>8} {'week4':>8} {'w4/w1':>7}")
for b in ["짧음(<7일)","중간(7-21일)","김(>21일)"]:
    a=R[R.band==b]
    if len(a)<1000: continue
    m=[a[f"w{i}"].mean() for i in [1,2,3,4]]
    print(f"{b:<14} {len(a):>10,} "+" ".join(f"{x*100:>7.1f}%" for x in m)+f" {m[3]/m[0]:>7.2f}")
m=[R[f"w{i}"].mean() for i in [1,2,3,4]]
print(f"{'전체':<14} {len(R):>10,} "+" ".join(f"{x*100:>7.1f}%" for x in m)+f" {m[3]/m[0]:>7.2f}")

# 주차 간 상관 — 같은 값을 내놓을 수밖에 없나
print(f"\n주차 라벨 간 상관: w1-w2 {R.w1.corr(R.w2):.3f}  w1-w4 {R.w1.corr(R.w4):.3f}  w3-w4 {R.w3.corr(R.w4):.3f}")
