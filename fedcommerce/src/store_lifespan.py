import numpy as np, pandas as pd, pyreadr, json
tx = pyreadr.read_r("data/transactions.rds")[None]; tx.columns=[c.lower() for c in tx.columns]
pr = pyreadr.read_r("data/products.rda")["products"]; pr.columns=[c.lower() for c in pr.columns]
tx = tx.merge(pr[["product_id","product_category"]],on="product_id",how="left")
tx = tx[tx.product_category.notna()&(tx.quantity>0)&(tx.product_category!="COUPON/MISC ITEMS")]
tx["ts"]=pd.to_datetime(tx.transaction_timestamp); tx["wk"]=tx.week.astype(int)
hs=tx.groupby(["household_id","store_id"]).basket_id.nunique().rename("b").reset_index()
prim=hs.sort_values("b",ascending=False).groupby("household_id").head(1)[["household_id","store_id"]].rename(columns={"store_id":"ps"})
ex=tx.merge(prim,on="household_id"); ex=ex[ex.store_id==ex.ps]
bk=ex.groupby("store_id").basket_id.nunique(); cl=bk[bk>=300].index; ex=ex[ex.store_id.isin(cl)]
print(f"client {len(cl)}개, 주차 범위 {ex.wk.min()}~{ex.wk.max()}\n")

W=ex.groupby(["store_id","wk"]).basket_id.nunique().unstack(fill_value=0)
rows=[]
for s in W.index:
    r=W.loc[s]; act=r[r>0].index
    span=act.max()-act.min()+1
    rows.append(dict(store=s, baskets=int(r.sum()), first=int(act.min()), last=int(act.max()),
                     n_active=len(act), span=span, gap_weeks=int(span-len(act)),
                     coverage=len(act)/52))
L=pd.DataFrame(rows).set_index("store")
print("활성 주 수 분포:"); print(L.n_active.describe().round(1).to_string())
print(f"\n52주 전부 활성: {(L.n_active==52).sum()}개 ({(L.n_active==52).mean()*100:.0f}%)")
print(f"48주 이상    : {(L.n_active>=48).sum()}개 ({(L.n_active>=48).mean()*100:.0f}%)")
print(f"40주 미만    : {(L.n_active<40).sum()}개 ({(L.n_active<40).mean()*100:.0f}%)")
print(f"\n시작이 1주가 아닌 store: {(L.first>1).sum()}개   끝이 52주가 아닌: {(L.last<52).sum()}개")
print(f"중간 공백 있는 store: {(L.gap_weeks>0).sum()}개 (공백 중앙 {L[L.gap_weeks>0].gap_weeks.median() if (L.gap_weeks>0).any() else 0:.0f}주)")
bad=L[L.n_active<44].sort_values("n_active")
if len(bad): print(f"\n활성 44주 미만 (문제 소지) {len(bad)}개:"); print(bad.head(12).to_string())

print("\n주별 전체 바스켓 (연말 성수기 확인):")
tot=ex.groupby("wk").basket_id.nunique()
for a,b in [(1,13),(14,26),(27,39),(40,52)]:
    seg=tot.loc[a:b]; print(f"  {a:>2}-{b:>2}주: 평균 {seg.mean():,.0f}/주  min {seg.min():,}  max {seg.max():,}")
print(f"  최대 주: {tot.idxmax()}주 {tot.max():,}   최소 주: {tot.idxmin()}주 {tot.min():,}")
print(f"  마지막 4주(49-52) 평균 {tot.loc[49:52].mean():,.0f}  vs 전체 평균 {tot.mean():,.0f}  ({tot.loc[49:52].mean()/tot.mean():.2f}x)")
L.to_csv("out/het/store_lifespan.csv")
print("\n-> out/het/store_lifespan.csv")
