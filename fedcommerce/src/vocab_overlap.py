import re, json, numpy as np, pandas as pd, pyreadr
from collections import Counter

pr = pyreadr.read_r("data/products.rda")["products"]; pr.columns=[c.lower() for c in pr.columns]
dh_cat = sorted(pr.product_category.dropna().unique())
dh_dep = sorted(pr.department.dropna().unique())
ai = pd.read_csv("data/instacart/aisles.csv"); dp = pd.read_csv("data/instacart/departments.csv")
ic_ais = sorted(ai.aisle.unique()); ic_dep = sorted(dp.department.unique())
print(f"dunnhumby: department {len(dh_dep)}  category {len(dh_cat)}")
print(f"Instacart: department {len(ic_dep)}  aisle {len(ic_ais)}")

ABBR = {"frzn":"frozen","refrigerated":"refrigerated","bkry":"bakery","veg":"vegetables","vegetables":"vegetables",
        "bev":"beverage","beverages":"beverage","chkn":"chicken","pwdr":"powder","crystl":"crystal",
        "drnk":"drink","mx":"mix","dinners":"dinner","prod":"produce","hh":"household","gm":"general",
        "spec":"specialty","rfrg":"refrigerated","juc":"juice","ss":"single","pkg":"package",
        "cnd":"canned","dry":"dry","gro":"grocery","misc":"misc","choc":"chocolate","candy":"candy"}
STOP = {"and","the","of","with","in","other","misc","all","no","non"}
def toks(s):
    w = re.sub(r"[^a-z0-9 ]", " ", str(s).lower()).split()
    out=[]
    for t in w:
        t = ABBR.get(t, t)
        if t in STOP or t.isdigit() or len(t)<2: continue
        out.append(t)
    return set(out)

def vocab(xs): 
    c=Counter()
    for x in xs: c.update(toks(x))
    return c

for lvl,(A,B,na,nb) in {"department":(dh_dep,ic_dep,"dunnhumby","Instacart"),
                        "fine (category vs aisle)":(dh_cat,ic_ais,"dunnhumby","Instacart")}.items():
    va,vb = vocab(A), vocab(B)
    sa,sb = set(va), set(vb)
    inter = sa & sb
    # 라벨 수준 커버리지: 상대 어휘와 토큰을 하나라도 공유하는 라벨 비율
    ca = sum(1 for x in A if toks(x) & sb)/len(A)
    cb = sum(1 for x in B if toks(x) & sa)/len(B)
    print(f"\n=== {lvl} ===")
    print(f"  토큰 수  {na} {len(sa)}   {nb} {len(sb)}   공통 {len(inter)}   Jaccard {len(inter)/len(sa|sb):.3f}")
    print(f"  라벨 커버리지: {na} 라벨의 {ca*100:.0f}% / {nb} 라벨의 {cb*100:.0f}% 가 상대와 토큰 공유")
    top = sorted(inter, key=lambda t:-(va[t]+vb[t]))[:18]
    print(f"  공통 토큰(빈도순): {', '.join(top)}")
    print(f"  {na} 전용 상위: {', '.join([t for t,_ in va.most_common(40) if t not in inter][:12])}")
    print(f"  {nb} 전용 상위: {', '.join([t for t,_ in vb.most_common(40) if t not in inter][:12])}")

print("\n=== 매칭 예시 (fine) ===")
vb_ic = {x: toks(x) for x in ic_ais}
shown=0
for x in dh_cat:
    tx = toks(x)
    best = max(ic_ais, key=lambda y: len(tx & vb_ic[y])/max(len(tx|vb_ic[y]),1))
    j = len(tx & vb_ic[best])/max(len(tx|vb_ic[best]),1)
    if j >= 0.4 and shown < 14:
        print(f"  {x:<34} <-> {best:<32} J={j:.2f}"); shown+=1
print("\n=== 매칭 실패 예시 ===")
shown=0
for x in dh_cat:
    tx = toks(x)
    if not (tx & set().union(*vb_ic.values())) and shown < 8:
        print(f"  {x}"); shown+=1
