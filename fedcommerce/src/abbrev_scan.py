"""dunnhumby 라벨의 약어를 실제 데이터에서 추출. 사전은 추측이 아니라 이 목록에서 만든다."""
import re, json
from collections import Counter
import pandas as pd, pyreadr

pr = pyreadr.read_r("data/products.rda")["products"]; pr.columns=[c.lower() for c in pr.columns]
tx = pyreadr.read_r("data/transactions.rds")[None]; tx.columns=[c.lower() for c in tx.columns]
tx = tx.merge(pr[["product_id","product_category","department"]], on="product_id", how="left")
tx = tx[tx.product_category.notna()]
freq = tx.product_category.value_counts()          # 거래량 가중치

ai = pd.read_csv("data/instacart/aisles.csv"); dp = pd.read_csv("data/instacart/departments.csv")
ic_tokens = set()
for s in list(ai.aisle)+list(dp.department):
    ic_tokens |= set(re.sub(r"[^a-z0-9 ]"," ",s.lower()).split())

def toks(s): return [t for t in re.sub(r"[^a-z0-9 ]"," ",str(s).lower()).split() if len(t)>1 and not t.isdigit()]

# 카테고리 토큰을 거래량으로 가중
w = Counter()
for cat, n in freq.items():
    for t in set(toks(cat)): w[t] += n
print(f"dunnhumby category 토큰 {len(w)}개 (거래량 가중)\n")

# Instacart 에 없는 토큰 = 정렬 실패 후보
missing = [(t,n) for t,n in w.most_common() if t not in ic_tokens]
hit = sum(n for t,n in w.items() if t in ic_tokens); tot = sum(w.values())
print(f"현재 토큰 커버리지(거래량 가중): {hit/tot*100:.1f}%\n")
print(f"{'미매칭 토큰':<18} {'거래량':>10}   (상위 45개 — 사전 후보)")
print("-"*60)
for t,n in missing[:45]:
    print(f"  {t:<16} {n:>10,}")
json.dump([t for t,_ in missing[:120]], open("out/abbrev_candidates.json","w"), indent=1)
print(f"\n-> out/abbrev_candidates.json ({len(missing)} 개 중 상위 120)")
