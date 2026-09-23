import pandas as pd, pyreadr, json
pr = pyreadr.read_r("data/products.rda")["products"]; pr.columns=[c.lower() for c in pr.columns]
tx = pyreadr.read_r("data/transactions.rds")[None]; tx.columns=[c.lower() for c in tx.columns]
tx = tx.merge(pr[["product_id","product_category"]], on="product_id", how="left")
f = tx[tx.product_category.notna()].product_category.value_counts()
print("=== Instacart aisles (134) ===")
ai = pd.read_csv("data/instacart/aisles.csv")
print(" | ".join(sorted(ai.aisle)))
print("\n=== Instacart departments (21) ===")
print(" | ".join(sorted(pd.read_csv("data/instacart/departments.csv").department)))
print(f"\n=== dunnhumby product_category 상위 90 (거래량, 전체 {len(f)}개) ===")
for i,(c,n) in enumerate(f.head(90).items()):
    print(f"  {c:<38}{n:>8,}", end="\n" if i%2 else "")
