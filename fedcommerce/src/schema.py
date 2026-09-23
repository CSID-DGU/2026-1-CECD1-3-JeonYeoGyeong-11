import pandas as pd, pyreadr
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)

tx = pyreadr.read_r("data/transactions.rds")[None]; tx.columns=[c.lower() for c in tx.columns]
pr = pyreadr.read_r("data/products.rda")["products"]; pr.columns=[c.lower() for c in pr.columns]
dm = pyreadr.read_r("data/demographics.rda")["demographics"]; dm.columns=[c.lower() for c in dm.columns]

for name, df in [("transactions", tx), ("products", pr), ("demographics", dm)]:
    print("="*100); print(f"{name}   shape={df.shape}")
    print("-"*100)
    print(pd.DataFrame({"dtype": df.dtypes.astype(str), "nunique": df.nunique(),
                        "n_null": df.isna().sum(), "example": df.iloc[0].astype(str)}).to_string())
    print(f"\n[상위 3행]"); print(df.head(3).to_string()); print()

print("="*100); print("한 바스켓(방문 1회) 예시 — 다음 구매 예측의 입력 단위")
b = tx[tx.basket_id == tx.basket_id.iloc[0]].merge(pr, on="product_id", how="left")
print(b[["household_id","store_id","basket_id","transaction_timestamp","product_id",
         "quantity","sales_value","department","product_category","product_type"]].to_string(index=False))
