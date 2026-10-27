
import pandas as pd
import os

os.makedirs("data/clean", exist_ok=True)

products = pd.read_csv("data/raw/products.csv")

print("Raw products shape:", products.shape)
print("Duplicate rows:", products.duplicated().sum())

products = products.drop_duplicates()

trimmed_values = 0
for column in ["product_id", "product_name", "category"]:
    original = products[column].astype(str)
    standardized = original.str.strip()
    trimmed_values += int(original.ne(standardized).sum())
    products[column] = standardized
print("Product text values trimmed:", trimmed_values)

prices = pd.to_numeric(products["price_inr"], errors="coerce")
print("Catalogue prices converted:", int(prices.notna().sum()))
print("Unparseable catalogue prices:", int(prices.isna().sum()))
products["price_inr"] = prices

products.to_csv("data/clean/products.csv", index=False)

print("Cleaned products shape:", products.shape)
print("Saved: data/clean/products.csv")
