
import pandas as pd
from pathlib import Path

RAW = Path("data/raw")
CLEAN = Path("data/clean")
CLEAN.mkdir(parents=True, exist_ok=True)

# Load datasets
orders = pd.read_csv(RAW / "orders.csv")
products = pd.read_csv(CLEAN / "products.csv")

print("Raw orders shape:", orders.shape)

# Convert timestamps before deduplication so chronological ordering is explicit.
orders["order_ts"] = pd.to_datetime(orders["order_ts"], errors="coerce")
print("Order timestamps converted:", int(orders["order_ts"].notna().sum()))

# Remove exact duplicate rows.
before = len(orders)
orders = orders.drop_duplicates()
print("Exact duplicates removed:", before - len(orders))

# Keep the latest record for each order_id, including the final status record.
before = len(orders)
orders = (
    orders.sort_values("order_ts", kind="stable")
    .drop_duplicates(subset="order_id", keep="last")
)
print("Duplicate order records removed:", before - len(orders))
print("Pending/earlier duplicate records discarded:", before - len(orders))

# Convert numeric columns
unparseable_amounts = pd.to_numeric(
    orders["order_amount"], errors="coerce"
).isna()
print("Unparseable/missing order amounts after order dedupe:", int(unparseable_amounts.sum()))
for column in ["order_amount", "quantity", "discount_pct", "delivery_minutes"]:
    orders[column] = pd.to_numeric(orders[column], errors="coerce")
for column in ["quantity", "discount_pct"]:
    print(
        f"{column} values parsed as numeric:",
        int(orders[column].notna().sum()),
        "| unparseable:",
        int(orders[column].isna().sum())
    )

# Add product information to calculate missing order amounts
orders = orders.merge(
    products[["product_id", "price_inr", "category"]],
    on="product_id",
    how="left",
    validate="many_to_one"
)

calculated_amount = (
    orders["price_inr"]
    * orders["quantity"]
    * (1 - orders["discount_pct"] / 100)
).round(2)

missing_amount_count = int(orders["order_amount"].isna().sum())
orders["order_amount"] = orders["order_amount"].fillna(calculated_amount).round(2)
print("Missing order amounts filled from catalogue price:", missing_amount_count)

# Replace invalid negative delivery times
negative_delivery_count = int(orders["delivery_minutes"].lt(0).sum())
orders.loc[orders["delivery_minutes"] < 0, "delivery_minutes"] = float("nan")
print("Negative delivery times set missing:", negative_delivery_count)

# Fill missing delivery times for completed orders using category median
completed = orders["status"].eq("completed")
category_medians = (
    orders.loc[completed]
    .groupby("category")["delivery_minutes"]
    .transform("median")
)

completed_missing_before = int(
    orders.loc[completed, "delivery_minutes"].isna().sum()
)
orders.loc[completed, "delivery_minutes"] = (
    orders.loc[completed, "delivery_minutes"].fillna(category_medians)
)
print("Completed delivery times imputed:", completed_missing_before)

# Cancelled orders should not have delivery times
orders.loc[
    orders["status"].eq("cancelled"), "delivery_minutes"
] = float("nan")
print(
    "Cancelled delivery times left missing:",
    int(orders["status"].eq("cancelled").sum())
)

# Save clean data
orders = orders.drop(columns=["price_inr", "category"])
orders["order_ts"] = orders["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
orders.to_csv(CLEAN / "orders.csv", index=False)

print("\n===== CLEAN ORDERS CHECK =====")
print("Clean orders shape:", orders.shape)
print("\nOrder status counts:")
print(orders["status"].value_counts())
print("\nMissing order amounts:", orders["order_amount"].isna().sum())
print(
    "Missing delivery times in completed orders:",
    orders.loc[orders["status"].eq("completed"), "delivery_minutes"].isna().sum()
)
print(
    "Missing delivery times in cancelled orders:",
    orders.loc[orders["status"].eq("cancelled"), "delivery_minutes"].isna().sum()
)
print(
    "Completed revenue:",
    round(orders.loc[orders["status"].eq("completed"), "order_amount"].sum(), 2)
)
print("\nSaved: data/clean/orders.csv")
