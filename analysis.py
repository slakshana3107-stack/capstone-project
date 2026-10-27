
import pandas as pd

# 1. Load cleaned data
orders = pd.read_csv("data/clean/orders.csv")
customers = pd.read_csv("data/clean/customers.csv")
products = pd.read_csv("data/clean/products.csv")

# 2. Keep completed orders only
completed = orders[
    orders["status"].str.lower() == "completed"
].copy()

# 3. Convert order amount to numeric
completed["order_amount"] = pd.to_numeric(
    completed["order_amount"], errors="coerce"
)

# 4. Add customer city
completed = completed.merge(
    customers[["customer_id", "city"]],
    on="customer_id",
    how="left"
)

# 5. Add product category
completed = completed.merge(
    products[["product_id", "category"]],
    on="product_id",
    how="left"
)

# 6. City-wise revenue summary
city_summary = completed.groupby("city").agg(
    total_amount_inr=("order_amount", "sum"),
    mean_amount_inr=("order_amount", "mean"),
    order_count=("order_amount", "count"),
)

print("\nCITY-WISE REVENUE SUMMARY")
print(city_summary.round(2))

# 7. Pivot table: City-wise and category-wise revenue
pivot = pd.pivot_table(
    completed,
    values="order_amount",
    index="city",
    columns="category",
    aggfunc="sum",
    fill_value=0,
    margins=True,
    margins_name="Total"
)

print("\nCITY-WISE CATEGORY REVENUE")
print(pivot.round(2))

# SQL-equivalent pivot cross-check.
sql_total = completed["order_amount"].sum()
print("\nPIVOT GRAND TOTAL:", round(float(pivot.loc["Total", "Total"]), 2))
print("SQL/Pandas completed revenue:", round(float(sql_total), 2))
assert round(float(pivot.loc["Total", "Total"]), 2) == round(float(sql_total), 2)

# 8. Overall revenue check
print("\nTOTAL COMPLETED REVENUE")
print(round(completed["order_amount"].sum(), 2))

print("\nPANDAS ANALYSIS COMPLETED SUCCESSFULLY!")

# Order amount statistical analysis
amount = completed["order_amount"].dropna()

print("\nORDER AMOUNT STATISTICS")
print("Mean:", round(amount.mean(), 2))
print("Median:", round(amount.median(), 2))
print("Minimum:", round(amount.min(), 2))
print("Maximum:", round(amount.max(), 2))

# Calculate quartiles and IQR
Q1 = amount.quantile(0.25)
Q3 = amount.quantile(0.75)
IQR = Q3 - Q1

lower_fence = Q1 - 1.5 * IQR
upper_fence = Q3 + 1.5 * IQR

outliers = amount[
    (amount < lower_fence) | (amount > upper_fence)
]

print("Q1:", round(Q1, 2))
print("Q3:", round(Q3, 2))
print("IQR:", round(IQR, 2))
print("Lower fence:", round(lower_fence, 2))
print("Upper fence:", round(upper_fence, 2))
print("Skewness:", round(amount.skew(), 2))
print("Outliers below lower fence:", int((amount < lower_fence).sum()))
print("Outliers above upper fence:", int((amount > upper_fence).sum()))
print("Number of outliers:", len(outliers))

# Correlation between order amount, quantity and discount
print("\nCORRELATION ANALYSIS")

numeric_columns = [
    "order_amount",
    "quantity",
    "discount_pct",
    "delivery_minutes",
]

correlation = completed[numeric_columns].corr()

print(correlation.round(3))

print("\nCorrelation between order amount and discount_pct:")
print(round(correlation.loc["order_amount", "discount_pct"], 3))

pair_values = correlation.where(
    ~pd.DataFrame(
        [[i == j for j in correlation.columns] for i in correlation.index],
        index=correlation.index,
        columns=correlation.columns,
    )
).stack()
largest_pair = pair_values.abs().idxmax()
print(
    "Largest absolute correlation:",
    largest_pair[0],
    "with",
    largest_pair[1],
    round(float(pair_values.loc[largest_pair]), 3),
)
