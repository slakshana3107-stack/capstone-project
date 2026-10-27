
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# 1. Set project paths
BASE_DIR = Path(__file__).resolve().parent
ORDERS_PATH = BASE_DIR / "data" / "clean" / "orders.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 2. Set the RFM snapshot date
SNAPSHOT = pd.Timestamp("2025-10-01 00:00:00")

# 3. Load cleaned orders
orders = pd.read_csv(ORDERS_PATH)

orders["order_ts"] = pd.to_datetime(
    orders["order_ts"], errors="coerce"
)
orders["order_amount"] = pd.to_numeric(
    orders["order_amount"], errors="coerce"
)

# 4. Keep completed orders only
completed = orders[
    orders["status"].str.lower().eq("completed")
].copy()

# 5. Calculate RFM
rfm = (
    completed.groupby("customer_id")
    .agg(
        last_order_ts=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum"),
    )
    .reset_index()
)

rfm["recency"] = (
    SNAPSHOT - rfm["last_order_ts"]
).dt.days

rfm = rfm[
    ["customer_id", "recency", "frequency", "monetary"]
].copy()

print("=" * 60)
print("RFM SUMMARY")
print("=" * 60)
print("Customers with completed orders:", len(rfm))
print(rfm[["recency", "frequency", "monetary"]].describe().round(2))

# 6. Standardise the RFM values
features = ["recency", "frequency", "monetary"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(rfm[features])

# 7. Evaluate K-Means for K = 2 to 8
print("\n" + "=" * 60)
print("K-MEANS EVALUATION")
print("=" * 60)

for k in range(2, 9):
    model = KMeans(
        n_clusters=k,
        n_init=10,
        random_state=42
    )

    labels = model.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)

    print(
        f"K={k}: Inertia={model.inertia_:.2f}, "
        f"Silhouette={score:.3f}"
    )

# 8. Fit the final model with K = 4
final_model = KMeans(
    n_clusters=4,
    n_init=10,
    random_state=42
)

rfm["segment_id"] = final_model.fit_predict(X_scaled)

# 9. Create segment profiles
profile = (
    rfm.groupby("segment_id")
    .agg(
        customer_count=("customer_id", "count"),
        mean_recency=("recency", "mean"),
        mean_frequency=("frequency", "mean"),
        mean_monetary=("monetary", "mean"),
    )
    .round(2)
)

print("\n" + "=" * 60)
print("SEGMENT PROFILES")
print("=" * 60)
print(profile)

# 10. Assign segment names based on behaviour
premium_id = profile["mean_monetary"].idxmax()

other_ids = [
    sid for sid in profile.index if sid != premium_id
]

frequent_id = profile.loc[
    other_ids, "mean_frequency"
].idxmax()

remaining_ids = [
    sid for sid in other_ids if sid != frequent_id
]

risk_id = profile.loc[
    remaining_ids, "mean_recency"
].idxmax()

regular_id = [
    sid for sid in remaining_ids if sid != risk_id
][0]

segment_names = {
    premium_id: "Premium High-Value",
    frequent_id: "Frequent Shoppers",
    risk_id: "At-Risk Occasional",
    regular_id: "Regular Customers",
}

rfm["segment_name"] = rfm["segment_id"].map(segment_names)
rfm["is_premium"] = rfm["segment_id"].eq(premium_id)

# 11. Print business actions for each segment
actions = {
    "Premium High-Value":
        "Provide personalised service and exclusive offers.",
    "Frequent Shoppers":
        "Encourage repeat purchases with loyalty rewards.",
    "At-Risk Occasional":
        "Send a re-engagement offer to encourage another order.",
    "Regular Customers":
        "Recommend relevant products and suitable bundles.",
}

print("\nSEGMENT NAMES AND RECOMMENDED ACTIONS")

for segment_id, row in profile.iterrows():
    name = segment_names[segment_id]

    print(
        f"\nSegment {segment_id}: {name}"
        f"\nCustomers: {int(row['customer_count'])}"
        f"\nMean recency: {row['mean_recency']}"
        f"\nMean frequency: {row['mean_frequency']}"
        f"\nMean monetary: {row['mean_monetary']}"
        f"\nAction: {actions[name]}"
    )

# 12. PCA: retain two principal components
pca = PCA(n_components=2)
pca.fit(X_scaled)

variance = pca.explained_variance_ratio_

print("\n" + "=" * 60)
print("PCA RESULTS")
print("=" * 60)
print("PC1 variance:", round(variance[0], 4))
print("PC2 variance:", round(variance[1], 4))
print("Total variance:", round(variance.sum(), 4))

# 13. Save the required output CSV
output = rfm[
    [
        "customer_id",
        "recency",
        "frequency",
        "monetary",
        "segment_id",
        "segment_name",
        "is_premium",
    ]
].copy()

output["is_premium"] = output["is_premium"].astype(bool)

output_path = OUTPUT_DIR / "customer_segments.csv"
output.to_csv(output_path, index=False)

print("\nSaved:", output_path)
print("Output rows:", len(output))
print("Premium customers:", int(output["is_premium"].sum()))
print("RFM segmentation completed.")
