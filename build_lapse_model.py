
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    GridSearchCV,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils import resample


# --------------------------------------------------
# 1. PROJECT PATHS AND SETTINGS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

ORDERS_PATH = BASE_DIR / "data" / "clean" / "orders.csv"
CUSTOMERS_PATH = BASE_DIR / "data" / "clean" / "customers.csv"
PRODUCTS_PATH = BASE_DIR / "data" / "clean" / "products.csv"
SEGMENTS_PATH = BASE_DIR / "outputs" / "customer_segments.csv"

OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = OUTPUT_DIR / "lapse_model.joblib"
RETENTION_PATH = OUTPUT_DIR / "retention_list.csv"

CUTOFF = pd.Timestamp("2025-09-01 00:00:00")
LABEL_END = pd.Timestamp("2025-10-01 00:00:00")


# --------------------------------------------------
# 2. LOAD AND PREPARE DATA
# --------------------------------------------------

orders = pd.read_csv(ORDERS_PATH)
customers = pd.read_csv(CUSTOMERS_PATH)
products = pd.read_csv(PRODUCTS_PATH)

orders["order_ts"] = pd.to_datetime(
    orders["order_ts"], errors="coerce"
)

orders["status"] = orders["status"].astype(str).str.lower()

for column in [
    "order_amount",
    "discount_pct",
    "delivery_minutes",
    "quantity",
]:
    if column in orders.columns:
        orders[column] = pd.to_numeric(
            orders[column], errors="coerce"
        )

# Join product category to each order.
if "category" not in orders.columns:
    orders = orders.merge(
        products[["product_id", "category"]],
        on="product_id",
        how="left",
    )

# Use orders strictly before September 1 for features.
history = orders[orders["order_ts"] < CUTOFF].copy()

completed_history = history[
    history["status"].eq("completed")
].copy()

# Define customers who ordered before the cutoff.
eligible_ids = completed_history["customer_id"].unique()

# Label: 1 = no completed order during September.
september_completed = orders[
    (orders["order_ts"] >= CUTOFF)
    & (orders["order_ts"] < LABEL_END)
    & (orders["status"].eq("completed"))
]

ordered_in_september = set(
    september_completed["customer_id"].unique()
)

eligible_customers = customers[
    customers["customer_id"].isin(eligible_ids)
].copy()

eligible_customers["lapsed"] = (
    ~eligible_customers["customer_id"].isin(
        ordered_in_september
    )
).astype(int)

print("=" * 65)
print("LAPSE PREDICTION - DATASET")
print("=" * 65)
print("Eligible customers:", len(eligible_customers))
print("Lapsed customers:", int(eligible_customers["lapsed"].sum()))
print(
    "Lapse rate:",
    round(eligible_customers["lapsed"].mean() * 100, 2),
    "%",
)


# --------------------------------------------------
# 3. BUILD FEATURES USING PRE-CUTOFF DATA ONLY
# --------------------------------------------------

# Completed-order RFM and shopping behaviour.
rfm = (
    completed_history.groupby("customer_id")
    .agg(
        last_order_ts=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum"),
        avg_discount=("discount_pct", "mean"),
        avg_delivery=("delivery_minutes", "mean"),
    )
    .reset_index()
)

rfm["recency"] = (
    CUTOFF - rfm["last_order_ts"]
).dt.days

rfm = rfm[
    [
        "customer_id",
        "recency",
        "frequency",
        "monetary",
        "avg_discount",
        "avg_delivery",
    ]
]

# Cancellation rate: cancelled / all final-status orders before cutoff.
status_counts = (
    history.groupby("customer_id")["status"]
    .agg(
        total_orders="count",
        cancelled_orders=lambda values: values.eq("cancelled").sum(),
    )
    .reset_index()
)

status_counts["cancellation_rate"] = (
    status_counts["cancelled_orders"]
    / status_counts["total_orders"].replace(0, np.nan)
)

status_counts = status_counts[
    ["customer_id", "cancellation_rate"]
]

# Most frequently ordered category, using completed orders only.
category_counts = (
    completed_history.dropna(subset=["category"])
    .groupby(["customer_id", "category"])
    .size()
    .reset_index(name="category_count")
    .sort_values(
        ["customer_id", "category_count", "category"],
        ascending=[True, False, True],
    )
)

favorite_category = (
    category_counts.drop_duplicates("customer_id")
    [["customer_id", "category"]]
    .rename(columns={"category": "favorite_category"})
)

# Combine the customer data and features.
dataset = (
    eligible_customers
    .merge(rfm, on="customer_id", how="left")
    .merge(status_counts, on="customer_id", how="left")
    .merge(favorite_category, on="customer_id", how="left")
)
dataset["signup_date"] = pd.to_datetime(dataset["signup_date"], errors="coerce")
dataset["tenure_days"] = (CUTOFF - dataset["signup_date"]).dt.days

# Build the feature matrix without looking at post-cutoff orders. Imputation is
# fitted inside each training pipeline, never on the full cohort or test rows.
numeric_features = [
    "recency",
    "frequency",
    "monetary",
    "avg_discount",
    "cancellation_rate",
    "avg_delivery",
    "tenure_days",
    "age",
]

categorical_features = [
    "city",
    "gender",
    "favorite_category",
]

for column in numeric_features:
    dataset[column] = pd.to_numeric(dataset[column], errors="coerce")

for column in categorical_features:
    dataset[column] = dataset[column].astype("string")

print("\nFeature rows:", len(dataset))
feature_columns = numeric_features + categorical_features
missing_by_feature = dataset[feature_columns].isna().sum()
print("Missing feature values:", int(missing_by_feature.sum()))
if missing_by_feature.any():
    raise ValueError(
        "Feature construction produced missing values: "
        + str(missing_by_feature[missing_by_feature > 0].to_dict())
    )


# --------------------------------------------------
# 4. TRAIN / TEST SPLIT
# --------------------------------------------------

X = dataset[numeric_features + categorical_features].copy()
y = dataset["lapsed"].copy()
customer_ids = dataset["customer_id"].copy()

X_train, X_test, y_train, y_test, ids_train, ids_test = (
    train_test_split(
        X,
        y,
        customer_ids,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )
)

print("\n" + "=" * 65)
print("TRAIN / TEST SPLIT")
print("=" * 65)
print("Train rows:", len(X_train))
print("Test rows:", len(X_test))
print("Train class counts:")
print(y_train.value_counts().sort_index())
print("Test class counts:")
print(y_test.value_counts().sort_index())


# --------------------------------------------------
# 5. PREPROCESSING PIPELINE
# --------------------------------------------------

# The imputers, scaler and encoder are fitted by the pipeline on training folds.
preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
            numeric_features,
        ),
        (
            "categorical",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "encoder",
                        OneHotEncoder(
                            drop="first",
                            handle_unknown="ignore",
                        ),
                    ),
                ]
            ),
            categorical_features,
        ),
    ]
)


# --------------------------------------------------
# 6. BASELINE: ALWAYS PREDICT NOT LAPSED
# --------------------------------------------------

baseline_predictions = np.zeros(len(y_test), dtype=int)

print("\n" + "=" * 65)
print("BASELINE RESULTS")
print("=" * 65)
print(
    "Accuracy:",
    round(accuracy_score(y_test, baseline_predictions), 4),
)
print(
    "Lapsed F1:",
    round(
        f1_score(
            y_test,
            baseline_predictions,
            zero_division=0,
        ),
        4,
    ),
)


# --------------------------------------------------
# 7. LOGISTIC REGRESSION WITH GRID SEARCH
# --------------------------------------------------

logistic_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            LogisticRegression(max_iter=1000),
        ),
    ]
)

logistic_search = GridSearchCV(
    estimator=logistic_pipeline,
    param_grid={
        "model__C": [0.01, 0.1, 1, 10, 100],
    },
    scoring="f1",
    cv=5,
    n_jobs=-1,
)

logistic_search.fit(X_train, y_train)

best_logistic = logistic_search.best_estimator_

print("\n" + "=" * 65)
print("LOGISTIC REGRESSION")
print("=" * 65)
print("Best parameters:", logistic_search.best_params_)
print("Best CV F1:", round(logistic_search.best_score_, 4))


# --------------------------------------------------
# 8. RANDOM FOREST WITH GRID SEARCH
# --------------------------------------------------

forest_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            RandomForestClassifier(
                n_estimators=100,
                random_state=42,
            ),
        ),
    ]
)

forest_search = GridSearchCV(
    estimator=forest_pipeline,
    param_grid={
        "model__max_depth": [3, 5, 8],
        "model__min_samples_leaf": [5, 10, 20],
    },
    scoring="f1",
    cv=5,
    n_jobs=-1,
)

forest_search.fit(X_train, y_train)

best_forest = forest_search.best_estimator_

print("\n" + "=" * 65)
print("RANDOM FOREST")
print("=" * 65)
print("Best parameters:", forest_search.best_params_)
print("Best CV F1:", round(forest_search.best_score_, 4))


# --------------------------------------------------
# 9. TEST METRICS
# --------------------------------------------------

def report_metrics(name, model, X_eval, y_eval):
    predictions = model.predict(X_eval)

    print(f"\n{name}")
    print("-" * 40)
    print("Accuracy:", round(
        accuracy_score(y_eval, predictions), 4
    ))
    print("Precision:", round(
        precision_score(y_eval, predictions, zero_division=0), 4
    ))
    print("Recall:", round(
        recall_score(y_eval, predictions, zero_division=0), 4
    ))
    print("F1:", round(
        f1_score(y_eval, predictions, zero_division=0), 4
    ))
    print("Confusion matrix:")
    print(confusion_matrix(y_eval, predictions))

    return predictions


print("\n" + "=" * 65)
print("TEST SET EVALUATION")
print("=" * 65)

logistic_predictions = report_metrics(
    "Logistic Regression", best_logistic, X_test, y_test
)

forest_predictions = report_metrics(
    "Random Forest", best_forest, X_test, y_test
)


# --------------------------------------------------
# 10. CLASS-BALANCE EXPERIMENT
# --------------------------------------------------

train_data = X_train.copy()
train_data["lapsed"] = y_train.to_numpy()

majority_count = int(train_data["lapsed"].value_counts().max())

lapsed_rows = train_data[train_data["lapsed"] == 1]
not_lapsed_rows = train_data[train_data["lapsed"] == 0]

if len(lapsed_rows) < len(not_lapsed_rows):
    lapsed_extra = resample(
        lapsed_rows,
        replace=True,
        n_samples=len(not_lapsed_rows) - len(lapsed_rows),
        random_state=42,
    )
    balanced_train = pd.concat(
        [train_data, lapsed_extra],
        ignore_index=True,
    )
elif len(not_lapsed_rows) < len(lapsed_rows):
    not_lapsed_extra = resample(
        not_lapsed_rows,
        replace=True,
        n_samples=len(lapsed_rows) - len(not_lapsed_rows),
        random_state=42,
    )
    balanced_train = pd.concat(
        [train_data, not_lapsed_extra],
        ignore_index=True,
    )
else:
    balanced_train = train_data.copy()

X_balanced = balanced_train.drop(columns=["lapsed"])
y_balanced = balanced_train["lapsed"]

balanced_logistic = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            LogisticRegression(
                max_iter=1000,
                C=logistic_search.best_params_["model__C"],
            ),
        ),
    ]
)

balanced_logistic.fit(X_balanced, y_balanced)

print("\n" + "=" * 65)
print("CLASS-BALANCE EXPERIMENT")
print("=" * 65)
print("Balanced training counts:")
print(y_balanced.value_counts().sort_index())

report_metrics(
    "Balanced Logistic Regression on unchanged test set",
    balanced_logistic,
    X_test,
    y_test,
)


# --------------------------------------------------
# 11. FEATURE IMPORTANCE
# --------------------------------------------------

# Random Forest feature importance.
forest_preprocessor = best_forest.named_steps["preprocessor"]
forest_model = best_forest.named_steps["model"]

feature_names = forest_preprocessor.get_feature_names_out()

importance = pd.DataFrame({
    "feature": feature_names,
    "importance": forest_model.feature_importances_,
}).sort_values("importance", ascending=False)

print("\n" + "=" * 65)
print("RANDOM FOREST - TOP 5 FEATURES")
print("=" * 65)
print(importance.head(5).to_string(index=False))

# Logistic regression: show the three largest absolute coefficients.
log_preprocessor = best_logistic.named_steps["preprocessor"]
log_model = best_logistic.named_steps["model"]

log_feature_names = log_preprocessor.get_feature_names_out()

coefficients = pd.DataFrame({
    "feature": log_feature_names,
    "coefficient": log_model.coef_[0],
})

coefficients["absolute_coefficient"] = (
    coefficients["coefficient"].abs()
)

coefficients = coefficients.sort_values(
    "absolute_coefficient",
    ascending=False,
)

print("\nLOGISTIC REGRESSION - TOP 3 COEFFICIENTS")
print(coefficients.head(3).to_string(index=False))
print("Positive coefficient: associated with higher lapse odds.")
print("Negative coefficient: associated with lower lapse odds.")


# --------------------------------------------------
# 12. SELECT AND SAVE THE MODEL
# --------------------------------------------------

# Choose the model with the higher cross-validation F1.
if forest_search.best_score_ >= logistic_search.best_score_:
    selected_model = best_forest
    selected_name = "Random Forest"
else:
    selected_model = best_logistic
    selected_name = "Logistic Regression"

selected_model.fit(X_train, y_train)
joblib.dump(selected_model, MODEL_PATH)

# Reload and check identical test predictions.
reloaded_model = joblib.load(MODEL_PATH)

pred_before = selected_model.predict(X_test)
pred_after = reloaded_model.predict(X_test)

print("\n" + "=" * 65)
print("MODEL PERSISTENCE")
print("=" * 65)
print("Selected model:", selected_name)
print("Saved model:", MODEL_PATH)
print(
    "Reloaded predictions identical:",
    bool(np.array_equal(pred_before, pred_after)),
)

# TRAINING-ONLY SCALER MEAN CHECK

fitted_preprocessor = best_logistic.named_steps["preprocessor"]
numeric_pipeline = fitted_preprocessor.named_transformers_["numeric"]
fitted_scaler = numeric_pipeline.named_steps["scaler"]

frequency_index = numeric_features.index("frequency")

training_frequency_mean = fitted_scaler.mean_[frequency_index]
full_cohort_frequency_mean = dataset["frequency"].mean()

print("\n" + "=" * 65)
print("DATA LEAKAGE CHECK: FREQUENCY MEAN")
print("=" * 65)

print(
    "Training-only scaler frequency mean:",
    round(training_frequency_mean, 4),
)

print(
    "Full cohort frequency mean:",
    round(full_cohort_frequency_mean, 4),
)
print("The means differ because the scaler was fitted only on training rows.")


# --------------------------------------------------
# 13. PREMIUM CUSTOMER RETENTION LIST
# --------------------------------------------------

segments = pd.read_csv(SEGMENTS_PATH)

premium_ids = set(
    segments.loc[
        segments["is_premium"].astype(str).str.lower().eq("true"),
        "customer_id",
    ]
)

test_results = pd.DataFrame({
    "customer_id": ids_test.to_numpy(),
    "actually_lapsed": y_test.to_numpy(),
    "predicted_lapse": pred_after,
})

test_results["lapse_probability"] = (
    reloaded_model.predict_proba(X_test)[:, 1]
)

# Keep Premium customers in the held-out test set.
retention = test_results[
    test_results["customer_id"].isin(premium_ids)
].copy()

retention = retention.sort_values(
    "lapse_probability",
    ascending=False,
).head(10)

retention = retention[
    [
        "customer_id",
        "lapse_probability",
        "actually_lapsed",
    ]
]

retention.to_csv(RETENTION_PATH, index=False)

print("\n" + "=" * 65)
print("RETENTION LIST")
print("=" * 65)
print(retention.to_string(index=False))
print("Saved:", RETENTION_PATH)
print(
    "Truly lapsed customers in top retention list:",
    int(retention["actually_lapsed"].sum()),
)

print("\nLAPSE MODELING STEP COMPLETED.")
