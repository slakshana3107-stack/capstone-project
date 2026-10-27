
import pandas as pd
import sqlite3
from pathlib import Path

# File paths
customers_path = Path("data/clean/customers.csv")
products_path = Path("data/clean/products.csv")
orders_path = Path("data/clean/orders.csv")
db_path = Path("data/instamart.db")

# Load cleaned data
customers = pd.read_csv(customers_path)
products = pd.read_csv(products_path)
orders = pd.read_csv(orders_path)

completed = orders.loc[orders["status"].eq("completed")]
cancelled = orders.loc[orders["status"].eq("cancelled")]

expected_city_counts = {
    "Pune": 227,
    "Delhi": 210,
    "Bengaluru": 197,
    "Chennai": 191,
    "Hyderabad": 191,
    "Mumbai": 184,
}
expected_gender_counts = {"female": 610, "male": 560, "unknown": 30}

checks = {
    "customer row count": len(customers) == 1200,
    "product row count": len(products) == 40,
    "cleaned order row count": len(orders) == 12309,
    "unique customer IDs": customers["customer_id"].is_unique,
    "unique product IDs": products["product_id"].is_unique,
    "unique order IDs": orders["order_id"].is_unique,
    "standardized city counts":
        customers["city"].value_counts().to_dict() == expected_city_counts,
    "standardized gender counts":
        customers["gender"].value_counts().to_dict() == expected_gender_counts,
    "no missing customer ages": customers["age"].notna().all(),
    "ages are within 18–60": customers["age"].between(18, 60).all(),
    "no missing order amounts": orders["order_amount"].notna().all(),
    "completed/cancelled row counts":
        len(completed) == 11110 and len(cancelled) == 1199,
    "no pending orders": set(orders["status"]) == {"completed", "cancelled"},
    "completed orders have delivery minutes":
        completed["delivery_minutes"].notna().all(),
    "cancelled orders retain missing delivery minutes":
        cancelled["delivery_minutes"].isna().all(),
    "completed revenue is INR 2,998,668.80":
        round(completed["order_amount"].sum(), 2) == 2998668.80,
    "SQLite database exists": db_path.exists(),
}

if db_path.exists():
    with sqlite3.connect(db_path) as conn:
        tables = pd.read_sql_query(
            "SELECT name FROM sqlite_master WHERE type='table'", conn
        )["name"].tolist()
        db_counts = {
            table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("customers", "products", "orders")
            if table in tables
        }
    checks["SQLite tables and row counts"] = (
        {"customers", "products", "orders"}.issubset(tables)
        and db_counts == {"customers": 1200, "products": 40, "orders": 12309}
    )
else:
    checks["SQLite tables and row counts"] = False

failed = [name for name, passed in checks.items() if not passed]
if failed:
    for name in failed:
        print(f"CHECK FAILED: {name}")
    raise SystemExit(1)

print("ALL CHECKPOINTS PASSED")
