from pathlib import Path
import pandas as pd

# Project folder
BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"

print("=" * 65)
print("INSTAMART RAW DATA AUDIT")
print("=" * 65)

# Check raw data folder
if not RAW_DIR.exists():
    raise FileNotFoundError(f"Raw data folder not found: {RAW_DIR}")

# Read every CSV file in data/raw
csv_files = sorted(RAW_DIR.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError("No CSV files found inside data/raw")

dataframes = {}

for file_path in csv_files:
    df = pd.read_csv(
        file_path,
        keep_default_na=file_path.name != "tickets.csv",
    )
    dataframes[file_path.stem] = df

    print(f"\n{'=' * 65}")
    print(f"FILE: {file_path.name}")
    print("=" * 65)

    # Dataset size
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    # Column-wise audit
    missing = df.isna().copy()
    if file_path.name == "tickets.csv":
        for column in df.select_dtypes(include=["object", "string"]).columns:
            missing[column] |= df[column].astype(str).str.strip().eq("")
    audit = pd.DataFrame({
        "data_type": df.dtypes.astype(str),
        "missing_count": missing.sum(),
        "missing_percentage": (
            missing.sum() / len(df) * 100
        ).round(2)
    })

    print("\nColumn-wise audit:")
    print(audit.to_string())

    # Duplicate checks
    print(f"\nExact duplicate rows: {df.duplicated().sum()}")

    # Blank text values
    blank_counts = {}

    for column in df.select_dtypes(include=["object", "string"]).columns:
        blank_counts[column] = (
            df[column].astype("string").str.strip().eq("").sum()
        )

    blank_counts = {
        column: count
        for column, count in blank_counts.items()
        if count > 0
    }

    print(f"Blank text values by column: {blank_counts}")

# Additional checks for orders
if "orders" in dataframes:
    orders = dataframes["orders"].copy()

    print(f"\n{'=' * 65}")
    print("ORDERS REFERENCE AUDIT")
    print("=" * 65)

    # Make column-name lookup case-insensitive
    column_lookup = {
        column.lower().strip(): column
        for column in orders.columns
    }

    # Order ID checks
    order_id_col = column_lookup.get("order_id")

    if order_id_col:
        print(
            "Distinct order IDs:",
            orders[order_id_col].nunique()
        )

        print(
            "Rows belonging to repeated order IDs:",
            orders[order_id_col].duplicated(keep=False).sum()
        )

    # Status checks
    status_col = column_lookup.get("status")

    if status_col:
        status = (
            orders[status_col]
            .astype("string")
            .str.strip()
            .str.lower()
        )

        print("\nOrder status counts:")
        print(status.value_counts(dropna=False).to_string())

        print(
            "\nPending order rows:",
            status.eq("pending").sum()
        )

    # Order amount checks
    amount_col = column_lookup.get("order_amount")

    if amount_col:
        amount = pd.to_numeric(
            orders[amount_col], errors="coerce"
        )

        print(
            "\nMissing or non-numeric order amounts:",
            amount.isna().sum()
        )

        print(
            "Negative order amounts:",
            amount.lt(0).sum()
        )

    # Delivery time checks
    delivery_col = column_lookup.get("delivery_minutes")

    if delivery_col:
        delivery = pd.to_numeric(
            orders[delivery_col], errors="coerce"
        )

        print(
            "\nBlank delivery times:",
            (
                orders[delivery_col].isna()
                | orders[delivery_col].astype(str).str.strip().eq("")
            ).sum()
        )
        print(
            "\nMissing or non-numeric delivery times:",
            delivery.isna().sum()
        )

        print(
            "Negative delivery times:",
            delivery.lt(0).sum()
        )

# Additional checks for customers
if "customers" in dataframes:
    customers = dataframes["customers"].copy()

    print(f"\n{'=' * 65}")
    print("CUSTOMERS REFERENCE AUDIT")
    print("=" * 65)

    column_lookup = {
        column.lower().strip(): column
        for column in customers.columns
    }

    age_col = column_lookup.get("age")

    if age_col:
        age = pd.to_numeric(customers[age_col], errors="coerce")

        print(
            "Blank ages:",
            (
                customers[age_col].isna()
                | customers[age_col].astype(str).str.strip().eq("")
            ).sum()
        )
        print("Missing or non-numeric ages:", age.isna().sum())
        print(
            "Age values outside 10–100:",
            (~age.between(10, 100) & age.notna()).sum()
        )

    city_col = column_lookup.get("city")
    if city_col:
        print(
            "Distinct raw city spellings:",
            customers[city_col].nunique(dropna=True)
        )

    gender_col = column_lookup.get("gender")
    if gender_col:
        gender = customers[gender_col].astype(str).str.strip()
        print(
            "Distinct non-blank raw gender spellings:",
            gender[gender.ne("")].nunique()
        )
        print(
            "Blank gender values:",
            (customers[gender_col].isna() | gender.eq("")).sum()
        )

print(f"\n{'=' * 65}")
print("RAW DATA AUDIT COMPLETED")
print("=" * 65)