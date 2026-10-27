
import pandas as pd
import os

os.makedirs("data/clean", exist_ok=True)

# Load raw customer data
customers = pd.read_csv("data/raw/customers.csv")
original_city = customers["city"].copy()
original_gender = customers["gender"].copy()

# 1. Standardise city names
customers["city"] = customers["city"].str.strip().str.title()
print("City values standardized:", int(original_city.ne(customers["city"]).sum()))

# 2. Standardise gender
customers["gender"] = (
    customers["gender"]
    .fillna("")
    .str.strip()
    .str.lower()
    .map({
        "male": "male",
        "m": "male",
        "female": "female",
        "f": "female"
    })
    .fillna("unknown")
)
print(
    "Gender values changed to canonical labels:",
    int(original_gender.fillna("").ne(customers["gender"]).sum())
)

# 3. Convert age to numeric and mark invalid ages as missing
customers["age"] = pd.to_numeric(
    customers["age"], errors="coerce"
)

invalid_age_count = int(
    (customers["age"].notna() & ~customers["age"].between(10, 100)).sum()
)
customers.loc[
    ~customers["age"].between(10, 100), "age"
] = float("nan")

# 4. Fill missing ages using city-wise median
missing_age_count = int(customers["age"].isna().sum())
customers["age"] = customers["age"].fillna(
    customers.groupby("city")["age"].transform("median")
)
print("Ages outside 10–100 set missing:", invalid_age_count)
print("Missing/invalid ages imputed by city median:", missing_age_count)

# 5. Convert signup_date to datetime
customers["signup_date"] = pd.to_datetime(
    customers["signup_date"], errors="coerce"
)
print("Signup dates converted:", int(customers["signup_date"].notna().sum()))

# Save cleaned customers
customers.to_csv("data/clean/customers.csv", index=False)

# Verify results
print("City counts:")
print(customers["city"].value_counts())

print("\nGender counts:")
print(customers["gender"].value_counts())

print("\nMissing ages:", customers["age"].isna().sum())
print("Minimum age:", customers["age"].min())
print("Maximum age:", customers["age"].max())
print("\nCleaned customers saved successfully!")
