
import sqlite3
import pandas as pd
from pathlib import Path

CLEAN = Path("data/clean")
DB_FOLDER = Path("data")
DB_FOLDER.mkdir(parents=True, exist_ok=True)

DB_PATH = DB_FOLDER / "instamart.db"

# Create SQLite database
connection = sqlite3.connect(DB_PATH)

# Load cleaned CSV files into SQLite tables
for table_name in ["customers", "products", "orders"]:
    csv_path = CLEAN / f"{table_name}.csv"
    df = pd.read_csv(csv_path)

    df.to_sql(
        table_name,
        connection,
        if_exists="replace",
        index=False
    )

    print(f"{table_name} table created: {len(df)} rows")

# Verify tables
cursor = connection.cursor()
cursor.execute(
    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
)
print("\nTables in database:")
for row in cursor.fetchall():
    print(row[0])

connection.close()

print(f"\nDatabase saved: {DB_PATH}")
