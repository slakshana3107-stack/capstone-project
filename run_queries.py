
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
db_path = ROOT / "data" / "instamart.db"
sql_path = ROOT / "sql" / "queries.sql"

if not db_path.exists():
    raise FileNotFoundError(f"Database not found: {db_path}")
if not sql_path.exists():
    raise FileNotFoundError(f"SQL query file not found: {sql_path}")

sql_text = sql_path.read_text(encoding="utf-8")

queries = [
    query.strip()
    for query in sql_text.split(";")
    if query.strip()
]

if len(queries) != 7:
    raise ValueError(f"Expected 7 SQL queries, found {len(queries)}")

with sqlite3.connect(db_path) as connection:
    for number, query in enumerate(queries, start=1):
        result = pd.read_sql_query(query, connection)
        print(f"\n===== QUERY {number} RESULT ({len(result)} rows) =====")
        print(result.to_string(index=False))

print("\nAll SQL queries executed successfully!")
