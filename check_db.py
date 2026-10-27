
import sqlite3

connection = sqlite3.connect("data/instamart.db")

cursor = connection.cursor()

cursor.execute("SELECT name FROM sqlite_master")

tables = cursor.fetchall()

print("Database tables:")

for table in tables:
    print(table[0])

connection.close()
