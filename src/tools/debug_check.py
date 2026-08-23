# src/tools/debug_check.py
import sqlite3
import os
from src.config import DB_PATH

print("DB_PATH:", DB_PATH)
print("Absolute:", os.path.abspath(DB_PATH))
print("Exists:", os.path.exists(DB_PATH))
print()

conn = sqlite3.connect(DB_PATH)
c = conn.cursor()

c.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = c.fetchall()
print("Tables found:", tables)
print()

if ('accounts',) in tables:
    c.execute("SELECT account_id, created_at, follower_count FROM accounts LIMIT 3")
    rows = c.fetchall()
    print("Sample accounts:")
    for row in rows:
        print(" ", row)
else:
    print("accounts table does NOT exist")

conn.close()