from src.db import get_conn

conn = get_conn()

for table in [
    "posts",
    "comments",
    "account_activity",
    "coordination_events"
]:
    print(f"\n=== {table} ===")
    for row in conn.execute(f"PRAGMA table_info({table})"):
        print(tuple(row))

conn.close()