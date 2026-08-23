# test_profile_retry.py — temporary, just to test the new batching/delay
from src.collector import collect_user_profiles, save_user_profiles
from src.db import get_conn

conn = get_conn()
c = conn.cursor()

# Pull usernames for accounts still missing profile data.
# NOTE: this only works if STORE_RAW_USERNAME=true was set when they were
# collected -- otherwise username is NULL and we can't retry them by name.
# This is exactly the raw-username-cache gap we talked about earlier.
c.execute("SELECT username FROM accounts WHERE created_utc IS NULL AND username IS NOT NULL")
usernames = [row[0] for row in c.fetchall()]
conn.close()

print(f"Found {len(usernames)} accounts missing profile data with a known username.")

if usernames:
    items, dataset_id = collect_user_profiles(usernames[:15])  # test on a small slice
    if items:
        stats = save_user_profiles(items)
        print(stats)
else:
    print("No retryable usernames -- STORE_RAW_USERNAME was probably false when these were collected.")