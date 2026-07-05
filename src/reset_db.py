# src/reset_db.py
import os
from config import DB_PATH

if os.path.exists(DB_PATH):
    os.remove(DB_PATH)
    print("Deleted:", DB_PATH)
else:
    print("No existing database found")