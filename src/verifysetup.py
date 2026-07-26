# src/verify_setup.py
# Run this to confirm your entire setup is working

import sys
print("Python version:", sys.version)

import pandas as pd
import numpy as np
import sklearn
import hdbscan
import networkx as nx
import streamlit
import plotly
import sqlite3


from config import DB_PATH, WEIGHTS
from benchmark_database import get_connection

print("\n--- Library Versions ---")
print(f"pandas:     {pd.__version__}")
print(f"sklearn:    {sklearn.__version__}")
print(f"networkx:   {nx.__version__}")
print(f"hdbscan:    {hdbscan.__version__}")

print("\n--- Database Check ---")
conn = get_connection()
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()
print("Tables in database:", [t[0] for t in tables])
conn.close()

print("\n--- Config Check ---")
print("Influence Score weights:", WEIGHTS)
print(f"Weights sum to: {sum(WEIGHTS.values())} (must be 1.0)")

print("\n✅ Setup complete. Ready to load benchmark data.")