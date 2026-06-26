# src/config.py

import os

# Base paths
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(BASE_DIR, 'data')
DB_PATH     = os.path.join(DATA_DIR, 'influence.db')
BENCHMARK   = os.path.join(DATA_DIR, 'benchmark')
RAW         = os.path.join(DATA_DIR, 'raw')
PROCESSED   = os.path.join(DATA_DIR, 'processed')
OUTPUTS     = os.path.join(BASE_DIR, 'outputs')

# Model parameters (you will tune these later)
ISOLATION_FOREST = {
    'n_estimators': 100,
    'contamination': 0.1,   # expect ~10% anomalous accounts
    'random_state': 42
}

HDBSCAN_PARAMS = {
    'min_cluster_size': 3,   # minimum accounts to form a cluster
    'min_samples': 2
}

COSINE_THRESHOLD = 0.90      # above this = near-duplicate content

# Influence Score weights (must sum to 1.0)
WEIGHTS = {
    'anomaly':       0.40,
    'coordination':  0.40,
    'duplication':   0.10,
    'network':       0.10
}

# Score tiers
TIERS = {
    'organic':     (0,  30),
    'suspicious':  (31, 60),
    'coordinated': (61, 100)
}

print("Config loaded.")