from __future__ import annotations

import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = Path(
    os.getenv("DATA_DIR", BASE_DIR / "data")
).resolve()

OUTPUTS = Path(
    os.getenv("OUTPUTS_DIR", BASE_DIR / "outputs")
).resolve()

DB_PATH = Path(
    os.getenv("INFLUENCE_DB_PATH", DATA_DIR / "influence.db")
).resolve()

BENCHMARK_DB_PATH = Path(
    os.getenv("BENCHMARK_DB_PATH", DATA_DIR / "benchmark.db")
).resolve()

BENCHMARK = Path(
    os.getenv("BENCHMARK_DIR", DATA_DIR / "benchmark")
).resolve()

RAW = Path(
    os.getenv("RAW_DATA_DIR", DATA_DIR / "raw")
).resolve()

PROCESSED = Path(
    os.getenv("PROCESSED_DATA_DIR", DATA_DIR / "processed")
).resolve()

REDDIT_COLLECTION = {
    "min_posts": 5000,
    "min_accounts": 1000,

    "target_posts": 8000,
    "target_accounts": 1500,

    "subreddits": [
        "Nepal",
        "NepaliPolitics",
        "nepalinews",
        "NepalSocial",
    ],

    "search_terms": [
        "election",
        "government",
        "politics",
        "parliament",
        "prime minister",
        "minister",
        "policy",
        "protest",
        "Nepal",
    ],

    "min_account_posts": 3,
}

ISOLATION_FOREST = {
    "n_estimators": int(os.getenv("IF_N_ESTIMATORS", "100")),
    "contamination": float(os.getenv("IF_CONTAMINATION", "0.24")),
    "random_state": int(os.getenv("IF_RANDOM_STATE", "42")),
}

HDBSCAN_PARAMS = {
    "min_cluster_size": int(
        os.getenv("HDBSCAN_MIN_CLUSTER_SIZE", "3")
    ),
    "min_samples": int(
        os.getenv("HDBSCAN_MIN_SAMPLES", "2")
    ),
}

COSINE_THRESHOLD = float(
    os.getenv("COSINE_THRESHOLD", "0.80")
)


WEIGHTS = {
    "anomaly": 0.40,
    "coordination": 0.40,
    "duplication": 0.10,
    "network": 0.10,
}


TIERS = {
    "organic": (0, 30),
    "suspicious": (31, 60),
    "coordinated": (61, 100),
}


def ensure_directories():
    for path in (
        DATA_DIR,
        OUTPUTS,
        BENCHMARK,
        RAW,
        PROCESSED,
    ):
        path.mkdir(parents=True, exist_ok=True)


def validate_config():
    if abs(sum(WEIGHTS.values()) - 1.0) > 1e-9:
        raise ValueError(
            f"Influence Score weights must sum to 1.0: {WEIGHTS}"
        )

    if not 0 < ISOLATION_FOREST["contamination"] <= 0.5:
        raise ValueError(
            "Isolation Forest contamination must be in (0, 0.5]."
        )

    if HDBSCAN_PARAMS["min_cluster_size"] < 2:
        raise ValueError(
            "HDBSCAN min_cluster_size must be >= 2."
        )

    if HDBSCAN_PARAMS["min_samples"] < 1:
        raise ValueError(
            "HDBSCAN min_samples must be >= 1."
        )

    if not 0 < COSINE_THRESHOLD <= 1:
        raise ValueError(
            "COSINE_THRESHOLD must be in (0, 1]."
        )


def ensure_runtime_ready():
    ensure_directories()
    validate_config()