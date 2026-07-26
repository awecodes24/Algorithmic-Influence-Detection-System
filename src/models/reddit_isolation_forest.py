# src/models/reddit_isolation_forest.py
# Runs Isolation Forest on REAL Reddit data (no ground-truth labels).
# Outputs anomaly scores + a ranked list of the most suspicious accounts.
# Cresci-2017 remains the only labeled dataset for AUC-ROC validation --
# this script is for actual detection, not accuracy measurement.

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from datetime import datetime, timezone
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

try:
    from src.db import get_conn
except ModuleNotFoundError:
    from db import get_conn

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


FEATURE_COLUMNS = [
    'age_days', 'posts_per_day', 'comments_per_day', 'comment_ratio',
    'karma_score', 'avg_score', 'subreddit_count', 'active_days',
    'hour_entropy', 'duplicate_ratio', 'avg_post_interval',
    'avg_comment_interval', 'night_activity_ratio', 'burstiness_score',
    'engagement_rate'
]

# Contamination: since there's no ground truth, this is an assumption --
# "what fraction of accounts do we expect to be anomalous". 0.1 (10%) is
# a reasonable starting default for real-world data (unlike Cresci-2017,
# which was ~76% bots by construction). Adjust after reviewing results --
# see the note printed at the end of run_isolation_forest().
CONTAMINATION = 0.1


def load_features():
    conn = get_conn()
    df = pd.read_sql(f"""
        SELECT account_id, {', '.join(FEATURE_COLUMNS)}
        FROM features
    """, conn)
    conn.close()

    logger.info(f"Loaded {len(df)} accounts with computed features")

    # Rows with any NULL feature (e.g. burstiness_score needs 3+ posts)
    # can't be scored reliably -- drop them, but report how many.
    before = len(df)
    df = df.dropna(subset=FEATURE_COLUMNS)
    dropped = before - len(df)
    if dropped:
        logger.warning(
            f"Dropped {dropped} accounts with incomplete features "
            f"(likely too few posts/comments to compute burstiness_score "
            f"or avg_post_interval)"
        )

    return df


def run_isolation_forest(df):
    if len(df) < 10:
        raise ValueError(
            f"Only {len(df)} accounts have complete features -- "
            f"too few to train a meaningful model. Collect more data first."
        )

    X = df[FEATURE_COLUMNS].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    logger.info(f"Training Isolation Forest (contamination={CONTAMINATION})...")
    model = IsolationForest(
        n_estimators=100,
        contamination=CONTAMINATION,
        random_state=42
    )
    model.fit(X_scaled)

    raw_scores = model.decision_function(X_scaled)

    # Normalize to 0-1, higher = more anomalous.
    # NOTE: unlike the Cresci-2017 benchmark run, there's no is_bot label
    # here to check the direction against -- this assumes the minority
    # pattern IS the anomaly, which is the standard Isolation Forest
    # assumption and should hold for real data (bots as minority), unlike
    # Cresci-2017 where bots were artificially the majority class.
    anomaly_scores = 1 - (raw_scores - raw_scores.min()) / \
                         (raw_scores.max() - raw_scores.min())

    df = df.copy()
    df['anomaly_score'] = anomaly_scores

    return df.sort_values('anomaly_score', ascending=False)


def save_scores(df):
    conn = get_conn()
    c = conn.cursor()

    now = datetime.now(timezone.utc).isoformat()
    saved = 0

    for _, row in df.iterrows():
        c.execute("""
            INSERT INTO scores (account_id, anomaly_score, scored_at)
            VALUES (?, ?, ?)
            ON CONFLICT(account_id) DO UPDATE SET
                anomaly_score = excluded.anomaly_score,
                scored_at = excluded.scored_at
        """, (row['account_id'], float(row['anomaly_score']), now))
        saved += 1

    conn.commit()
    conn.close()
    logger.info(f"Saved anomaly scores for {saved} accounts to scores table")


def print_top_suspicious(df, n=20):
    print(f"\n{'━'*70}")
    print(f"  TOP {n} MOST ANOMALOUS ACCOUNTS")
    print(f"{'━'*70}")
    print(f"{'account_id':<20}{'score':>8}{'posts/day':>12}{'burstiness':>12}{'karma':>10}")
    print(f"{'-'*70}")

    for _, row in df.head(n).iterrows():
        print(
            f"{row['account_id'][:18]:<20}"
            f"{row['anomaly_score']:>8.4f}"
            f"{row['posts_per_day']:>12.3f}"
            f"{row['burstiness_score']:>12.3f}"
            f"{row['karma_score']:>10.1f}"
        )
    print(f"{'━'*70}\n")


if __name__ == "__main__":
    logger.info("Running Isolation Forest on real Reddit data\n")

    df = load_features()
    df = run_isolation_forest(df)
    save_scores(df)
    print_top_suspicious(df)

    logger.info(
        "NOTE: contamination=0.1 is an assumption, not a measured value -- "
        "there's no ground truth for real data. After reviewing the top "
        "suspicious accounts above, manually spot-check a few (read their "
        "actual posts/comments) to sanity-check whether the ranking looks "
        "reasonable before trusting it further."
    )