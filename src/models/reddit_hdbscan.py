# src/models/reddit_hdbscan.py
# HDBSCAN coordination clustering on REAL Reddit data.
# Counterpart to reddit_isolation_forest.py, and to models/hdbscan_model.py
# (which does the same job against the Cresci-2017 benchmark instead).
#
# Coordination Score follows the proposal's Eq. 4.5: normalized HDBSCAN
# cluster persistence, with noise (label = -1) scored 0.

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import hdbscan
from datetime import datetime, timezone
from sklearn.preprocessing import StandardScaler

try:
    from src.db import get_conn
except ModuleNotFoundError:
    from db import get_conn

try:
    from src.config import HDBSCAN_PARAMS
except ModuleNotFoundError:
    from config import HDBSCAN_PARAMS

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# Deliberately a SUBSET of the 15 features reddit_preprocessor.py computes --
# same principle as models/hdbscan_model.py: the goal here is "do these
# accounts behave like a coordinated group", not "is this account
# individually weird" (that's reddit_isolation_forest.py's job). Posting
# rhythm + topical-spread features only:
#   - posts_per_day / comments_per_day : shared volume patterns
#   - hour_entropy / night_activity_ratio / burstiness_score : shared
#     timing patterns -- the strongest synchrony signal available without
#     a pairwise comparison (that's what the temporal_similarity table is
#     for, and is out of scope for this script)
#   - duplicate_ratio : shared template/copy-paste behavior
#   - subreddit_count : shared narrow-focus vs. broad-organic usage
#
# Deliberately excluded: karma_score, avg_score, engagement_rate, age_days
# -- these describe an account's own history/reception, already covered by
# Isolation Forest, not whether it moves in sync with others.
#
# Also excluded: avg_post_interval / avg_comment_interval. Both need 2+
# timestamps of ONE specific type (posts-only or comments-only), so an
# account that only comments -- common on Reddit -- would get dropped by
# avg_post_interval's NaN alone. burstiness_score and hour_entropy are both
# computed from combined post+comment timestamps in reddit_preprocessor.py,
# so the "posting rhythm" concept is already covered without that bias.
FEATURE_COLUMNS = [
    'posts_per_day',
    'comments_per_day',
    'hour_entropy',
    'night_activity_ratio',
    'burstiness_score',
    'duplicate_ratio',
    'subreddit_count',
]

# burstiness_score is the only column here that can be NULL (needs 3+
# combined post+comment timestamps) -- everything else always resolves
# once an account has any activity at all. Same situation
# reddit_isolation_forest.py hit with its own three conditional columns:
# on the actual collected data, 149/178 accounts miss that 3+ bar at
# once, and dropping every one of them throws away most of the pool this
# script exists to cluster. Median-impute instead, consistent with
# reddit_isolation_forest.py's fix -- see that file's comment above
# CONDITIONAL_COLUMNS for the full reasoning.
CONDITIONAL_COLUMNS = ['burstiness_score']
CORE_COLUMNS = [c for c in FEATURE_COLUMNS if c not in CONDITIONAL_COLUMNS]


def load_features():
    conn = get_conn()
    df = pd.read_sql(f"""
        SELECT account_id, {', '.join(FEATURE_COLUMNS)}
        FROM features
    """, conn)
    conn.close()

    logger.info(f"Loaded {len(df)} accounts with computed features")

    # Core features should never be NULL for a row that exists at all --
    # if one is, something upstream is actually broken (not just sparse
    # data), so this still drops the account and says so.
    before = len(df)
    df = df.dropna(subset=CORE_COLUMNS)
    dropped = before - len(df)
    if dropped:
        logger.warning(
            f"Dropped {dropped} accounts missing CORE features -- "
            f"unexpected for an account with any recorded activity, worth "
            f"checking reddit_preprocessor.py output for those account_ids"
        )

    # burstiness_score is legitimately undefined below the 3-timestamp
    # bar, not broken -- report how many, then impute the column median
    # so those accounts still get clustered on their other 6 (perfectly
    # good) synchrony features instead of being dropped entirely.
    for col in CONDITIONAL_COLUMNS:
        n_missing = int(df[col].isna().sum())
        if not n_missing:
            continue
        median = df[col].median()
        if pd.isna(median):
            median = 0.0
            logger.warning(
                f"'{col}': 0/{len(df)} accounts have this feature -- "
                f"imputing 0.0 for all of them (revisit once more data "
                f"makes a real median possible)"
            )
        else:
            logger.info(
                f"'{col}': imputing {n_missing}/{len(df)} accounts missing "
                f"this feature with the column median ({median:.4f})"
            )
        df[col] = df[col].fillna(median)

    return df


def run_hdbscan(df):
    if len(df) < 10:
        raise ValueError(
            f"Only {len(df)} accounts have complete features -- "
            f"too few to cluster meaningfully. Collect more data first."
        )

    X = df[FEATURE_COLUMNS].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    logger.info(
        f"Running HDBSCAN (min_cluster_size={HDBSCAN_PARAMS['min_cluster_size']}, "
        f"min_samples={HDBSCAN_PARAMS['min_samples']})..."
    )
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=HDBSCAN_PARAMS['min_cluster_size'],
        min_samples=HDBSCAN_PARAMS['min_samples'],
    )
    labels = clusterer.fit_predict(X_scaled)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())

    logger.info(
        f"Clusters found: {n_clusters} | "
        f"Noise points: {n_noise}/{len(labels)} "
        f"({100 * n_noise / len(labels):.1f}%)"
    )
    if len(labels) > 0 and n_noise / len(labels) > 0.9:
        logger.warning(
            "Over 90% of accounts landed as noise -- with real data this "
            "usually means min_cluster_size/min_samples in config.py "
            "(currently tuned as placeholders, before any real data existed) "
            "are too strict for your actual sample size. Consider lowering "
            "them once you see how much data you actually have."
        )

    df = df.copy()
    df['cluster_id'] = labels

    # Coordination Score, per proposal Eq. 4.5: normalized cluster
    # persistence (HDBSCAN's stability measure). Noise (-1) scores 0.
    persistence = clusterer.cluster_persistence_
    if len(persistence) > 0 and persistence.max() > persistence.min():
        norm_persistence = (persistence - persistence.min()) / \
                            (persistence.max() - persistence.min())
    else:
        norm_persistence = np.ones_like(persistence)

    def score_for_label(label):
        if label == -1:
            return 0.0
        return float(norm_persistence[label])

    df['coord_score'] = df['cluster_id'].apply(score_for_label)

    if n_clusters > 0:
        logger.info("Cluster sizes (top 10 by size):")
        sizes = df[df['cluster_id'] != -1]['cluster_id'].value_counts().head(10)
        for cid, size in sizes.items():
            logger.info(
                f"  cluster {cid:>4} -> {size:>4} accounts "
                f"(coord_score={score_for_label(cid):.3f})"
            )

    return df.sort_values('coord_score', ascending=False)


def save_scores(df):
    conn = get_conn()
    c = conn.cursor()

    now = datetime.now(timezone.utc).isoformat()
    saved = 0

    for _, row in df.iterrows():
        # Same pattern as reddit_isolation_forest.py's save_scores(): only
        # touches coord_score/cluster_id/scored_at, never overwrites
        # anomaly_score if Isolation Forest already wrote it for this
        # account (or vice versa, if this runs first).
        c.execute("""
            INSERT INTO scores (account_id, coord_score, cluster_id, scored_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(account_id) DO UPDATE SET
                coord_score = excluded.coord_score,
                cluster_id  = excluded.cluster_id,
                scored_at   = excluded.scored_at
        """, (row['account_id'], float(row['coord_score']), int(row['cluster_id']), now))
        saved += 1

    conn.commit()
    conn.close()
    logger.info(f"Saved cluster_id + coord_score for {saved} accounts to scores table")


def print_top_coordinated(df, n=20):
    print(f"\n{'━'*70}")
    print(f"  TOP {n} MOST COORDINATED ACCOUNTS")
    print(f"{'━'*70}")
    print(f"{'account_id':<20}{'coord_score':>12}{'cluster':>9}{'posts/day':>12}{'hour_ent':>10}")
    print(f"{'-'*70}")

    for _, row in df.head(n).iterrows():
        print(
            f"{row['account_id'][:18]:<20}"
            f"{row['coord_score']:>12.4f}"
            f"{row['cluster_id']:>9}"
            f"{row['posts_per_day']:>12.3f}"
            f"{row['hour_entropy']:>10.3f}"
        )
    print(f"{'━'*70}\n")


if __name__ == "__main__":
    logger.info("Running HDBSCAN coordination clustering on real Reddit data\n")

    df = load_features()
    df = run_hdbscan(df)
    save_scores(df)
    print_top_coordinated(df)

    logger.info(
        "NOTE: min_cluster_size=3/min_samples=2 in config.py were set before "
        "real data existed. Once you see the noise percentage above, sanity- "
        "check a couple of top clusters by hand (do those accounts actually "
        "look coordinated?) before trusting coord_score further -- same "
        "spirit as the contamination note in reddit_isolation_forest.py."
    )