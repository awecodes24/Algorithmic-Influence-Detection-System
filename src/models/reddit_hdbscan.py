# src/models/reddit_hdbscan.py
#
# HDBSCAN coordination clustering on real Reddit data.
#
# Coordination detection requires enough behavioral activity to produce
# meaningful timing and activity features. Sparse accounts are therefore
# NOT median-imputed and forced into clusters.
#
# Accounts with insufficient activity receive:
#     coord_score = NULL
#     cluster_id = NULL
#
# This means "insufficient evidence for coordination analysis", NOT
# "organic" and NOT "zero coordination".

import logging
from datetime import datetime, timezone

import hdbscan
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import HDBSCAN_PARAMS
from src.db import get_conn


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

# Minimum number of distinct active days required before an account is
# considered for coordination clustering.
MIN_ACTIVE_DAYS_FOR_COORDINATION = 3

# Minimum combined behavioral activity rate.
#
# This prevents accounts with only a handful of interactions across a very
# long account lifetime from being clustered simply because they share
# coarse feature values.
MIN_COMBINED_ACTIVITY_RATE = 0.01


FEATURE_COLUMNS = [
    "posts_per_day",
    "comments_per_day",
    "hour_entropy",
    "night_activity_ratio",
    "burstiness_score",
    "duplicate_ratio",
    "subreddit_count",
]


# ---------------------------------------------------------------------
# LOAD FEATURES
# ---------------------------------------------------------------------

def load_features():
    """
    Load behavioral features and determine which accounts have enough
    meaningful activity for coordination analysis.

    An account must satisfy all of the following:

    1. burstiness_score is available
    2. active on at least MIN_ACTIVE_DAYS_FOR_COORDINATION distinct days
    3. combined posts/day + comments/day meets the minimum threshold
    """

    conn = get_conn()

    df = pd.read_sql(
        f"""
        SELECT
            account_id,
            active_days,
            posts_per_day,
            comments_per_day,
            {", ".join([
                col
                for col in FEATURE_COLUMNS
                if col not in (
                    "posts_per_day",
                    "comments_per_day",
                )
            ])}
        FROM features
        """,
        conn,
    )

    conn.close()

    logger.info(
        "Loaded %d accounts with computed features",
        len(df),
    )

    # -------------------------------------------------------------
    # Calculate combined activity rate
    # -------------------------------------------------------------

    df["combined_activity_rate"] = (
        df["posts_per_day"].fillna(0)
        + df["comments_per_day"].fillna(0)
    )

    # -------------------------------------------------------------
    # Coordination eligibility
    # -------------------------------------------------------------

    df["coordination_eligible"] = (
        df["burstiness_score"].notna()
        & (
            df["active_days"]
            >= MIN_ACTIVE_DAYS_FOR_COORDINATION
        )
        & (
            df["combined_activity_rate"]
            >= MIN_COMBINED_ACTIVITY_RATE
        )
    )

    eligible_df = df[
        df["coordination_eligible"]
    ].copy()

    logger.info(
        "Eligibility requirements:"
    )

    logger.info(
        "  burstiness_score available"
    )

    logger.info(
        "  active_days >= %d",
        MIN_ACTIVE_DAYS_FOR_COORDINATION,
    )

    logger.info(
        "  combined_activity_rate >= %.4f",
        MIN_COMBINED_ACTIVITY_RATE,
    )

    logger.info(
        "Coordination eligibility: %d/%d accounts eligible",
        len(eligible_df),
        len(df),
    )

    logger.info(
        "%d/%d accounts marked as insufficient data for coordination",
        len(df) - len(eligible_df),
        len(df),
    )

    # -------------------------------------------------------------
    # Ensure all clustering features are available
    # -------------------------------------------------------------

    before = len(eligible_df)

    eligible_df = eligible_df.dropna(
        subset=FEATURE_COLUMNS
    )

    dropped = before - len(eligible_df)

    if dropped:

        logger.warning(
            "Dropped %d eligible accounts because one or more "
            "coordination features were missing",
            dropped,
        )

        missing_ids = set(
            df.loc[
                df["coordination_eligible"],
                "account_id",
            ]
        ) - set(
            eligible_df["account_id"]
        )

        df.loc[
            df["account_id"].isin(missing_ids),
            "coordination_eligible",
        ] = False

    return df, eligible_df


# ---------------------------------------------------------------------
# RUN HDBSCAN
# ---------------------------------------------------------------------

def run_hdbscan(all_df, eligible_df):
    """
    Run HDBSCAN only on accounts with sufficient behavioral activity.

    Accounts that are not eligible remain in the result with:

        cluster_id = NaN
        coord_score = NaN
        coordination_status = "INSUFFICIENT_DATA"

    Noise points among eligible accounts receive:

        cluster_id = -1
        coord_score = 0.0
        coordination_status = "NO_CLUSTER"

    Clustered eligible accounts receive:

        cluster_id >= 0
        coord_score = normalized persistence
        coordination_status = "ANALYZED"
    """

    result = all_df.copy()

    # Default state for all accounts.
    result["cluster_id"] = np.nan
    result["coord_score"] = np.nan
    result["coordination_status"] = "INSUFFICIENT_DATA"

    # -----------------------------------------------------------------
    # Safety check
    # -----------------------------------------------------------------

    if len(eligible_df) < 10:
        logger.warning(
            "Only %d accounts have sufficient activity for coordination "
            "clustering. At least 10 are required for meaningful HDBSCAN. "
            "All accounts will remain INSUFFICIENT_DATA.",
            len(eligible_df),
        )

        return result

    # -----------------------------------------------------------------
    # Prepare eligible feature matrix
    # -----------------------------------------------------------------

    X = eligible_df[FEATURE_COLUMNS].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    logger.info(
        "Running HDBSCAN on %d eligible accounts "
        "(min_cluster_size=%s, min_samples=%s)",
        len(eligible_df),
        HDBSCAN_PARAMS["min_cluster_size"],
        HDBSCAN_PARAMS["min_samples"],
    )

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=HDBSCAN_PARAMS[
            "min_cluster_size"
        ],
        min_samples=HDBSCAN_PARAMS[
            "min_samples"
        ],
    )

    labels = clusterer.fit_predict(X_scaled)

    # -----------------------------------------------------------------
    # Cluster statistics
    # -----------------------------------------------------------------

    n_clusters = len(
        set(labels)
    ) - (
        1 if -1 in labels else 0
    )

    n_noise = int(
        (labels == -1).sum()
    )

    logger.info(
        "Clusters found: %d | Noise points: %d/%d (%.1f%%)",
        n_clusters,
        n_noise,
        len(labels),
        100 * n_noise / len(labels),
    )

    # -----------------------------------------------------------------
    # Calculate persistence-based coordination score
    # -----------------------------------------------------------------

    persistence = clusterer.cluster_persistence_

    if (
        len(persistence) > 0
        and persistence.max() > persistence.min()
    ):
        norm_persistence = (
            persistence - persistence.min()
        ) / (
            persistence.max() - persistence.min()
        )
    elif len(persistence) > 0:
        norm_persistence = np.ones_like(
            persistence
        )
    else:
        norm_persistence = np.array([])

    def score_for_label(label):
        if label == -1:
            return 0.0

        return float(
            norm_persistence[label]
        )

    # -----------------------------------------------------------------
    # Build results for eligible accounts
    # -----------------------------------------------------------------

    eligible_result = eligible_df[
        ["account_id"]
    ].copy()

    eligible_result["cluster_id"] = labels

    eligible_result["coord_score"] = (
        eligible_result["cluster_id"]
        .apply(score_for_label)
    )

    eligible_result["coordination_status"] = (
        np.where(
            eligible_result["cluster_id"] == -1,
            "NO_CLUSTER",
            "ANALYZED",
        )
    )

    # -----------------------------------------------------------------
    # Merge eligible results back into all accounts
    # -----------------------------------------------------------------

    eligible_result = eligible_result.set_index(
        "account_id"
    )

    result = result.set_index(
        "account_id"
    )

    result.update(
        eligible_result
    )

    result = result.reset_index()

    # -----------------------------------------------------------------
    # Logging
    # -----------------------------------------------------------------

    analyzed_count = int(
        (
            result["coordination_status"]
            == "ANALYZED"
        ).sum()
    )

    noise_count = int(
        (
            result["coordination_status"]
            == "NO_CLUSTER"
        ).sum()
    )

    insufficient_count = int(
        (
            result["coordination_status"]
            == "INSUFFICIENT_DATA"
        ).sum()
    )

    logger.info(
        "Coordination summary:"
    )

    logger.info(
        "  ANALYZED          : %d",
        analyzed_count,
    )

    logger.info(
        "  NO_CLUSTER        : %d",
        noise_count,
    )

    logger.info(
        "  INSUFFICIENT_DATA : %d",
        insufficient_count,
    )

    # -----------------------------------------------------------------
    # Show largest clusters
    # -----------------------------------------------------------------

    clustered = result[
        result["cluster_id"].notna()
        & (result["cluster_id"] != -1)
    ]

    if not clustered.empty:
        logger.info(
            "Cluster sizes (top 10):"
        )

        sizes = (
            clustered["cluster_id"]
            .value_counts()
            .head(10)
        )

        for cluster_id, size in sizes.items():

            cluster_score = clustered.loc[
                clustered["cluster_id"] == cluster_id,
                "coord_score",
            ].iloc[0]

            logger.info(
                "  cluster %4d -> %4d accounts "
                "(coord_score=%.3f)",
                int(cluster_id),
                int(size),
                cluster_score,
            )

    return result.sort_values(
        "coord_score",
        ascending=False,
        na_position="last",
    )


# ---------------------------------------------------------------------
# SAVE SCORES
# ---------------------------------------------------------------------

def save_scores(df):
    """
    Save HDBSCAN coordination results.

    Important:
    - NULL coord_score means insufficient data.
    - 0.0 coord_score means the account was analyzed but landed as noise.
    """

    conn = get_conn()
    cursor = conn.cursor()

    now = datetime.now(
        timezone.utc
    ).isoformat()

    saved = 0

    for _, row in df.iterrows():

        coord_score = (
            None
            if pd.isna(row["coord_score"])
            else float(row["coord_score"])
        )

        cluster_id = (
            None
            if pd.isna(row["cluster_id"])
            else int(row["cluster_id"])
        )

        cursor.execute(
            """
            INSERT INTO scores (
                account_id,
                coord_score,
                cluster_id,
                scored_at
            )
            VALUES (?, ?, ?, ?)

            ON CONFLICT(account_id)
            DO UPDATE SET
                coord_score = excluded.coord_score,
                cluster_id = excluded.cluster_id,
                scored_at = excluded.scored_at
            """,
            (
                row["account_id"],
                coord_score,
                cluster_id,
                now,
            ),
        )

        saved += 1

    conn.commit()
    conn.close()

    logger.info(
        "Saved coordination results for %d accounts",
        saved,
    )


# ---------------------------------------------------------------------
# DISPLAY RESULTS
# ---------------------------------------------------------------------

def print_top_coordinated(df, n=20):

    coordinated = df[
        df["coordination_status"] == "ANALYZED"
    ].copy()

    if coordinated.empty:

        print(
            "\nNo coordinated accounts were identified."
        )

        return

    coordinated = coordinated.sort_values(
        "coord_score",
        ascending=False,
    )

    print(
        f"\n{'━' * 78}"
    )

    print(
        f"  TOP {min(n, len(coordinated))} "
        f"MOST COORDINATED ACCOUNTS"
    )

    print(
        f"{'━' * 78}"
    )

    print(
        f"{'account_id':<20}"
        f"{'coord_score':>14}"
        f"{'cluster':>10}"
        f"{'posts/day':>14}"
        f"{'hour_ent':>12}"
    )

    print(
        f"{'-' * 78}"
    )

    for _, row in coordinated.head(n).iterrows():

        print(
            f"{str(row['account_id'])[:18]:<20}"
            f"{row['coord_score']:>14.4f}"
            f"{int(row['cluster_id']):>10}"
            f"{row['posts_per_day']:>14.3f}"
            f"{row['hour_entropy']:>12.3f}"
        )

    print(
        f"{'━' * 78}\n"
    )


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

if __name__ == "__main__":

    logger.info(
        "Running HDBSCAN coordination clustering "
        "on real Reddit data\n"
    )

    all_df, eligible_df = load_features()

    result_df = run_hdbscan(
        all_df,
        eligible_df,
    )

    save_scores(
        result_df
    )

    print_top_coordinated(
        result_df
    )

    logger.info(
        "IMPORTANT INTERPRETATION:"
    )

    logger.info(
        "coord_score = NULL means insufficient behavioral data "
        "for coordination analysis."
    )

    logger.info(
        "coord_score = 0.0 means the account was analyzed but "
        "did not belong to a stable HDBSCAN cluster."
    )

    logger.info(
        "coord_score > 0.0 means the account belongs to an "
        "identified behavioral cluster."
    )
