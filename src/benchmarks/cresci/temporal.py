from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import pandas as pd


DEFAULT_WINDOW = 5

TEMPORAL_FEATURE_COLUMNS = [
    "account_id",
    "mean_temporal_similarity",
    "max_temporal_similarity",
    "high_coordination_count",
    "temporal_events_per_tweet",
    "temporal_neighbors_per_tweet",
    "high_coordination_ratio",
    "temporal_coordination_score",
]


def table_exists(
    conn: sqlite3.Connection,
    table_name: str,
) -> bool:
    row = conn.execute(
        """
        SELECT 1
        FROM sqlite_master
        WHERE type = 'table'
          AND name = ?
        """,
        (table_name,),
    ).fetchone()

    return row is not None


def create_table(
    conn: sqlite3.Connection,
) -> None:
    """
    Create the final 7-feature temporal table.

    This function assumes the table does not already exist.
    """

    conn.execute(
        """
        CREATE TABLE temporal_features_final (
            account_id INTEGER PRIMARY KEY,

            mean_temporal_similarity REAL NOT NULL DEFAULT 0.0,

            max_temporal_similarity REAL NOT NULL DEFAULT 0.0,

            high_coordination_count INTEGER NOT NULL DEFAULT 0,

            temporal_events_per_tweet REAL NOT NULL DEFAULT 0.0,

            temporal_neighbors_per_tweet REAL NOT NULL DEFAULT 0.0,

            high_coordination_ratio REAL NOT NULL DEFAULT 0.0,

            temporal_coordination_score REAL NOT NULL DEFAULT 0.0,

            FOREIGN KEY(account_id)
                REFERENCES accounts(account_id)
        )
        """
    )

    conn.execute(
        """
        CREATE INDEX idx_temporal_features_final_account
        ON temporal_features_final(account_id)
        """
    )

    conn.commit()


def load_partition(
    conn: sqlite3.Connection,
    source_group: str,
    split_name: str,
) -> pd.DataFrame:
    """
    Load tweets belonging only to:

        source_group
        split_name

    This prevents temporal activity from another split
    from contributing to this partition's features.
    """

    df = pd.read_sql_query(
        """
        SELECT
            t.tweet_id,
            t.account_id,
            t.created_at_utc

        FROM tweets t

        INNER JOIN benchmark_splits s
            ON t.account_id = s.account_id

        WHERE t.source_group = ?
          AND s.split = ?
          AND t.created_at_utc IS NOT NULL

        ORDER BY
            t.created_at_utc,
            t.tweet_id
        """,
        conn,
        params=(
            source_group,
            split_name,
        ),
    )

    if df.empty:
        return df

    df["created_at_utc"] = pd.to_datetime(
        df["created_at_utc"],
        errors="coerce",
        utc=True,
    )

    df = df.dropna(
        subset=["created_at_utc"]
    )

    if df.empty:
        return df

    # Integer Unix seconds.
    #
    # Cresci timestamps are effectively second-level
    # for the final temporal analysis.
    df["epoch_seconds"] = (
        df["created_at_utc"]
        .astype("int64")
        // 1_000_000_000
    )

    return df[
        [
            "tweet_id",
            "account_id",
            "epoch_seconds",
        ]
    ].reset_index(drop=True)


def calculate_temporal_features(
    df: pd.DataFrame,
    window_seconds: int,
) -> pd.DataFrame:
    """
    For every tweet, find the nearest event from a
    different account before and after it.

    Similarity:

        similarity = 1 - delta / window

    where delta is the actual time difference.

    Only similarities within the configured window
    are counted.
    """

    if df.empty:
        return pd.DataFrame(
            columns=TEMPORAL_FEATURE_COLUMNS
        )

    timestamps = (
        df["epoch_seconds"]
        .astype("int64")
        .to_numpy()
    )

    accounts = (
        df["account_id"]
        .astype("int64")
        .to_numpy()
    )

    n = len(df)

    # Nearest different-account event before/after
    # each tweet.
    before = [None] * n
    after = [None] * n

    # --------------------------------------------------
    # Forward pass
    # --------------------------------------------------

    # Most recent event for each of the two most recent
    # distinct accounts.
    recent = []

    for i in range(n):

        account = int(accounts[i])
        current = int(timestamps[i])

        # Search recent distinct accounts.
        best_distance = None

        for recent_account, recent_time in reversed(
            recent
        ):
            if recent_account == account:
                continue

            distance = current - recent_time

            if distance > window_seconds:
                break

            best_distance = distance
            break

        before[i] = best_distance

        # Update most recent event for this account.
        updated = False

        for j, (
            recent_account,
            recent_time,
        ) in enumerate(recent):

            if recent_account == account:
                recent[j] = (
                    account,
                    current,
                )
                updated = True
                break

        if not updated:
            recent.append(
                (
                    account,
                    current,
                )
            )

        # Keep recent accounts bounded.
        recent.sort(
            key=lambda x: x[1]
        )

        # Keep enough recent accounts to cover
        # the temporal window.
        cutoff = (
            current
            - window_seconds
        )

        recent = [
            item
            for item in recent
            if item[1] >= cutoff
        ]

    # --------------------------------------------------
    # Backward pass
    # --------------------------------------------------

    recent = []

    for i in range(
        n - 1,
        -1,
        -1,
    ):

        account = int(accounts[i])
        current = int(timestamps[i])

        best_distance = None

        for recent_account, recent_time in reversed(
            recent
        ):
            if recent_account == account:
                continue

            distance = recent_time - current

            if distance > window_seconds:
                break

            best_distance = distance
            break

        after[i] = best_distance

        updated = False

        for j, (
            recent_account,
            recent_time,
        ) in enumerate(recent):

            if recent_account == account:
                recent[j] = (
                    account,
                    current,
                )
                updated = True
                break

        if not updated:
            recent.append(
                (
                    account,
                    current,
                )
            )

        recent.sort(
            key=lambda x: x[1]
        )

        cutoff = (
            current
            + window_seconds
        )

        recent = [
            item
            for item in recent
            if item[1] <= cutoff
        ]

    # --------------------------------------------------
    # Account-level aggregation
    # --------------------------------------------------

    stats = {}

    def ensure_account(
        account_id: int,
    ):
        if account_id not in stats:
            stats[account_id] = {
                "events": 0,
                "similarity_sum": 0.0,
                "max_similarity": 0.0,
                "high_count": 0,
            }

        return stats[account_id]

    for i in range(n):

        account_id = int(
            accounts[i]
        )

        distances = []

        if before[i] is not None:
            distances.append(
                before[i]
            )

        if after[i] is not None:
            distances.append(
                after[i]
            )

        if not distances:
            continue

        item = ensure_account(
            account_id
        )

        for distance in distances:

            similarity = (
                1.0
                - (
                    float(distance)
                    / float(window_seconds)
                )
            )

            similarity = max(
                0.0,
                min(1.0, similarity),
            )

            item["events"] += 1

            item["similarity_sum"] += (
                similarity
            )

            item["max_similarity"] = max(
                item["max_similarity"],
                similarity,
            )

            if similarity >= 0.80:
                item["high_count"] += 1

    # Tweet count by account.
    tweet_counts = (
        df.groupby("account_id")
        .size()
        .to_dict()
    )

    rows = []

    for account_id, item in stats.items():

        events = item["events"]

        tweet_count = max(
            int(
                tweet_counts.get(
                    account_id,
                    1,
                )
            ),
            1,
        )

        mean_similarity = (
            item["similarity_sum"]
            / max(events, 1)
        )

        max_similarity = (
            item["max_similarity"]
        )

        high_count = (
            item["high_count"]
        )

        events_per_tweet = (
            events
            / tweet_count
        )

        neighbors_per_tweet = (
            events
            / tweet_count
        )

        high_ratio = (
            high_count
            / max(events, 1)
        )

        coordination_score = (
            0.40 * mean_similarity
            + 0.30 * max_similarity
            + 0.30 * high_ratio
        )

        rows.append(
            (
                account_id,
                mean_similarity,
                max_similarity,
                high_count,
                events_per_tweet,
                neighbors_per_tweet,
                high_ratio,
                coordination_score,
            )
        )

    return pd.DataFrame(
        rows,
        columns=TEMPORAL_FEATURE_COLUMNS,
    )


def save_features(
    conn: sqlite3.Connection,
    features: pd.DataFrame,
) -> None:

    if features.empty:
        return

    rows = list(
        features.itertuples(
            index=False,
            name=None,
        )
    )

    conn.executemany(
        """
        INSERT OR REPLACE INTO temporal_features_final (
            account_id,
            mean_temporal_similarity,
            max_temporal_similarity,
            high_coordination_count,
            temporal_events_per_tweet,
            temporal_neighbors_per_tweet,
            high_coordination_ratio,
            temporal_coordination_score
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )

    conn.commit()


def fill_missing_accounts(
    conn: sqlite3.Connection,
) -> None:
    """
    Accounts without temporal activity receive zeros.
    """

    conn.execute(
        """
        INSERT OR IGNORE INTO temporal_features_final (
            account_id,
            mean_temporal_similarity,
            max_temporal_similarity,
            high_coordination_count,
            temporal_events_per_tweet,
            temporal_neighbors_per_tweet,
            high_coordination_ratio,
            temporal_coordination_score
        )
        SELECT
            account_id,
            0.0,
            0.0,
            0,
            0.0,
            0.0,
            0.0,
            0.0
        FROM accounts
        """
    )

    conn.commit()


def validate_table(
    conn: sqlite3.Connection,
) -> None:

    total_accounts = conn.execute(
        """
        SELECT COUNT(*)
        FROM accounts
        """
    ).fetchone()[0]

    temporal_accounts = conn.execute(
        """
        SELECT COUNT(*)
        FROM temporal_features_final
        """
    ).fetchone()[0]

    if total_accounts != temporal_accounts:
        raise RuntimeError(
            "Temporal feature table does not contain "
            "exactly one row per account:\n"
            f"accounts={total_accounts:,}\n"
            f"temporal={temporal_accounts:,}"
        )

    columns = {
        row[1]
        for row in conn.execute(
            """
            PRAGMA table_info(
                temporal_features_final
            )
            """
        )
    }

    missing = [
        column
        for column in TEMPORAL_FEATURE_COLUMNS
        if column not in columns
    ]

    if missing:
        raise RuntimeError(
            "Missing temporal columns: "
            f"{missing}"
        )

    print(
        "Temporal feature table passed validation."
    )


def print_summary(
    conn: sqlite3.Connection,
) -> None:

    count = conn.execute(
        """
        SELECT COUNT(*)
        FROM temporal_features_final
        """
    ).fetchone()[0]

    stats = conn.execute(
        """
        SELECT
            AVG(mean_temporal_similarity),
            AVG(max_temporal_similarity),
            AVG(high_coordination_count),
            AVG(temporal_events_per_tweet),
            AVG(temporal_neighbors_per_tweet),
            AVG(high_coordination_ratio),
            AVG(temporal_coordination_score)
        FROM temporal_features_final
        """
    ).fetchone()

    print(
        "\n"
        + "=" * 70
    )

    print(
        "CRESCI-2017 TEMPORAL FEATURE SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        f"Accounts: {count:,}"
    )

    labels = [
        "mean_temporal_similarity",
        "max_temporal_similarity",
        "high_coordination_count",
        "temporal_events_per_tweet",
        "temporal_neighbors_per_tweet",
        "high_coordination_ratio",
        "temporal_coordination_score",
    ]

    for label, value in zip(
        labels,
        stats,
    ):
        print(
            f"{label:<32}"
            f"{value:.6f}"
        )

    print(
        "=" * 70
    )


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Build split-aware Cresci-2017 "
            "temporal coordination features."
        )
    )

    parser.add_argument(
        "--database",
        type=Path,
        default=Path(
            "data/benchmarks/cresci-2017.db"
        ),
    )

    parser.add_argument(
        "--window",
        type=int,
        default=DEFAULT_WINDOW,
    )

    parser.add_argument(
        "--rebuild",
        action="store_true",
        help=(
            "Replace temporal_features_final. "
            "Required when the table already exists."
        ),
    )

    args = parser.parse_args()

    if not args.database.exists():
        raise FileNotFoundError(
            f"Database not found:\n"
            f"  {args.database}"
        )

    if args.window <= 0:
        raise ValueError(
            "Window must be greater than zero."
        )

    conn = sqlite3.connect(
        args.database
    )

    try:

        # ------------------------------------------------
        # Protect existing table
        # ------------------------------------------------

        if table_exists(
            conn,
            "temporal_features_final",
        ):

            if not args.rebuild:

                print(
                    "\nExisting "
                    "temporal_features_final found."
                )

                validate_table(
                    conn
                )

                print_summary(
                    conn
                )

                print(
                    "\nNo changes made."
                )

                print(
                    "Use --rebuild to intentionally "
                    "regenerate temporal features."
                )

                return

            print(
                "\nWARNING:"
            )

            print(
                "Rebuilding temporal_features_final."
            )

            answer = input(
                "\nType REBUILD to continue: "
            ).strip()

            if answer != "REBUILD":
                print(
                    "Cancelled."
                )
                return

            conn.execute(
                """
                DROP TABLE temporal_features_final
                """
            )

            conn.commit()

        # ------------------------------------------------
        # Create table
        # ------------------------------------------------

        create_table(
            conn
        )

        # ------------------------------------------------
        # Determine groups
        # ------------------------------------------------

        groups = [
            row[0]
            for row in conn.execute(
                """
                SELECT DISTINCT source_group
                FROM tweets
                WHERE created_at_utc IS NOT NULL
                ORDER BY source_group
                """
            )
        ]

        splits = [
            "train",
            "validation",
            "test",
        ]

        print(
            f"Temporal window: "
            f"{args.window} seconds"
        )

        print(
            f"Source groups: "
            f"{len(groups)}"
        )

        print(
            "Split-aware processing enabled."
        )

        # ------------------------------------------------
        # Process independently by source group + split
        # ------------------------------------------------

        for split_name in splits:

            print(
                "\n"
                + "-" * 70
            )

            print(
                f"PROCESSING {split_name.upper()}"
            )

            print(
                "-" * 70
            )

            for group in groups:

                print(
                    f"\n{group}"
                )

                df = load_partition(
                    conn,
                    group,
                    split_name,
                )

                if df.empty:

                    print(
                        "  No timestamped tweets."
                    )

                    continue

                print(
                    f"  Tweets: {len(df):,}"
                )

                features = (
                    calculate_temporal_features(
                        df,
                        args.window,
                    )
                )

                print(
                    "  Accounts with temporal "
                    f"activity: {len(features):,}"
                )

                save_features(
                    conn,
                    features,
                )

        # ------------------------------------------------
        # Accounts with no temporal activity
        # ------------------------------------------------

        fill_missing_accounts(
            conn
        )

        # ------------------------------------------------
        # Validate
        # ------------------------------------------------

        validate_table(
            conn
        )

        print_summary(
            conn
        )

    finally:
        conn.close()


if __name__ == "__main__":
    main()