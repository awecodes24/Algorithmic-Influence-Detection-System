from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import pandas as pd


DEFAULT_WINDOW = 5


def create_table(conn: sqlite3.Connection) -> None:
    conn.execute("DROP TABLE IF EXISTS temporal_features")

    conn.execute(
        """
        CREATE TABLE temporal_features (
            account_id INTEGER PRIMARY KEY,

            temporal_event_count INTEGER NOT NULL DEFAULT 0,

            temporal_neighbor_count INTEGER NOT NULL DEFAULT 0,

            mean_temporal_similarity REAL NOT NULL DEFAULT 0.0,

            max_temporal_similarity REAL NOT NULL DEFAULT 0.0,

            high_coordination_count INTEGER NOT NULL DEFAULT 0,

            FOREIGN KEY(account_id)
                REFERENCES accounts(account_id)
        )
        """
    )

    conn.commit()


def load_group(
    conn: sqlite3.Connection,
    source_group: str,
) -> pd.DataFrame:

    df = pd.read_sql_query(
        """
        SELECT
            tweet_id,
            account_id,
            created_at_utc
        FROM tweets
        WHERE source_group = ?
          AND created_at_utc IS NOT NULL
        ORDER BY created_at_utc, tweet_id
        """,
        conn,
        params=(source_group,),
    )

    if df.empty:
        return df

    df["created_at_utc"] = pd.to_datetime(
        df["created_at_utc"],
        utc=True,
        errors="coerce",
    )

    df = df.dropna(
        subset=["created_at_utc"]
    ).reset_index(drop=True)

    df["epoch"] = (
        df["created_at_utc"]
        .astype("int64")
        // 1_000_000_000
    )

    return df


def nearest_cross_account_features(
    df: pd.DataFrame,
    window_seconds: int,
) -> pd.DataFrame:

    if df.empty:
        return pd.DataFrame()

    timestamps = df["epoch"].to_numpy()
    accounts = df["account_id"].astype(int).to_numpy()

    n = len(df)

    # For each tweet, store nearest different-account
    # event before and after it.
    nearest_before = [None] * n
    nearest_after = [None] * n

    # --------------------------------------------------
    # Forward pass
    #
    # Find the nearest previous timestamp belonging
    # to a different account.
    # --------------------------------------------------

    last_account = None
    last_time = None

    second_last_account = None
    second_last_time = None

    for i in range(n):

        account = accounts[i]
        current = timestamps[i]

        if (
            last_account is not None
            and last_account != account
            and current - last_time <= window_seconds
        ):
            nearest_before[i] = current - last_time

        elif (
            second_last_account is not None
            and second_last_account != account
            and current - second_last_time <= window_seconds
        ):
            nearest_before[i] = current - second_last_time

        # Update recent event history.
        second_last_account = last_account
        second_last_time = last_time

        last_account = account
        last_time = current

    # --------------------------------------------------
    # Backward pass
    #
    # Find nearest following event from another account.
    # --------------------------------------------------

    last_account = None
    last_time = None

    second_last_account = None
    second_last_time = None

    for i in range(n - 1, -1, -1):

        account = accounts[i]
        current = timestamps[i]

        if (
            last_account is not None
            and last_account != account
            and last_time - current <= window_seconds
        ):
            nearest_after[i] = last_time - current

        elif (
            second_last_account is not None
            and second_last_account != account
            and second_last_time - current <= window_seconds
        ):
            nearest_after[i] = second_last_time - current

        second_last_account = last_account
        second_last_time = last_time

        last_account = account
        last_time = current

    # --------------------------------------------------
    # Convert tweet-level nearest distances into
    # account-level features.
    # --------------------------------------------------

    event_count = {}
    neighbor_count = {}
    similarity_sum = {}
    similarity_max = {}
    high_coordination = {}

    for i, account in enumerate(accounts):

        distances = []

        if nearest_before[i] is not None:
            distances.append(
                nearest_before[i]
            )

        if nearest_after[i] is not None:
            distances.append(
                nearest_after[i]
            )

        if not distances:
            continue

        # A tweet can contribute at most two nearest
        # cross-account events.
        for distance in distances:

            similarity = max(
                0.0,
                1.0 - (
                    distance
                    / window_seconds
                ),
            )

            account = int(account)

            event_count[account] = (
                event_count.get(account, 0) + 1
            )

            neighbor_count[account] = (
                neighbor_count.get(account, 0) + 1
            )

            similarity_sum[account] = (
                similarity_sum.get(account, 0.0)
                + similarity
            )

            similarity_max[account] = max(
                similarity_max.get(account, 0.0),
                similarity,
            )

            if similarity >= 0.80:
                high_coordination[account] = (
                    high_coordination.get(account, 0)
                    + 1
                )

    rows = []

    for account in event_count:

        events = event_count[account]

        rows.append(
            (
                account,
                events,
                neighbor_count.get(
                    account,
                    0,
                ),
                similarity_sum[account] / events,
                similarity_max[account],
                high_coordination.get(
                    account,
                    0,
                ),
            )
        )

    return pd.DataFrame(
        rows,
        columns=[
            "account_id",
            "temporal_event_count",
            "temporal_neighbor_count",
            "mean_temporal_similarity",
            "max_temporal_similarity",
            "high_coordination_count",
        ],
    )


def save_features(
    conn: sqlite3.Connection,
    features: pd.DataFrame,
) -> None:

    if features.empty:
        return

    conn.executemany(
        """
        INSERT OR REPLACE INTO temporal_features (
            account_id,
            temporal_event_count,
            temporal_neighbor_count,
            mean_temporal_similarity,
            max_temporal_similarity,
            high_coordination_count
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        list(
            features.itertuples(
                index=False,
                name=None,
            )
        ),
    )

    conn.commit()


def fill_missing_accounts(
    conn: sqlite3.Connection,
) -> None:

    conn.execute(
        """
        INSERT OR IGNORE INTO temporal_features (
            account_id,
            temporal_event_count,
            temporal_neighbor_count,
            mean_temporal_similarity,
            max_temporal_similarity,
            high_coordination_count
        )
        SELECT
            account_id,
            0,
            0,
            0.0,
            0.0,
            0
        FROM accounts
        """
    )

    conn.commit()


def main():

    parser = argparse.ArgumentParser()

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

    args = parser.parse_args()

    if not args.database.exists():
        raise FileNotFoundError(
            args.database
        )

    if args.window <= 0:
        raise ValueError(
            "Window must be positive."
        )

    conn = sqlite3.connect(
        args.database
    )

    try:

        create_table(conn)

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

        print(
            f"Window: {args.window} seconds"
        )

        for group in groups:

            print(
                f"\nProcessing {group}"
            )

            df = load_group(
                conn,
                group,
            )

            print(
                f"Tweets: {len(df):,}"
            )

            features = (
                nearest_cross_account_features(
                    df,
                    args.window,
                )
            )

            print(
                f"Accounts with coordination: "
                f"{len(features):,}"
            )

            save_features(
                conn,
                features,
            )

        fill_missing_accounts(
            conn
        )

        total = conn.execute(
            """
            SELECT COUNT(*)
            FROM temporal_features
            """
        ).fetchone()[0]

        print(
            f"\nTemporal features created for "
            f"{total:,} accounts."
        )

        print("\nSummary:")

        row = conn.execute(
            """
            SELECT
                AVG(temporal_event_count),
                AVG(mean_temporal_similarity),
                MAX(max_temporal_similarity),
                AVG(high_coordination_count)
            FROM temporal_features
            """
        ).fetchone()

        print(
            f"Average events      : {row[0]:.2f}"
        )
        print(
            f"Average similarity  : {row[1]:.4f}"
        )
        print(
            f"Maximum similarity  : {row[2]:.4f}"
        )
        print(
            f"High coordination   : {row[3]:.2f}"
        )

    finally:
        conn.close()


if __name__ == "__main__":
    main()