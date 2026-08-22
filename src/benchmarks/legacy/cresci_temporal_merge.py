from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import pandas as pd


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--database",
        type=Path,
        default=Path(
            "data/benchmarks/cresci-2017.db"
        ),
    )

    args = parser.parse_args()

    if not args.database.exists():
        raise FileNotFoundError(
            f"Database not found: {args.database}"
        )

    conn = sqlite3.connect(args.database)

    try:

        print("Loading temporal features...")

        temporal = pd.read_sql_query(
            """
            SELECT
                account_id,
                temporal_event_count,
                temporal_neighbor_count,
                mean_temporal_similarity,
                max_temporal_similarity,
                high_coordination_count
            FROM temporal_features
            """,
            conn,
        )

        print(
            f"Temporal accounts: "
            f"{len(temporal):,}"
        )

        print("Loading base features...")

        base = pd.read_sql_query(
            """
            SELECT *
            FROM account_features
            """,
            conn,
        )

        print(
            f"Base accounts: "
            f"{len(base):,}"
        )

        # ------------------------------------------------
        # Merge
        # ------------------------------------------------

        df = base.merge(
            temporal,
            on="account_id",
            how="left",
        )

        # Missing temporal values become zero.
        temporal_columns = [
            "temporal_event_count",
            "temporal_neighbor_count",
            "mean_temporal_similarity",
            "max_temporal_similarity",
            "high_coordination_count",
        ]

        for column in temporal_columns:
            df[column] = df[column].fillna(0.0)

        # ------------------------------------------------
        # Normalized features
        # ------------------------------------------------

        tweet_count = (
            df["tweet_count"]
            .clip(lower=1)
        )

        df[
            "temporal_events_per_tweet"
        ] = (
            df["temporal_event_count"]
            / tweet_count
        )

        df[
            "temporal_neighbors_per_tweet"
        ] = (
            df["temporal_neighbor_count"]
            / tweet_count
        )

        df[
            "high_coordination_ratio"
        ] = (
            df["high_coordination_count"]
            / df["temporal_event_count"]
            .clip(lower=1)
        )

        # Combined coordination score.
        df[
            "temporal_coordination_score"
        ] = (
            0.4
            * df[
                "mean_temporal_similarity"
            ]
            +
            0.3
            * df[
                "max_temporal_similarity"
            ]
            +
            0.3
            * df[
                "high_coordination_ratio"
            ]
        )

        # ------------------------------------------------
        # Replace account_features
        # ------------------------------------------------

        conn.execute(
            "DROP TABLE IF EXISTS account_features_temporal"
        )

        df.to_sql(
            "account_features_temporal",
            conn,
            if_exists="replace",
            index=False,
        )

        # Index
        conn.execute(
            """
            CREATE INDEX idx_account_features_temporal_account
            ON account_features_temporal(account_id)
            """
        )

        conn.commit()

        print(
            "\nCreated:"
        )

        print(
            "account_features_temporal"
        )

        print(
            f"Rows: {len(df):,}"
        )

        print("\nNew temporal features:")

        for column in [
            "temporal_events_per_tweet",
            "temporal_neighbors_per_tweet",
            "high_coordination_ratio",
            "temporal_coordination_score",
        ]:

            print(
                f"\n{column}"
            )

            print(
                df[column].describe()
            )

    finally:
        conn.close()


if __name__ == "__main__":
    main()