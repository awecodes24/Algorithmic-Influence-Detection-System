from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import pandas as pd


BASE_FEATURES = [
    "followers_count",
    "friends_count",
    "statuses_count",
    "favourites_count",
    "listed_count",
    "verified",
    "protected",
    "default_profile",
    "default_profile_image",
    "geo_enabled",
    "has_description",
    "has_location",
    "account_age_days",
    "statuses_per_day",
    "followers_per_day",
    "friends_per_day",
    "followers_friends_ratio",
    "friends_followers_ratio",
    "favorites_status_ratio",
    "listed_followers_ratio",
    "tweet_count",
    "avg_retweets",
    "avg_replies",
    "avg_favorites",
    "avg_hashtags",
    "avg_urls",
    "avg_mentions",
    "retweet_ratio",
    "reply_ratio",
    "avg_text_length",
    "avg_words",
    "url_ratio",
    "mention_ratio",
    "hashtag_ratio",
    "duplicate_tweet_ratio",
]


TEMPORAL_FEATURES = [
    "high_coordination_count",
    "temporal_events_per_tweet",
    "temporal_neighbors_per_tweet",
    "high_coordination_ratio",
    "temporal_coordination_score",
]

FINAL_FEATURES = (
    BASE_FEATURES
    + TEMPORAL_FEATURES
)


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


def get_columns(
    conn: sqlite3.Connection,
    table_name: str,
) -> set[str]:
    return {
        row[1]
        for row in conn.execute(
            f'PRAGMA table_info("{table_name}")'
        )
    }


def validate_source_tables(
    conn: sqlite3.Connection,
) -> None:
    required_tables = [
        "accounts",
        "account_features",
        "temporal_features_final",
        "benchmark_splits",
    ]

    for table in required_tables:
        if not table_exists(
            conn,
            table,
        ):
            raise RuntimeError(
                f"Required table missing: {table}"
            )

    base_columns = get_columns(
        conn,
        "account_features",
    )

    missing_base = [
        column
        for column in BASE_FEATURES
        if column not in base_columns
    ]

    if missing_base:
        raise RuntimeError(
            "account_features is missing "
            f"columns: {missing_base}"
        )

    temporal_columns = get_columns(
        conn,
        "temporal_features_final",
    )

    missing_temporal = [
        column
        for column in TEMPORAL_FEATURES
        if column not in temporal_columns
    ]

    if missing_temporal:
        raise RuntimeError(
            "temporal_features_final is missing "
            f"columns: {missing_temporal}"
        )


def build_final_features(
    conn: sqlite3.Connection,
) -> pd.DataFrame:
    base_columns = ", ".join(
        f'b."{column}"'
        for column in BASE_FEATURES
    )

    temporal_columns = ", ".join(
        f't."{column}"'
        for column in TEMPORAL_FEATURES
    )

    query = f"""
        SELECT
            b.account_id,

            {base_columns},

            {temporal_columns},

            s.split,
            s.label

        FROM account_features b

        INNER JOIN temporal_features_final t
            ON b.account_id = t.account_id

        INNER JOIN benchmark_splits s
            ON b.account_id = s.account_id

        INNER JOIN accounts a
            ON b.account_id = a.account_id

        ORDER BY b.account_id
    """

    df = pd.read_sql_query(
        query,
        conn,
    )

    if df.empty:
        raise RuntimeError(
            "No rows were produced for "
            "account_features_final."
        )

    # Fill temporal values for accounts without
    # detected temporal activity.
    for column in TEMPORAL_FEATURES:
        df[column] = (
            df[column]
            .fillna(0.0)
        )

    # --------------------------------------------------
    # Basic integrity checks
    # --------------------------------------------------

    if df["account_id"].duplicated().any():
        duplicates = int(
            df["account_id"].duplicated().sum()
        )

        raise RuntimeError(
            f"Found {duplicates} duplicate "
            "account IDs."
        )

    account_count = conn.execute(
        """
        SELECT COUNT(*)
        FROM accounts
        """
    ).fetchone()[0]

    if len(df) != account_count:
        raise RuntimeError(
            "Final feature count does not match "
            "account count:\n"
            f"accounts={account_count:,}\n"
            f"features={len(df):,}"
        )

    # Check exact feature presence.
    missing = [
        column
        for column in FINAL_FEATURES
        if column not in df.columns
    ]

    if missing:
        raise RuntimeError(
            "Missing final features: "
            f"{missing}"
        )

    return df


def create_final_table(
    conn: sqlite3.Connection,
) -> None:
    column_defs = []

    column_defs.append(
        "account_id INTEGER PRIMARY KEY"
    )

    for column in BASE_FEATURES:
        column_defs.append(
            f'"{column}" REAL'
        )

    for column in TEMPORAL_FEATURES:
        column_defs.append(
            f'"{column}" REAL'
        )

    column_defs.extend(
        [
            "split TEXT NOT NULL",
            "label INTEGER NOT NULL",
        ]
    )

    sql = f"""
        CREATE TABLE account_features_final (
            {", ".join(column_defs)}
        )
    """

    conn.execute(sql)

    conn.execute(
        """
        CREATE INDEX idx_final_features_account
        ON account_features_final(account_id)
        """
    )

    conn.execute(
        """
        CREATE INDEX idx_final_features_split
        ON account_features_final(split)
        """
    )

    conn.execute(
        """
        CREATE INDEX idx_final_features_label
        ON account_features_final(label)
        """
    )

    conn.commit()


def save_final_features(
    conn: sqlite3.Connection,
    df: pd.DataFrame,
) -> None:
    columns = [
        "account_id",
        *BASE_FEATURES,
        *TEMPORAL_FEATURES,
        "split",
        "label",
    ]

    placeholders = ", ".join(
        ["?"] * len(columns)
    )

    column_sql = ", ".join(
        f'"{column}"'
        for column in columns
    )

    sql = f"""
        INSERT INTO account_features_final (
            {column_sql}
        )
        VALUES ({placeholders})
    """

    rows = list(
        df[columns].itertuples(
            index=False,
            name=None,
        )
    )

    conn.executemany(
        sql,
        rows,
    )

    conn.commit()


def validate_existing_final_table(
    conn: sqlite3.Connection,
) -> None:
    print(
        "\nExisting account_features_final "
        "table found."
    )

    count = conn.execute(
        """
        SELECT COUNT(*)
        FROM account_features_final
        """
    ).fetchone()[0]

    accounts = conn.execute(
        """
        SELECT COUNT(*)
        FROM accounts
        """
    ).fetchone()[0]

    print(
        f"Accounts       : {accounts:,}"
    )

    print(
        f"Final features : {count:,}"
    )

    if count != accounts:
        raise RuntimeError(
            "account_features_final does not "
            "contain exactly one row per account."
        )

    columns = get_columns(
        conn,
        "account_features_final",
    )

    required = [
        "account_id",
        *FINAL_FEATURES,
        "split",
        "label",
    ]

    missing = [
        column
        for column in required
        if column not in columns
    ]

    if missing:
        raise RuntimeError(
            "account_features_final is missing "
            f"columns: {missing}"
        )

    duplicate_count = conn.execute(
        """
        SELECT COUNT(*)
        FROM (
            SELECT account_id
            FROM account_features_final
            GROUP BY account_id
            HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0]

    if duplicate_count:
        raise RuntimeError(
            f"Found {duplicate_count} "
            "duplicate account IDs."
        )

    missing_accounts = conn.execute(
        """
        SELECT COUNT(*)
        FROM accounts a
        LEFT JOIN account_features_final f
            ON a.account_id = f.account_id
        WHERE f.account_id IS NULL
        """
    ).fetchone()[0]

    if missing_accounts:
        raise RuntimeError(
            f"Found {missing_accounts} "
            "accounts missing from final features."
        )

    label_mismatch = conn.execute(
        """
        SELECT COUNT(*)
        FROM accounts a
        INNER JOIN account_features_final f
            ON a.account_id = f.account_id
        WHERE a.label != f.label
        """
    ).fetchone()[0]

    if label_mismatch:
        raise RuntimeError(
            f"Found {label_mismatch} label mismatches."
        )

    split_missing = conn.execute(
        """
        SELECT COUNT(*)
        FROM account_features_final
        WHERE split NOT IN (
            'train',
            'validation',
            'test'
        )
        """
    ).fetchone()[0]

    if split_missing:
        raise RuntimeError(
            f"Found {split_missing} invalid split values."
        )

    print(
        "Existing account_features_final "
        "passed validation."
    )


def print_summary(
    conn: sqlite3.Connection,
) -> None:
    print(
        "\n"
        + "=" * 70
    )

    print(
        "CRESCI-2017 FINAL FEATURE SUMMARY"
    )

    print(
        "=" * 70
    )

    count = conn.execute(
        """
        SELECT COUNT(*)
        FROM account_features_final
        """
    ).fetchone()[0]

    print(
        f"Rows           : {count:,}"
    )

    print(
        f"Base features  : {len(BASE_FEATURES)}"
    )

    print(
        f"Temporal       : {len(TEMPORAL_FEATURES)}"
    )

    print(
        f"Total features : {len(FINAL_FEATURES)}"
    )

    print(
        "\nSplit distribution:"
    )

    for split, count in conn.execute(
        """
        SELECT
            split,
            COUNT(*)
        FROM account_features_final
        GROUP BY split
        ORDER BY
            CASE split
                WHEN 'train' THEN 1
                WHEN 'validation' THEN 2
                WHEN 'test' THEN 3
            END
        """
    ):
        print(
            f"  {split:<12} {count:,}"
        )

    print(
        "\nFinal temporal features:"
    )

    for feature in TEMPORAL_FEATURES:
        print(
            f"  {feature}"
        )

    print(
        "=" * 70
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create the final Cresci-2017 "
            "42-feature account table."
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
        "--rebuild",
        action="store_true",
        help=(
            "Explicitly replace "
            "account_features_final."
        ),
    )

    args = parser.parse_args()

    if not args.database.exists():
        raise FileNotFoundError(
            f"Database not found:\n"
            f"  {args.database}"
        )

    conn = sqlite3.connect(
        args.database
    )

    try:
        validate_source_tables(
            conn
        )

        # ------------------------------------------------
        # Existing final table
        # ------------------------------------------------

        if table_exists(
            conn,
            "account_features_final",
        ):

            if not args.rebuild:

                validate_existing_final_table(
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
                    "recreate the final table."
                )

                return

            print(
                "\nWARNING:"
            )

            print(
                "Rebuilding account_features_final."
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
                DROP TABLE account_features_final
                """
            )

            conn.commit()

        # ------------------------------------------------
        # Build
        # ------------------------------------------------

        print(
            "\nBuilding final 42-feature table..."
        )

        df = build_final_features(
            conn
        )

        print(
            f"Rows generated: {len(df):,}"
        )

        print(
            f"Feature count : "
            f"{len(FINAL_FEATURES)}"
        )

        create_final_table(
            conn
        )

        save_final_features(
            conn,
            df,
        )

        validate_existing_final_table(
            conn
        )

        print_summary(
            conn
        )

    finally:
        conn.close()


if __name__ == "__main__":
    main()