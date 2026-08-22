from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import pandas as pd


EPSILON = 1e-6

FEATURE_COLUMNS = [
    "account_id",
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


def safe_divide(a, b):
    return a / (b + EPSILON)


def create_feature_table(
    conn: sqlite3.Connection,
) -> None:
    """
    Create account_features.

    Assumes the table does not already exist.
    """

    conn.execute(
        """
        CREATE TABLE account_features (
            account_id INTEGER PRIMARY KEY,

            followers_count REAL,
            friends_count REAL,
            statuses_count REAL,
            favourites_count REAL,
            listed_count REAL,

            verified REAL,
            protected REAL,
            default_profile REAL,
            default_profile_image REAL,
            geo_enabled REAL,

            has_description REAL,
            has_location REAL,

            account_age_days REAL,

            statuses_per_day REAL,
            followers_per_day REAL,
            friends_per_day REAL,

            followers_friends_ratio REAL,
            friends_followers_ratio REAL,
            favorites_status_ratio REAL,
            listed_followers_ratio REAL,

            tweet_count REAL,
            avg_retweets REAL,
            avg_replies REAL,
            avg_favorites REAL,

            avg_hashtags REAL,
            avg_urls REAL,
            avg_mentions REAL,

            retweet_ratio REAL,
            reply_ratio REAL,

            avg_text_length REAL,
            avg_words REAL,

            url_ratio REAL,
            mention_ratio REAL,
            hashtag_ratio REAL,

            duplicate_tweet_ratio REAL,

            FOREIGN KEY(account_id)
                REFERENCES accounts(account_id)
        )
        """
    )

    conn.execute(
        """
        CREATE INDEX idx_account_features_account
        ON account_features(account_id)
        """
    )

    conn.commit()


def load_account_data(
    conn: sqlite3.Connection,
) -> pd.DataFrame:
    return pd.read_sql_query(
        """
        SELECT
            account_id,
            statuses_count,
            followers_count,
            friends_count,
            favourites_count,
            listed_count,

            verified,
            protected,
            default_profile,
            default_profile_image,
            geo_enabled,

            description,
            location,
            created_at
        FROM accounts
        """,
        conn,
    )


def load_tweet_statistics(
    conn: sqlite3.Connection,
) -> pd.DataFrame:

    query = """
        SELECT
            account_id,

            COUNT(*) AS tweet_count,

            AVG(
                COALESCE(retweet_count, 0)
            ) AS avg_retweets,

            AVG(
                COALESCE(reply_count, 0)
            ) AS avg_replies,

            AVG(
                COALESCE(favorite_count, 0)
            ) AS avg_favorites,

            AVG(
                COALESCE(num_hashtags, 0)
            ) AS avg_hashtags,

            AVG(
                COALESCE(num_urls, 0)
            ) AS avg_urls,

            AVG(
                COALESCE(num_mentions, 0)
            ) AS avg_mentions,

            AVG(
                CASE
                    WHEN text IS NOT NULL
                    THEN LENGTH(text)
                    ELSE 0
                END
            ) AS avg_text_length,

            AVG(
                CASE
                    WHEN text IS NOT NULL
                    THEN
                        LENGTH(text)
                        - LENGTH(
                            REPLACE(text, ' ', '')
                        )
                        + 1
                    ELSE 0
                END
            ) AS avg_words,

            SUM(
                CASE
                    WHEN retweeted_status_id IS NOT NULL
                    THEN 1
                    ELSE 0
                END
            ) * 1.0 / COUNT(*) AS retweet_ratio,

            SUM(
                CASE
                    WHEN in_reply_to_user_id IS NOT NULL
                    THEN 1
                    ELSE 0
                END
            ) * 1.0 / COUNT(*) AS reply_ratio,

            SUM(
                CASE
                    WHEN text LIKE '%http://%'
                      OR text LIKE '%https://%'
                    THEN 1
                    ELSE 0
                END
            ) * 1.0 / COUNT(*) AS url_ratio,

            SUM(
                CASE
                    WHEN text LIKE '%@%'
                    THEN 1
                    ELSE 0
                END
            ) * 1.0 / COUNT(*) AS mention_ratio,

            SUM(
                CASE
                    WHEN text LIKE '%#%'
                    THEN 1
                    ELSE 0
                END
            ) * 1.0 / COUNT(*) AS hashtag_ratio

        FROM tweets
        GROUP BY account_id
    """

    return pd.read_sql_query(
        query,
        conn,
    )


def load_duplicate_statistics(
    conn: sqlite3.Connection,
) -> pd.DataFrame:

    query = """
        SELECT
            account_id,
            COUNT(*) AS total_tweets,
            COUNT(DISTINCT text) AS unique_tweets
        FROM tweets
        WHERE text IS NOT NULL
        GROUP BY account_id
    """

    df = pd.read_sql_query(
        query,
        conn,
    )

    if df.empty:
        return pd.DataFrame(
            columns=[
                "account_id",
                "duplicate_tweet_ratio",
            ]
        )

    df["duplicate_tweet_ratio"] = (
        1.0
        - (
            df["unique_tweets"]
            / df["total_tweets"].clip(
                lower=1
            )
        )
    )

    return df[
        [
            "account_id",
            "duplicate_tweet_ratio",
        ]
    ]


def calculate_account_age(
    created_at: pd.Series,
) -> pd.Series:

    dates = pd.to_datetime(
        created_at,
        errors="coerce",
        utc=True,
    )

    reference_date = dates.max()

    if pd.isna(reference_date):
        return pd.Series(
            3650.0,
            index=created_at.index,
        )

    age = (
        reference_date - dates
    ).dt.total_seconds() / 86400.0

    return age.clip(
        lower=1.0
    )


def build_features(
    accounts: pd.DataFrame,
    tweets: pd.DataFrame,
    duplicates: pd.DataFrame,
) -> pd.DataFrame:

    df = accounts.copy()

    # --------------------------------------------------
    # Account age
    # --------------------------------------------------

    df["account_age_days"] = (
        calculate_account_age(
            df["created_at"]
        )
    )

    # --------------------------------------------------
    # Activity rates
    # --------------------------------------------------

    df["statuses_per_day"] = safe_divide(
        df["statuses_count"].fillna(0),
        df["account_age_days"],
    )

    df["followers_per_day"] = safe_divide(
        df["followers_count"].fillna(0),
        df["account_age_days"],
    )

    df["friends_per_day"] = safe_divide(
        df["friends_count"].fillna(0),
        df["account_age_days"],
    )

    # --------------------------------------------------
    # Ratios
    # --------------------------------------------------

    df["followers_friends_ratio"] = safe_divide(
        df["followers_count"].fillna(0),
        df["friends_count"].fillna(0),
    )

    df["friends_followers_ratio"] = safe_divide(
        df["friends_count"].fillna(0),
        df["followers_count"].fillna(0),
    )

    df["favorites_status_ratio"] = safe_divide(
        df["favourites_count"].fillna(0),
        df["statuses_count"].fillna(0),
    )

    df["listed_followers_ratio"] = safe_divide(
        df["listed_count"].fillna(0),
        df["followers_count"].fillna(0),
    )

    # --------------------------------------------------
    # Profile flags
    # --------------------------------------------------

    df["has_description"] = (
        df["description"]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
        .astype(float)
    )

    df["has_location"] = (
        df["location"]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
        .astype(float)
    )

    df = df.drop(
        columns=[
            "description",
            "location",
            "created_at",
        ]
    )

    # --------------------------------------------------
    # Tweet statistics
    # --------------------------------------------------

    df = df.merge(
        tweets,
        on="account_id",
        how="left",
    )

    df = df.merge(
        duplicates,
        on="account_id",
        how="left",
    )

    # --------------------------------------------------
    # Missing tweet values
    # --------------------------------------------------

    tweet_columns = [
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

    for column in tweet_columns:
        if column in df.columns:
            df[column] = (
                df[column]
                .fillna(0.0)
            )

    # --------------------------------------------------
    # Final feature columns
    # --------------------------------------------------

    df = df[
        FEATURE_COLUMNS
    ].copy()

    # --------------------------------------------------
    # Sanity checks
    # --------------------------------------------------

    if df["account_id"].duplicated().any():
        raise RuntimeError(
            "Duplicate account IDs found "
            "in generated feature table."
        )

    if len(df) == 0:
        raise RuntimeError(
            "Generated account_features is empty."
        )

    return df


def save_features(
    conn: sqlite3.Connection,
    features: pd.DataFrame,
) -> None:

    placeholders = ", ".join(
        ["?"] * len(
            FEATURE_COLUMNS
        )
    )

    columns = ", ".join(
        FEATURE_COLUMNS
    )

    sql = f"""
        INSERT INTO account_features (
            {columns}
        )
        VALUES ({placeholders})
    """

    rows = list(
        features.itertuples(
            index=False,
            name=None,
        )
    )

    conn.executemany(
        sql,
        rows,
    )

    conn.commit()


def validate_existing_features(
    conn: sqlite3.Connection,
) -> None:

    print(
        "\nExisting account_features table found."
    )

    account_count = conn.execute(
        """
        SELECT COUNT(*)
        FROM accounts
        """
    ).fetchone()[0]

    feature_count = conn.execute(
        """
        SELECT COUNT(*)
        FROM account_features
        """
    ).fetchone()[0]

    print(
        f"Accounts : {account_count:,}"
    )

    print(
        f"Features : {feature_count:,}"
    )

    if account_count != feature_count:
        raise RuntimeError(
            "account_features does not contain "
            "exactly one row per account."
        )

    duplicate_count = conn.execute(
        """
        SELECT COUNT(*)
        FROM (
            SELECT account_id
            FROM account_features
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

    missing_count = conn.execute(
        """
        SELECT COUNT(*)
        FROM accounts a
        LEFT JOIN account_features f
            ON a.account_id = f.account_id
        WHERE f.account_id IS NULL
        """
    ).fetchone()[0]

    if missing_count:
        raise RuntimeError(
            f"Found {missing_count} accounts "
            "without features."
        )

    # Verify expected columns.
    existing_columns = {
        row[1]
        for row in conn.execute(
            "PRAGMA table_info(account_features)"
        )
    }

    missing_columns = [
        column
        for column in FEATURE_COLUMNS
        if column not in existing_columns
    ]

    if missing_columns:
        raise RuntimeError(
            "Existing account_features is missing "
            f"columns: {missing_columns}"
        )

    print(
        "Existing account_features "
        "passed validation."
    )


def print_summary(
    conn: sqlite3.Connection,
) -> None:

    count = conn.execute(
        """
        SELECT COUNT(*)
        FROM account_features
        """
    ).fetchone()[0]

    print(
        "\n"
        + "=" * 70
    )

    print(
        "CRESCI-2017 FEATURE SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        f"Feature rows : {count:,}"
    )

    print(
        f"Feature count: "
        f"{len(FEATURE_COLUMNS) - 1}"
    )

    print(
        "\nFeature columns:"
    )

    for column in FEATURE_COLUMNS:
        if column != "account_id":
            print(
                f"  {column}"
            )

    print(
        "=" * 70
    )


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Create or validate the Cresci-2017 "
            "account-level feature table."
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
            "Explicitly replace account_features "
            "using the current accounts/tweets tables."
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

        # ------------------------------------------------
        # Existing feature table
        # ------------------------------------------------

        if table_exists(
            conn,
            "account_features",
        ):

            if not args.rebuild:

                validate_existing_features(
                    conn
                )

                print_summary(
                    conn
                )

                print(
                    "\nNo changes made."
                )

                return

            print(
                "\nWARNING:"
            )

            print(
                "Rebuilding account_features."
            )

            print(
                "This will replace the existing "
                "35-feature table."
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
                "DROP TABLE account_features"
            )

            conn.commit()

        # ------------------------------------------------
        # Load raw account information
        # ------------------------------------------------

        print(
            "\nLoading account data..."
        )

        accounts = load_account_data(
            conn
        )

        print(
            f"Accounts loaded: "
            f"{len(accounts):,}"
        )

        # ------------------------------------------------
        # Load tweet statistics
        # ------------------------------------------------

        print(
            "\nCalculating tweet statistics..."
        )

        tweets = load_tweet_statistics(
            conn
        )

        print(
            f"Tweet-stat rows: "
            f"{len(tweets):,}"
        )

        # ------------------------------------------------
        # Duplicate statistics
        # ------------------------------------------------

        print(
            "\nCalculating duplicate statistics..."
        )

        duplicates = (
            load_duplicate_statistics(
                conn
            )
        )

        print(
            f"Duplicate-stat rows: "
            f"{len(duplicates):,}"
        )

        # ------------------------------------------------
        # Build
        # ------------------------------------------------

        print(
            "\nBuilding features..."
        )

        features = build_features(
            accounts,
            tweets,
            duplicates,
        )

        print(
            f"Generated feature rows: "
            f"{len(features):,}"
        )

        # ------------------------------------------------
        # Create table
        # ------------------------------------------------

        create_feature_table(
            conn
        )

        save_features(
            conn,
            features,
        )

        # ------------------------------------------------
        # Validate
        # ------------------------------------------------

        validate_existing_features(
            conn
        )

        print_summary(
            conn
        )

    finally:
        conn.close()


if __name__ == "__main__":
    main()