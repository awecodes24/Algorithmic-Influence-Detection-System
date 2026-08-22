from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import pandas as pd


# ------------------------------------------------------------
# Cresci-2017 source groups
#
# 0 = genuine / human
# 1 = bot
# ------------------------------------------------------------

GROUP_LABELS = {
    "genuine_accounts.csv": 0,

    "social_spambots_1.csv": 1,
    "social_spambots_2.csv": 1,
    "social_spambots_3.csv": 1,

    "traditional_spambots_1.csv": 1,
    "traditional_spambots_2.csv": 1,
    "traditional_spambots_3.csv": 1,
    "traditional_spambots_4.csv": 1,

    "fake_followers.csv": 1,
}


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def clean(value):
    """Convert pandas NaN / empty strings to None."""

    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if isinstance(value, str):
        value = value.strip()

        if not value:
            return None

    return value


def safe_int(value):
    """Safely convert a value to integer."""

    value = clean(value)

    if value is None:
        return None

    try:
        return int(float(value))
    except (ValueError, TypeError):
        return None


def find_csv(
    group_dir: Path,
    filename: str,
) -> Path | None:
    """
    Find a dataset CSV recursively.

    This handles both:

        group/users.csv

    and the nested fake-followers structure:

        group/fake_followers/users.csv
    """

    direct = group_dir / filename

    if direct.exists():
        return direct

    matches = list(
        group_dir.rglob(filename)
    )

    if matches:
        return matches[0]

    return None


# ------------------------------------------------------------
# Database initialization
# ------------------------------------------------------------

def initialize_database(
    db_path: Path,
    overwrite: bool = False,
):
    """
    Create a fresh Cresci database.

    IMPORTANT:
    Existing databases are protected by default.
    """

    db_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if db_path.exists() and not overwrite:
        raise FileExistsError(
            f"\nDatabase already exists:\n"
            f"  {db_path}\n\n"
            f"Refusing to overwrite it.\n"
            f"Use --overwrite only when intentionally "
            f"rebuilding the database from raw Cresci data."
        )

    conn = sqlite3.connect(
        db_path
    )

    conn.execute(
        "PRAGMA journal_mode=WAL"
    )

    conn.execute(
        "PRAGMA synchronous=NORMAL"
    )

    conn.execute(
        "PRAGMA foreign_keys=ON"
    )

    conn.executescript(
        """
        CREATE TABLE accounts (
            account_id INTEGER PRIMARY KEY,

            name TEXT,
            screen_name TEXT,

            statuses_count INTEGER,
            followers_count INTEGER,
            friends_count INTEGER,
            favourites_count INTEGER,
            listed_count INTEGER,

            lang TEXT,
            time_zone TEXT,
            location TEXT,

            default_profile INTEGER,
            default_profile_image INTEGER,
            geo_enabled INTEGER,

            verified INTEGER,
            protected INTEGER,

            description TEXT,

            created_at TEXT,
            timestamp TEXT,

            source_group TEXT NOT NULL,
            label INTEGER NOT NULL
        );

        CREATE TABLE tweets (
            tweet_id INTEGER PRIMARY KEY,

            account_id INTEGER NOT NULL,

            text TEXT,
            source TEXT,

            created_at TEXT,
            timestamp TEXT,

            retweet_count INTEGER,
            reply_count INTEGER,
            favorite_count INTEGER,

            num_hashtags INTEGER,
            num_urls INTEGER,
            num_mentions INTEGER,

            retweeted_status_id INTEGER,
            in_reply_to_user_id INTEGER,

            source_group TEXT NOT NULL,
            label INTEGER NOT NULL,

            FOREIGN KEY(account_id)
                REFERENCES accounts(account_id)
        );

        CREATE INDEX idx_accounts_label
            ON accounts(label);

        CREATE INDEX idx_accounts_group
            ON accounts(source_group);

        CREATE INDEX idx_tweets_account
            ON tweets(account_id);

        CREATE INDEX idx_tweets_timestamp
            ON tweets(timestamp);

        CREATE INDEX idx_tweets_group
            ON tweets(source_group);

        CREATE INDEX idx_tweets_account_timestamp
            ON tweets(account_id, timestamp);
        """
    )

    conn.commit()

    return conn


# ------------------------------------------------------------
# Users
# ------------------------------------------------------------

def process_users(
    conn: sqlite3.Connection,
    dataset_root: Path,
):
    sql = """
        INSERT OR IGNORE INTO accounts (
            account_id,
            name,
            screen_name,
            statuses_count,
            followers_count,
            friends_count,
            favourites_count,
            listed_count,
            lang,
            time_zone,
            location,
            default_profile,
            default_profile_image,
            geo_enabled,
            verified,
            protected,
            description,
            created_at,
            timestamp,
            source_group,
            label
        )
        VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
    """

    total = 0

    for group_name, label in GROUP_LABELS.items():

        group_dir = (
            dataset_root / group_name
        )

        if not group_dir.exists():
            print(
                f"[WARN] Missing group: "
                f"{group_dir}"
            )
            continue

        users_file = find_csv(
            group_dir,
            "users.csv",
        )

        if users_file is None:
            print(
                f"[WARN] users.csv not found "
                f"for {group_name}"
            )
            continue

        print(
            f"\n[USERS] {group_name}"
        )

        print(
            f"        {users_file}"
        )

        group_total = 0

        for chunk in pd.read_csv(
            users_file,
            chunksize=50_000,
            encoding="utf-8",
            encoding_errors="replace",
            low_memory=False,
            on_bad_lines="skip",
        ):

            rows = []

            for _, row in chunk.iterrows():

                account_id = safe_int(
                    row.get("id")
                )

                if account_id is None:
                    continue

                rows.append(
                    (
                        account_id,

                        clean(
                            row.get("name")
                        ),

                        clean(
                            row.get("screen_name")
                        ),

                        safe_int(
                            row.get(
                                "statuses_count"
                            )
                        ),

                        safe_int(
                            row.get(
                                "followers_count"
                            )
                        ),

                        safe_int(
                            row.get(
                                "friends_count"
                            )
                        ),

                        safe_int(
                            row.get(
                                "favourites_count"
                            )
                        ),

                        safe_int(
                            row.get(
                                "listed_count"
                            )
                        ),

                        clean(
                            row.get("lang")
                        ),

                        clean(
                            row.get("time_zone")
                        ),

                        clean(
                            row.get("location")
                        ),

                        safe_int(
                            row.get(
                                "default_profile"
                            )
                        ),

                        safe_int(
                            row.get(
                                "default_profile_image"
                            )
                        ),

                        safe_int(
                            row.get(
                                "geo_enabled"
                            )
                        ),

                        safe_int(
                            row.get("verified")
                        ),

                        safe_int(
                            row.get("protected")
                        ),

                        clean(
                            row.get("description")
                        ),

                        clean(
                            row.get("created_at")
                        ),

                        clean(
                            row.get("timestamp")
                        ),

                        group_name,

                        label,
                    )
                )

            if rows:

                conn.executemany(
                    sql,
                    rows,
                )

                conn.commit()

                group_total += len(rows)
                total += len(rows)

        print(
            f"        Rows processed: "
            f"{group_total:,}"
        )

    print(
        f"\n[USERS] Total rows processed: "
        f"{total:,}"
    )


# ------------------------------------------------------------
# Tweets
# ------------------------------------------------------------

def process_tweets(
    conn: sqlite3.Connection,
    dataset_root: Path,
):
    sql = """
        INSERT OR IGNORE INTO tweets (
            tweet_id,
            account_id,
            text,
            source,
            created_at,
            timestamp,
            retweet_count,
            reply_count,
            favorite_count,
            num_hashtags,
            num_urls,
            num_mentions,
            retweeted_status_id,
            in_reply_to_user_id,
            source_group,
            label
        )
        VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?
        )
    """

    total = 0

    for group_name, label in GROUP_LABELS.items():

        group_dir = (
            dataset_root / group_name
        )

        if not group_dir.exists():
            continue

        tweets_file = find_csv(
            group_dir,
            "tweets.csv",
        )

        if tweets_file is None:
            print(
                f"\n[TWEETS] "
                f"{group_name}: no tweets.csv"
            )
            continue

        print(
            f"\n[TWEETS] {group_name}"
        )

        print(
            f"         {tweets_file}"
        )

        group_total = 0

        for chunk in pd.read_csv(
            tweets_file,
            chunksize=50_000,
            encoding="utf-8",
            encoding_errors="replace",
            low_memory=False,
            on_bad_lines="skip",
        ):

            rows = []

            for _, row in chunk.iterrows():

                tweet_id = safe_int(
                    row.get("id")
                )

                account_id = safe_int(
                    row.get("user_id")
                )

                if (
                    tweet_id is None
                    or account_id is None
                ):
                    continue

                rows.append(
                    (
                        tweet_id,

                        account_id,

                        clean(
                            row.get("text")
                        ),

                        clean(
                            row.get("source")
                        ),

                        clean(
                            row.get("created_at")
                        ),

                        clean(
                            row.get("timestamp")
                        ),

                        safe_int(
                            row.get(
                                "retweet_count"
                            )
                        ),

                        safe_int(
                            row.get(
                                "reply_count"
                            )
                        ),

                        safe_int(
                            row.get(
                                "favorite_count"
                            )
                        ),

                        safe_int(
                            row.get(
                                "num_hashtags"
                            )
                        ),

                        safe_int(
                            row.get(
                                "num_urls"
                            )
                        ),

                        safe_int(
                            row.get(
                                "num_mentions"
                            )
                        ),

                        safe_int(
                            row.get(
                                "retweeted_status_id"
                            )
                        ),

                        safe_int(
                            row.get(
                                "in_reply_to_user_id"
                            )
                        ),

                        group_name,

                        label,
                    )
                )

            if rows:

                conn.executemany(
                    sql,
                    rows,
                )

                conn.commit()

                group_total += len(rows)
                total += len(rows)

        print(
            f"         Rows processed: "
            f"{group_total:,}"
        )

    print(
        f"\n[TWEETS] Total rows processed: "
        f"{total:,}"
    )


# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

def print_summary(
    conn: sqlite3.Connection,
):
    print(
        "\n"
        + "=" * 70
    )

    print(
        "CRESCI-2017 DATABASE SUMMARY"
    )

    print(
        "=" * 70
    )

    accounts = conn.execute(
        """
        SELECT COUNT(*)
        FROM accounts
        """
    ).fetchone()[0]

    tweets = conn.execute(
        """
        SELECT COUNT(*)
        FROM tweets
        """
    ).fetchone()[0]

    print(
        f"Accounts : {accounts:,}"
    )

    print(
        f"Tweets   : {tweets:,}"
    )

    print(
        "\nClass distribution:"
    )

    for label, count in conn.execute(
        """
        SELECT
            label,
            COUNT(*)
        FROM accounts
        GROUP BY label
        ORDER BY label
        """
    ):

        name = (
            "Human"
            if label == 0
            else "Bot"
        )

        print(
            f"  {name:<8} "
            f"{count:,}"
        )

    print(
        "\nSource groups:"
    )

    for (
        group,
        label,
        count,
    ) in conn.execute(
        """
        SELECT
            source_group,
            label,
            COUNT(*)
        FROM accounts
        GROUP BY
            source_group,
            label
        ORDER BY source_group
        """
    ):

        name = (
            "Human"
            if label == 0
            else "Bot"
        )

        print(
            f"  {group:<32} "
            f"{name:<6} "
            f"{count:,}"
        )

    print(
        "\nTweet counts:"
    )

    for (
        group,
        count,
    ) in conn.execute(
        """
        SELECT
            source_group,
            COUNT(*)
        FROM tweets
        GROUP BY source_group
        ORDER BY source_group
        """
    ):

        print(
            f"  {group:<32} "
            f"{count:,}"
        )

    print(
        "=" * 70
    )


# ------------------------------------------------------------
# Integrity check
# ------------------------------------------------------------

def integrity_check(
    conn: sqlite3.Connection,
):
    result = conn.execute(
        "PRAGMA integrity_check"
    ).fetchone()[0]

    print(
        f"\nDatabase integrity: {result}"
    )

    if result != "ok":
        raise RuntimeError(
            f"SQLite integrity check failed: "
            f"{result}"
        )


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Build a clean Cresci-2017 SQLite "
            "benchmark database from raw CSV files."
        )
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help=(
            "Path to datasets_full.csv containing "
            "the Cresci source-group directories."
        ),
    )

    parser.add_argument(
        "--database",
        type=Path,
        default=Path(
            "data/benchmarks/cresci-2017.db"
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Intentionally rebuild the database. "
            "WARNING: this destroys the existing "
            "Cresci database."
        ),
    )

    args = parser.parse_args()

    if not args.dataset.exists():
        raise FileNotFoundError(
            f"Dataset directory not found:\n"
            f"  {args.dataset}"
        )

    print(
        "Dataset root:"
    )

    print(
        f"  {args.dataset.resolve()}"
    )

    print(
        "\nDatabase:"
    )

    print(
        f"  {args.database.resolve()}"
    )

    if args.database.exists() and not args.overwrite:

        print(
            "\nDatabase already exists."
        )

        print(
            "No changes made."
        )

        print(
            "\nThis is intentional: "
            "the existing database contains "
            "your split/features/results."
        )

        print(
            "Use --overwrite only to rebuild "
            "everything from raw CSV files."
        )

        return

    conn = initialize_database(
        args.database,
        overwrite=args.overwrite,
    )

    try:

        print(
            "\nProcessing accounts..."
        )

        process_users(
            conn,
            args.dataset,
        )

        print(
            "\nProcessing tweets..."
        )

        process_tweets(
            conn,
            args.dataset,
        )

        print_summary(
            conn
        )

        integrity_check(
            conn
        )

    finally:

        conn.close()


if __name__ == "__main__":
    main()