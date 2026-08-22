from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42

SPLIT_NAMES = {
    "train",
    "validation",
    "test",
}


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


def create_split_table(
    conn: sqlite3.Connection,
) -> None:
    """
    Create the benchmark_splits table.

    This function assumes the table does not already exist.
    """

    conn.execute(
        """
        CREATE TABLE benchmark_splits (
            account_id INTEGER PRIMARY KEY,

            split TEXT NOT NULL CHECK (
                split IN (
                    'train',
                    'validation',
                    'test'
                )
            ),

            label INTEGER NOT NULL,

            FOREIGN KEY(account_id)
                REFERENCES accounts(account_id)
        )
        """
    )

    conn.execute(
        """
        CREATE INDEX idx_benchmark_splits_split
        ON benchmark_splits(split)
        """
    )

    conn.execute(
        """
        CREATE INDEX idx_benchmark_splits_label
        ON benchmark_splits(label)
        """
    )

    conn.commit()


def load_accounts(
    conn: sqlite3.Connection,
) -> pd.DataFrame:
    df = pd.read_sql_query(
        """
        SELECT
            account_id,
            label
        FROM accounts
        ORDER BY account_id
        """,
        conn,
    )

    if df.empty:
        raise RuntimeError(
            "accounts table is empty."
        )

    if df["label"].isna().any():
        raise RuntimeError(
            "Some accounts have missing labels."
        )

    if df["label"].nunique() < 2:
        raise RuntimeError(
            "At least two classes are required "
            "for stratified splitting."
        )

    return df


def make_split(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create a deterministic account-level:

        70% train
        15% validation
        15% test

    split stratified by label.
    """

    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        stratify=df["label"],
        random_state=RANDOM_STATE,
    )

    validation_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["label"],
        random_state=RANDOM_STATE,
    )

    train_df = train_df.copy()
    validation_df = validation_df.copy()
    test_df = test_df.copy()

    train_df["split"] = "train"
    validation_df["split"] = "validation"
    test_df["split"] = "test"

    result = pd.concat(
        [
            train_df,
            validation_df,
            test_df,
        ],
        ignore_index=True,
    )

    result = result[
        [
            "account_id",
            "split",
            "label",
        ]
    ]

    result = result.sort_values(
        "account_id"
    ).reset_index(drop=True)

    return result


def save_split(
    conn: sqlite3.Connection,
    split_df: pd.DataFrame,
) -> None:
    rows = list(
        split_df.itertuples(
            index=False,
            name=None,
        )
    )

    conn.executemany(
        """
        INSERT INTO benchmark_splits (
            account_id,
            split,
            label
        )
        VALUES (?, ?, ?)
        """,
        rows,
    )

    conn.commit()


def validate_existing_split(
    conn: sqlite3.Connection,
) -> None:
    """
    Validate an existing split without modifying it.
    """

    print(
        "\nExisting benchmark_splits table found."
    )

    total_accounts = conn.execute(
        "SELECT COUNT(*) FROM accounts"
    ).fetchone()[0]

    total_splits = conn.execute(
        "SELECT COUNT(*) FROM benchmark_splits"
    ).fetchone()[0]

    if total_accounts != total_splits:
        raise RuntimeError(
            "Existing split is invalid:\n"
            f"  accounts          = {total_accounts:,}\n"
            f"  benchmark_splits = {total_splits:,}\n\n"
            "The split does not cover every account."
        )

    invalid_split_count = conn.execute(
        """
        SELECT COUNT(*)
        FROM benchmark_splits
        WHERE split NOT IN (
            'train',
            'validation',
            'test'
        )
        """
    ).fetchone()[0]

    if invalid_split_count:
        raise RuntimeError(
            f"Found {invalid_split_count} "
            "rows with invalid split names."
        )

    duplicate_count = conn.execute(
        """
        SELECT COUNT(*)
        FROM (
            SELECT account_id
            FROM benchmark_splits
            GROUP BY account_id
            HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0]

    if duplicate_count:
        raise RuntimeError(
            f"Found {duplicate_count} duplicate "
            "account assignments."
        )

    missing_count = conn.execute(
        """
        SELECT COUNT(*)
        FROM accounts a
        LEFT JOIN benchmark_splits s
            ON a.account_id = s.account_id
        WHERE s.account_id IS NULL
        """
    ).fetchone()[0]

    if missing_count:
        raise RuntimeError(
            f"Found {missing_count} accounts "
            "without a split."
        )

    mismatch_count = conn.execute(
        """
        SELECT COUNT(*)
        FROM accounts a
        JOIN benchmark_splits s
            ON a.account_id = s.account_id
        WHERE a.label != s.label
        """
    ).fetchone()[0]

    if mismatch_count:
        raise RuntimeError(
            f"Found {mismatch_count} rows where "
            "account label != split label."
        )

    print(
        "Existing split passed validation."
    )


def check_leakage(
    conn: sqlite3.Connection,
) -> None:
    print(
        "\nLeakage checks:"
    )

    duplicate_accounts = conn.execute(
        """
        SELECT account_id
        FROM benchmark_splits
        GROUP BY account_id
        HAVING COUNT(*) > 1
        """
    ).fetchall()

    if duplicate_accounts:
        raise RuntimeError(
            f"Found {len(duplicate_accounts)} "
            "accounts assigned to multiple splits."
        )

    print(
        "  ✓ No account appears in multiple splits."
    )

    missing = conn.execute(
        """
        SELECT COUNT(*)
        FROM accounts a
        LEFT JOIN benchmark_splits s
            ON a.account_id = s.account_id
        WHERE s.account_id IS NULL
        """
    ).fetchone()[0]

    if missing:
        raise RuntimeError(
            f"Found {missing} accounts missing "
            "from benchmark_splits."
        )

    print(
        "  ✓ Every account has exactly one split."
    )

    label_mismatch = conn.execute(
        """
        SELECT COUNT(*)
        FROM accounts a
        JOIN benchmark_splits s
            ON a.account_id = s.account_id
        WHERE a.label != s.label
        """
    ).fetchone()[0]

    if label_mismatch:
        raise RuntimeError(
            f"Found {label_mismatch} label mismatches."
        )

    print(
        "  ✓ Labels match accounts."
    )


def print_summary(
    conn: sqlite3.Connection,
) -> None:
    print(
        "\n"
        + "=" * 70
    )
    print(
        "CRESCI-2017 SPLIT SUMMARY"
    )
    print(
        "=" * 70
    )

    total = conn.execute(
        """
        SELECT COUNT(*)
        FROM benchmark_splits
        """
    ).fetchone()[0]

    print(
        f"Total accounts: {total:,}"
    )

    print(
        "\nSplit distribution:"
    )

    rows = conn.execute(
        """
        SELECT
            split,
            COUNT(*) AS count,
            ROUND(
                100.0 * COUNT(*)
                / (
                    SELECT COUNT(*)
                    FROM benchmark_splits
                ),
                2
            ) AS percentage
        FROM benchmark_splits
        GROUP BY split
        ORDER BY
            CASE split
                WHEN 'train' THEN 1
                WHEN 'validation' THEN 2
                WHEN 'test' THEN 3
            END
        """
    ).fetchall()

    for split, count, percentage in rows:
        print(
            f"  {split:<12}"
            f"{count:>8,}"
            f"  {percentage:>6.2f}%"
        )

    print(
        "\nClass distribution:"
    )

    rows = conn.execute(
        """
        SELECT
            split,
            label,
            COUNT(*) AS count
        FROM benchmark_splits
        GROUP BY split, label
        ORDER BY
            CASE split
                WHEN 'train' THEN 1
                WHEN 'validation' THEN 2
                WHEN 'test' THEN 3
            END,
            label
        """
    ).fetchall()

    for split, label, count in rows:
        class_name = (
            "Human"
            if label == 0
            else "Bot"
        )

        print(
            f"  {split:<12}"
            f"{class_name:<8}"
            f"{count:>8,}"
        )

    print(
        "\nLeakage check:"
    )

    check_leakage(conn)

    print(
        "=" * 70
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create or validate the Cresci-2017 "
            "account-level train/validation/test split."
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
        "--recreate",
        action="store_true",
        help=(
            "Explicitly delete and recreate "
            "benchmark_splits."
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
        # Existing split
        # ------------------------------------------------

        if table_exists(
            conn,
            "benchmark_splits",
        ):

            if args.recreate:

                print(
                    "\nWARNING:"
                )

                print(
                    "Recreating benchmark_splits."
                )

                print(
                    "This changes the benchmark "
                    "train/validation/test assignments."
                )

                answer = input(
                    "\nType RECREATE to continue: "
                ).strip()

                if answer != "RECREATE":
                    print(
                        "Cancelled."
                    )
                    return

                conn.execute(
                    "DROP TABLE benchmark_splits"
                )

                conn.commit()

            else:

                validate_existing_split(
                    conn
                )

                print_summary(
                    conn
                )

                print(
                    "\nNo changes made."
                )

                return

        # ------------------------------------------------
        # Create a new split
        # ------------------------------------------------

        accounts = load_accounts(
            conn
        )

        print(
            f"\nLoaded {len(accounts):,} "
            "accounts."
        )

        print(
            "Creating deterministic "
            "70/15/15 account-level split..."
        )

        split_df = make_split(
            accounts
        )

        create_split_table(
            conn
        )

        save_split(
            conn,
            split_df,
        )

        validate_existing_split(
            conn
        )

        print_summary(
            conn
        )

    finally:
        conn.close()


if __name__ == "__main__":
    main()