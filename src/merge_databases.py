#!/usr/bin/env python3
"""Merge friend SQLite databases into the current master database.

Only raw collection tables and dataset metadata are merged. Derived analysis
results are intentionally NOT imported because they must be recomputed on the
merged corpus.
"""
import argparse
from pathlib import Path
import sqlite3
import os

RAW_TABLES = ("posts", "comments", "accounts", "dataset_metadata")
DERIVED_TABLES = (
    "features", "edges", "content_similarity", "temporal_similarity",
    "account_pairs", "communities", "scores", "predictions",
    "coordination_events", "model_metrics", "experiments",
)


def merge_db(master: Path, sources: list[Path]) -> None:
    conn = sqlite3.connect(master, timeout=60)
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=60000")

    try:
        # Ensure the master schema exists before attachments.
        os.environ["INFLUENCE_DB_PATH"] = str(master.resolve())
        from src.db import init_db  # works from project root
        conn.close()
        init_db()
        conn = sqlite3.connect(master, timeout=60)
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=60000")

        before_posts = conn.execute("SELECT COUNT(*) FROM posts").fetchone()[0]
        before_comments = conn.execute("SELECT COUNT(*) FROM comments").fetchone()[0]
        before_accounts = conn.execute("SELECT COUNT(*) FROM accounts").fetchone()[0]

        for idx, source in enumerate(sources):
            alias = f"src{idx}"
            source = source.resolve()
            if source.resolve() == master.resolve():
                continue
            conn.execute(f"ATTACH DATABASE ? AS {alias}", (str(source),))
            try:
                conn.execute(f"INSERT OR IGNORE INTO accounts SELECT * FROM {alias}.accounts")
                conn.execute(f"INSERT OR IGNORE INTO posts SELECT * FROM {alias}.posts")
                conn.execute(f"INSERT OR IGNORE INTO comments SELECT * FROM {alias}.comments")

                # Rebuild account_activity later from the merged posts/comments.
                conn.execute(f"INSERT INTO dataset_metadata SELECT * FROM {alias}.dataset_metadata")
            finally:
                conn.execute(f"DETACH DATABASE {alias}")

        # Rebuild account_activity from final raw rows so duplicate keyword
        # discoveries cannot duplicate activity, while distinct posts/comments
        # that happen at the same second remain distinct rows.
        conn.execute("DELETE FROM account_activity")
        conn.execute("""
            INSERT INTO account_activity (account_id, activity_type, subreddit, created_utc)
            SELECT account_id, 'post', subreddit, created_utc FROM posts
        """)
        conn.execute("""
            INSERT INTO account_activity (account_id, activity_type, subreddit, created_utc)
            SELECT account_id, 'comment', subreddit, created_utc FROM comments
        """)

        # Recompute denormalized account totals from the final merged raw data.
        conn.execute("""
            UPDATE accounts
            SET total_posts = (SELECT COUNT(*) FROM posts WHERE posts.account_id = accounts.id),
                total_comments = (SELECT COUNT(*) FROM comments WHERE comments.account_id = accounts.id)
        """)

        # Previously computed coordination outputs are invalid after merging a
        # larger corpus. Clear them; model_metrics/experiments are retained as
        # historical experiment records.
        for table in (
            "features", "edges", "content_similarity", "temporal_similarity",
            "account_pairs", "communities", "scores", "predictions",
            "coordination_events",
        ):
            conn.execute(f"DELETE FROM {table}")

        conn.commit()

        after_posts = conn.execute("SELECT COUNT(*) FROM posts").fetchone()[0]
        after_comments = conn.execute("SELECT COUNT(*) FROM comments").fetchone()[0]
        after_accounts = conn.execute("SELECT COUNT(*) FROM accounts").fetchone()[0]

        print(f"Master before: posts={before_posts}, comments={before_comments}, accounts={before_accounts}")
        print(f"Master after:  posts={after_posts}, comments={after_comments}, accounts={after_accounts}")
        print("Derived analysis tables were NOT merged; recompute them from the final master DB.")
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("master", type=Path, help="Master SQLite database")
    parser.add_argument("sources", nargs="+", type=Path, help="Friend SQLite databases")
    args = parser.parse_args()
    merge_db(args.master, args.sources)


if __name__ == "__main__":
    main()
