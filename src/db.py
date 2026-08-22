# src/db.py

from __future__ import annotations

import sqlite3
from pathlib import Path

from src.config import DB_PATH, ensure_directories


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# Keep the project's database location explicit.
DB_PATH = DATA_DIR / "influence.db"


# ---------------------------------------------------------------------
# DATABASE CONNECTION
# ---------------------------------------------------------------------

def get_conn(db_path: str | Path | None = None):
    """
    Return a configured SQLite connection.

    Parameters
    ----------
    db_path:
        Optional custom database path. If omitted, the project's
        default influence.db database is used.
    """

    ensure_directories()

    path = Path(db_path) if db_path else DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(path)

    conn.row_factory = sqlite3.Row

    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA busy_timeout = 60000")

    return conn


# ---------------------------------------------------------------------
# SCHEMA MIGRATIONS
# ---------------------------------------------------------------------

def _migrate_schema(conn):
    """
    Apply idempotent schema migrations.

    SQLite CREATE TABLE IF NOT EXISTS does not modify an already
    existing table. Therefore, columns introduced in newer versions
    must be added explicitly with ALTER TABLE.

    This function is safe to run repeatedly.
    Existing data is preserved.
    """

    c = conn.cursor()

    # ================================================================
    # COORDINATION EVENTS MIGRATIONS
    # ================================================================

    c.execute(
        """
        PRAGMA table_info(coordination_events)
        """
    )

    coordination_event_columns = {
        row["name"]
        for row in c.fetchall()
    }

    required_coordination_event_columns = {
        "target_post_id": "TEXT",
        "source_content_type": "TEXT",
        "target_content_type": "TEXT",
    }

    for column, column_type in required_coordination_event_columns.items():

        if column not in coordination_event_columns:

            print(
                f"[DB MIGRATION] Adding "
                f"coordination_events.{column}"
            )

            c.execute(
                f"""
                ALTER TABLE coordination_events
                ADD COLUMN {column} {column_type}
                """
            )

            coordination_event_columns.add(column)

    # ================================================================
    # SCORES TABLE MIGRATIONS
    # ================================================================

    c.execute(
        """
        PRAGMA table_info(scores)
        """
    )

    existing_score_columns = {
        row["name"]
        for row in c.fetchall()
    }

    # All columns that may be missing from older versions of the
    # scores table.
    required_score_columns = {
        "network_score_topic_scoped": "REAL",
        "temporal_score": "REAL",
        "confidence_level": "TEXT",
        "evidence_status": "TEXT",
        "assessment": "TEXT",
    }

    for column, column_type in required_score_columns.items():

        if column not in existing_score_columns:

            print(
                f"[DB MIGRATION] Adding "
                f"scores.{column}"
            )

            c.execute(
                f"""
                ALTER TABLE scores
                ADD COLUMN {column} {column_type}
                """
            )

            existing_score_columns.add(column)
    
    # ================================================================
    # ACCOUNT PAIRS TABLE MIGRATIONS
    # ================================================================

    c.execute(
        """
        PRAGMA table_info(account_pairs)
        """
    )

    existing_account_pair_columns = {
        row["name"]
        for row in c.fetchall()
    }

    required_account_pair_columns = {
        "network_volume_score": "REAL",
        "network_reciprocity_score": "REAL",
        "network_concentration_score": "REAL",
    }

    for column, column_type in (
        required_account_pair_columns.items()
    ):

        if column not in existing_account_pair_columns:

            print(
                f"[DB MIGRATION] Adding "
                f"account_pairs.{column}"
            )

            c.execute(
                f"""
                ALTER TABLE account_pairs
                ADD COLUMN {column} {column_type}
                """
            )

            existing_account_pair_columns.add(
                column
            )

    conn.commit()


# ---------------------------------------------------------------------
# DATABASE INITIALIZATION
# ---------------------------------------------------------------------

def init_db():
    """
    Create all database tables and indexes.

    Existing databases are preserved. Missing columns are added through
    _migrate_schema().
    """

    conn = get_conn()
    c = conn.cursor()

    c.executescript(
        """

        ----------------------------------------------------
        -- RAW SCRAPED DATA
        ----------------------------------------------------

        CREATE TABLE IF NOT EXISTS posts (
            id              TEXT PRIMARY KEY,
            account_id      TEXT NOT NULL,
            subreddit       TEXT,
            title           TEXT,
            text            TEXT,
            content_hash    TEXT,
            created_utc     REAL,
            scraped_at      REAL,
            score           INTEGER,
            num_comments    INTEGER,
            permalink       TEXT,
            edited          INTEGER DEFAULT 0,

            topic           TEXT,
            topic_score     REAL,
            sentiment       REAL,
            language        TEXT,
            is_relevant     INTEGER DEFAULT 1
        );


        CREATE TABLE IF NOT EXISTS comments (
            id              TEXT PRIMARY KEY,
            account_id      TEXT NOT NULL,
            post_id         TEXT,
            parent_id       TEXT,
            subreddit       TEXT,
            text            TEXT,
            content_hash    TEXT,
            created_utc     REAL,
            scraped_at      REAL,
            score           INTEGER,
            edited          INTEGER DEFAULT 0,

            topic           TEXT,
            topic_score     REAL,
            sentiment       REAL,
            language        TEXT,
            is_relevant     INTEGER DEFAULT 1
        );


        ----------------------------------------------------
        -- ACCOUNTS
        ----------------------------------------------------

        CREATE TABLE IF NOT EXISTS accounts (
            id              TEXT PRIMARY KEY,
            username        TEXT,
            created_utc     REAL,
            comment_karma   INTEGER,
            link_karma      INTEGER,
            total_posts     INTEGER DEFAULT 0,
            total_comments  INTEGER DEFAULT 0
        );


        ----------------------------------------------------
        -- ACCOUNT ACTIVITY
        ----------------------------------------------------

        CREATE TABLE IF NOT EXISTS account_activity (
            id INTEGER PRIMARY KEY AUTOINCREMENT,

            account_id TEXT,
            activity_type TEXT,
            subreddit TEXT,
            created_utc REAL,

            UNIQUE(
                account_id,
                activity_type,
                subreddit,
                created_utc
            )
        );


        ----------------------------------------------------
        -- ENGINEERED FEATURES
        ----------------------------------------------------

        CREATE TABLE IF NOT EXISTS features (
            account_id              TEXT PRIMARY KEY,

            age_days                REAL,
            posts_per_day           REAL,
            comments_per_day        REAL,
            comment_ratio           REAL,

            karma_score             REAL,
            avg_score               REAL,

            subreddit_count         INTEGER,
            active_days             INTEGER,

            hour_entropy            REAL,
            duplicate_ratio         REAL,

            avg_post_interval       REAL,
            avg_comment_interval    REAL,

            night_activity_ratio    REAL,
            burstiness_score        REAL,
            engagement_rate         REAL,

            computed_at             TEXT
        );


        ----------------------------------------------------
        -- NETWORK EDGES
        ----------------------------------------------------

        CREATE TABLE IF NOT EXISTS edges (
            source_account_id   TEXT,
            target_account_id   TEXT,
            edge_type           TEXT,
            weight              REAL DEFAULT 1,

            PRIMARY KEY (
                source_account_id,
                target_account_id,
                edge_type
            )
        ) WITHOUT ROWID;


        ----------------------------------------------------
        -- CONTENT SIMILARITY
        ----------------------------------------------------

        CREATE TABLE IF NOT EXISTS content_similarity (
            source_account_id   TEXT,
            target_account_id   TEXT,
            similarity          REAL,
            method              TEXT,

            PRIMARY KEY (
                source_account_id,
                target_account_id
            )
        ) WITHOUT ROWID;


        ----------------------------------------------------
        -- TEMPORAL COORDINATION
        ----------------------------------------------------

        CREATE TABLE IF NOT EXISTS temporal_similarity (
            source_account_id   TEXT,
            target_account_id   TEXT,
            similarity          REAL,
            avg_time_diff       REAL,

            PRIMARY KEY (
                source_account_id,
                target_account_id
            )
        ) WITHOUT ROWID;


        ----------------------------------------------------
        -- ACCOUNT PAIRS
        ----------------------------------------------------

        CREATE TABLE IF NOT EXISTS account_pairs (
            source_account_id   TEXT,
            target_account_id   TEXT,

            content_score               REAL,
            temporal_score              REAL,

            network_score               REAL,
            network_volume_score        REAL,
            network_reciprocity_score   REAL,
            network_concentration_score REAL,

            final_score                 REAL,

            coordination_type   TEXT,

            PRIMARY KEY (
                source_account_id,
                target_account_id
            )
        ) WITHOUT ROWID;


        ----------------------------------------------------
        -- COMMUNITY DETECTION
        ----------------------------------------------------

        CREATE TABLE IF NOT EXISTS communities (
            account_id              TEXT PRIMARY KEY,
            community_id            INTEGER,
            centrality              REAL,
            pagerank                REAL,
            coordination_strength   REAL
        );


        ----------------------------------------------------
        -- FINAL SCORES
        ----------------------------------------------------

        CREATE TABLE IF NOT EXISTS scores (
            account_id                  TEXT PRIMARY KEY,

            anomaly_score               REAL,
            coord_score                 REAL,
            temporal_score              REAL,
            dup_score                   REAL,
            network_score               REAL,
            network_score_topic_scoped  REAL,

            influence_score             REAL,

            tier                        TEXT,
            cluster_id                  INTEGER,

            evidence_status             TEXT,
            confidence_level            TEXT,

            assessment                  TEXT,

            scored_at                   TEXT
        );


        ----------------------------------------------------
        -- MODEL PREDICTIONS
        ----------------------------------------------------

        CREATE TABLE IF NOT EXISTS predictions (
            account_id          TEXT PRIMARY KEY,
            predicted_label     TEXT,
            confidence          REAL,
            reason              TEXT,
            predicted_at        TEXT
        );


        ----------------------------------------------------
        -- COORDINATION EVENTS
        ----------------------------------------------------

        CREATE TABLE IF NOT EXISTS coordination_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,

            source_account_id TEXT NOT NULL,
            target_account_id TEXT NOT NULL,

            source_post_id TEXT,
            target_post_id TEXT,

            event_type TEXT NOT NULL,
            similarity REAL,
            event_time REAL,

            source_content_type TEXT,
            target_content_type TEXT,

            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );


        ----------------------------------------------------
        -- DATASET METADATA
        ----------------------------------------------------

        CREATE TABLE IF NOT EXISTS dataset_metadata (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,

            collection_date     TEXT,
            subreddits          TEXT,

            total_posts         INTEGER,
            total_comments      INTEGER,
            total_accounts      INTEGER,

            notes               TEXT
        );


        ----------------------------------------------------
        -- MODEL METRICS
        ----------------------------------------------------

        CREATE TABLE IF NOT EXISTS model_metrics (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,

            model_name          TEXT,
            accuracy            REAL,
            precision           REAL,
            recall              REAL,
            f1                  REAL,
            roc_auc             REAL,

            created_at          TEXT DEFAULT CURRENT_TIMESTAMP
        );


        ----------------------------------------------------
        -- EXPERIMENT TRACKING
        ----------------------------------------------------

        CREATE TABLE IF NOT EXISTS experiments (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,

            model_name          TEXT,
            parameters          TEXT,

            accuracy            REAL,
            precision_score     REAL,
            recall_score        REAL,
            f1_score            REAL,

            created_at          TEXT DEFAULT CURRENT_TIMESTAMP
        );


        ----------------------------------------------------
        -- COLLECTION RUNS
        ----------------------------------------------------

        CREATE TABLE IF NOT EXISTS collection_runs (
            run_id              TEXT PRIMARY KEY,

            started_at          TEXT NOT NULL,
            finished_at         TEXT,

            collection_method   TEXT NOT NULL,
            mode                TEXT NOT NULL,

            subreddits          TEXT,
            search_terms        TEXT,

            fetched_posts       INTEGER DEFAULT 0,
            inserted_posts      INTEGER DEFAULT 0,
            duplicate_posts     INTEGER DEFAULT 0,
            rejected_posts      INTEGER DEFAULT 0,

            fetched_comments    INTEGER DEFAULT 0,
            inserted_comments   INTEGER DEFAULT 0,

            new_accounts        INTEGER DEFAULT 0,

            total_posts         INTEGER DEFAULT 0,
            total_accounts      INTEGER DEFAULT 0,

            status              TEXT DEFAULT 'running',
            error_message       TEXT
        );


        ----------------------------------------------------
        -- INDEXES
        ----------------------------------------------------

        CREATE INDEX IF NOT EXISTS idx_posts_account
            ON posts(account_id);

        CREATE INDEX IF NOT EXISTS idx_comments_account
            ON comments(account_id);

        CREATE INDEX IF NOT EXISTS idx_comments_post
            ON comments(post_id);

        CREATE INDEX IF NOT EXISTS idx_comments_parent
            ON comments(parent_id);

        CREATE INDEX IF NOT EXISTS idx_posts_created
            ON posts(created_utc);

        CREATE INDEX IF NOT EXISTS idx_comments_created
            ON comments(created_utc);

        CREATE INDEX IF NOT EXISTS idx_posts_hash
            ON posts(content_hash);

        CREATE INDEX IF NOT EXISTS idx_comments_hash
            ON comments(content_hash);

        CREATE INDEX IF NOT EXISTS idx_post_topic
            ON posts(topic);

        CREATE INDEX IF NOT EXISTS idx_comment_topic
            ON comments(topic);

        CREATE INDEX IF NOT EXISTS idx_post_relevance
            ON posts(is_relevant);

        CREATE INDEX IF NOT EXISTS idx_comment_relevance
            ON comments(is_relevant);

        CREATE INDEX IF NOT EXISTS idx_post_language
            ON posts(language);

        CREATE INDEX IF NOT EXISTS idx_comment_language
            ON comments(language);

        CREATE INDEX IF NOT EXISTS idx_post_sentiment
            ON posts(sentiment);

        CREATE INDEX IF NOT EXISTS idx_comment_sentiment
            ON comments(sentiment);

        CREATE INDEX IF NOT EXISTS idx_post_score
            ON posts(score);

        CREATE INDEX IF NOT EXISTS idx_comment_score
            ON comments(score);

        CREATE INDEX IF NOT EXISTS idx_account_activity
            ON account_activity(account_id);

        CREATE INDEX IF NOT EXISTS idx_account_activity_time
            ON account_activity(created_utc);

        CREATE INDEX IF NOT EXISTS idx_edges_source
            ON edges(source_account_id);

        CREATE INDEX IF NOT EXISTS idx_edges_target
            ON edges(target_account_id);

        CREATE INDEX IF NOT EXISTS idx_pair_score
            ON account_pairs(final_score);

        CREATE INDEX IF NOT EXISTS idx_pair_type
            ON account_pairs(coordination_type);

        CREATE INDEX IF NOT EXISTS idx_community_id
            ON communities(community_id);

        CREATE INDEX IF NOT EXISTS idx_influence_score
            ON scores(influence_score);

        CREATE INDEX IF NOT EXISTS idx_coord_score
            ON scores(coord_score);

        CREATE INDEX IF NOT EXISTS idx_prediction_label
            ON predictions(predicted_label);

        CREATE INDEX IF NOT EXISTS idx_event_type
            ON coordination_events(event_type);

        CREATE INDEX IF NOT EXISTS idx_event_time
            ON coordination_events(event_time);

        CREATE INDEX IF NOT EXISTS idx_collection_runs_status
            ON collection_runs(status);

        CREATE INDEX IF NOT EXISTS idx_collection_runs_started
            ON collection_runs(started_at);
        """
    )

    # ------------------------------------------------------------
    # RUN SCHEMA MIGRATIONS
    # ------------------------------------------------------------

    _migrate_schema(conn)

    # ------------------------------------------------------------
    # INDEXES FOR NEWER SCORES COLUMNS
    # ------------------------------------------------------------

    c.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_confidence_level
        ON scores(confidence_level)
        """
    )

    c.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_evidence_status
        ON scores(evidence_status)
        """
    )

    c.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_assessment
        ON scores(assessment)
        """
    )

    conn.commit()
    conn.close()


# ---------------------------------------------------------------------
# DIRECT EXECUTION
# ---------------------------------------------------------------------

if __name__ == "__main__":

    init_db()

    print(f"Database initialised at:\n{DB_PATH}")

    conn = get_conn()
    c = conn.cursor()

    c.execute("SELECT COUNT(*) FROM posts")
    total_posts = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM comments")
    total_comments = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM accounts")
    total_users = c.fetchone()[0]

    print(f"Database total posts: {total_posts}")
    print(f"Database total comments: {total_comments}")
    print(f"Database total users: {total_users}")

    conn.close()