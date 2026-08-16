# src/db.py

from pathlib import Path
import sqlite3

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "influence.db"


def get_conn():
    DATA_DIR.mkdir(exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def _migrate_schema(conn):
    """
    Idempotent migrations for columns added after a table's initial
    CREATE TABLE. CREATE TABLE IF NOT EXISTS silently does nothing on a
    DB that already has the table -- which is every existing local DB at
    this point -- so a schema change like coordination_events.
    target_post_id needs an explicit ALTER TABLE to actually land there.
    Safe to call every time init_db() runs: only adds a missing column,
    never touches an existing table's rows or any other table.
    """
    c = conn.cursor()
    c.execute("PRAGMA table_info(coordination_events)")
    existing_cols = {row[1] for row in c.fetchall()}
    if 'target_post_id' not in existing_cols:
        c.execute("ALTER TABLE coordination_events ADD COLUMN target_post_id TEXT")


def init_db():
    conn = get_conn()
    c = conn.cursor()

    c.executescript("""
                    
    -- RAW SCRAPED DATA

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

    -- `id` is the SHA-256-derived pseudonym and is what every other table
    -- joins on; it is always populated. `username` is left NULL by the
    -- collector unless STORE_RAW_USERNAME=true is set locally -- storing
    -- the raw username here next to the hash would make the "anonymized"
    -- claim in the proposal (Sec 3.4.1) untrue, since the mapping would be
    -- trivially reversible by anyone with DB access. Only turn that flag on
    -- for your own local manual verification, and never commit, share, or
    -- expose influence.db (or an export built from it) while it's on.
    CREATE TABLE IF NOT EXISTS accounts (
        id              TEXT PRIMARY KEY,
        username        TEXT,
        created_utc     REAL,
        comment_karma   INTEGER,
        link_karma      INTEGER,
        total_posts     INTEGER DEFAULT 0,
        total_comments  INTEGER DEFAULT 0
    );

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

        age_days               REAL,
        posts_per_day          REAL,
        comments_per_day       REAL,
        comment_ratio          REAL,

        karma_score            REAL,
        avg_score               REAL,

        subreddit_count        INTEGER,
        active_days            INTEGER,

        hour_entropy            REAL,
        duplicate_ratio         REAL,

        avg_post_interval       REAL,
        avg_comment_interval    REAL,

        night_activity_ratio    REAL,
        burstiness_score        REAL,
        engagement_rate         REAL,

        computed_at              TEXT
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
        method               TEXT,

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

        content_score       REAL,
        temporal_score      REAL,
        network_score       REAL,
        final_score         REAL,

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
        account_id          TEXT PRIMARY KEY,

        anomaly_score       REAL,
        coord_score         REAL,
        dup_score           REAL,
        network_score       REAL,
        influence_score     REAL,

        tier                TEXT,
        cluster_id          INTEGER,

        scored_at           TEXT
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
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        source_account_id   TEXT,
        target_account_id   TEXT,
        source_post_id      TEXT,
        target_post_id      TEXT,
        event_type          TEXT,
        similarity          REAL,
        event_time          REAL,
        created_at          TEXT DEFAULT CURRENT_TIMESTAMP
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
    """)

    _migrate_schema(conn)

    conn.commit()
    conn.close()


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