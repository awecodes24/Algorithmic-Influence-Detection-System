# src/database.py
# Run this ONCE to initialize your SQLite database

import sqlite3
import os
from config import BENCHMARK_DB_PATH

def get_connection():
    return sqlite3.connect(BENCHMARK_DB_PATH)

def initialize_database():
    conn = get_connection()
    cursor = conn.cursor()

    # ACCOUNTS table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS accounts (
            account_id       TEXT PRIMARY KEY,
            platform         TEXT,
            username         TEXT,
            created_at       TEXT,
            follower_count   INTEGER DEFAULT 0,
            following_count  INTEGER DEFAULT 0,
            total_posts      INTEGER DEFAULT 0,
            account_age_days INTEGER DEFAULT 0,
            is_verified      INTEGER DEFAULT 0,
            language         TEXT DEFAULT 'unknown',
            collected_at     TEXT
        )
    ''')

    # POSTS table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS posts (
            post_id         TEXT PRIMARY KEY,
            account_id      TEXT,
            platform        TEXT,
            content         TEXT,
            topic_label     TEXT,
            hashtags        TEXT,
            posted_at       TEXT,
            likes           INTEGER DEFAULT 0,
            shares          INTEGER DEFAULT 0,
            comments_count  INTEGER DEFAULT 0,
            engagement_rate REAL DEFAULT 0.0,
            collected_at    TEXT,
            FOREIGN KEY (account_id) REFERENCES accounts(account_id)
        )
    ''')

    # INTERACTIONS table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS interactions (
            interaction_id   TEXT PRIMARY KEY,
            source_account   TEXT,
            target_account   TEXT,
            post_id          TEXT,
            interaction_type TEXT,
            platform         TEXT,
            occurred_at      TEXT
        )
    ''')

    # FEATURES table (computed during preprocessing)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS features (
            account_id           TEXT PRIMARY KEY,
            platform             TEXT,
            is_bot               INTEGER DEFAULT -1,

            -- Universal behavioral features
            posts_per_day        REAL DEFAULT 0.0,
            follower_ratio       REAL DEFAULT 0.0,
            followers_per_day    REAL DEFAULT 0.0,
            is_empty_account     INTEGER DEFAULT 0,
            log_followers        REAL DEFAULT 0.0,
            log_following        REAL DEFAULT 0.0,
            log_posts            REAL DEFAULT 0.0,
            account_age_days     INTEGER DEFAULT 0,

            -- Extra Cresci-specific fields
            favourites_count     INTEGER DEFAULT 0,
            listed_count         INTEGER DEFAULT 0,
            favourites_ratio     REAL DEFAULT 0.0,
            listed_ratio         REAL DEFAULT 0.0,
            log_favourites       REAL DEFAULT 0.0,

            -- Computed after tweet/post analysis
            avg_posting_hour     REAL DEFAULT 0.0,
            posting_hour_std     REAL DEFAULT 0.0,
            avg_engagement       REAL DEFAULT 0.0,
            content_length_avg   REAL DEFAULT 0.0,
            hashtag_rate         REAL DEFAULT 0.0,
            

            -- From model outputs
            pagerank_score       REAL DEFAULT 0.0,

            FOREIGN KEY (account_id) REFERENCES accounts(account_id)
        )
    ''')

    # RESULTS table (model outputs)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS results (
            account_id          TEXT PRIMARY KEY,
            anomaly_score       REAL DEFAULT 0.0,
            coordination_score  REAL DEFAULT 0.0,
            duplication_score   REAL DEFAULT 0.0,
            network_score       REAL DEFAULT 0.0,
            influence_score     REAL DEFAULT 0.0,
            tier                TEXT DEFAULT 'organic',
            cluster_id          INTEGER DEFAULT -1,
            processed_at        TEXT,
            FOREIGN KEY (account_id) REFERENCES accounts(account_id)
        )
    ''')

    conn.commit()
    conn.close()
    print("Database initialized successfully at:", os.path.abspath(BENCHMARK_DB_PATH))
    print("Tables created: accounts, posts, interactions, features, results")

if __name__ == "__main__":
    initialize_database()