# src/database.py
# Run this ONCE to initialize your SQLite database

import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'influence.db')

def get_connection():
    return sqlite3.connect(DB_PATH)

def initialize_database():
    conn = get_connection()
    cursor = conn.cursor()

    # ACCOUNTS table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS features (
            account_id                TEXT PRIMARY KEY,
            posts_per_day             REAL DEFAULT 0.0,
            avg_posting_hour          REAL DEFAULT 0.0,
            posting_hour_std          REAL DEFAULT 0.0,
            hashtag_variety           REAL DEFAULT 0.0,
            avg_engagement_rate       REAL DEFAULT 0.0,
            content_length_avg        REAL DEFAULT 0.0,
            platform                  TEXT,
            is_bot                    INTEGER DEFAULT -1,
            follower_following_ratio  REAL DEFAULT 0.0,
            followers_per_day         REAL DEFAULT 0.0,
            is_empty_account          INTEGER DEFAULT 0,
            log_followers             REAL DEFAULT 0.0,
            log_following             REAL DEFAULT 0.0,
            log_posts                 REAL DEFAULT 0.0,
            account_age_days          INTEGER DEFAULT 0,
            favourites_count          INTEGER DEFAULT 0,
            listed_count              INTEGER DEFAULT 0,
            FOREIGN KEY (account_id) REFERENCES accounts(account_id)
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
            account_id          TEXT PRIMARY KEY,
            posts_per_day       REAL DEFAULT 0.0,
            avg_posting_hour    REAL DEFAULT 0.0,
            posting_hour_std    REAL DEFAULT 0.0,
            hashtag_variety     REAL DEFAULT 0.0,
            avg_engagement_rate REAL DEFAULT 0.0,
            content_length_avg  REAL DEFAULT 0.0,
            platform            TEXT,
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
    print("Database initialized successfully at:", os.path.abspath(DB_PATH))
    print("Tables created: accounts, posts, interactions, features, results")

if __name__ == "__main__":
    initialize_database()