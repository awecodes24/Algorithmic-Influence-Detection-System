# src/preprocessor.py
# Computes behavioral features per account from raw data
# Run this AFTER data_loader.py

import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime
from config import DB_PATH


def get_connection():
    return sqlite3.connect(DB_PATH)


def compute_account_age(created_at_str):
    """
    Compute account age in days from created_at string.
    Cresci-2017 uses Twitter date format: 'Mon Jan 01 00:00:00 +0000 2015'
    """
    if not created_at_str or created_at_str == 'nan':
        return 365
    try:
        dt = datetime.strptime(created_at_str, '%a %b %d %H:%M:%S +0000 %Y')
        age = (datetime.now() - dt).days
        return max(age, 1)
    except (ValueError, TypeError):
        try:
            dt = datetime.fromisoformat(created_at_str)
            age = (datetime.now() - dt).days
            return max(age, 1)
        except Exception:
            return 365


def compute_features():
    """
    Computes universal behavioral features regardless of platform.
    Writes results into the features table for use by all models.
    """
    conn = get_connection()

    print("📊 Loading accounts from database...")
    df = pd.read_sql("""
        SELECT 
            a.account_id,
            a.platform,
            a.created_at,
            a.follower_count,
            a.following_count,
            a.total_posts,
            a.is_verified,
            f.is_bot
        FROM accounts a
        LEFT JOIN features f ON a.account_id = f.account_id
    """, conn)

    print(f"   Found {len(df)} accounts")
    print(f"   Platforms: {df['platform'].unique().tolist()}")

    if len(df) == 0:
        print("   ❌ No accounts found. Run data_loader.py first.")
        conn.close()
        return

    # ── Feature Engineering ──────────────────────────────────────────────────
    print("⚙️  Computing features...")

    df['account_age_days'] = df['created_at'].apply(compute_account_age)

    df['posts_per_day'] = df.apply(
        lambda r: round(r['total_posts'] / max(r['account_age_days'], 1), 4),
        axis=1
    )

    df['follower_ratio'] = df.apply(
        lambda r: round(r['follower_count'] / max(r['following_count'], 1), 4),
        axis=1
    )

    df['followers_per_day'] = df.apply(
        lambda r: round(r['follower_count'] / max(r['account_age_days'], 1), 4),
        axis=1
    )

    df['is_empty_account'] = (
        (df['follower_count'] == 0) & (df['total_posts'] == 0)
    ).astype(int)

    df['log_followers'] = np.log1p(df['follower_count'])
    df['log_following'] = np.log1p(df['following_count'])
    df['log_posts']     = np.log1p(df['total_posts'])

    df = df.replace([np.inf, -np.inf], 0)
    df = df.fillna(0)

    print("   Features computed:")
    feature_cols = [
        'posts_per_day', 'follower_ratio', 'followers_per_day',
        'is_empty_account', 'log_followers', 'log_following',
        'log_posts', 'account_age_days'
    ]
    for col in feature_cols:
        print(f"   ✓ {col:<25} mean={df[col].mean():.4f}")

    # ── Write back to features table ─────────────────────────────────────────
    print("\n💾 Writing features to database...")
    cursor = conn.cursor()
    saved = 0

    for _, row in df.iterrows():
        cursor.execute('''
            INSERT OR REPLACE INTO features (
                account_id, platform, is_bot,
                posts_per_day, follower_ratio, followers_per_day,
                is_empty_account, log_followers, log_following,
                log_posts, account_age_days
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            row['account_id'],
            row['platform'],
            int(row['is_bot']) if row['is_bot'] != -1 else -1,
            row['posts_per_day'],
            row['follower_ratio'],
            row['followers_per_day'],
            row['is_empty_account'],
            row['log_followers'],
            row['log_following'],
            row['log_posts'],
            row['account_age_days']
        ))
        saved += 1

    conn.commit()
    conn.close()

    print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Preprocessing Complete
   Accounts processed:  {saved}
   Features computed:   {len(feature_cols)}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """)

    return df


def get_feature_matrix():
    """
    Returns a clean numpy matrix ready for model input.
    Call this from your model scripts.
    """
    conn = get_connection()

    df = pd.read_sql("""
        SELECT 
            account_id,
            posts_per_day,
            follower_ratio,
            followers_per_day,
            is_empty_account,
            log_followers,
            log_following,
            log_posts,
            account_age_days
        FROM features
    """, conn)
    conn.close()

    if len(df) == 0:
        print("❌ No features found. Run preprocessor.py first.")
        return None, None

    account_ids = df['account_id'].values
    feature_matrix = df.drop('account_id', axis=1).values

    print(f"✅ Feature matrix ready: {feature_matrix.shape}")
    print(f"   {feature_matrix.shape[0]} accounts × "
          f"{feature_matrix.shape[1]} features")

    return account_ids, feature_matrix


if __name__ == "__main__":
    print("🚀 Starting preprocessing...\n")
    compute_features()