# src/models/hdbscan_model.py
# HDBSCAN coordination clustering, run against the CRESCI-2017 BENCHMARK —
# this is the counterpart to models/isolationforest.py (benchmark AUC
# validation), not the real-data pipeline. For real Reddit data, use
# models/reddit_hdbscan.py instead, which clusters on the 15 features
# reddit_preprocessor.py computes.
#
# FIX (see progress log): this used to import DB_PATH (the Reddit DB) by
# mistake. log_posts / account_age_days / is_empty_account / is_bot and the
# `results` table only exist in benchmark.db, not influence.db, which is
# almost certainly why this never ran successfully before. Import
# BENCHMARK_DB_PATH instead, and nothing else in this file needs to change.
#
# Run this AFTER benchmark_preprocessor.py (needs the features table populated)

import sqlite3
from datetime import datetime

import hdbscan
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import BENCHMARK_DB_PATH, HDBSCAN_PARAMS


def get_connection():
    return sqlite3.connect(BENCHMARK_DB_PATH)


def load_features():
    """
    Pulls behavioral features for ALL accounts (unlike Isolation Forest's
    training step, clustering doesn't need labels — is_bot is only pulled
    here so we can validate cluster purity afterward, not for the clustering
    itself).
    """
    conn = get_connection()
    df = pd.read_sql(
        """SELECT
            account_id,
            posts_per_day,
            log_posts,
            account_age_days,
            is_empty_account,
            is_bot,
            source_dataset
            FROM features
        """,
        conn,
    )
    conn.close()
    
    print(f"Loaded {len(df)} accounts for clustering")
    return df


def run_hdbscan(df):
    """
    Clusters accounts using synchrony-oriented features — deliberately NOT
    reusing every feature Isolation Forest used, since the goal here is
    "do these accounts behave like a coordinated group", not "is this
    account individually weird".
 
    NOTE: posts_per_day, log_posts, and account_age_days are what's
    currently available in the features table. If/when preprocessor.py
    is extended to compute avg_posting_hour / posting_hour_std (columns
    already reserved in the features table schema), add those here —
    posting-time synchrony is a much stronger coordination signal than
    account age or post volume alone.
    """
    feature_cols = [
        'posts_per_day',
        'log_posts',
        'account_age_days',
        'is_empty_account'
    ]
    
    X = df[feature_cols].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("Running HDBSCAN...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=HDBSCAN_PARAMS['min_cluster_size'],
        min_samples=HDBSCAN_PARAMS['min_samples'],
    )
    labels = clusterer.fit_predict(X_scaled)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())

    print(f"\n{'━'*50}")
    print(f"  Clusters found: {n_clusters}")
    print(f"  Noise points:   {n_noise} / {len(labels)} "
          f"({100 * n_noise / len(labels):.1f}%)")
    print(f"{'━'*50}")
    
    df['cluster_id'] = labels
    
    persistence = clusterer.cluster_persistence_
    
    if len(persistence) > 0 and persistence.max() > persistence.min():
        norm_persistence = (persistence - persistence.min()) / (persistence.max() - persistence.min())
    
    else:
        norm_persistence = np.ones_like(persistence)


    def score_for_label(label):
        if label == -1:
            return 0.0
        return float(norm_persistence[label])
    
    df['coordination_score'] = df['cluster_id'].apply(score_for_label)

    print("\nCluster sizes (top 10 by size):")
    sizes = df[df['cluster_id'] != -1]['cluster_id'].value_counts().head(10)
    for cid, size in sizes.items():
        score = score_for_label(cid)
        print(f" cluster {cid:>4} -> {size:>4} accounts "
              f"(coordination_score={score:.3f})")
        
    return df

def evaluate_against_bot_labels(df):
    """
    Quick validation using is_bot as ground truth (Cresci-2017 only).
    This checks whether HDBSCAN clusters are bot-dominated or human-dominated
    on average -- coarser than per-campaign purity below, but still useful
    as a sanity check on its own.
    """
    labeled = df[df['is_bot'] != -1]
    clustered = labeled[labeled['cluster_id'] != -1]
 
    if len(clustered) == 0:
        print("\nNo non-noise clusters with labeled accounts to evaluate.")
        return
 
    print(f"\n{'━'*50}")
    print("  Cluster composition (bot vs human) — coarse check")
    print(f"{'━'*50}")
 
    for cid, group in clustered.groupby('cluster_id'):
        bot_frac = (group['is_bot'] == 1).mean()
        dominant = "bot" if bot_frac >= 0.5 else "human"
        purity = bot_frac if dominant == "bot" else (1 - bot_frac)
        flag = "✅" if purity >= 0.75 else "⚠️ "
        print(f"  cluster {cid:>4}  n={len(group):<5}  "
              f"dominant={dominant:<6}  purity={purity:.3f}  {flag}")


def evaluate_per_campaign_purity(df):
    """
    Per-campaign cluster purity, per the proposal's Ch. 5 target (>= 0.75):
    for each non-noise cluster, what fraction of its accounts came from
    the SAME source_dataset (e.g. all from social_spambots_1.csv, or all
    from genuine_accounts.csv)? This is stricter than
    evaluate_against_bot_labels() above, which only checks bot-vs-human --
    a cluster mixing social_spambots_1 and traditional_spambots_2 accounts
    would pass that check (both are "bot") but fail this one (two
    different campaigns), which is the more meaningful signal for
    "did HDBSCAN find a specific coordinated campaign".
    """
    labeled = df[df['source_dataset'] != 'unknown']
    clustered = labeled[labeled['cluster_id'] != -1]

    if len(clustered) == 0:
        print("\nNo non-noise clusters with source_dataset info to evaluate.")
        return

    print(f"\n{'━'*50}")
    print("  Cluster composition (per-campaign) — proposal Ch. 5 metric")
    print(f"{'━'*50}")

    purities = []
    for cid, group in clustered.groupby('cluster_id'):
        top_dataset, top_count = group['source_dataset'].value_counts().idxmax(), \
                                  group['source_dataset'].value_counts().max()
        purity = top_count / len(group)
        purities.append(purity)
        flag = "✅" if purity >= 0.75 else "⚠️ "
        print(f"  cluster {cid:>4}  n={len(group):<5}  "
              f"dominant_dataset={top_dataset:<28}  purity={purity:.3f}  {flag}")

    mean_purity = sum(purities) / len(purities)
    print(f"{'-'*50}")
    print(f"  Mean per-campaign purity across {len(purities)} clusters: {mean_purity:.3f}  ", end="")
    print("✅ Meets target (≥ 0.75)" if mean_purity >= 0.75 else "⚠️  Below target (≥ 0.75)")
    print(f"{'━'*50}")
        
def save_results(df):
    conn = get_connection()
    cursor = conn.cursor()
 
    saved = 0
    for _, row in df.iterrows():
        # Only update cluster_id + coordination_score — do NOT overwrite
        # anomaly_score, which Isolation Forest already wrote. UPDATE, not
        # INSERT OR REPLACE, or you'll wipe that column back to its default.
        cursor.execute(
            """
            UPDATE results
            SET cluster_id = ?, coordination_score = ?, processed_at = ?
            WHERE account_id = ?
            """,
            (
                int(row['cluster_id']),
                float(row['coordination_score']),
                datetime.now().isoformat(),
                row['account_id'],
            )
        )
        # If the account has no results row yet (Isolation Forest hasn't
        # run for it, or this is being run standalone), insert one instead.
        if cursor.rowcount == 0:
            cursor.execute(
                """
                INSERT OR IGNORE INTO results
                    (account_id, cluster_id, coordination_score, processed_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    row['account_id'],
                    int(row['cluster_id']),
                    float(row['coordination_score']),
                    datetime.now().isoformat(),
                )
            )
        saved += 1
 
    conn.commit()
    conn.close()
    print(f"\nSaved cluster_id + coordination_score for {saved} accounts")
   
    
if __name__ == "__main__":
    print("Running HDBSCAN Coordination Clustering\n")
    df = load_features()
    df = run_hdbscan(df)
    evaluate_against_bot_labels(df)
    evaluate_per_campaign_purity(df)
    save_results(df)
    print("\nDone.")
 