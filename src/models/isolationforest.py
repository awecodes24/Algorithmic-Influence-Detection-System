# src/models/isolationforest.py

import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

from src.config import BENCHMARK_DB_PATH, ISOLATION_FOREST


def get_connection():
    return sqlite3.connect(BENCHMARK_DB_PATH)


def load_features_with_labels():
    conn = get_connection()
    df = pd.read_sql(
        """
        SELECT
            account_id,   
            posts_per_day,
            follower_ratio,
            followers_per_day,
            is_empty_account,
            log_followers,
            log_following,
            log_posts,
            account_age_days,
            favourites_ratio,
            listed_ratio,
            log_favourites,
            is_bot
        FROM features
        WHERE is_bot != -1
        """,
        conn,
    )
    conn.close()

    print(f"Loaded {len(df)} labeled accounts")
    print(f" Bots:   {(df['is_bot'] == 1).sum()}")
    print(f" Humans: {(df['is_bot'] == 0).sum()}")

    return df


def run_isolation_forest(df):
    feature_col = [
        'posts_per_day', 'follower_ratio', 'followers_per_day',
        'is_empty_account', 'log_followers', 'log_following',
        'log_posts', 'account_age_days','favourites_ratio','listed_ratio','log_favourites'
    ]

    X = df[feature_col].values
    y = df['is_bot'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("Training isolation forest...")
    model = IsolationForest(
        n_estimators=ISOLATION_FOREST['n_estimators'],
        contamination=ISOLATION_FOREST['contamination'],
        random_state=ISOLATION_FOREST['random_state']
    )
    model.fit(X_scaled)

    raw_scores  = model.decision_function(X_scaled)
    predictions = model.predict(X_scaled)

    anomaly_scores = 1 - (raw_scores - raw_scores.min()) / \
                         (raw_scores.max() - raw_scores.min())

    # ── Check both directions since class imbalance can flip meaning ────────
    auc_normal  = roc_auc_score(y, anomaly_scores)
    auc_flipped = roc_auc_score(y, 1 - anomaly_scores)

    print(f"\nAUC (anomaly=bot):    {auc_normal:.4f}")
    print(f"AUC (anomaly=human):  {auc_flipped:.4f}")

    if auc_flipped > auc_normal:
        print("⚠️  Flipping direction — bots are majority class here, "
              "so 'anomaly' originally meant 'human'")
        anomaly_scores = 1 - anomaly_scores
        auc = auc_flipped
    else:
        auc = auc_normal

    predicted_labels = (anomaly_scores >= 0.5).astype(int)

    print(f"\n{'━'*50}")
    print(f"  Final AUC-ROC Score: {auc:.4f}  ", end="")
    print("✅ Meets target (≥ 0.80)" if auc >= 0.80 else "⚠️  Below target (≥ 0.80)")
    print(f"{'━'*50}")

    print("\nClassification Report:")
    print(classification_report(y, predicted_labels, target_names=['Human', 'Bot']))

    print("Confusion Matrix:")
    cm = confusion_matrix(y, predicted_labels)
    print(f"                Predicted")
    print(f"                Human  Bot")
    print(f"  Actual Human  {cm[0][0]:<6} {cm[0][1]}")
    print(f"  Actual Bot    {cm[1][0]:<6} {cm[1][1]}")

    df['anomaly_score']    = anomaly_scores
    df['predicted_is_bot'] = predicted_labels

    return df, auc


def save_results(df):
    conn = get_connection()
    cursor = conn.cursor()

    saved = 0
    for _, row in df.iterrows():
        cursor.execute(
            "INSERT OR REPLACE INTO results (account_id, anomaly_score, processed_at) VALUES (?, ?, ?)",
            (
                row['account_id'],
                row['anomaly_score'],
                datetime.now().isoformat()
            )
        )
        saved += 1

    conn.commit()
    conn.close()
    print(f"\nSaved anomaly scores for {saved} accounts to results table")


if __name__ == "__main__":
    print("Running Isolation Forest\n")
    df = load_features_with_labels()
    df, auc = run_isolation_forest(df)
    save_results(df)
    print(f"\nDone. Final AUC-ROC: {auc:.4f}")