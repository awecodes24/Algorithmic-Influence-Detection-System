# src/benchmark/validate.py
# Validates models directly on Cresci-2017
# No database, no loader, just pandas → model → score

import pandas as pd
import numpy as np
import os
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix
)

# ── Paths ────────────────────────────────────────────────────────────────────
CRESCI_PATH = os.path.join('data', 'benchmark', 'cresci-2017')

DATASETS = {
    'genuine_accounts.csv':       0,
    'fake_followers.csv':         1,
    'social_spambots_1.csv':      1,
    'social_spambots_2.csv':      1,
    'social_spambots_3.csv':      1,
    'traditional_spambots_1.csv': 1,
    'traditional_spambots_2.csv': 1,
    'traditional_spambots_3.csv': 1,
    'traditional_spambots_4.csv': 1,
}


def safe_int(val, default=0):
    try:
        return int(float(val))
    except:
        return default


def find_users_csv(folder_name):
    """Find users.csv handling Cresci's nested folder structure."""
    # Try nested first: folder/folder/users.csv
    nested = os.path.join(CRESCI_PATH, folder_name, folder_name, 'users.csv')
    if os.path.exists(nested):
        return nested
    # Try direct: folder/users.csv
    direct = os.path.join(CRESCI_PATH, folder_name, 'users.csv')
    if os.path.exists(direct):
        return direct
    return None


def load_all():
    """
    Load all Cresci-2017 users.csv files into one dataframe.
    Computes features on the spot — no database needed.
    """
    frames = []

    for folder_name, is_bot in DATASETS.items():
        path = find_users_csv(folder_name)

        if path is None:
            print(f"⚠️  Not found: {folder_name}")
            continue

        df = pd.read_csv(path, low_memory=False)

        # ── Keep only columns we need ────────────────────────────────────────
        df = df[['id',
                 'statuses_count',
                 'followers_count',
                 'friends_count',
                 'favourites_count',
                 'listed_count',
                 'created_at']].copy()

        df['is_bot'] = is_bot
        df['source'] = folder_name
        frames.append(df)

        label = '🤖 BOT  ' if is_bot else '👤 HUMAN'
        print(f"{label}  {folder_name:<35} {len(df):>6} accounts")

    combined = pd.concat(frames, ignore_index=True)
    print(f"\n✅ Total loaded: {len(combined)} accounts\n")
    return combined


def compute_features(df):
    """
    Compute universal behavioral features from raw columns.
    These same features will be computed from real scraped data later.
    """

    def account_age(created_at_str):
        from datetime import datetime
        try:
            dt = datetime.strptime(
                str(created_at_str),
                '%a %b %d %H:%M:%S +0000 %Y'
            )
            return max((datetime.now() - dt).days, 1)
        except:
            return 365

    df['account_age_days']  = df['created_at'].apply(account_age)
    df['posts_per_day']     = df['statuses_count'] / df['account_age_days']
    df['follower_ratio']    = df['followers_count'] / \
                              df['friends_count'].clip(lower=1)
    df['followers_per_day'] = df['followers_count'] / df['account_age_days']
    df['log_followers']     = np.log1p(df['followers_count'])
    df['log_following']     = np.log1p(df['friends_count'])
    df['log_posts']         = np.log1p(df['statuses_count'])
    df['is_empty']          = (
                                (df['followers_count'] == 0) &
                                (df['statuses_count'] == 0)
                              ).astype(int)

    df = df.replace([np.inf, -np.inf], 0).fillna(0)

    print("✅ Features computed:")
    features = ['posts_per_day', 'follower_ratio', 'followers_per_day',
                'log_followers', 'log_following', 'log_posts',
                'is_empty', 'account_age_days']
    for f in features:
        print(f"   {f:<25} mean={df[f].mean():.4f}")

    return df, features


def run_isolation_forest(df, feature_cols):
    """
    Run Isolation Forest and print AUC-ROC.
    This is your first real model result.
    """
    print("\n🌲 Running Isolation Forest...")

    X = df[feature_cols].values
    y = df['is_bot'].values

    # ── Scale features ───────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Train model ──────────────────────────────────────────────────────────
    model = IsolationForest(
        n_estimators=100,
        contamination=0.5,    # ~50% bots in dataset
        random_state=42
    )
    model.fit(X_scaled)

    # ── Get anomaly scores ───────────────────────────────────────────────────
    # Isolation Forest returns -1 (anomaly) or 1 (normal)
    # decision_function returns raw scores — higher = more normal
    raw_scores    = model.decision_function(X_scaled)
    predictions   = model.predict(X_scaled)

    # ── Convert to bot probability (0 to 1) ─────────────────────────────────
    # Flip and normalize: anomaly = bot = 1
    anomaly_scores = 1 - (raw_scores - raw_scores.min()) / \
                         (raw_scores.max() - raw_scores.min())

    # ── Convert predictions: -1 → 1 (bot), 1 → 0 (human) ───────────────────
    predicted_labels = np.where(predictions == -1, 1, 0)

    # ── Metrics ──────────────────────────────────────────────────────────────
    auc = roc_auc_score(y, anomaly_scores)

    print(f"\n{'━'*45}")
    print(f"  AUC-ROC Score:  {auc:.4f}  ", end="")
    if auc >= 0.80:
        print("✅ Meets target (≥ 0.80)")
    else:
        print("⚠️  Below target (≥ 0.80)")

    print(f"\n  Classification Report:")
    print(classification_report(
        y, predicted_labels,
        target_names=['Human', 'Bot']
    ))

    print(f"  Confusion Matrix:")
    cm = confusion_matrix(y, predicted_labels)
    print(f"                Predicted")
    print(f"                Human  Bot")
    print(f"  Actual Human  {cm[0][0]:<6} {cm[0][1]}")
    print(f"  Actual Bot    {cm[1][0]:<6} {cm[1][1]}")
    print(f"{'━'*45}\n")

    # ── Add scores back to dataframe ─────────────────────────────────────────
    df['anomaly_score']    = anomaly_scores
    df['predicted_is_bot'] = predicted_labels

    return df, model, auc


if __name__ == "__main__":
    print("🚀 Cresci-2017 Benchmark Validation\n")
    print("━" * 45)

    # Step 1 — Load
    df = load_all()

    # Step 2 — Features
    df, feature_cols = compute_features(df)

    # Step 3 — Model
    df, model, auc = run_isolation_forest(df, feature_cols)

    # Step 4 — Save scores for inspection
    output_path = os.path.join('outputs', 'reports', 'benchmark_scores.csv')
    df[['id', 'source', 'is_bot',
        'anomaly_score', 'predicted_is_bot']].to_csv(
        output_path, index=False
    )
    print(f"📄 Scores saved to: {output_path}")