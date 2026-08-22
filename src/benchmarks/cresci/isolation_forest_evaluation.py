from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline


RANDOM_STATE = 42


# Use the same behavioral/profile features that
# your Cresci baseline uses.
FEATURES = [
    "followers_count",
    "friends_count",
    "statuses_count",
    "favourites_count",
    "listed_count",
    "verified",
    "protected",
    "default_profile",
    "default_profile_image",
    "geo_enabled",
    "has_description",
    "has_location",
    "account_age_days",
    "statuses_per_day",
    "followers_per_day",
    "friends_per_day",
    "followers_friends_ratio",
    "friends_followers_ratio",
    "favorites_status_ratio",
    "listed_followers_ratio",
    "tweet_count",
    "avg_retweets",
    "avg_replies",
    "avg_favorites",
    "avg_hashtags",
    "avg_urls",
    "avg_mentions",
    "retweet_ratio",
    "reply_ratio",
    "avg_text_length",
    "avg_words",
    "url_ratio",
    "mention_ratio",
    "hashtag_ratio",
    "duplicate_tweet_ratio",
]


def load_data(
    conn: sqlite3.Connection,
) -> pd.DataFrame:

    columns = ", ".join(
        f'"{feature}"'
        for feature in FEATURES
    )

    query = f"""
        SELECT
            account_id,
            {columns},
            split,
            label
        FROM account_features_final
        ORDER BY account_id
    """

    df = pd.read_sql_query(
        query,
        conn,
    )

    if df.empty:
        raise RuntimeError(
            "No Cresci final feature records found."
        )

    return df


def build_model() -> Pipeline:
    return Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "isolation_forest",
                IsolationForest(
                    n_estimators=300,
                    contamination="auto",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Isolation Forest on the "
            "Cresci-2017 held-out test set."
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
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/cresci/final"
        ),
    )

    args = parser.parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not args.database.exists():
        raise FileNotFoundError(
            f"Database not found:\n"
            f"  {args.database}"
        )

    conn = sqlite3.connect(
        args.database
    )

    try:
        df = load_data(conn)
    finally:
        conn.close()

    train = df[
        df["split"] == "train"
    ].copy()

    test = df[
        df["split"] == "test"
    ].copy()

    if train.empty:
        raise RuntimeError(
            "Training split is empty."
        )

    if test.empty:
        raise RuntimeError(
            "Test split is empty."
        )

    X_train = train[FEATURES]
    X_test = test[FEATURES]

    y_test = (
        test["label"]
        .astype(int)
        .to_numpy()
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "CRESCI-2017 ISOLATION FOREST EVALUATION"
    )

    print(
        "=" * 70
    )

    print(
        f"Features    : {len(FEATURES)}"
    )

    print(
        f"Train       : {len(train):,}"
    )

    print(
        f"Test        : {len(test):,}"
    )

    print(
        "\nTraining Isolation Forest on TRAIN only..."
    )

    model = build_model()

    model.fit(
        X_train
    )

    # sklearn decision_function:
    # larger = more normal
    #
    # ROC-AUC needs larger = more bot/anomalous.
    anomaly_score = -model.decision_function(
        X_test
    )

    auc_roc = roc_auc_score(
        y_test,
        anomaly_score,
    )

    print(
        f"\nIsolation Forest AUC-ROC: "
        f"{auc_roc:.6f}"
    )

    # --------------------------------------------------
    # Optional threshold evaluation
    # --------------------------------------------------

    threshold = float(
        np.quantile(
            anomaly_score,
            0.90,
        )
    )

    prediction = (
        anomaly_score >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_test,
        prediction,
    )

    precision = precision_score(
        y_test,
        prediction,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        prediction,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        prediction,
        zero_division=0,
    )

    # --------------------------------------------------
    # ROC curve
    # --------------------------------------------------

    fpr, tpr, thresholds = roc_curve(
        y_test,
        anomaly_score,
    )

    roc_df = pd.DataFrame(
        {
            "false_positive_rate": fpr,
            "true_positive_rate": tpr,
            "threshold": thresholds,
        }
    )

    # --------------------------------------------------
    # Save metrics
    # --------------------------------------------------

    metrics = {
        "benchmark": "Cresci-2017",
        "model": "IsolationForest",
        "feature_count": len(FEATURES),
        "train_accounts": len(train),
        "test_accounts": len(test),
        "auc_roc": float(auc_roc),
        "threshold": threshold,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "random_state": RANDOM_STATE,
    }

    metrics_path = (
        args.output_dir
        / "isolation_forest_metrics.json"
    )

    with metrics_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metrics,
            f,
            indent=2,
        )

    # --------------------------------------------------
    # Save predictions
    # --------------------------------------------------

    conn = sqlite3.connect(args.database)

    try:
        source_groups = pd.read_sql_query(
            """
            SELECT
                account_id,
                source_group
            FROM accounts
            """,
            conn,
        )
    finally:
        conn.close()

    predictions = (
        test[
            [
                "account_id",
                "label",
            ]
        ]
        .merge(
            source_groups,
            on="account_id",
            how="left",
        )
    )

    predictions[
        "anomaly_score"
    ] = anomaly_score

    predictions[
        "predicted_label"
    ] = prediction

    predictions[
        "correct"
    ] = (
        predictions["label"]
        == predictions["predicted_label"]
    )

    predictions.to_csv(
        args.output_dir
        / "isolation_forest_predictions.csv",
        index=False,
    )

    # --------------------------------------------------
    # Save ROC data
    # --------------------------------------------------

    roc_df.to_csv(
        args.output_dir
        / "isolation_forest_roc_curve.csv",
        index=False,
    )

    # --------------------------------------------------
    # Save model
    # --------------------------------------------------

    joblib.dump(
        {
            "pipeline": model,
            "features": FEATURES,
            "auc_roc": float(auc_roc),
            "random_state": RANDOM_STATE,
        },
        args.output_dir
        / "cresci_isolation_forest.joblib",
    )

    # --------------------------------------------------
    # Plot ROC curve
    # --------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(8, 7)
    )

    ax.plot(
        fpr,
        tpr,
        label=f"Isolation Forest (AUC = {auc_roc:.4f})",
        linewidth=2,
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random classifier",
    )

    ax.set_xlabel(
        "False Positive Rate"
    )

    ax.set_ylabel(
        "True Positive Rate"
    )

    ax.set_title(
        "Cresci-2017 Isolation Forest ROC Curve"
    )

    ax.legend(
        loc="lower right"
    )

    ax.grid(
        alpha=0.25
    )

    fig.tight_layout()

    fig.savefig(
        args.output_dir
        / "isolation_forest_roc_curve.png",
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        "\nSaved:"
    )

    print(
        f"  {metrics_path}"
    )

    print(
        "  outputs/cresci/final/"
        "isolation_forest_predictions.csv"
    )

    print(
        "  outputs/cresci/final/"
        "isolation_forest_roc_curve.csv"
    )

    print(
        "  outputs/cresci/final/"
        "isolation_forest_roc_curve.png"
    )

    print(
        "  outputs/cresci/final/"
        "cresci_isolation_forest.joblib"
    )


if __name__ == "__main__":
    main()