from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

RANDOM_STATE = 42


# These are the actual predictive features.
# IMPORTANT:
# - account_id is an identifier, not a feature
# - label is the target, never a feature
# - split is only used to select train/validation/test rows
FEATURE_COLUMNS = [
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


def load_dataset(
    conn: sqlite3.Connection,
) -> pd.DataFrame:

    query = """
        SELECT
            f.account_id,

            f.followers_count,
            f.friends_count,
            f.statuses_count,
            f.favourites_count,
            f.listed_count,

            f.verified,
            f.protected,
            f.default_profile,
            f.default_profile_image,
            f.geo_enabled,

            f.has_description,
            f.has_location,

            f.account_age_days,

            f.statuses_per_day,
            f.followers_per_day,
            f.friends_per_day,

            f.followers_friends_ratio,
            f.friends_followers_ratio,
            f.favorites_status_ratio,
            f.listed_followers_ratio,

            f.tweet_count,
            f.avg_retweets,
            f.avg_replies,
            f.avg_favorites,

            f.avg_hashtags,
            f.avg_urls,
            f.avg_mentions,

            f.retweet_ratio,
            f.reply_ratio,

            f.avg_text_length,
            f.avg_words,

            f.url_ratio,
            f.mention_ratio,
            f.hashtag_ratio,

            f.duplicate_tweet_ratio,

            s.split,
            s.label

        FROM account_features f

        INNER JOIN benchmark_splits s
            ON f.account_id = s.account_id
    """

    df = pd.read_sql_query(query, conn)

    if df.empty:
        raise RuntimeError(
            "No rows found. "
            "Make sure account_features and benchmark_splits exist."
        )

    return df


def prepare_xy(
    df: pd.DataFrame,
    split_name: str,
):
    subset = df[
        df["split"] == split_name
    ].copy()

    if subset.empty:
        raise RuntimeError(
            f"No rows available for split: {split_name}"
        )

    X = subset[FEATURE_COLUMNS].copy()
    y = subset["label"].astype(int)

    return X, y, subset


def build_pipeline() -> Pipeline:

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "model",
                model,
            ),
        ]
    )

    return pipeline


def calculate_metrics(
    y_true,
    probabilities,
    threshold=0.5,
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    metrics = {
        "threshold": float(threshold),
        "accuracy": float(
            accuracy_score(
                y_true,
                predictions,
            )
        ),
        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
    }

    try:
        metrics["roc_auc"] = float(
            roc_auc_score(
                y_true,
                probabilities,
            )
        )
    except ValueError:
        metrics["roc_auc"] = None

    return metrics


def choose_validation_threshold(
    y_true,
    probabilities,
):
    """
    Select the threshold that gives the best validation F1.

    TEST is never used here.
    """

    thresholds = np.arange(
        0.10,
        0.96,
        0.01,
    )

    best_threshold = 0.50
    best_f1 = -1.0

    results = []

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        f1 = f1_score(
            y_true,
            predictions,
            zero_division=0,
        )

        precision = precision_score(
            y_true,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            y_true,
            predictions,
            zero_division=0,
        )

        results.append(
            {
                "threshold": float(threshold),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }
        )

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)

    return best_threshold, results


def save_json(
    data,
    path: Path,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            data,
            f,
            indent=2,
        )


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Train Cresci-2017 bot detector "
            "using TRAIN data only."
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
        "--model-output",
        type=Path,
        default=Path(
            "outputs/cresci_random_forest.joblib"
        ),
    )

    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=Path(
            "outputs/cresci_validation_metrics.json"
        ),
    )

    parser.add_argument(
        "--threshold-output",
        type=Path,
        default=Path(
            "outputs/cresci_validation_thresholds.csv"
        ),
    )

    args = parser.parse_args()

    if not args.database.exists():
        raise FileNotFoundError(
            f"Database not found: {args.database}"
        )

    args.model_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    conn = sqlite3.connect(
        args.database
    )

    try:

        print("Loading Cresci-2017 data...")

        df = load_dataset(conn)

        print(
            f"Total usable accounts: {len(df):,}"
        )

        X_train, y_train, train_df = prepare_xy(
            df,
            "train",
        )

        X_val, y_val, val_df = prepare_xy(
            df,
            "validation",
        )

        print(
            f"Train accounts:      {len(X_train):,}"
        )

        print(
            f"Validation accounts: {len(X_val):,}"
        )

        print("\nTrain class distribution:")

        print(
            y_train.value_counts()
            .sort_index()
            .rename(
                index={
                    0: "Human",
                    1: "Bot",
                }
            )
        )

        print("\nValidation class distribution:")

        print(
            y_val.value_counts()
            .sort_index()
            .rename(
                index={
                    0: "Human",
                    1: "Bot",
                }
            )
        )

        print(
            "\nBuilding Random Forest pipeline..."
        )

        pipeline = build_pipeline()

        print(
            "Fitting model on TRAIN only..."
        )

        pipeline.fit(
            X_train,
            y_train,
        )

        print(
            "Generating validation probabilities..."
        )

        val_probabilities = pipeline.predict_proba(
            X_val
        )[:, 1]

        print(
            "Selecting threshold using VALIDATION..."
        )

        best_threshold, threshold_results = (
            choose_validation_threshold(
                y_val,
                val_probabilities,
            )
        )

        validation_metrics = calculate_metrics(
            y_val,
            val_probabilities,
            best_threshold,
        )

        validation_metrics[
            "train_accounts"
        ] = int(len(X_train))

        validation_metrics[
            "validation_accounts"
        ] = int(len(X_val))

        validation_metrics[
            "feature_count"
        ] = int(len(FEATURE_COLUMNS))

        print("\nValidation results:")
        print(
            json.dumps(
                validation_metrics,
                indent=2,
            )
        )

        print(
            f"\nSelected threshold: "
            f"{best_threshold:.2f}"
        )

        print(
            "\nSaving trained model..."
        )

        joblib.dump(
            {
                "pipeline": pipeline,
                "features": FEATURE_COLUMNS,
                "threshold": best_threshold,
                "random_state": RANDOM_STATE,
            },
            args.model_output,
        )

        print(
            f"Model saved to: "
            f"{args.model_output}"
        )

        save_json(
            validation_metrics,
            args.metrics_output,
        )

        pd.DataFrame(
            threshold_results
        ).to_csv(
            args.threshold_output,
            index=False,
        )

        print(
            f"Validation metrics saved to: "
            f"{args.metrics_output}"
        )

        print(
            f"Threshold table saved to: "
            f"{args.threshold_output}"
        )

        print(
            "\nTEST SET WAS NOT USED."
        )

    finally:
        conn.close()


if __name__ == "__main__":
    main()