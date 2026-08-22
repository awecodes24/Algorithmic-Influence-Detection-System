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


BASE_FEATURES = [
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

TEMPORAL_FEATURES = [
    "temporal_event_count",
    "temporal_neighbor_count",
    "mean_temporal_similarity",
    "max_temporal_similarity",
    "high_coordination_count",
    "temporal_events_per_tweet",
    "temporal_neighbors_per_tweet",
    "high_coordination_ratio",
    "temporal_coordination_score",
]


def load_dataset(
    conn: sqlite3.Connection,
    table: str,
    features: list[str],
) -> pd.DataFrame:

    columns = ", ".join(
        f'f."{x}"' for x in features
    )

    query = f"""
        SELECT
            f.account_id,
            {columns},
            s.split,
            s.label

        FROM "{table}" f

        INNER JOIN benchmark_splits s
            ON f.account_id = s.account_id
    """

    return pd.read_sql_query(
        query,
        conn,
    )


def build_model() -> Pipeline:

    classifier = RandomForestClassifier(
        n_estimators=300,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "model",
                classifier,
            ),
        ]
    )


def split_data(
    df: pd.DataFrame,
    features: list[str],
):

    train = df[df["split"] == "train"]
    validation = df[df["split"] == "validation"]
    test = df[df["split"] == "test"]

    X_train = train[features]
    y_train = train["label"].astype(int)

    X_val = validation[features]
    y_val = validation["label"].astype(int)

    X_test = test[features]
    y_test = test["label"].astype(int)

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    )


def select_threshold(
    y_true,
    probabilities,
):

    best_threshold = 0.50
    best_f1 = -1

    for threshold in np.arange(
        0.10,
        0.96,
        0.01,
    ):

        predictions = (
            probabilities >= threshold
        ).astype(int)

        f1 = f1_score(
            y_true,
            predictions,
            zero_division=0,
        )

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(
                threshold
            )

    return best_threshold


def calculate_metrics(
    y_true,
    probabilities,
    threshold,
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    return {
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
        "roc_auc": float(
            roc_auc_score(
                y_true,
                probabilities,
            )
        ),
    }


def train_and_evaluate(
    name: str,
    table: str,
    features: list[str],
    conn: sqlite3.Connection,
    output_dir: Path,
):

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    df = load_dataset(
        conn,
        table,
        features,
    )

    print(
        f"Accounts: {len(df):,}"
    )

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    ) = split_data(
        df,
        features,
    )

    print(
        f"Train      : {len(X_train):,}"
    )

    print(
        f"Validation : {len(X_val):,}"
    )

    print(
        f"Test       : {len(X_test):,}"
    )

    model = build_model()

    print(
        "\nTraining on TRAIN only..."
    )

    model.fit(
        X_train,
        y_train,
    )

    val_probabilities = (
        model.predict_proba(X_val)[:, 1]
    )

    threshold = select_threshold(
        y_val,
        val_probabilities,
    )

    val_metrics = calculate_metrics(
        y_val,
        val_probabilities,
        threshold,
    )

    print(
        f"\nValidation threshold: "
        f"{threshold:.2f}"
    )

    print(
        "Validation F1:",
        f"{val_metrics['f1']:.4f}",
    )

    # Final test prediction.
    test_probabilities = (
        model.predict_proba(X_test)[:, 1]
    )

    test_metrics = calculate_metrics(
        y_test,
        test_probabilities,
        threshold,
    )

    test_metrics[
        "test_accounts"
    ] = int(len(X_test))

    test_metrics[
        "feature_count"
    ] = int(len(features))

    print("\nFINAL TEST RESULTS")

    for key, value in test_metrics.items():

        if isinstance(value, float):
            print(
                f"{key:25}: {value:.4f}"
            )
        else:
            print(
                f"{key:25}: {value}"
            )

    prefix = (
        "baseline"
        if table == "account_features"
        else "temporal"
    )

    joblib.dump(
        {
            "pipeline": model,
            "features": features,
            "threshold": threshold,
            "model_name": name,
        },
        output_dir
        / f"cresci_{prefix}_model.joblib",
    )

    with open(
        output_dir
        / f"cresci_{prefix}_metrics.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            {
                "model": name,
                "validation": val_metrics,
                "test": test_metrics,
            },
            f,
            indent=2,
        )

    return test_metrics


def main():

    parser = argparse.ArgumentParser()

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
        default=Path("outputs"),
    )

    args = parser.parse_args()

    if not args.database.exists():
        raise FileNotFoundError(
            args.database
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    conn = sqlite3.connect(
        args.database
    )

    try:

        baseline = train_and_evaluate(
            name="BASELINE: Profile + Behavior",
            table="account_features",
            features=BASE_FEATURES,
            conn=conn,
            output_dir=args.output_dir,
        )

        temporal = train_and_evaluate(
            name=(
                "TEMPORAL: Profile + Behavior "
                "+ Coordination"
            ),
            table="account_features_temporal",
            features=(
                BASE_FEATURES
                + TEMPORAL_FEATURES
            ),
            conn=conn,
            output_dir=args.output_dir,
        )

        comparison = pd.DataFrame(
            [
                {
                    "model": "Baseline",
                    **baseline,
                },
                {
                    "model": "Temporal",
                    **temporal,
                },
            ]
        )

        comparison.to_csv(
            args.output_dir
            / "cresci_model_comparison.csv",
            index=False,
        )

        print("\n" + "=" * 70)
        print("MODEL COMPARISON")
        print("=" * 70)

        print(
            comparison[
                [
                    "model",
                    "accuracy",
                    "precision",
                    "recall",
                    "f1",
                    "roc_auc",
                ]
            ].to_string(
                index=False
            )
        )

        print(
            "\nSaved:",
            args.output_dir
            / "cresci_model_comparison.csv",
        )

    finally:
        conn.close()


if __name__ == "__main__":
    main()