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
    "high_coordination_count",
    "temporal_events_per_tweet",
    "temporal_neighbors_per_tweet",
    "high_coordination_ratio",
    "temporal_coordination_score",
]

ALL_FEATURES = (
    BASE_FEATURES
    + TEMPORAL_FEATURES
)

EXPECTED_FEATURE_COUNT = 40


def table_exists(
    conn: sqlite3.Connection,
    table_name: str,
) -> bool:
    row = conn.execute(
        """
        SELECT 1
        FROM sqlite_master
        WHERE type = 'table'
          AND name = ?
        """,
        (table_name,),
    ).fetchone()

    return row is not None


def validate_database(
    conn: sqlite3.Connection,
) -> None:
    required_tables = [
        "account_features_final",
        "benchmark_splits",
    ]

    for table in required_tables:
        if not table_exists(conn, table):
            raise RuntimeError(
                f"Required table missing: {table}"
            )

    columns = {
        row[1]
        for row in conn.execute(
            """
            PRAGMA table_info(
                account_features_final
            )
            """
        )
    }

    missing_features = [
        feature
        for feature in ALL_FEATURES
        if feature not in columns
    ]

    if missing_features:
        raise RuntimeError(
            "account_features_final is missing "
            f"features: {missing_features}"
        )

    account_count = conn.execute(
        """
        SELECT COUNT(*)
        FROM accounts
        """
    ).fetchone()[0]

    final_count = conn.execute(
        """
        SELECT COUNT(*)
        FROM account_features_final
        """
    ).fetchone()[0]

    if account_count != final_count:
        raise RuntimeError(
            "Final feature count does not match "
            f"account count: "
            f"{account_count} vs {final_count}"
        )


def load_data(
    conn: sqlite3.Connection,
) -> pd.DataFrame:
    feature_columns = ", ".join(
        f'f."{feature}"'
        for feature in ALL_FEATURES
    )

    query = f"""
        SELECT
            f.account_id,
            {feature_columns},
            f.split,
            f.label
        FROM account_features_final f
        INNER JOIN benchmark_splits s
            ON f.account_id = s.account_id
        ORDER BY f.account_id
    """

    df = pd.read_sql_query(
        query,
        conn,
    )

    if df.empty:
        raise RuntimeError(
            "account_features_final returned no rows."
        )

    return df


def validate_split_data(
    df: pd.DataFrame,
) -> None:
    required_splits = {
        "train",
        "validation",
        "test",
    }

    actual_splits = set(
        df["split"].unique()
    )

    missing = (
        required_splits
        - actual_splits
    )

    if missing:
        raise RuntimeError(
            f"Missing splits: {sorted(missing)}"
        )

    if df["account_id"].duplicated().any():
        raise RuntimeError(
            "Duplicate account IDs found."
        )

    split_counts = (
        df.groupby("split")["account_id"]
        .nunique()
    )

    for split_name in required_splits:
        if split_counts.get(
            split_name,
            0,
        ) == 0:
            raise RuntimeError(
                f"Split is empty: {split_name}"
            )

    # Confirm account-level isolation.
    for left, right in [
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ]:

        a = set(
            df.loc[
                df["split"] == left,
                "account_id",
            ]
        )

        b = set(
            df.loc[
                df["split"] == right,
                "account_id",
            ]
        )

        overlap = a.intersection(b)

        if overlap:
            raise RuntimeError(
                f"Leakage detected between "
                f"{left} and {right}: "
                f"{len(overlap)} accounts."
            )


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
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    min_samples_split=4,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def select_threshold(
    y_true,
    probabilities,
):
    best_threshold = 0.50
    best_f1 = -1.0

    for threshold in np.arange(
        0.10,
        0.96,
        0.01,
    ):
        predictions = (
            probabilities >= threshold
        ).astype(int)

        score = f1_score(
            y_true,
            predictions,
            zero_division=0,
        )

        if score > best_f1:
            best_f1 = score
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


def save_json(
    path: Path,
    data: dict,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            data,
            handle,
            indent=2,
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train the final Cresci-2017 "
            "42-feature Random Forest."
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

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Allow overwriting an existing "
            "final model and metrics."
        ),
    )

    args = parser.parse_args()

    if not args.database.exists():
        raise FileNotFoundError(
            f"Database not found:\n"
            f"  {args.database}"
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        args.output_dir
        / "cresci_final_model.joblib"
    )

    metrics_path = (
        args.output_dir
        / "cresci_final_metrics.json"
    )

    prediction_path = (
        args.output_dir
        / "test_predictions.csv"
    )

    if (
        (
            model_path.exists()
            or metrics_path.exists()
            or prediction_path.exists()
        )
        and not args.overwrite
    ):
        raise FileExistsError(
            "\nFinal Cresci artifacts already exist:\n"
            f"  {model_path}\n"
            f"  {metrics_path}\n"
            f"  {prediction_path}\n\n"
            "Use --overwrite only when intentionally "
            "retraining the final model."
        )

    conn = sqlite3.connect(
        args.database
    )

    try:

        validate_database(
            conn
        )

        df = load_data(
            conn
        )

    finally:
        conn.close()

    # --------------------------------------------------
    # Validate dataset
    # --------------------------------------------------

    if (
        len(ALL_FEATURES)
        != EXPECTED_FEATURE_COUNT
    ):
        raise RuntimeError(
            f"Expected {EXPECTED_FEATURE_COUNT} "
            f"features but configured "
            f"{len(ALL_FEATURES)}."
        )

    if "account_id" in ALL_FEATURES:
        raise RuntimeError(
            "account_id must never be used "
            "as a model feature."
        )

    if "label" in ALL_FEATURES:
        raise RuntimeError(
            "label must never be used "
            "as a model feature."
        )

    if "split" in ALL_FEATURES:
        raise RuntimeError(
            "split must never be used "
            "as a model feature."
        )

    validate_split_data(
        df
    )

    train = df[
        df["split"] == "train"
    ].copy()

    validation = df[
        df["split"] == "validation"
    ].copy()

    test = df[
        df["split"] == "test"
    ].copy()

    X_train = train[
        ALL_FEATURES
    ]

    y_train = train[
        "label"
    ].astype(int)

    X_validation = validation[
        ALL_FEATURES
    ]

    y_validation = validation[
        "label"
    ].astype(int)

    X_test = test[
        ALL_FEATURES
    ]

    y_test = test[
        "label"
    ].astype(int)

    print(
        "\n"
        + "=" * 70
    )

    print(
        "CRESCI-2017 FINAL TRAINING"
    )

    print(
        "=" * 70
    )

    print(
        f"Features    : {len(ALL_FEATURES)}"
    )

    print(
        f"Train       : {len(train):,}"
    )

    print(
        f"Validation  : {len(validation):,}"
    )

    print(
        f"Test        : {len(test):,}"
    )

    print(
        "\nTraining class distribution:"
    )

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

    print(
        "\nValidation class distribution:"
    )

    print(
        y_validation.value_counts()
        .sort_index()
        .rename(
            index={
                0: "Human",
                1: "Bot",
            }
        )
    )

    print(
        "\nTest class distribution:"
    )

    print(
        y_test.value_counts()
        .sort_index()
        .rename(
            index={
                0: "Human",
                1: "Bot",
            }
        )
    )

    # --------------------------------------------------
    # Train
    # --------------------------------------------------

    model = build_model()

    print(
        "\nFitting Random Forest on TRAIN only..."
    )

    model.fit(
        X_train,
        y_train,
    )

    # --------------------------------------------------
    # Validation threshold
    # --------------------------------------------------

    validation_probability = (
        model.predict_proba(
            X_validation
        )[:, 1]
    )

    threshold = select_threshold(
        y_validation,
        validation_probability,
    )

    validation_metrics = calculate_metrics(
        y_validation,
        validation_probability,
        threshold,
    )

    print(
        f"\nValidation threshold: "
        f"{threshold:.2f}"
    )

    print(
        f"Validation F1: "
        f"{validation_metrics['f1']:.6f}"
    )

    # --------------------------------------------------
    # FINAL TEST
    # --------------------------------------------------

    print(
        "\nEvaluating untouched TEST set..."
    )

    test_probability = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    test_prediction = (
        test_probability >= threshold
    ).astype(int)

    test_metrics = calculate_metrics(
        y_test,
        test_probability,
        threshold,
    )

    test_metrics["test_accounts"] = int(
        len(test)
    )

    test_metrics["feature_count"] = int(
        len(ALL_FEATURES)
    )

    test_metrics["random_state"] = (
        RANDOM_STATE
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "FINAL CRESCI-2017 TEST RESULTS"
    )

    print(
        "=" * 70
    )

    for key, value in test_metrics.items():

        if isinstance(value, float):
            print(
                f"{key:<20}: {value:.6f}"
            )

        else:
            print(
                f"{key:<20}: {value}"
            )

    print(
        "=" * 70
    )

    # --------------------------------------------------
    # Account-level predictions
    # --------------------------------------------------

    predictions = test[
        [
            "account_id",
            "label",
            "split",
        ]
    ].copy()

    predictions[
        "bot_probability"
    ] = test_probability

    predictions[
        "predicted_label"
    ] = test_prediction

    predictions[
        "correct"
    ] = (
        predictions["label"]
        == predictions["predicted_label"]
    )

    predictions.to_csv(
        prediction_path,
        index=False,
    )

    # --------------------------------------------------
    # Model package
    # --------------------------------------------------

    package = {
        "pipeline": model,
        "features": ALL_FEATURES,
        "threshold": threshold,
        "random_state": RANDOM_STATE,
        "feature_count": len(
            ALL_FEATURES
        ),
        "model_type": (
            "RandomForestClassifier"
        ),
        "benchmark": "Cresci-2017",
    }

    joblib.dump(
        package,
        model_path,
    )

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------

    final_metrics = {
        "benchmark": "Cresci-2017",
        "model_type": (
            "RandomForestClassifier"
        ),
        "feature_count": len(
            ALL_FEATURES
        ),
        "train_accounts": len(
            train
        ),
        "validation_accounts": len(
            validation
        ),
        "test_accounts": len(
            test
        ),
        "validation": validation_metrics,
        "test": test_metrics,
    }

    save_json(
        metrics_path,
        final_metrics,
    )

    print(
        "\nSaved final artifacts:"
    )

    print(
        f"  Model       : {model_path}"
    )

    print(
        f"  Metrics     : {metrics_path}"
    )

    print(
        f"  Predictions : {prediction_path}"
    )

    print(
        "\nTEST SET WAS USED ONLY FOR FINAL EVALUATION."
    )


if __name__ == "__main__":
    main()