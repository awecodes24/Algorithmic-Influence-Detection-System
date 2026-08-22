from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

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
    "mean_temporal_similarity",
    "max_temporal_similarity",
    "high_coordination_count",
    "temporal_events_per_tweet",
    "temporal_neighbors_per_tweet",
    "high_coordination_ratio",
    "temporal_coordination_score",
]


# ------------------------------------------------------------
# Ablation definitions
#
# All experiments use the same:
#   - database
#   - account split
#   - Random Forest configuration
#   - validation threshold selection
#
# Only the feature subset changes.
# ------------------------------------------------------------

ABLATIONS = {
    "baseline": [],

    "full_temporal": [
        "mean_temporal_similarity",
        "max_temporal_similarity",
        "high_coordination_count",
        "temporal_events_per_tweet",
        "temporal_neighbors_per_tweet",
        "high_coordination_ratio",
        "temporal_coordination_score",
    ],

    "without_similarity": [
        "high_coordination_count",
        "temporal_events_per_tweet",
        "temporal_neighbors_per_tweet",
        "high_coordination_ratio",
        "temporal_coordination_score",
    ],

    "without_high_coordination": [
        "mean_temporal_similarity",
        "max_temporal_similarity",
        "temporal_events_per_tweet",
        "temporal_neighbors_per_tweet",
    ],

    "only_temporal_score": [
        "temporal_coordination_score",
    ],
}


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


def load_data(
    conn: sqlite3.Connection,
) -> pd.DataFrame:

    if not table_exists(
        conn,
        "account_features_final",
    ):
        raise RuntimeError(
            "account_features_final does not exist."
        )

    all_features = (
        BASE_FEATURES
        + TEMPORAL_FEATURES
    )

    columns = ", ".join(
        f'f."{feature}"'
        for feature in all_features
    )

    query = f"""
        SELECT
            f.account_id,
            {columns},
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
            "account_features_final is empty."
        )

    return df


def validate_data(
    df: pd.DataFrame,
) -> None:

    expected_splits = {
        "train",
        "validation",
        "test",
    }

    actual_splits = set(
        df["split"].unique()
    )

    missing = (
        expected_splits
        - actual_splits
    )

    if missing:
        raise RuntimeError(
            f"Missing splits: {sorted(missing)}"
        )

    if df["account_id"].duplicated().any():
        raise RuntimeError(
            "Duplicate account IDs detected."
        )

    for left, right in [
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ]:

        left_ids = set(
            df.loc[
                df["split"] == left,
                "account_id",
            ]
        )

        right_ids = set(
            df.loc[
                df["split"] == right,
                "account_id",
            ]
        )

        overlap = left_ids.intersection(
            right_ids
        )

        if overlap:
            raise RuntimeError(
                f"Split leakage between "
                f"{left} and {right}: "
                f"{len(overlap)} accounts."
            )


def build_model() -> Pipeline:

    return Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
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
) -> float:

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


def run_experiment(
    name: str,
    temporal_features: list[str],
    df: pd.DataFrame,
) -> dict:

    features = (
        BASE_FEATURES
        + temporal_features
    )

    train = df[
        df["split"] == "train"
    ]

    validation = df[
        df["split"] == "validation"
    ]

    test = df[
        df["split"] == "test"
    ]

    X_train = train[
        features
    ]

    y_train = train[
        "label"
    ].astype(int)

    X_validation = validation[
        features
    ]

    y_validation = validation[
        "label"
    ].astype(int)

    X_test = test[
        features
    ]

    y_test = test[
        "label"
    ].astype(int)

    model = build_model()

    model.fit(
        X_train,
        y_train,
    )

    # Threshold is selected ONLY on validation.
    validation_probability = (
        model.predict_proba(
            X_validation
        )[:, 1]
    )

    threshold = select_threshold(
        y_validation,
        validation_probability,
    )

    # Test is evaluated only after threshold selection.
    test_probability = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    predictions = (
        test_probability >= threshold
    ).astype(int)

    return {
        "experiment": name,
        "feature_count": len(features),
        "temporal_feature_count": len(
            temporal_features
        ),
        "threshold": float(threshold),

        "accuracy": float(
            accuracy_score(
                y_test,
                predictions,
            )
        ),

        "precision": float(
            precision_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),

        "recall": float(
            recall_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),

        "f1": float(
            f1_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),

        "roc_auc": float(
            roc_auc_score(
                y_test,
                test_probability,
            )
        ),
    }


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Run Cresci-2017 feature ablation "
            "experiments."
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
        "--output",
        type=Path,
        default=Path(
            "outputs/cresci/ablation/"
            "cresci_ablation.csv"
        ),
    )

    args = parser.parse_args()

    if not args.database.exists():
        raise FileNotFoundError(
            f"Database not found:\n"
            f"  {args.database}"
        )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    conn = sqlite3.connect(
        args.database
    )

    try:

        df = load_data(
            conn
        )

    finally:

        conn.close()

    validate_data(
        df
    )

    print(
        "\n"
        + "=" * 75
    )

    print(
        "CRESCI-2017 ABLATION STUDY"
    )

    print(
        "=" * 75
    )

    results = []

    for name, temporal_features in (
        ABLATIONS.items()
    ):

        print(
            f"\nRunning: {name}"
        )

        result = run_experiment(
            name,
            temporal_features,
            df,
        )

        results.append(
            result
        )

        print(
            f"  Features : "
            f"{result['feature_count']}"
        )

        print(
            f"  Threshold: "
            f"{result['threshold']:.2f}"
        )

        print(
            f"  F1       : "
            f"{result['f1']:.6f}"
        )

        print(
            f"  ROC-AUC  : "
            f"{result['roc_auc']:.6f}"
        )

    results_df = pd.DataFrame(
        results
    )

    # --------------------------------------------------
    # Add changes relative to baseline
    # --------------------------------------------------

    baseline = results_df[
        results_df["experiment"]
        == "baseline"
    ].iloc[0]

    results_df[
        "f1_change_vs_baseline"
    ] = (
        results_df["f1"]
        - baseline["f1"]
    )

    results_df[
        "roc_auc_change_vs_baseline"
    ] = (
        results_df["roc_auc"]
        - baseline["roc_auc"]
    )

    results_df[
        "accuracy_change_vs_baseline"
    ] = (
        results_df["accuracy"]
        - baseline["accuracy"]
    )

    results_df[
        "recall_change_vs_baseline"
    ] = (
        results_df["recall"]
        - baseline["recall"]
    )

    results_df.to_csv(
        args.output,
        index=False,
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "FINAL ABLATION RESULTS"
    )

    print(
        "=" * 100
    )

    print(
        results_df[
            [
                "experiment",
                "feature_count",
                "threshold",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "f1_change_vs_baseline",
                "roc_auc_change_vs_baseline",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nSaved:"
    )

    print(
        args.output
    )


if __name__ == "__main__":
    main()