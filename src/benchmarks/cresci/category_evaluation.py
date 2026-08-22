from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


EXPECTED_FEATURE_COUNT = 40

EXPECTED_FEATURES = [
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
    "high_coordination_count",
    "temporal_events_per_tweet",
    "temporal_neighbors_per_tweet",
    "high_coordination_ratio",
    "temporal_coordination_score",
]


def calculate_binary_metrics(
    y_true,
    y_pred,
):
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    total = tn + fp + fn + tp

    accuracy = (
        (tn + tp) / total
        if total > 0
        else 0.0
    )

    false_positive_rate = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0.0
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    false_negative_rate = (
        fn / (fn + tp)
        if (fn + tp) > 0
        else 0.0
    )

    return {
        "accuracy": float(accuracy),
        "precision": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "specificity": float(
            specificity
        ),
        "false_positive_rate": float(
            false_positive_rate
        ),
        "false_negative_rate": float(
            false_negative_rate
        ),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def load_model(
    model_path: Path,
):
    package = joblib.load(
        model_path
    )

    required_keys = {
        "pipeline",
        "features",
        "threshold",
    }

    missing_keys = (
        required_keys
        - set(package.keys())
    )

    if missing_keys:
        raise RuntimeError(
            "Final model package is missing: "
            f"{sorted(missing_keys)}"
        )

    features = package["features"]

    if features != EXPECTED_FEATURES:
        raise RuntimeError(
            "Final model feature list does not match "
            "the approved Cresci feature set.\n\n"
            f"Expected:\n{EXPECTED_FEATURES}\n\n"
            f"Found:\n{features}"
        )

    return (
        package["pipeline"],
        features,
        float(package["threshold"]),
    )


def load_test_data(
    conn: sqlite3.Connection,
    features: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    source_groups = [
        row[0]
        for row in conn.execute(
            """
            SELECT DISTINCT source_group
            FROM accounts
            ORDER BY source_group
            """
        )
    ]

    if not source_groups:
        raise RuntimeError(
            "No source groups found."
        )

    feature_sql = ", ".join(
        f'f."{feature}"'
        for feature in features
    )

    df = pd.read_sql_query(
        f"""
        SELECT
            f.account_id,
            {feature_sql},
            a.source_group,
            f.label,
            f.split

        FROM account_features_final f

        INNER JOIN accounts a
            ON f.account_id = a.account_id

        WHERE f.split = 'test'

        ORDER BY f.account_id
        """,
        conn,
    )

    return df, source_groups


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the final Cresci-2017 "
            "model on every TEST source group."
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
        "--model",
        type=Path,
        default=Path(
            "outputs/cresci/final/"
            "cresci_final_model.joblib"
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
            "Allow replacing existing category "
            "evaluation files."
        ),
    )

    args = parser.parse_args()

    if not args.database.exists():
        raise FileNotFoundError(
            f"Database not found:\n"
            f"  {args.database}"
        )

    if not args.model.exists():
        raise FileNotFoundError(
            f"Model not found:\n"
            f"  {args.model}"
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_files = [
        args.output_dir
        / "category_results.csv",

        args.output_dir
        / "category_predictions.csv",

        args.output_dir
        / "category_misclassifications.csv",

        args.output_dir
        / "category_summary.json",
    ]

    if (
        any(path.exists() for path in result_files)
        and not args.overwrite
    ):
        raise FileExistsError(
            "\nCategory evaluation artifacts already exist.\n"
            "Use --overwrite to replace them."
        )

    # --------------------------------------------------
    # Load model
    # --------------------------------------------------

    model, features, threshold = load_model(
        args.model
    )

    print(
        f"Loaded threshold: {threshold:.4f}"
    )

    print(
        f"Model features: {len(features)}"
    )

    # --------------------------------------------------
    # Load TEST data
    # --------------------------------------------------

    conn = sqlite3.connect(
        args.database
    )

    try:

        df, source_groups = load_test_data(
            conn,
            features,
        )

    finally:
        conn.close()

    if df.empty:
        raise RuntimeError(
            "No TEST accounts found."
        )

    if set(df["split"].unique()) != {"test"}:
        raise RuntimeError(
            "Category evaluator received "
            "non-test rows."
        )

    print(
        "\nDetected source groups:"
    )

    for group in source_groups:
        print(
            f"  {group}"
        )

    print(
        f"\nTest accounts: {len(df):,}"
    )

    # --------------------------------------------------
    # Generate predictions
    # --------------------------------------------------

    probability = model.predict_proba(
        df[features]
    )[:, 1]

    prediction = (
        probability >= threshold
    ).astype(int)

    df["bot_probability"] = probability

    df["predicted_label"] = prediction

    df["correct"] = (
        df["label"]
        == df["predicted_label"]
    )

    # --------------------------------------------------
    # Per-category metrics
    # --------------------------------------------------

    results = []

    for group in source_groups:

        subset = df[
            df["source_group"] == group
        ].copy()

        if subset.empty:
            continue

        y_true = subset[
            "label"
        ].astype(int)

        y_pred = subset[
            "predicted_label"
        ].astype(int)

        metrics = calculate_binary_metrics(
            y_true,
            y_pred,
        )

        unique_labels = sorted(
            subset["label"]
            .unique()
            .tolist()
        )

        # Human-only category.
        if unique_labels == [0]:

            result = {
                "source_group": group,
                "class": "human",
                "accounts": len(subset),

                "accuracy": metrics[
                    "accuracy"
                ],

                "precision": None,
                "recall": None,
                "f1": None,

                "specificity": metrics[
                    "specificity"
                ],

                "false_positive_rate": metrics[
                    "false_positive_rate"
                ],

                "false_negative_rate": None,

                "true_negative": metrics[
                    "true_negative"
                ],

                "false_positive": metrics[
                    "false_positive"
                ],

                "false_negative": None,

                "true_positive": None,
            }

        # Bot-only category.
        elif unique_labels == [1]:

            result = {
                "source_group": group,
                "class": "bot",
                "accounts": len(subset),

                "accuracy": metrics[
                    "accuracy"
                ],

                "precision": metrics[
                    "precision"
                ],

                "recall": metrics[
                    "recall"
                ],

                "f1": metrics[
                    "f1"
                ],

                "specificity": None,
                "false_positive_rate": None,

                "false_negative_rate": metrics[
                    "false_negative_rate"
                ],

                "true_negative": None,
                "false_positive": None,

                "false_negative": metrics[
                    "false_negative"
                ],

                "true_positive": metrics[
                    "true_positive"
                ],
            }

        # Mixed category.
        else:

            result = {
                "source_group": group,
                "class": "mixed",
                "accounts": len(subset),

                "accuracy": metrics[
                    "accuracy"
                ],

                "precision": metrics[
                    "precision"
                ],

                "recall": metrics[
                    "recall"
                ],

                "f1": metrics[
                    "f1"
                ],

                "specificity": metrics[
                    "specificity"
                ],

                "false_positive_rate": metrics[
                    "false_positive_rate"
                ],

                "false_negative_rate": metrics[
                    "false_negative_rate"
                ],

                "true_negative": metrics[
                    "true_negative"
                ],

                "false_positive": metrics[
                    "false_positive"
                ],

                "false_negative": metrics[
                    "false_negative"
                ],

                "true_positive": metrics[
                    "true_positive"
                ],
            }

        results.append(
            result
        )

    results_df = pd.DataFrame(
        results
    )

    # --------------------------------------------------
    # Account-level predictions
    # --------------------------------------------------

    prediction_columns = [
        "account_id",
        "source_group",
        "label",
        "predicted_label",
        "bot_probability",
        "correct",
    ]

    predictions_df = df[
        prediction_columns
    ].copy()

    # --------------------------------------------------
    # Misclassifications
    # --------------------------------------------------

    errors_df = predictions_df[
        ~predictions_df["correct"]
    ].copy()

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    summary = {
        "benchmark": "Cresci-2017",
        "test_accounts": int(
            len(predictions_df)
        ),
        "source_groups": int(
            len(source_groups)
        ),
        "misclassified_accounts": int(
            len(errors_df)
        ),
        "threshold": threshold,
        "feature_count": len(features),
    }

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    results_df.to_csv(
        args.output_dir
        / "category_results.csv",
        index=False,
    )

    predictions_df.to_csv(
        args.output_dir
        / "category_predictions.csv",
        index=False,
    )

    errors_df.to_csv(
        args.output_dir
        / "category_misclassifications.csv",
        index=False,
    )

    with open(
        args.output_dir
        / "category_summary.json",
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            summary,
            handle,
            indent=2,
        )

    # --------------------------------------------------
    # Print
    # --------------------------------------------------

    print(
        "\n"
        + "=" * 110
    )

    print(
        "CRESCI-2017 PER-CATEGORY TEST RESULTS"
    )

    print(
        "=" * 110
    )

    print(
        results_df.to_string(
            index=False
        )
    )

    print(
        "\n"
        + "=" * 110
    )

    print(
        "SUMMARY"
    )

    print(
        "=" * 110
    )

    print(
        f"Source groups evaluated : "
        f"{len(source_groups)}"
    )

    print(
        f"Test accounts           : "
        f"{len(predictions_df):,}"
    )

    print(
        f"Misclassified accounts  : "
        f"{len(errors_df):,}"
    )

    print(
        "\nSaved:"
    )

    for path in result_files:
        print(
            f"  {path}"
        )


if __name__ == "__main__":
    main()