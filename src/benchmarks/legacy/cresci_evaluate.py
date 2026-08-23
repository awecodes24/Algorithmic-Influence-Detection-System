from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def load_test_data(
    conn: sqlite3.Connection,
    features: list[str],
) -> pd.DataFrame:

    feature_sql = ",\n            ".join(
        f"f.{feature}"
        for feature in features
    )

    query = f"""
        SELECT
            f.account_id,
            {feature_sql},
            s.label,
            s.split

        FROM account_features f

        INNER JOIN benchmark_splits s
            ON f.account_id = s.account_id

        WHERE s.split = 'test'
    """

    return pd.read_sql_query(query, conn)


def evaluate(
    y_true,
    probabilities,
    threshold: float,
):
    predictions = (
        probabilities >= threshold
    ).astype(int)

    metrics = {
        "threshold": float(threshold),
        "test_accounts": int(len(y_true)),
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

    return metrics, predictions


def main():

    parser = argparse.ArgumentParser(
        description="Final held-out Cresci-2017 evaluation."
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
            "outputs/cresci_random_forest.joblib"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/cresci_evaluation"),
    )

    args = parser.parse_args()

    if not args.database.exists():
        raise FileNotFoundError(
            f"Database not found: {args.database}"
        )

    if not args.model.exists():
        raise FileNotFoundError(
            f"Model not found: {args.model}"
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------
    # Load the EXACT model saved during training
    # --------------------------------------------------

    package = joblib.load(
        args.model
    )

    model = package["pipeline"]
    features = package["features"]
    threshold = float(
        package["threshold"]
    )

    print(
        f"Using validation-selected threshold: "
        f"{threshold:.2f}"
    )

    # --------------------------------------------------
    # Load TEST only
    # --------------------------------------------------

    conn = sqlite3.connect(
        args.database
    )

    try:

        test_df = load_test_data(
            conn,
            features,
        )

    finally:
        conn.close()

    if test_df.empty:
        raise RuntimeError(
            "TEST set is empty."
        )

    X_test = test_df[features]
    y_test = test_df["label"].astype(int)

    print(
        f"TEST accounts: {len(test_df):,}"
    )

    print("\nTEST class distribution:")

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
    # FINAL TEST PREDICTION
    # --------------------------------------------------

    print(
        "\nGenerating final TEST predictions..."
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    metrics, predictions = evaluate(
        y_test,
        probabilities,
        threshold,
    )

    # --------------------------------------------------
    # Print results
    # --------------------------------------------------

    print(
        "\n" + "=" * 60
    )
    print(
        "FINAL CRESCI-2017 TEST RESULTS"
    )
    print(
        "=" * 60
    )

    for key, value in metrics.items():

        if isinstance(value, float):
            print(
                f"{key:12}: {value:.4f}"
            )
        else:
            print(
                f"{key:12}: {value}"
            )

    print(
        "=" * 60
    )

    # --------------------------------------------------
    # Classification report
    # --------------------------------------------------

    report = classification_report(
        y_test,
        predictions,
        target_names=[
            "Human",
            "Bot",
        ],
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(
        report
    ).transpose()

    report_path = (
        args.output_dir
        / "classification_report.csv"
    )

    report_df.to_csv(
        report_path
    )

    # --------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------

    cm = confusion_matrix(
        y_test,
        predictions,
    )

    cm_df = pd.DataFrame(
        cm,
        index=[
            "Actual Human",
            "Actual Bot",
        ],
        columns=[
            "Predicted Human",
            "Predicted Bot",
        ],
    )

    cm_df.to_csv(
        args.output_dir
        / "confusion_matrix.csv"
    )

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "Human",
            "Bot",
        ],
    )

    display.plot(
        ax=ax,
        values_format="d",
    )

    ax.set_title(
        "Cresci-2017 Held-out Test Confusion Matrix"
    )

    fig.tight_layout()

    fig.savefig(
        args.output_dir
        / "confusion_matrix.png",
        dpi=200,
    )

    plt.close(fig)

    # --------------------------------------------------
    # Save account-level predictions
    # --------------------------------------------------

    prediction_df = test_df[
        [
            "account_id",
            "label",
        ]
    ].copy()

    prediction_df[
        "bot_probability"
    ] = probabilities

    prediction_df[
        "predicted_label"
    ] = predictions

    prediction_df[
        "correct"
    ] = (
        prediction_df["label"]
        == prediction_df["predicted_label"]
    )

    prediction_df.to_csv(
        args.output_dir
        / "test_predictions.csv",
        index=False,
    )

    # --------------------------------------------------
    # Save metrics
    # --------------------------------------------------

    with open(
        args.output_dir
        / "test_metrics.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metrics,
            f,
            indent=2,
        )

    print(
        "\nSaved:"
    )

    print(
        args.output_dir
        / "test_metrics.json"
    )

    print(
        args.output_dir
        / "classification_report.csv"
    )

    print(
        args.output_dir
        / "confusion_matrix.csv"
    )

    print(
        args.output_dir
        / "confusion_matrix.png"
    )

    print(
        args.output_dir
        / "test_predictions.csv"
    )

    print(
        "\nTEST SET WAS USED ONLY FOR FINAL EVALUATION."
    )


if __name__ == "__main__":
    main()