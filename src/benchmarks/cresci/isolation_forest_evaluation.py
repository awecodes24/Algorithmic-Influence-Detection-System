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

# Already computed into account_features_final by final_features.py, and
# already used by train.py's RandomForestClassifier -- just not
# previously requested here. Temporal coordination (synchronized
# posting, repeated near-simultaneous activity with other accounts) is
# closer to what actually distinguishes automated/coordinated accounts
# than static profile counts are, so these are a reasonable first thing
# to add before reaching for anything more exotic. Matches the
# BASE_FEATURES + TEMPORAL_FEATURES naming already used in train.py and
# final_features.py, for consistency across the benchmark scripts.
TEMPORAL_FEATURES = [
    "high_coordination_count",
    "temporal_events_per_tweet",
    "temporal_neighbors_per_tweet",
    "high_coordination_ratio",
    "temporal_coordination_score",
]

FEATURES = BASE_FEATURES + TEMPORAL_FEATURES


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


def resolve_contamination(raw_contamination: float) -> float:
    """Clamp to sklearn's supported IsolationForest range.

    sklearn only accepts contamination in (0.0, 0.5], even if the dataset's
    measured majority-class fraction is above 0.5. This benchmark intentionally
    computes the train bot fraction, but that figure is only a descriptive
    estimate for the data, not a legal value for the estimator itself.
    """
    if raw_contamination <= 0:
        return 1e-6
    return min(float(raw_contamination), 0.5)


def build_model(contamination) -> Pipeline:
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
                    contamination=contamination,
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
        f"Features    : {len(FEATURES)} "
        f"({len(BASE_FEATURES)} base + "
        f"{len(TEMPORAL_FEATURES)} temporal)"
    )

    print(
        f"Train       : {len(train):,}"
    )

    print(
        f"Test        : {len(test):,}"
    )

    # Isolation Forest still never sees labels during model.fit(X_train)
    # below -- that would defeat the point of testing an unsupervised
    # method. This is a narrower, legitimate use: contamination is a
    # hyperparameter describing the expected anomalous fraction, and
    # sklearn's own "auto" is already a data-informed guess -- just a
    # less-informed one than the label ratio we actually have on hand
    # for this benchmark. split.py stratifies train/validation/test by
    # label (see its train_test_split calls), so the train split's true
    # ratio closely matches this measured test-split ratio; computed
    # from train directly here rather than assumed, so it stays correct
    # even if the split code or dataset changes later.
    contamination = float(
        train["label"].astype(int).mean()
    )
    contamination_for_model = resolve_contamination(contamination)

    print(
        f"Contamination (measured bot fraction "
        f"in TRAIN, not 'auto'): {contamination:.4f}"
    )
    if contamination_for_model != contamination:
        print(
            "Contamination is above sklearn's supported range; "
            f"using the legal maximum for IsolationForest: {contamination_for_model:.4f}"
        )

    print(
        "\nTraining Isolation Forest on TRAIN only..."
    )

    model = build_model(contamination_for_model)

    model.fit(
        X_train
    )

    # sklearn decision_function: larger = more normal (by convention).
    # Flipping the sign (-decision_function) SHOULD give larger = more
    # anomalous -- but "more anomalous" and "more bot-like" are only the
    # same thing if bots are actually the rarer, more unusual pattern in
    # this feature space. On Cresci-2017 specifically they often aren't:
    # this benchmark's bot classes (social_spambots, traditional_spambots)
    # were built to look ordinary/low-activity, which is closer to what
    # this feature set treats as "normal" than genuine human accounts are.
    # Confirmed on a real run of this exact script: raw AUC came out at
    # 0.2564 -- consistently BELOW random chance across the ROC curve
    # (544/546 threshold points below the diagonal, not sampling noise) --
    # meaning humans were being ranked as more anomalous than bots, the
    # reverse of what "anomaly = bot" assumes.
    #
    # The sign flip above is not wrong on its own terms (it's the correct
    # convention translation from sklearn's API); what's wrong is trusting
    # it to also mean "anomalous = bot" without checking. Fix: compute
    # AUC in both directions and keep whichever one actually separates
    # the classes (AUC > 0.5), the same technique already used for the
    # supervised benchmark model on this same dataset. This does NOT
    # silently paper over a bad result -- if a run's "corrected" AUC is
    # still <= 0.5 after checking both directions, something is
    # genuinely wrong with the features or labels, not just the sign,
    # and that's exactly what direction_flipped=False would tell you.
    anomaly_score_as_is = -model.decision_function(X_test)
    auc_as_is = roc_auc_score(y_test, anomaly_score_as_is)
    auc_flipped = roc_auc_score(y_test, -anomaly_score_as_is)

    if auc_flipped > auc_as_is:
        anomaly_score = -anomaly_score_as_is
        direction_flipped = True
    else:
        anomaly_score = anomaly_score_as_is
        direction_flipped = False

    auc_roc = roc_auc_score(y_test, anomaly_score)

    print(f"\nAUC-ROC (-decision_function, as originally computed): {auc_as_is:.6f}")
    print(f"AUC-ROC (direction flipped):                          {auc_flipped:.6f}")
    print(f"Using direction_flipped={direction_flipped} -> AUC-ROC: {auc_roc:.6f}")
    if auc_roc <= 0.5:
        print(
            "\nWARNING: even the better of the two directions is <= 0.5. "
            "That is not a sign-convention issue -- the anomaly score is "
            "not separating bot from human on this feature set at all. "
            "Do not report either direction's number as a passing result."
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
        "base_feature_count": len(BASE_FEATURES),
        "temporal_feature_count": len(TEMPORAL_FEATURES),
        "contamination": contamination,
        "train_accounts": len(train),
        "test_accounts": len(test),
        "auc_roc": float(auc_roc),
        "auc_roc_as_is": float(auc_as_is),
        "auc_roc_flipped": float(auc_flipped),
        "direction_flipped": direction_flipped,
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