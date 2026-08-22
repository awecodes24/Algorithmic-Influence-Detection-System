"""
Content Coordination Threshold Sweep

Tests the distribution of cross-account TF-IDF cosine similarity
at multiple thresholds without changing the production detector.

This diagnostic is used to determine whether the current
MIN_CONTENT_SIMILARITY threshold is supported by the dataset.

Thresholds tested:

0.70
0.75
0.80
0.85
0.90

The script does not modify the database.
The script does not modify production detector configuration.
"""

import logging
from collections import defaultdict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.models.reddit_content_coordination import (
    load_content,
    MAX_FEATURES,
)


# ---------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

THRESHOLDS = [
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
]


# ---------------------------------------------------------------------
# THRESHOLD SWEEP
# ---------------------------------------------------------------------

def run_threshold_sweep(
    df,
):
    """
    Calculate cross-account content similarity distribution
    across multiple thresholds.

    The comparison remains subreddit-local, matching the
    production content coordination detector.
    """

    threshold_occurrences = defaultdict(
        int
    )

    threshold_pairs = {
        threshold: set()
        for threshold in THRESHOLDS
    }

    total_comparisons = 0

    # -------------------------------------------------------------
    # PROCESS EACH SUBREDDIT
    # -------------------------------------------------------------

    grouped = df.groupby(
        "subreddit"
    )

    for subreddit, group in grouped:

        group = group.reset_index(
            drop=True
        )

        if len(group) < 2:

            continue

        accounts = group[
            "account_id"
        ].astype(
            str
        ).tolist()

        texts = group[
            "text"
        ].astype(
            str
        ).tolist()

        # ---------------------------------------------------------
        # TF-IDF
        # ---------------------------------------------------------

        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=MAX_FEATURES,
        )

        matrix = vectorizer.fit_transform(
            texts
        )

        similarity_matrix = cosine_similarity(
            matrix
        )

        # ---------------------------------------------------------
        # COMPARE CROSS-ACCOUNT PAIRS
        # ---------------------------------------------------------

        for i in range(
            len(group)
        ):

            source_account = accounts[
                i
            ]

            for j in range(
                i + 1,
                len(group),
            ):

                target_account = accounts[
                    j
                ]

                # Ignore same-account comparisons.

                if (
                    source_account
                    == target_account
                ):

                    continue

                total_comparisons += 1

                similarity = float(
                    similarity_matrix[
                        i,
                        j,
                    ]
                )

                pair = tuple(
                    sorted(
                        [
                            source_account,
                            target_account,
                        ]
                    )
                )

                for threshold in THRESHOLDS:

                    if (
                        similarity
                        >= threshold
                    ):

                        threshold_occurrences[
                            threshold
                        ] += 1

                        threshold_pairs[
                            threshold
                        ].add(
                            pair
                        )

    return {
        "occurrences":
            threshold_occurrences,

        "pairs":
            threshold_pairs,

        "comparisons":
            total_comparisons,
    }


# ---------------------------------------------------------------------
# PRINT RESULTS
# ---------------------------------------------------------------------

def print_results(
    results,
):
    """
    Print threshold sweep results.
    """

    print(
        "\n"
        + "=" * 90
    )

    print(
        "CONTENT SIMILARITY THRESHOLD SWEEP"
    )

    print(
        "=" * 90
    )

    print(
        f"\nTotal cross-account comparisons: "
        f"{results['comparisons']}"
    )

    print(
        "\n"
        + "-" * 90
    )

    print(
        f"{'THRESHOLD':<15}"
        f"{'OCCURRENCES':>20}"
        f"{'ACCOUNT PAIRS':>20}"
    )

    print(
        "-" * 90
    )

    for threshold in THRESHOLDS:

        occurrence_count = (
            results[
                "occurrences"
            ].get(
                threshold,
                0,
            )
        )

        pair_count = len(
            results[
                "pairs"
            ][
                threshold
            ]
        )

        print(
            f"{threshold:<15.2f}"
            f"{occurrence_count:>20}"
            f"{pair_count:>20}"
        )

    print(
        "=" * 90
        + "\n"
    )


# ---------------------------------------------------------------------
# SHOW CANDIDATE PAIRS
# ---------------------------------------------------------------------

def print_candidate_pairs(
    df,
    minimum_threshold=0.70,
):
    """
    Print candidate pairs and their strongest similarity.

    This helps determine whether lower-threshold candidates
    are meaningful or simply generic text.
    """

    pair_best_similarity = {}

    grouped = df.groupby(
        "subreddit"
    )

    for subreddit, group in grouped:

        group = group.reset_index(
            drop=True
        )

        if len(group) < 2:

            continue

        accounts = group[
            "account_id"
        ].astype(
            str
        ).tolist()

        texts = group[
            "text"
        ].astype(
            str
        ).tolist()

        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=MAX_FEATURES,
        )

        matrix = vectorizer.fit_transform(
            texts
        )

        similarity_matrix = cosine_similarity(
            matrix
        )

        for i in range(
            len(group)
        ):

            for j in range(
                i + 1,
                len(group),
            ):

                source_account = accounts[
                    i
                ]

                target_account = accounts[
                    j
                ]

                if (
                    source_account
                    == target_account
                ):

                    continue

                similarity = float(
                    similarity_matrix[
                        i,
                        j,
                    ]
                )

                if (
                    similarity
                    < minimum_threshold
                ):

                    continue

                pair = tuple(
                    sorted(
                        [
                            source_account,
                            target_account,
                        ]
                    )
                )

                current = pair_best_similarity.get(
                    pair
                )

                if (
                    current is None
                    or similarity
                    > current[
                        "similarity"
                    ]
                ):

                    pair_best_similarity[
                        pair
                    ] = {
                        "similarity":
                            similarity,

                        "subreddit":
                            subreddit,
                    }

    sorted_pairs = sorted(
        pair_best_similarity.items(),
        key=lambda item: (
            item[
                1
            ][
                "similarity"
            ]
        ),
        reverse=True,
    )

    print(
        "\n"
        + "=" * 90
    )

    print(
        f"TOP CANDIDATE PAIRS "
        f"(SIMILARITY >= {minimum_threshold:.2f})"
    )

    print(
        "=" * 90
    )

    if not sorted_pairs:

        print(
            "\nNo candidate pairs found."
        )

        return

    print(
        f"\n{'SOURCE':<22}"
        f"{'TARGET':<22}"
        f"{'SIMILARITY':>15}"
        f"{'SUBREDDIT':>20}"
    )

    print(
        "-" * 90
    )

    for pair, data in sorted_pairs[:30]:

        print(
            f"{str(pair[0])[:20]:<22}"
            f"{str(pair[1])[:20]:<22}"
            f"{data['similarity']:>15.4f}"
            f"{str(data['subreddit'])[:18]:>20}"
        )


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():

    print(
        "\n"
        + "=" * 90
    )

    print(
        "PHASE 1.1 — CONTENT THRESHOLD VALIDATION"
    )

    print(
        "=" * 90
    )

    df = load_content()

    if df.empty:

        logger.warning(
            "No usable content found."
        )

        return

    results = run_threshold_sweep(
        df
    )

    print_results(
        results
    )

    print_candidate_pairs(
        df,
        minimum_threshold=0.70,
    )


if __name__ == "__main__":

    main()