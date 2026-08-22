"""
Content Coordination Diagnostics

This module inspects candidate content-similarity relationships before
changing the main coordination detector.

It does NOT modify the database.

The diagnostics answer:

1. How much usable content exists?
2. How many subreddit groups are being analysed?
3. How many cross-account content pairs reach different
   similarity ranges?
4. How many pairs are rejected because they do not have repeated
   occurrences?
5. Which account pairs are closest to becoming valid content
   coordination evidence?
"""

import logging
from collections import defaultdict

import pandas as pd

from src.models.reddit_content_coordination import (
    load_content,
    clean_text,
    MIN_TEXT_LENGTH,
    MIN_CONTENT_SIMILARITY,
    MIN_SIMILAR_OCCURRENCES,
    MAX_RECORDS_PER_SUBREDDIT,
    MAX_FEATURES,
)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# SIMILARITY BANDS
# ---------------------------------------------------------------------

SIMILARITY_BANDS = [
    (0.95, 1.01, "0.95 - 1.00"),
    (0.90, 0.95, "0.90 - 0.95"),
    (0.85, 0.90, "0.85 - 0.90"),
    (0.80, 0.85, "0.80 - 0.85"),
    (0.75, 0.80, "0.75 - 0.80"),
    (0.70, 0.75, "0.70 - 0.75"),
]


# ---------------------------------------------------------------------
# LOAD AND INSPECT CONTENT
# ---------------------------------------------------------------------

def inspect_content(df):
    """
    Print basic information about the content available for
    coordination analysis.
    """

    print(
        "\n"
        + "=" * 100
    )

    print(
        "CONTENT DATASET INSPECTION"
    )

    print(
        "=" * 100
    )

    print(
        f"\nTotal loaded activity records : {len(df)}"
    )

    if df.empty:

        print(
            "\nNo content records available."
        )

        return pd.DataFrame()

    print(
        f"Unique accounts                : "
        f"{df['account_id'].nunique()}"
    )

    print(
        f"Unique subreddits              : "
        f"{df['subreddit'].nunique()}"
    )

    print(
        "\nActivity types:"
    )

    print(
        df[
            "activity_type"
        ].value_counts()
    )

    # -------------------------------------------------------------
    # TEXT LENGTH INSPECTION
    # -------------------------------------------------------------

    working_df = df.copy()

    working_df[
        "cleaned_text"
    ] = working_df[
        "text"
    ].apply(
        clean_text
    )

    working_df[
        "text_length"
    ] = working_df[
        "cleaned_text"
    ].str.len()

    usable_df = working_df[
        working_df[
            "text_length"
        ]
        >= MIN_TEXT_LENGTH
    ].copy()

    rejected_short = (
        len(working_df)
        - len(usable_df)
    )

    print(
        "\nTEXT FILTERING"
    )

    print(
        "-" * 100
    )

    print(
        f"Minimum text length            : "
        f"{MIN_TEXT_LENGTH}"
    )

    print(
        f"Usable text records            : "
        f"{len(usable_df)}"
    )

    print(
        f"Rejected short/empty records   : "
        f"{rejected_short}"
    )

    if not usable_df.empty:

        print(
            f"Average usable text length     : "
            f"{usable_df['text_length'].mean():.2f}"
        )

        print(
            f"Median usable text length      : "
            f"{usable_df['text_length'].median():.2f}"
        )

        print(
            f"Maximum usable text length     : "
            f"{usable_df['text_length'].max()}"
        )

    return usable_df


# ---------------------------------------------------------------------
# INSPECT SUBREDDIT DISTRIBUTION
# ---------------------------------------------------------------------

def inspect_subreddits(df):
    """
    Inspect how much usable content exists inside each subreddit.
    """

    print(
        "\n"
        + "=" * 100
    )

    print(
        "SUBREDDIT DISTRIBUTION"
    )

    print(
        "=" * 100
    )

    if df.empty:

        print(
            "\nNo usable subreddit groups."
        )

        return

    subreddit_stats = (
        df.groupby(
            "subreddit"
        )
        .agg(
            records=(
                "account_id",
                "size",
            ),
            accounts=(
                "account_id",
                "nunique",
            ),
        )
        .reset_index()
        .sort_values(
            "records",
            ascending=False,
        )
    )

    print(
        f"\nSubreddits with usable content: "
        f"{len(subreddit_stats)}"
    )

    print(
        "\nTOP 20 SUBREDDITS BY USABLE CONTENT"
    )

    print(
        "-" * 100
    )

    print(
        subreddit_stats
        .head(20)
        .to_string(
            index=False
        )
    )

    single_account_subreddits = len(
        subreddit_stats[
            subreddit_stats[
                "accounts"
            ]
            < 2
        ]
    )

    insufficient_records = len(
        subreddit_stats[
            subreddit_stats[
                "records"
            ]
            < 2
        ]
    )

    print(
        "\nSUBREDDIT LIMITATIONS"
    )

    print(
        "-" * 100
    )

    print(
        f"Groups with fewer than 2 accounts: "
        f"{single_account_subreddits}"
    )

    print(
        f"Groups with fewer than 2 records : "
        f"{insufficient_records}"
    )


# ---------------------------------------------------------------------
# GET SIMILARITY BAND
# ---------------------------------------------------------------------

def get_similarity_band(similarity):
    """
    Return the diagnostic band for a similarity value.
    """

    for lower, upper, label in SIMILARITY_BANDS:

        if (
            similarity >= lower
            and similarity < upper
        ):

            return label

    return None


# ---------------------------------------------------------------------
# ANALYSE CONTENT CANDIDATES
# ---------------------------------------------------------------------

def analyse_candidates(df):
    """
    Re-run candidate generation for diagnostics.

    This does not write anything to the database.
    """

    pair_stats = defaultdict(
        lambda: {
            "similarities": [],
            "subreddits": set(),
        }
    )

    band_counts = defaultdict(
        int
    )

    total_cross_account_comparisons = 0

    total_candidate_occurrences = 0

    processed_groups = 0

    skipped_groups = 0

    grouped = df.groupby(
        "subreddit"
    )

    total_groups = len(
        grouped
    )

    for subreddit, group in grouped:

        processed_groups += 1

        group = group.copy()

        if (
            len(group)
            > MAX_RECORDS_PER_SUBREDDIT
        ):

            group = group.sample(
                n=MAX_RECORDS_PER_SUBREDDIT,
                random_state=42,
            )

        if len(group) < 2:

            skipped_groups += 1

            continue

        if (
            group[
                "account_id"
            ]
            .nunique()
            < 2
        ):

            skipped_groups += 1

            continue

        texts = (
            group[
                "cleaned_text"
            ]
            .tolist()
        )

        try:

            vectorizer = TfidfVectorizer(
                stop_words="english",
                max_features=MAX_FEATURES,
            )

            matrix = vectorizer.fit_transform(
                texts
            )

        except ValueError:

            skipped_groups += 1

            continue

        similarity_matrix = cosine_similarity(
            matrix
        )

        records = group.reset_index(
            drop=True
        )

        group_size = len(
            records
        )

        for i in range(group_size):

            source_account = str(
                records.iloc[i][
                    "account_id"
                ]
            )

            for j in range(
                i + 1,
                group_size,
            ):

                target_account = str(
                    records.iloc[j][
                        "account_id"
                    ]
                )

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

                total_cross_account_comparisons += 1

                band = get_similarity_band(
                    similarity
                )

                if band is not None:

                    band_counts[
                        band
                    ] += 1

                # Store candidates starting from 0.70.
                #
                # This allows us to inspect whether the current
                # 0.80 threshold is discarding many near-candidates.

                if similarity < 0.70:

                    continue

                total_candidate_occurrences += 1

                pair = tuple(
                    sorted(
                        [
                            source_account,
                            target_account,
                        ]
                    )
                )

                pair_stats[
                    pair
                ][
                    "similarities"
                ].append(
                    similarity
                )

                pair_stats[
                    pair
                ][
                    "subreddits"
                ].add(
                    str(
                        subreddit
                    )
                )

        if (
            processed_groups % 10 == 0
        ):

            logger.info(
                "Processed %d / %d subreddit groups",
                processed_groups,
                total_groups,
            )

    return {
        "pair_stats": pair_stats,
        "band_counts": band_counts,
        "total_cross_account_comparisons":
            total_cross_account_comparisons,
        "total_candidate_occurrences":
            total_candidate_occurrences,
        "processed_groups":
            processed_groups,
        "skipped_groups":
            skipped_groups,
    }


# ---------------------------------------------------------------------
# PRINT SIMILARITY DISTRIBUTION
# ---------------------------------------------------------------------

def print_similarity_distribution(
    diagnostics,
):
    """
    Print similarity-band statistics.
    """

    print(
        "\n"
        + "=" * 100
    )

    print(
        "SIMILARITY DISTRIBUTION"
    )

    print(
        "=" * 100
    )

    total_comparisons = diagnostics[
        "total_cross_account_comparisons"
    ]

    print(
        f"\nTotal cross-account comparisons: "
        f"{total_comparisons}"
    )

    print()

    print(
        f"{'Similarity Range':<25}"
        f"{'Occurrences':>15}"
        f"{'Percentage':>15}"
    )

    print(
        "-" * 55
    )

    for _, _, label in SIMILARITY_BANDS:

        count = diagnostics[
            "band_counts"
        ].get(
            label,
            0,
        )

        percentage = (
            (
                count
                / total_comparisons
                * 100
            )
            if total_comparisons > 0
            else 0.0
        )

        print(
            f"{label:<25}"
            f"{count:>15}"
            f"{percentage:>14.4f}%"
        )


# ---------------------------------------------------------------------
# ANALYSE PAIR REPETITION
# ---------------------------------------------------------------------

def analyse_pair_repetition(
    pair_stats,
):
    """
    Determine how many candidate pairs are rejected because they
    lack repeated similar-content evidence.
    """

    print(
        "\n"
        + "=" * 100
    )

    print(
        "PAIR REPETITION ANALYSIS"
    )

    print(
        "=" * 100
    )

    occurrence_distribution = defaultdict(
        int
    )

    pairs_above_threshold = 0

    valid_pairs = []

    rejected_once = []

    for pair, data in pair_stats.items():

        occurrences = len(
            data[
                "similarities"
            ]
        )

        occurrence_distribution[
            occurrences
        ] += 1

        similarities_above_threshold = [

            similarity

            for similarity in data[
                "similarities"
            ]

            if similarity
            >= MIN_CONTENT_SIMILARITY

        ]

        threshold_occurrences = len(
            similarities_above_threshold
        )

        if threshold_occurrences > 0:

            pairs_above_threshold += 1

        if (
            threshold_occurrences
            >= MIN_SIMILAR_OCCURRENCES
        ):

            valid_pairs.append(
                {
                    "pair": pair,
                    "occurrences":
                        threshold_occurrences,
                    "avg_similarity":
                        sum(
                            similarities_above_threshold
                        )
                        /
                        threshold_occurrences,
                    "max_similarity":
                        max(
                            similarities_above_threshold
                        ),
                    "subreddit_count":
                        len(
                            data[
                                "subreddits"
                            ]
                        ),
                }
            )

        elif threshold_occurrences == 1:

            rejected_once.append(
                {
                    "pair": pair,
                    "similarity":
                        similarities_above_threshold[
                            0
                        ],
                    "subreddit_count":
                        len(
                            data[
                                "subreddits"
                            ]
                        ),
                }
            )

    print(
        f"\nCandidate account pairs "
        f"(similarity >= 0.70): "
        f"{len(pair_stats)}"
    )

    print(
        f"Pairs with at least one occurrence "
        f">= {MIN_CONTENT_SIMILARITY:.2f}: "
        f"{pairs_above_threshold}"
    )

    print(
        f"Pairs surviving repetition requirement "
        f"({MIN_SIMILAR_OCCURRENCES}+ occurrences): "
        f"{len(valid_pairs)}"
    )

    print(
        f"Pairs rejected because they only have "
        f"one strong occurrence: "
        f"{len(rejected_once)}"
    )

    print(
        "\nOCCURRENCE DISTRIBUTION"
    )

    print(
        "-" * 100
    )

    for occurrences in sorted(
        occurrence_distribution
    ):

        print(
            f"{occurrences} occurrence(s): "
            f"{occurrence_distribution[occurrences]} pairs"
        )

    return (
        valid_pairs,
        rejected_once,
    )


# ---------------------------------------------------------------------
# PRINT TOP PAIRS
# ---------------------------------------------------------------------

def print_top_pairs(
    valid_pairs,
    rejected_once,
    limit=20,
):
    """
    Print the strongest valid and near-valid pairs.
    """

    print(
        "\n"
        + "=" * 100
    )

    print(
        "TOP VALID CONTENT COORDINATION PAIRS"
    )

    print(
        "=" * 100
    )

    valid_pairs = sorted(
        valid_pairs,
        key=lambda row: (
            row[
                "avg_similarity"
            ],
            row[
                "occurrences"
            ],
        ),
        reverse=True,
    )

    if not valid_pairs:

        print(
            "\nNo pairs currently satisfy all "
            "content coordination requirements."
        )

    else:

        print()

        for index, row in enumerate(
            valid_pairs[:limit],
            start=1,
        ):

            source, target = row[
                "pair"
            ]

            print(
                f"\n#{index}"
            )

            print(
                f"Accounts       : "
                f"{source} <-> {target}"
            )

            print(
                f"Occurrences    : "
                f"{row['occurrences']}"
            )

            print(
                f"Average sim.   : "
                f"{row['avg_similarity']:.4f}"
            )

            print(
                f"Maximum sim.   : "
                f"{row['max_similarity']:.4f}"
            )

            print(
                f"Subreddit count: "
                f"{row['subreddit_count']}"
            )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "NEAR-MISS PAIRS"
    )

    print(
        "=" * 100
    )

    rejected_once = sorted(
        rejected_once,
        key=lambda row: row[
            "similarity"
        ],
        reverse=True,
    )

    if not rejected_once:

        print(
            "\nNo pairs were rejected solely because "
            "they lacked repeated occurrences."
        )

    else:

        print(
            "\nPairs with exactly one occurrence "
            f">= {MIN_CONTENT_SIMILARITY:.2f}:"
        )

        for index, row in enumerate(
            rejected_once[:limit],
            start=1,
        ):

            source, target = row[
                "pair"
            ]

            print(
                f"\n#{index}"
            )

            print(
                f"Accounts       : "
                f"{source} <-> {target}"
            )

            print(
                f"Similarity     : "
                f"{row['similarity']:.4f}"
            )

            print(
                f"Subreddit count: "
                f"{row['subreddit_count']}"
            )


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():

    print(
        "\n"
        + "=" * 100
    )

    print(
        "CONTENT COORDINATION DIAGNOSTICS"
    )

    print(
        "=" * 100
    )

    print(
        "\nCURRENT DETECTOR CONFIGURATION"
    )

    print(
        "-" * 100
    )

    print(
        f"MIN_TEXT_LENGTH            : "
        f"{MIN_TEXT_LENGTH}"
    )

    print(
        f"MIN_CONTENT_SIMILARITY     : "
        f"{MIN_CONTENT_SIMILARITY}"
    )

    print(
        f"MIN_SIMILAR_OCCURRENCES    : "
        f"{MIN_SIMILAR_OCCURRENCES}"
    )

    print(
        f"MAX_RECORDS_PER_SUBREDDIT  : "
        f"{MAX_RECORDS_PER_SUBREDDIT}"
    )

    print(
        f"MAX_FEATURES               : "
        f"{MAX_FEATURES}"
    )

    # -------------------------------------------------------------
    # LOAD CONTENT
    # -------------------------------------------------------------

    df = load_content()

    if df.empty:

        print(
            "\nNo content available for diagnostics."
        )

        return

    # -------------------------------------------------------------
    # INSPECT DATA
    # -------------------------------------------------------------

    usable_df = inspect_content(
        df
    )

    if usable_df.empty:

        print(
            "\nNo usable content remains after filtering."
        )

        return

    # -------------------------------------------------------------
    # INSPECT SUBREDDITS
    # -------------------------------------------------------------

    inspect_subreddits(
        usable_df
    )

    # -------------------------------------------------------------
    # ANALYSE CANDIDATES
    # -------------------------------------------------------------

    diagnostics = analyse_candidates(
        usable_df
    )

    # -------------------------------------------------------------
    # PRINT DISTRIBUTION
    # -------------------------------------------------------------

    print_similarity_distribution(
        diagnostics
    )

    # -------------------------------------------------------------
    # ANALYSE REPETITION
    # -------------------------------------------------------------

    (
        valid_pairs,
        rejected_once,
    ) = analyse_pair_repetition(
        diagnostics[
            "pair_stats"
        ]
    )

    # -------------------------------------------------------------
    # PRINT TOP PAIRS
    # -------------------------------------------------------------

    print_top_pairs(
        valid_pairs,
        rejected_once,
    )

    # -------------------------------------------------------------
    # FINAL SUMMARY
    # -------------------------------------------------------------

    print(
        "\n"
        + "=" * 100
    )

    print(
        "DIAGNOSTIC SUMMARY"
    )

    print(
        "=" * 100
    )

    print(
        f"\nProcessed subreddit groups       : "
        f"{diagnostics['processed_groups']}"
    )

    print(
        f"Skipped subreddit groups         : "
        f"{diagnostics['skipped_groups']}"
    )

    print(
        f"Cross-account comparisons        : "
        f"{diagnostics['total_cross_account_comparisons']}"
    )

    print(
        f"Candidate occurrences >= 0.70    : "
        f"{diagnostics['total_candidate_occurrences']}"
    )

    print(
        f"Candidate account pairs >= 0.70  : "
        f"{len(diagnostics['pair_stats'])}"
    )

    print(
        f"Valid repeated pairs >= "
        f"{MIN_CONTENT_SIMILARITY:.2f}: "
        f"{len(valid_pairs)}"
    )

    print(
        "\nInterpretation:"
    )

    print(
        "This diagnostic does not change the detector."
    )

    print(
        "Use the similarity and repetition distributions to determine "
        "whether the current lack of content evidence is caused mainly "
        "by the similarity threshold, the repetition requirement, "
        "limited usable content, or the subreddit grouping structure."
    )

    print(
        "\n"
        + "=" * 100
        + "\n"
    )


if __name__ == "__main__":

    main()