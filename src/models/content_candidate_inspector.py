"""
Content Candidate Inspector

Inspects the actual text behind high-similarity
cross-account content pairs.

This diagnostic does not modify:

- the database
- production thresholds
- detector configuration

It is used to determine whether high similarity is caused by:

1. Genuine repeated/copied content.
2. Deleted or removed placeholders.
3. Boilerplate text.
4. Quoted/copied text.
5. Duplicate records.
6. Other preprocessing artifacts.
"""

import logging

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

MIN_INSPECTION_SIMILARITY = 0.70


# ---------------------------------------------------------------------
# FIND CANDIDATES
# ---------------------------------------------------------------------

def find_candidates(
    df,
):
    """
    Find all cross-account content pairs with similarity
    greater than or equal to the inspection threshold.

    Returns the original records so the actual text can
    be manually evaluated.
    """

    candidates = []

    grouped = df.groupby(
        "subreddit"
    )

    for subreddit, group in grouped:

        group = group.reset_index(
            drop=True
        )

        if len(group) < 2:

            continue

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

            source = group.iloc[
                i
            ]

            source_account = str(
                source[
                    "account_id"
                ]
            )

            for j in range(
                i + 1,
                len(group),
            ):

                target = group.iloc[
                    j
                ]

                target_account = str(
                    target[
                        "account_id"
                    ]
                )

                # Ignore same-account comparisons.

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
                    < MIN_INSPECTION_SIMILARITY
                ):

                    continue

                candidates.append(
                    {
                        "source":
                            source.to_dict(),

                        "target":
                            target.to_dict(),

                        "similarity":
                            similarity,

                        "subreddit":
                            subreddit,
                    }
                )

    candidates.sort(
        key=lambda item: (
            item[
                "similarity"
            ]
        ),
        reverse=True,
    )

    return candidates


# ---------------------------------------------------------------------
# PRINT TEXT
# ---------------------------------------------------------------------

def print_text(
    label,
    record,
):
    """
    Print a candidate activity record.
    """

    print(
        f"\n{label}"
    )

    print(
        "-" * 100
    )

    print(
        f"Account ID    : "
        f"{record.get('account_id')}"
    )

    print(
        f"Activity ID   : "
        f"{record.get('activity_id')}"
    )

    print(
        f"Activity Type : "
        f"{record.get('activity_type')}"
    )

    print(
        f"Subreddit     : "
        f"{record.get('subreddit')}"
    )

    print(
        f"Timestamp     : "
        f"{record.get('timestamp')}"
    )

    print(
        "\nTEXT:"
    )

    print(
        record.get(
            "text",
            "",
        )
    )


# ---------------------------------------------------------------------
# PRINT CANDIDATES
# ---------------------------------------------------------------------

def print_candidates(
    candidates,
):
    """
    Print complete evidence for every candidate pair.
    """

    print(
        "\n"
        + "=" * 100
    )

    print(
        "CONTENT CANDIDATE EVIDENCE INSPECTION"
    )

    print(
        "=" * 100
    )

    print(
        f"\nCandidates found: "
        f"{len(candidates)}"
    )

    if not candidates:

        print(
            "\nNo candidates found."
        )

        return

    for index, candidate in enumerate(
        candidates,
        start=1,
    ):

        print(
            "\n"
            + "#" * 100
        )

        print(
            f"CANDIDATE #{index}"
        )

        print(
            "#" * 100
        )

        print(
            f"\nSimilarity : "
            f"{candidate['similarity']:.4f}"
        )

        print(
            f"Subreddit  : "
            f"{candidate['subreddit']}"
        )

        print_text(
            "SOURCE",
            candidate[
                "source"
            ],
        )

        print_text(
            "TARGET",
            candidate[
                "target"
            ],
        )

        print(
            "\n"
            + "=" * 100
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
        "PHASE 1.2 — CONTENT CANDIDATE INSPECTION"
    )

    print(
        "=" * 100
    )

    df = load_content()

    if df.empty:

        logger.warning(
            "No usable content found."
        )

        return

    candidates = find_candidates(
        df
    )

    print_candidates(
        candidates
    )


if __name__ == "__main__":

    main()