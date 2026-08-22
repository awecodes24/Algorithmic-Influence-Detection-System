"""
Content Coordination Detection

Detects suspicious textual similarity between activities from
different Reddit accounts.

The detector uses two evidence levels:

1. Strong single-event duplication
   - Near-exact textual similarity
   - Different accounts
   - Same subreddit
   - Sufficiently long content
   - Close temporal proximity

2. Repeated similarity evidence
   - Multiple high-similarity activities between the same account pair

Important:
Content similarity is supporting evidence of possible coordination.
A single duplicate or repeated similarity alone does not prove
coordinated influence. The resulting evidence is later combined with
temporal and network evidence by build_account_pairs.py.
"""

import logging
import re
from collections import defaultdict

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.db import get_conn


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

# Ignore extremely short content because short comments can create
# misleading cosine similarity.
MIN_TEXT_LENGTH = 30


# Minimum similarity for normal candidate evidence.
MIN_CONTENT_SIMILARITY = 0.85


# Require repeated normal evidence between the same two accounts.
MIN_SIMILAR_OCCURRENCES = 2


# ---------------------------------------------------------------------
# STRONG SINGLE-EVENT DUPLICATION
# ---------------------------------------------------------------------

# A single event can be preserved when it is an extremely strong
# duplicate. This is intentionally much stricter than the normal
# similarity threshold.
STRONG_SINGLE_SIMILARITY = 0.98


# Long-form duplicated content is more meaningful than a short phrase.
MIN_STRONG_SINGLE_TEXT_LENGTH = 100


# Maximum allowed time difference between two strong duplicate events.
#
# 24 hours keeps the detector conservative. Exact or near-exact
# reposting years apart should not automatically be treated as
# coordinated timing evidence.
MAX_STRONG_SINGLE_TIME_DIFF_SECONDS = 24 * 60 * 60


# ---------------------------------------------------------------------
# PROCESSING LIMITS
# ---------------------------------------------------------------------

MAX_RECORDS_PER_SUBREDDIT = 2000

MAX_FEATURES = 10000


# ---------------------------------------------------------------------
# TEXT CLEANING
# ---------------------------------------------------------------------

def clean_text(text):
    """
    Normalize textual content before TF-IDF processing.

    The goal is to reduce accidental similarity caused by:

    - Reddit boilerplate
    - deleted or removed content
    - URLs
    - user and subreddit references
    - quoted text
    - duplicated titles or repeated sentences
    - malformed spacing
    """

    if text is None:
        return ""

    text = str(text)

    # -------------------------------------------------------------
    # REMOVE COMMON REDDIT PLACEHOLDERS
    # -------------------------------------------------------------

    placeholder_values = {
        "[deleted]",
        "[removed]",
        "deleted",
        "removed",
        "[unavailable]",
    }

    if text.strip().lower() in placeholder_values:
        return ""

    # -------------------------------------------------------------
    # REMOVE QUOTED LINES
    # -------------------------------------------------------------

    lines = text.splitlines()

    non_quoted_lines = []

    for line in lines:

        stripped_line = line.strip()

        if stripped_line.startswith(">"):
            continue

        non_quoted_lines.append(
            stripped_line
        )

    text = " ".join(
        non_quoted_lines
    )

    # -------------------------------------------------------------
    # FIX CAMEL-CASE WORD BOUNDARIES
    # -------------------------------------------------------------

    text = re.sub(
        r"([a-z])([A-Z])",
        r"\1 \2",
        text,
    )

    # -------------------------------------------------------------
    # LOWERCASE
    # -------------------------------------------------------------

    text = text.lower()

    # -------------------------------------------------------------
    # REMOVE URLS
    # -------------------------------------------------------------

    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text,
    )

    # -------------------------------------------------------------
    # REMOVE REDDIT USER REFERENCES
    # -------------------------------------------------------------

    text = re.sub(
        r"/?u/[a-z0-9_-]+",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    # -------------------------------------------------------------
    # REMOVE SUBREDDIT REFERENCES
    # -------------------------------------------------------------

    text = re.sub(
        r"/?r/[a-z0-9_]+",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    # -------------------------------------------------------------
    # KEEP ONLY LETTERS, NUMBERS AND SPACES
    # -------------------------------------------------------------

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text,
    )

    # -------------------------------------------------------------
    # NORMALIZE WHITESPACE
    # -------------------------------------------------------------

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    # -------------------------------------------------------------
    # REMOVE EXACT DUPLICATE HALVES
    # -------------------------------------------------------------

    tokens = text.split()

    if (
        len(tokens) >= 6
        and len(tokens) % 2 == 0
    ):

        midpoint = len(tokens) // 2

        first_half = tokens[:midpoint]

        second_half = tokens[midpoint:]

        if first_half == second_half:

            text = " ".join(
                first_half
            )

    return text


# ---------------------------------------------------------------------
# LOAD CONTENT
# ---------------------------------------------------------------------

def load_content():
    """
    Load textual activity from posts and comments.

    Returns a DataFrame containing:

    - activity_id
    - account_id
    - subreddit
    - timestamp
    - activity_type
    - text
    """

    conn = get_conn()

    records = []

    # -------------------------------------------------------------
    # POSTS
    # -------------------------------------------------------------

    try:

        posts = pd.read_sql(
            """
            SELECT
                id,
                account_id,
                subreddit,
                title,
                text,
                created_utc
            FROM posts
            WHERE account_id IS NOT NULL
            """,
            conn,
        )

        for _, row in posts.iterrows():

            title = (
                ""
                if pd.isna(row["title"])
                else str(row["title"]).strip()
            )

            body = (
                ""
                if pd.isna(row["text"])
                else str(row["text"]).strip()
            )

            normalized_title = clean_text(
                title
            )

            normalized_body = clean_text(
                body
            )

            # Avoid title duplication when the body already begins
            # with the title.
            if (
                normalized_title
                and normalized_body
            ):

                if normalized_body.startswith(
                    normalized_title
                ):

                    combined_text = body

                else:

                    combined_text = (
                        f"{title} {body}"
                    )

            elif title:

                combined_text = title

            else:

                combined_text = body

            cleaned = clean_text(
                combined_text
            )

            if len(cleaned) < MIN_TEXT_LENGTH:
                continue

            records.append(
                {
                    "activity_id": row["id"],
                    "account_id": str(
                        row["account_id"]
                    ),
                    "subreddit": str(
                        row["subreddit"]
                    ).lower(),
                    "timestamp": row[
                        "created_utc"
                    ],
                    "activity_type": "post",
                    "text": cleaned,
                }
            )

    except Exception as e:

        logger.warning(
            "Could not load posts: %s",
            e,
        )

    # -------------------------------------------------------------
    # COMMENTS
    # -------------------------------------------------------------

    try:

        comments = pd.read_sql(
            """
            SELECT
                id,
                account_id,
                subreddit,
                text,
                created_utc
            FROM comments
            WHERE account_id IS NOT NULL
            """,
            conn,
        )

        for _, row in comments.iterrows():

            raw_text = (
                ""
                if pd.isna(row["text"])
                else str(row["text"])
            )

            cleaned = clean_text(
                raw_text
            )

            if len(cleaned) < MIN_TEXT_LENGTH:
                continue

            records.append(
                {
                    "activity_id": row["id"],
                    "account_id": str(
                        row["account_id"]
                    ),
                    "subreddit": str(
                        row["subreddit"]
                    ).lower(),
                    "timestamp": row[
                        "created_utc"
                    ],
                    "activity_type": "comment",
                    "text": cleaned,
                }
            )

    except Exception as e:

        logger.warning(
            "Could not load comments: %s",
            e,
        )

    finally:

        conn.close()

    df = pd.DataFrame(
        records
    )

    if df.empty:

        logger.warning(
            "No usable textual content found."
        )

        return df

    df = df.dropna(
        subset=[
            "account_id",
            "text",
        ]
    )

    df = df[
        df["text"].str.len()
        >= MIN_TEXT_LENGTH
    ]

    df = df.reset_index(
        drop=True
    )

    logger.info(
        "Loaded %d usable text records from %d accounts",
        len(df),
        df["account_id"].nunique(),
    )

    logger.info(
        "Subreddits with content: %d",
        df["subreddit"].nunique(),
    )

    return df


# ---------------------------------------------------------------------
# HELPER: SAFE TIME DIFFERENCE
# ---------------------------------------------------------------------

def get_time_difference_seconds(
    source_timestamp,
    target_timestamp,
):
    """
    Return absolute time difference in seconds.

    Returns None when timestamps are unavailable or invalid.
    """

    try:

        if (
            pd.isna(source_timestamp)
            or pd.isna(target_timestamp)
        ):

            return None

        return abs(
            float(source_timestamp)
            - float(target_timestamp)
        )

    except (
        TypeError,
        ValueError,
    ):

        return None


# ---------------------------------------------------------------------
# HELPER: STRONG SINGLE EVENT
# ---------------------------------------------------------------------

def is_strong_single_event(
    source,
    target,
    similarity,
):
    """
    Determine whether one highly similar event is strong enough to be
    retained even when the account pair does not have repeated content.

    Requirements:

    1. Different accounts.
    2. Similarity >= STRONG_SINGLE_SIMILARITY.
    3. Both texts are sufficiently long.
    4. Same subreddit.
    5. Close temporal proximity.
    """

    if (
        source["account_id"]
        == target["account_id"]
    ):

        return False

    if similarity < STRONG_SINGLE_SIMILARITY:
        return False

    if (
        len(source["text"])
        < MIN_STRONG_SINGLE_TEXT_LENGTH
    ):

        return False

    if (
        len(target["text"])
        < MIN_STRONG_SINGLE_TEXT_LENGTH
    ):

        return False

    if (
        source["subreddit"]
        != target["subreddit"]
    ):

        return False

    time_difference = (
        get_time_difference_seconds(
            source["timestamp"],
            target["timestamp"],
        )
    )

    if time_difference is None:
        return False

    if (
        time_difference
        > MAX_STRONG_SINGLE_TIME_DIFF_SECONDS
    ):

        return False

    return True


# ---------------------------------------------------------------------
# DETECT SIMILAR CONTENT
# ---------------------------------------------------------------------

def detect_content_similarity(df):
    """
    Detect similar textual content between different accounts.

    Content is compared only within the same subreddit.

    Returns account-pair evidence.
    """

    if df.empty:
        return {}

    pair_stats = defaultdict(
        lambda: {
            "similarities": [],
            "events": [],
            "subreddits": set(),
            "strong_single_events": 0,
        }
    )

    subreddit_groups = df.groupby(
        "subreddit"
    )

    logger.info(
        "Processing %d subreddit groups",
        len(subreddit_groups),
    )

    for subreddit, group in subreddit_groups:

        group = group.copy()

        if (
            not subreddit
            or subreddit == "nan"
        ):

            continue

        if (
            len(group)
            > MAX_RECORDS_PER_SUBREDDIT
        ):

            group = group.sample(
                n=MAX_RECORDS_PER_SUBREDDIT,
                random_state=42,
            )

        if len(group) < 2:
            continue

        if (
            group["account_id"].nunique()
            < 2
        ):

            continue

        texts = group[
            "text"
        ].tolist()

        try:

            vectorizer = TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.95,
                max_features=MAX_FEATURES,
                sublinear_tf=True,
            )

            matrix = vectorizer.fit_transform(
                texts
            )

        except ValueError:

            continue

        similarity_matrix = cosine_similarity(
            matrix
        )

        group_records = group.to_dict(
            "records"
        )

        n = len(group_records)

        for i in range(n):

            for j in range(
                i + 1,
                n,
            ):

                source = group_records[i]
                target = group_records[j]

                if (
                    source["account_id"]
                    == target["account_id"]
                ):

                    continue

                similarity = float(
                    similarity_matrix[i, j]
                )

                if (
                    similarity
                    < MIN_CONTENT_SIMILARITY
                ):

                    continue

                pair = tuple(
                    sorted(
                        [
                            source["account_id"],
                            target["account_id"],
                        ]
                    )
                )

                event = {
                    "source_activity_id": source[
                        "activity_id"
                    ],
                    "target_activity_id": target[
                        "activity_id"
                    ],
                    "source_activity_type": source[
                        "activity_type"
                    ],
                    "target_activity_type": target[
                        "activity_type"
                    ],
                    "similarity": similarity,
                    "subreddit": subreddit,
                    "source_timestamp": source[
                        "timestamp"
                    ],
                    "target_timestamp": target[
                        "timestamp"
                    ],
                    "time_difference_seconds":
                        get_time_difference_seconds(
                            source["timestamp"],
                            target["timestamp"],
                        ),
                }

                pair_stats[
                    pair
                ]["similarities"].append(
                    similarity
                )

                pair_stats[
                    pair
                ]["events"].append(
                    event
                )

                pair_stats[
                    pair
                ]["subreddits"].add(
                    subreddit
                )

                if is_strong_single_event(
                    source,
                    target,
                    similarity,
                ):

                    pair_stats[
                        pair
                    ]["strong_single_events"] += 1

    logger.info(
        "Candidate content pairs: %d",
        len(pair_stats),
    )

    return pair_stats


# ---------------------------------------------------------------------
# BUILD RESULTS
# ---------------------------------------------------------------------

def build_results(pair_stats):
    """
    Convert content evidence into database-ready results.

    A pair is accepted when either:

    1. It has repeated high-similarity evidence, OR
    2. It contains at least one strong single-event duplicate.
    """

    results = []

    rejected_insufficient = 0

    repeated_count = 0

    strong_single_count = 0

    for pair, data in pair_stats.items():

        occurrences = len(
            data["similarities"]
        )

        strong_single_events = (
            data["strong_single_events"]
        )

        repeated_evidence = (
            occurrences
            >= MIN_SIMILAR_OCCURRENCES
        )

        strong_single_evidence = (
            strong_single_events
            > 0
        )

        if (
            not repeated_evidence
            and not strong_single_evidence
        ):

            rejected_insufficient += 1

            continue

        average_similarity = (
            sum(
                data["similarities"]
            )
            / occurrences
        )

        max_similarity = max(
            data["similarities"]
        )

        # ---------------------------------------------------------
        # DETERMINE METHOD
        # ---------------------------------------------------------

        if (
            repeated_evidence
            and strong_single_evidence
        ):

            method = (
                "tfidf_cosine_repeated"
                "_plus_strong_single"
            )

        elif repeated_evidence:

            method = (
                "tfidf_cosine_repeated"
            )

            repeated_count += 1

        else:

            method = (
                "tfidf_cosine_strong_single"
            )

            strong_single_count += 1

        # ---------------------------------------------------------
        # SIMILARITY SCORE
        # ---------------------------------------------------------
        #
        # For repeated evidence use average similarity.
        #
        # For a single strong duplicate use maximum similarity so an
        # exact duplicate is represented accurately rather than being
        # diluted by unrelated weaker candidates.
        # ---------------------------------------------------------

        if (
            strong_single_evidence
            and not repeated_evidence
        ):

            final_similarity = (
                max_similarity
            )

        else:

            final_similarity = (
                average_similarity
            )

        results.append(
            {
                "source_account_id": pair[0],
                "target_account_id": pair[1],
                "similarity": round(
                    float(final_similarity),
                    4,
                ),
                "method": method,
                "occurrences": occurrences,
                "strong_single_events":
                    strong_single_events,
            }
        )

    results.sort(
        key=lambda x: (
            x["similarity"],
            x["strong_single_events"],
            x["occurrences"],
        ),
        reverse=True,
    )

    logger.info(
        "Pairs rejected by insufficient evidence: %d",
        rejected_insufficient,
    )

    logger.info(
        "Pairs accepted through repeated evidence: %d",
        repeated_count,
    )

    logger.info(
        "Pairs accepted through strong single-event evidence: %d",
        strong_single_count,
    )

    logger.info(
        "Detected %d content evidence pairs",
        len(results),
    )

    return results


# ---------------------------------------------------------------------
# SAVE RESULTS
# ---------------------------------------------------------------------

def save_results(results):
    """
    Save content similarity evidence.
    """

    conn = get_conn()

    cursor = conn.cursor()

    try:

        cursor.execute(
            """
            DELETE FROM content_similarity
            """
        )

    except Exception as e:

        logger.warning(
            "Could not clear content_similarity: %s",
            e,
        )

    inserted = 0

    for row in results:

        try:

            cursor.execute(
                """
                INSERT INTO content_similarity (
                    source_account_id,
                    target_account_id,
                    similarity,
                    method
                )
                VALUES (?, ?, ?, ?)
                """,
                (
                    row[
                        "source_account_id"
                    ],
                    row[
                        "target_account_id"
                    ],
                    row[
                        "similarity"
                    ],
                    row[
                        "method"
                    ],
                ),
            )

            inserted += 1

        except Exception as e:

            logger.warning(
                "Could not save pair %s <-> %s: %s",
                row[
                    "source_account_id"
                ],
                row[
                    "target_account_id"
                ],
                e,
            )

    conn.commit()

    conn.close()

    logger.info(
        "Saved %d content similarity pairs",
        inserted,
    )


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():

    print(
        "\n"
        + "=" * 80
    )

    print(
        "CONTENT COORDINATION DETECTION"
    )

    print(
        "=" * 80
        + "\n"
    )

    # -------------------------------------------------------------
    # LOAD CONTENT
    # -------------------------------------------------------------

    df = load_content()

    if df.empty:

        logger.warning(
            "No usable content available."
        )

        return

    # -------------------------------------------------------------
    # DETECT SIMILARITY
    # -------------------------------------------------------------

    pair_stats = (
        detect_content_similarity(
            df
        )
    )

    # -------------------------------------------------------------
    # BUILD FINAL RESULTS
    # -------------------------------------------------------------

    results = build_results(
        pair_stats
    )

    # -------------------------------------------------------------
    # SAVE RESULTS
    # -------------------------------------------------------------

    save_results(
        results
    )

    # -------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------

    print(
        "\n"
        + "=" * 80
    )

    print(
        "CONTENT COORDINATION SUMMARY"
    )

    print(
        "=" * 80
    )

    print(
        f"Activity records scanned : {len(df)}"
    )

    print(
        f"Accounts analysed        : "
        f"{df['account_id'].nunique()}"
    )

    print(
        f"Candidate account pairs  : "
        f"{len(pair_stats)}"
    )

    print(
        f"Evidence pairs saved     : "
        f"{len(results)}"
    )

    print(
        "=" * 80
        + "\n"
    )

    # -------------------------------------------------------------
    # TOP RESULTS
    # -------------------------------------------------------------

    if results:

        print(
            "TOP CONTENT RELATIONSHIPS"
        )

        print(
            "-" * 80
        )

        for row in results[:20]:

            print(
                f"{row['source_account_id']} <-> "
                f"{row['target_account_id']} | "
                f"similarity={row['similarity']:.4f} | "
                f"occurrences={row['occurrences']} | "
                f"strong_single_events="
                f"{row['strong_single_events']} | "
                f"method={row['method']}"
            )

    else:

        print(
            "No account pairs satisfied either the repeated "
            "content requirement or the strong single-event "
            "duplication requirement."
        )


if __name__ == "__main__":

    main()