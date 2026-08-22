"""
Temporal Coordination Detection

Detects repeated independent bursts of cross-account activity.

Temporal coordination is treated as supporting evidence only.

A pair of accounts is considered to have temporal coordination evidence
only when the same pair participates together in multiple independent
temporal bursts separated by a minimum time interval.

This prevents:

1. One large burst from creating hundreds of coordinated pairs.
2. A single coincidence from being treated as coordination.
3. Multiple activities inside one burst from inflating pair evidence.
4. Burst boundaries from arbitrarily splitting dense activity.

Important:
The temporal_similarity table stores one aggregated relationship per
account pair.

The coordination_events table preserves individual independent burst
events for that relationship.
"""

import logging
import math
from collections import defaultdict
from itertools import combinations

import pandas as pd

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

BURST_WINDOW_SECONDS = 10 * 60

MIN_BURST_ACCOUNTS = 3

MIN_BURST_ACTIVITIES = 3

MAX_ACCOUNTS_PER_BURST = 25

MIN_BURSTS_PER_PAIR = 2

MIN_BURST_SEPARATION_SECONDS = 30 * 60

BURST_SCORE_CAP = 5

MIN_TEMPORAL_SIMILARITY = 0.35


# ---------------------------------------------------------------------
# LOAD ACTIVITY
# ---------------------------------------------------------------------

def load_activity():
    """
    Load posts and comments as timestamped activity records.
    """

    conn = get_conn()

    activities = []

    try:

        posts = pd.read_sql(
            """
            SELECT
                account_id,
                created_utc,
                subreddit
            FROM posts
            WHERE
                account_id IS NOT NULL
                AND created_utc IS NOT NULL
            """,
            conn,
        )

        for _, row in posts.iterrows():

            timestamp = pd.to_datetime(
                row["created_utc"],
                unit="s",
                utc=True,
                errors="coerce",
            )

            if pd.isna(timestamp):

                continue

            activities.append(
                {
                    "account_id": str(
                        row["account_id"]
                    ),
                    "timestamp": timestamp,
                    "subreddit": row["subreddit"],
                    "activity_type": "post",
                }
            )

    except Exception as e:

        logger.warning(
            "Could not load posts: %s",
            e,
        )

    try:

        comments = pd.read_sql(
            """
            SELECT
                account_id,
                created_utc,
                subreddit
            FROM comments
            WHERE
                account_id IS NOT NULL
                AND created_utc IS NOT NULL
            """,
            conn,
        )

        for _, row in comments.iterrows():

            timestamp = pd.to_datetime(
                row["created_utc"],
                unit="s",
                utc=True,
                errors="coerce",
            )

            if pd.isna(timestamp):

                continue

            activities.append(
                {
                    "account_id": str(
                        row["account_id"]
                    ),
                    "timestamp": timestamp,
                    "subreddit": row["subreddit"],
                    "activity_type": "comment",
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
        activities
    )

    if df.empty:

        logger.warning(
            "No activity records found."
        )

        return df

    df = df.dropna(
        subset=[
            "account_id",
            "timestamp",
        ]
    )

    df = df.sort_values(
        "timestamp"
    ).reset_index(
        drop=True
    )

    logger.info(
        "Loaded %d activity records from %d accounts",
        len(df),
        df["account_id"].nunique(),
    )

    logger.info(
        "Activity time range: %s -> %s",
        df["timestamp"].min(),
        df["timestamp"].max(),
    )

    return df


# ---------------------------------------------------------------------
# DETECT TEMPORAL BURSTS
# ---------------------------------------------------------------------

def detect_bursts(df):
    """
    Detect temporal bursts.

    A burst continues while consecutive activities remain within the
    configured burst window.
    """

    if df.empty:

        return []

    records = df.to_dict(
        "records"
    )

    bursts = []

    current_burst = []

    previous_time = None

    for record in records:

        timestamp = record[
            "timestamp"
        ]

        if previous_time is None:

            current_burst = [
                record
            ]

            previous_time = timestamp

            continue

        gap = (
            timestamp
            - previous_time
        ).total_seconds()

        if gap <= BURST_WINDOW_SECONDS:

            current_burst.append(
                record
            )

        else:

            if current_burst:

                bursts.append(
                    current_burst
                )

            current_burst = [
                record
            ]

        previous_time = timestamp

    if current_burst:

        bursts.append(
            current_burst
        )

    logger.info(
        "Detected %d raw temporal bursts",
        len(bursts),
    )

    return bursts


# ---------------------------------------------------------------------
# EVALUATE BURST
# ---------------------------------------------------------------------

def evaluate_burst(
    burst,
    burst_id,
):
    """
    Determine whether a burst contains meaningful cross-account activity.
    """

    if len(burst) < MIN_BURST_ACTIVITIES:

        return None

    accounts = sorted(
        {
            str(
                record["account_id"]
            )
            for record in burst
        }
    )

    if len(accounts) < MIN_BURST_ACCOUNTS:

        return None

    timestamps = [
        record["timestamp"]
        for record in burst
    ]

    start_time = min(
        timestamps
    )

    end_time = max(
        timestamps
    )

    duration = max(
        1.0,
        (
            end_time
            - start_time
        ).total_seconds(),
    )

    account_density = min(
        1.0,
        len(accounts) / 10.0,
    )

    temporal_density = max(
        0.0,
        1.0
        - (
            duration
            / BURST_WINDOW_SECONDS
        ),
    )

    subreddit_counts = defaultdict(
        int
    )

    for record in burst:

        subreddit = str(
            record.get(
                "subreddit",
                "",
            )
        ).lower()

        if subreddit:

            subreddit_counts[
                subreddit
            ] += 1

    dominant_subreddit_count = (
        max(
            subreddit_counts.values()
        )
        if subreddit_counts
        else 0
    )

    same_subreddit_ratio = (
        dominant_subreddit_count
        / len(burst)
        if burst
        else 0.0
    )

    burst_strength = (

        0.40
        * account_density

        +

        0.35
        * temporal_density

        +

        0.25
        * same_subreddit_ratio
    )

    return {
        "burst_id": burst_id,
        "accounts": accounts,
        "start_time": start_time,
        "end_time": end_time,
        "duration": duration,
        "activity_count": len(
            burst
        ),
        "account_count": len(
            accounts
        ),
        "same_subreddit_ratio": (
            same_subreddit_ratio
        ),
        "burst_strength": (
            burst_strength
        ),
    }


# ---------------------------------------------------------------------
# CHECK BURST INDEPENDENCE
# ---------------------------------------------------------------------

def is_independent_burst(
    pair_data,
    burst_start,
):
    """
    Check whether a burst is sufficiently separated from the previous
    burst counted for the same account pair.
    """

    last_burst_time = pair_data.get(
        "last_burst_time"
    )

    if last_burst_time is None:

        return True

    separation = (
        burst_start
        - last_burst_time
    ).total_seconds()

    return (
        separation
        >= MIN_BURST_SEPARATION_SECONDS
    )


# ---------------------------------------------------------------------
# BUILD PAIR EVIDENCE
# ---------------------------------------------------------------------

def build_pair_evidence(
    bursts,
):
    """
    Convert meaningful temporal bursts into account-pair evidence.

    The same pair must participate in multiple independent bursts.

    Each accepted burst is also preserved individually in burst_events
    so that coordination_events can store event-level evidence instead
    of collapsing all bursts into one pair-level row.
    """

    pair_stats = {}

    meaningful_burst_records = []

    for burst_id, burst in enumerate(
        bursts,
        start=1,
    ):

        info = evaluate_burst(
            burst,
            burst_id,
        )

        if info is None:

            continue

        meaningful_burst_records.append(
            {
                "info": info,
                "events": burst,
            }
        )

    meaningful_burst_count = len(
        meaningful_burst_records
    )

    # -------------------------------------------------------------
    # BURST INSPECTION
    # -------------------------------------------------------------

    print(
        "\n"
        + "=" * 80
    )

    print(
        "BURST INSPECTION"
    )

    print(
        "=" * 80
    )

    if not meaningful_burst_records:

        print(
            "No meaningful temporal bursts found."
        )

    else:

        for index, burst_data in enumerate(
            meaningful_burst_records[:10],
            start=1,
        ):

            info = burst_data[
                "info"
            ]

            events = burst_data[
                "events"
            ]

            accounts = set(
                info["accounts"]
            )

            subreddits = sorted(
                {
                    str(
                        event.get(
                            "subreddit",
                            "",
                        )
                    )
                    for event in events
                    if event.get(
                        "subreddit"
                    )
                }
            )

            print(
                f"\nBURST #{index}"
            )

            print(
                f"Burst ID         : "
                f"{info['burst_id']}"
            )

            print(
                f"Start time       : "
                f"{info['start_time']}"
            )

            print(
                f"End time         : "
                f"{info['end_time']}"
            )

            print(
                f"Duration         : "
                f"{info['duration']:.2f} seconds"
            )

            print(
                f"Activities       : "
                f"{info['activity_count']}"
            )

            print(
                f"Unique accounts  : "
                f"{info['account_count']}"
            )

            print(
                f"Burst strength   : "
                f"{info['burst_strength']:.4f}"
            )

            print(
                f"Same subreddit   : "
                f"{info['same_subreddit_ratio']:.4f}"
            )

            print(
                f"Subreddits       : "
                f"{', '.join(subreddits[:20])}"
            )

            print(
                f"Sample accounts  : "
                f"{', '.join(sorted(accounts)[:20])}"
            )

    print(
        "\n"
        + "=" * 80
    )

    # -------------------------------------------------------------
    # BUILD PAIR PARTICIPATION
    # -------------------------------------------------------------

    for burst_data in meaningful_burst_records:

        info = burst_data[
            "info"
        ]

        accounts = list(
            info["accounts"]
        )

        if (
            len(accounts)
            > MAX_ACCOUNTS_PER_BURST
        ):

            accounts = accounts[
                :MAX_ACCOUNTS_PER_BURST
            ]

        for source, target in combinations(
            accounts,
            2,
        ):

            pair = (
                source,
                target,
            )

            if pair not in pair_stats:

                pair_stats[
                    pair
                ] = {
                    "burst_count": 0,
                    "strengths": [],
                    "durations": [],
                    "same_subreddit_ratios": [],
                    "burst_ids": [],
                    "burst_events": [],
                    "last_burst_time": None,
                }

            data = pair_stats[
                pair
            ]

            if not is_independent_burst(
                data,
                info[
                    "start_time"
                ],
            ):

                continue

            data[
                "burst_count"
            ] += 1

            data[
                "strengths"
            ].append(
                info[
                    "burst_strength"
                ]
            )

            data[
                "durations"
            ].append(
                info[
                    "duration"
                ]
            )

            data[
                "same_subreddit_ratios"
            ].append(
                info[
                    "same_subreddit_ratio"
                ]
            )

            data[
                "burst_ids"
            ].append(
                info[
                    "burst_id"
                ]
            )

            # -----------------------------------------------------
            # PRESERVE THIS INDIVIDUAL BURST AS EVENT-LEVEL EVIDENCE
            # -----------------------------------------------------

            data[
                "burst_events"
            ].append(
                {
                    "burst_id": info[
                        "burst_id"
                    ],
                    "start_time": info[
                        "start_time"
                    ],
                    "end_time": info[
                        "end_time"
                    ],
                    "duration": info[
                        "duration"
                    ],
                    "activity_count": info[
                        "activity_count"
                    ],
                    "account_count": info[
                        "account_count"
                    ],
                    "same_subreddit_ratio": info[
                        "same_subreddit_ratio"
                    ],
                    "burst_strength": info[
                        "burst_strength"
                    ],
                }
            )

            data[
                "last_burst_time"
            ] = info[
                "start_time"
            ]

    logger.info(
        "Meaningful temporal bursts: %d",
        meaningful_burst_count,
    )

    logger.info(
        "Account pairs participating in bursts: %d",
        len(pair_stats),
    )

    return (
        pair_stats,
        meaningful_burst_count,
    )


# ---------------------------------------------------------------------
# CALCULATE TEMPORAL SIMILARITY
# ---------------------------------------------------------------------

def calculate_similarity(
    pair_data,
):
    """
    Calculate temporal similarity from:

    1. Repeated independent burst participation.
    2. Average burst strength.
    3. Same-subreddit consistency.
    """

    burst_count = pair_data[
        "burst_count"
    ]

    if (
        burst_count
        < MIN_BURSTS_PER_PAIR
    ):

        return None

    repetition_score = min(
        1.0,
        math.log1p(
            burst_count
        )
        /
        math.log1p(
            BURST_SCORE_CAP
        ),
    )

    strengths = pair_data[
        "strengths"
    ]

    average_strength = (
        sum(strengths)
        / len(strengths)
        if strengths
        else 0.0
    )

    subreddit_ratios = pair_data[
        "same_subreddit_ratios"
    ]

    average_subreddit_ratio = (
        sum(
            subreddit_ratios
        )
        / len(
            subreddit_ratios
        )
        if subreddit_ratios
        else 0.0
    )

    similarity = (

        0.50
        * repetition_score

        +

        0.30
        * average_strength

        +

        0.20
        * average_subreddit_ratio
    )

    durations = pair_data[
        "durations"
    ]

    avg_time_diff = (
        sum(
            durations
        )
        / len(
            durations
        )
        if durations
        else None
    )

    return {
        "similarity": round(
            min(
                1.0,
                similarity,
            ),
            4,
        ),

        "avg_time_diff": (
            round(
                float(avg_time_diff),
                2,
            )
            if avg_time_diff is not None
            else None
        ),

        "burst_count": burst_count,

        "burst_ids": pair_data[
            "burst_ids"
        ],
    }


# ---------------------------------------------------------------------
# BUILD RESULTS
# ---------------------------------------------------------------------

def build_results(
    pair_stats,
):
    """
    Convert pair evidence into database-ready results.

    Pair-level similarity remains aggregated, while individual burst
    metadata is carried forward for event-level storage.
    """

    results = []

    rejected_by_repetition = 0

    rejected_by_similarity = 0

    for pair, data in pair_stats.items():

        similarity_data = calculate_similarity(
            data
        )

        if similarity_data is None:

            rejected_by_repetition += 1

            continue

        similarity = similarity_data[
            "similarity"
        ]

        if (
            similarity
            < MIN_TEMPORAL_SIMILARITY
        ):

            rejected_by_similarity += 1

            continue

        results.append(
            {
                "source_account_id": pair[0],
                "target_account_id": pair[1],
                "similarity": similarity,
                "avg_time_diff": similarity_data[
                    "avg_time_diff"
                ],
                "burst_count": similarity_data[
                    "burst_count"
                ],
                "burst_ids": similarity_data[
                    "burst_ids"
                ],

                # Individual evidence for coordination_events.
                "burst_events": data[
                    "burst_events"
                ],
            }
        )

    results.sort(
        key=lambda row: row[
            "similarity"
        ],
        reverse=True,
    )

    logger.info(
        "Pairs rejected by insufficient repeated bursts: %d",
        rejected_by_repetition,
    )

    logger.info(
        "Pairs rejected by similarity threshold: %d",
        rejected_by_similarity,
    )

    logger.info(
        "Detected %d temporal coordination pairs",
        len(results),
    )

    return results


# ---------------------------------------------------------------------
# SAVE RESULTS
# ---------------------------------------------------------------------

def save_results(
    results,
):
    """
    Save aggregated temporal similarity evidence.
    """

    conn = get_conn()

    cursor = conn.cursor()

    try:

        cursor.execute(
            """
            DELETE FROM temporal_similarity
            """
        )

        conn.commit()

    except Exception as e:

        logger.warning(
            "Could not clear temporal_similarity: %s",
            e,
        )

    inserted = 0

    for row in results:

        try:

            cursor.execute(
                """
                INSERT INTO temporal_similarity (
                    source_account_id,
                    target_account_id,
                    similarity,
                    avg_time_diff
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
                        "avg_time_diff"
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
        "Saved %d temporal similarity pairs",
        inserted,
    )


# ---------------------------------------------------------------------
# UPDATE ACCOUNT TEMPORAL SCORES
# ---------------------------------------------------------------------

def update_account_temporal_scores():
    """
    Aggregate pair-level temporal similarity into an account-level
    temporal_score.
    """

    conn = get_conn()

    cursor = conn.cursor()

    try:

        cursor.execute(
            """
            UPDATE scores
            SET temporal_score = NULL
            """
        )

        rows = cursor.execute(
            """
            SELECT
                account_id,
                MAX(similarity) AS temporal_score
            FROM (
                SELECT
                    source_account_id AS account_id,
                    similarity
                FROM temporal_similarity

                UNION ALL

                SELECT
                    target_account_id AS account_id,
                    similarity
                FROM temporal_similarity
            )
            WHERE
                account_id IS NOT NULL
                AND similarity IS NOT NULL
            GROUP BY account_id
            """
        ).fetchall()

        updated = 0

        for row in rows:

            account_id = row[0]

            temporal_score = float(
                row[1]
            )

            cursor.execute(
                """
                UPDATE scores
                SET temporal_score = ?
                WHERE account_id = ?
                """,
                (
                    temporal_score,
                    account_id,
                ),
            )

            updated += cursor.rowcount

        conn.commit()

        logger.info(
            "Updated temporal_score for %d accounts",
            updated,
        )

        return updated

    except Exception as e:

        conn.rollback()

        logger.exception(
            "Could not update account temporal scores: %s",
            e,
        )

        return 0

    finally:

        conn.close()


# ---------------------------------------------------------------------
# SAVE COORDINATION EVENTS
# ---------------------------------------------------------------------

def save_coordination_events(
    results,
):
    """
    Save individual independent temporal bursts as coordination events.

    Important:

    temporal_similarity:
        One aggregated relationship per account pair.

    coordination_events:
        One row for every independent burst in which that pair
        participated.

    Therefore, if a pair has:

        burst_count = 3

    the coordination_events table receives:

        3 temporal_synchronization rows.
    """

    conn = get_conn()

    cursor = conn.cursor()

    cursor.execute(
        """
        DELETE FROM coordination_events
        WHERE event_type = 'temporal_synchronization'
        """
    )

    saved = 0

    for record in results:

        source_account_id = record.get(
            "source_account_id"
        )

        target_account_id = record.get(
            "target_account_id"
        )

        pair_similarity = record.get(
            "similarity"
        )

        burst_events = record.get(
            "burst_events",
            [],
        )

        if (
            source_account_id is None
            or target_account_id is None
            or pair_similarity is None
        ):

            continue

        # ---------------------------------------------------------
        # INSERT ONE DATABASE EVENT FOR EACH INDEPENDENT BURST
        # ---------------------------------------------------------

        for burst in burst_events:

            event_time = burst.get(
                "start_time"
            )

            if event_time is not None:

                event_time = event_time.timestamp()

            cursor.execute(
                """
                INSERT INTO coordination_events (
                    source_account_id,
                    target_account_id,
                    source_post_id,
                    event_type,
                    similarity,
                    event_time,
                    created_at,
                    target_post_id
                )
                VALUES (?, ?, ?, ?, ?, ?, datetime('now'), ?)
                """,
                (
                    str(source_account_id),
                    str(target_account_id),

                    # No individual post reference is currently
                    # available from the burst-level aggregation.
                    None,

                    "temporal_synchronization",

                    # Keep the pair-level similarity so the event
                    # remains connected to the relationship strength.
                    float(pair_similarity),

                    float(event_time)
                    if event_time is not None
                    else None,

                    None,
                ),
            )

            saved += 1

    conn.commit()

    conn.close()

    logger.info(
        "Saved %d temporal coordination events",
        saved,
    )

    return saved


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():

    print(
        "\n"
        + "=" * 80
    )

    print(
        "TEMPORAL COORDINATION DETECTION"
    )

    print(
        "=" * 80
        + "\n"
    )

    df = load_activity()

    if df.empty:

        logger.warning(
            "No activity available."
        )

        return

    bursts = detect_bursts(
        df
    )

    (
        pair_stats,
        meaningful_burst_count,
    ) = build_pair_evidence(
        bursts
    )

    results = build_results(
        pair_stats
    )

    save_results(
        results
    )

    saved_event_count = (
        save_coordination_events(
            results
        )
    )

    temporal_accounts_updated = (
        update_account_temporal_scores()
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "TEMPORAL COORDINATION SUMMARY"
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
        f"Raw bursts detected      : "
        f"{len(bursts)}"
    )

    print(
        f"Meaningful bursts        : "
        f"{meaningful_burst_count}"
    )

    print(
        f"Candidate pairs found    : "
        f"{len(pair_stats)}"
    )

    print(
        f"Evidence pairs saved     : "
        f"{len(results)}"
    )

    print(
        f"Coordination events saved: "
        f"{saved_event_count}"
    )

    print(
        f"Accounts with temporal signal : "
        f"{temporal_accounts_updated}"
    )

    print(
        "=" * 80
        + "\n"
    )

    if results:

        print(
            "TOP TEMPORAL RELATIONSHIPS"
        )

        print(
            "-" * 100
        )

        print(
            f"{'SOURCE':<20}"
            f"{'TARGET':<20}"
            f"{'SIMILARITY':>15}"
            f"{'BURSTS':>12}"
            f"{'AVG WINDOW':>18}"
        )

        print(
            "-" * 100
        )

        for row in results[:20]:

            print(
                f"{str(row['source_account_id'])[:18]:<20}"
                f"{str(row['target_account_id'])[:18]:<20}"
                f"{row['similarity']:>15.4f}"
                f"{row['burst_count']:>12}"
                f"{row['avg_time_diff']:>18.2f}"
            )

    else:

        print(
            "No account pairs satisfied the repeated "
            "temporal coordination requirement."
        )


if __name__ == "__main__":

    main()