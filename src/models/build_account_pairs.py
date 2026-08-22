# src/models/build_account_pairs.py
#
# Builds the account_pairs table by combining pair-level evidence from:
#
#   1. Content similarity
#   2. Temporal synchronization
#   3. Direct network interaction
#
# Important scoring principle:
#
# Network interaction alone is treated as structural association, not
# strong proof of coordination. Stronger coordination scores require
# independent support from temporal and/or content evidence.
#

from __future__ import annotations

import logging
import math

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

# Evidence weights.
#
# Content and temporal synchronization are more directly relevant to
# coordinated behavior than simple network interaction.
CONTENT_WEIGHT = 0.45
TEMPORAL_WEIGHT = 0.40
NETWORK_WEIGHT = 0.15


# ---------------------------------------------------------------------
# NETWORK COORDINATION PARAMETERS
# ---------------------------------------------------------------------

MIN_NETWORK_INTERACTIONS = 2.0

NETWORK_VOLUME_WEIGHT = 0.35
NETWORK_RECIPROCITY_WEIGHT = 0.30
NETWORK_CONCENTRATION_WEIGHT = 0.35

MIN_NETWORK_COORDINATION_SCORE = 0.20


# ---------------------------------------------------------------------
# EVIDENCE CONFIDENCE
# ---------------------------------------------------------------------

# Different evidence sources have different standalone meanings.
#
# Network interaction alone represents structural association and should
# remain conservative.
NETWORK_ONLY_FACTOR = 0.45

# Temporal synchronization is useful evidence, but isolated temporal
# similarity should not by itself imply coordinated influence.
TEMPORAL_ONLY_FACTOR = 0.65

# Strong near-duplicate content is more direct evidence of coordination.
CONTENT_ONLY_FACTOR = 0.75


# ---------------------------------------------------------------------
# MULTI-SOURCE EVIDENCE CONFIDENCE
# ---------------------------------------------------------------------
#
# Independent evidence sources reinforce each other. Instead of heavily
# suppressing the weighted base score, multi-source evidence receives
# an explicit convergence bonus.

TWO_EVIDENCE_FACTOR = 1.10
THREE_EVIDENCE_FACTOR = 1.20

TWO_EVIDENCE_BONUS = 0.15
THREE_EVIDENCE_BONUS = 0.25


# ---------------------------------------------------------------------
# NETWORK-ONLY SCORE LIMIT
# ---------------------------------------------------------------------

# Network evidence alone should remain structural evidence and should
# not independently produce a very high coordination score.
MAX_NETWORK_ONLY_FINAL_SCORE = 0.40


# ---------------------------------------------------------------------
# MINIMUM SCORE REQUIRED FOR EVIDENCE PRESENCE
# ---------------------------------------------------------------------

MIN_EVIDENCE_SCORE = 0.0

# ---------------------------------------------------------------------
# CANONICAL PAIR
# ---------------------------------------------------------------------

def canonical_pair(
    source_account_id,
    target_account_id,
):
    """
    Return a deterministic ordering for an account pair.

    A -> B and B -> A are represented as one canonical pair:

        A <-> B
    """

    if not source_account_id or not target_account_id:
        return None

    source_account_id = str(source_account_id)
    target_account_id = str(target_account_id)

    if source_account_id == target_account_id:
        return None

    return tuple(
        sorted(
            (
                source_account_id,
                target_account_id,
            )
        )
    )


# ---------------------------------------------------------------------
# LOAD CONTENT EVIDENCE
# ---------------------------------------------------------------------

def load_content_pairs(conn):
    """
    Load pair-level content similarity evidence.

    Returns
    -------
    dict
        {
            (account_a, account_b): similarity
        }
    """

    query = """
        SELECT
            source_account_id,
            target_account_id,
            similarity
        FROM content_similarity
        WHERE similarity IS NOT NULL
    """

    rows = conn.execute(
        query
    ).fetchall()

    pairs = {}
    skipped = 0

    for row in rows:

        pair = canonical_pair(
            row["source_account_id"],
            row["target_account_id"],
        )

        if pair is None:
            skipped += 1
            continue

        similarity = max(
            0.0,
            min(
                float(row["similarity"]),
                1.0,
            ),
        )

        # Keep strongest evidence for the pair.
        if (
            pair not in pairs
            or similarity > pairs[pair]
        ):
            pairs[pair] = similarity

    logger.info(
        "Loaded %d content evidence pairs",
        len(pairs),
    )

    if skipped:
        logger.info(
            "Skipped %d invalid content pairs",
            skipped,
        )

    return pairs


# ---------------------------------------------------------------------
# LOAD TEMPORAL EVIDENCE
# ---------------------------------------------------------------------

def load_temporal_pairs(conn):
    """
    Load pair-level temporal synchronization evidence.

    Returns
    -------
    dict
        {
            (account_a, account_b): similarity
        }
    """

    query = """
        SELECT
            source_account_id,
            target_account_id,
            similarity
        FROM temporal_similarity
        WHERE similarity IS NOT NULL
    """

    rows = conn.execute(
        query
    ).fetchall()

    pairs = {}
    skipped = 0

    for row in rows:

        pair = canonical_pair(
            row["source_account_id"],
            row["target_account_id"],
        )

        if pair is None:
            skipped += 1
            continue

        similarity = max(
            0.0,
            min(
                float(row["similarity"]),
                1.0,
            ),
        )

        # Keep strongest temporal evidence.
        if (
            pair not in pairs
            or similarity > pairs[pair]
        ):
            pairs[pair] = similarity

    logger.info(
        "Loaded %d temporal evidence pairs",
        len(pairs),
    )

    if skipped:
        logger.info(
            "Skipped %d invalid temporal pairs",
            skipped,
        )

    return pairs


# ---------------------------------------------------------------------
# LOAD NETWORK EVIDENCE
# ---------------------------------------------------------------------

def load_network_pairs(
    conn,
):
    """
    Build account-pair network coordination evidence.

    Network coordination is evaluated using:

        1. Interaction volume
        2. Interaction reciprocity
        3. Interaction concentration

    Concentration measures how important the interaction between two
    accounts is relative to their overall interaction activity.

    A pair that interacts frequently but where both accounts interact
    with many other accounts receives weaker evidence than a pair whose
    interactions are concentrated primarily toward each other.
    """

    query = """
        SELECT
            source_account_id,
            target_account_id,
            weight
        FROM edges
        WHERE
            edge_type IN (
                'comment_on_post',
                'reply_to_comment'
            )
            AND weight IS NOT NULL
            AND weight > 0
    """

    rows = conn.execute(
        query
    ).fetchall()

    directed_interactions = {}

    account_activity = {}

    skipped = 0

    # -------------------------------------------------------------
    # LOAD DIRECTED INTERACTIONS
    # -------------------------------------------------------------

    for row in rows:

        source_account_id = str(
            row["source_account_id"]
        )

        target_account_id = str(
            row["target_account_id"]
        )

        pair = canonical_pair(
            source_account_id,
            target_account_id,
        )

        if pair is None:

            skipped += 1
            continue

        weight = float(
            row["weight"]
        )

        direction = (
            source_account_id,
            target_account_id,
        )

        directed_interactions[direction] = (
            directed_interactions.get(
                direction,
                0.0,
            )
            + weight
        )

        # Track total interaction activity for concentration scoring.
        account_activity[source_account_id] = (
            account_activity.get(
                source_account_id,
                0.0,
            )
            + weight
        )

        account_activity[target_account_id] = (
            account_activity.get(
                target_account_id,
                0.0,
            )
            + weight
        )

    logger.info(
        "Loaded %d directed network interactions",
        len(directed_interactions),
    )

    if skipped:

        logger.info(
            "Skipped %d invalid network interactions",
            skipped,
        )

    if not directed_interactions:

        return {}

    # -------------------------------------------------------------
    # AGGREGATE INTO CANONICAL ACCOUNT PAIRS
    # -------------------------------------------------------------

    raw_pairs = {}

    for (
        source_account_id,
        target_account_id,
    ), weight in directed_interactions.items():

        pair = canonical_pair(
            source_account_id,
            target_account_id,
        )

        if pair is None:

            continue

        if pair not in raw_pairs:

            raw_pairs[pair] = {
                "forward": 0.0,
                "reverse": 0.0,
            }

        account_a, account_b = pair

        if (
            source_account_id == account_a
            and target_account_id == account_b
        ):

            raw_pairs[pair]["forward"] += weight

        else:

            raw_pairs[pair]["reverse"] += weight

    logger.info(
        "Generated %d raw network interaction pairs",
        len(raw_pairs),
    )

    # -------------------------------------------------------------
    # FILTER WEAK INTERACTION PAIRS
    # -------------------------------------------------------------

    filtered_pairs = {}

    rejected = 0

    for pair, values in raw_pairs.items():

        total_weight = (
            values["forward"]
            + values["reverse"]
        )

        if total_weight < MIN_NETWORK_INTERACTIONS:

            rejected += 1
            continue

        filtered_pairs[pair] = values

    logger.info(
        "Rejected %d pairs below minimum network interaction "
        "threshold (%.1f)",
        rejected,
        MIN_NETWORK_INTERACTIONS,
    )

    if not filtered_pairs:

        return {}

    # -------------------------------------------------------------
    # CALCULATE LOG-SCALED INTERACTION VOLUME
    # -------------------------------------------------------------

    transformed_volumes = {}

    for pair, values in filtered_pairs.items():

        total_weight = (
            values["forward"]
            + values["reverse"]
        )

        transformed_volumes[pair] = math.log1p(
            total_weight
        )

    max_volume = max(
        transformed_volumes.values()
    )

    # -------------------------------------------------------------
    # CALCULATE NETWORK COORDINATION EVIDENCE
    # -------------------------------------------------------------

    network_pairs = {}

    rejected_scores = 0

    for pair, values in filtered_pairs.items():

        account_a, account_b = pair

        forward = values["forward"]

        reverse = values["reverse"]

        total_weight = (
            forward
            + reverse
        )

        # ---------------------------------------------------------
        # 1. INTERACTION VOLUME SCORE
        # ---------------------------------------------------------

        if max_volume > 0:

            volume_score = (
                transformed_volumes[pair]
                / max_volume
            )

        else:

            volume_score = 0.0

        # ---------------------------------------------------------
        # 2. RECIPROCITY SCORE
        # ---------------------------------------------------------
        #
        # Completely one-directional = 0
        # Perfectly balanced        = 1
        # ---------------------------------------------------------

        if total_weight > 0:

            reciprocity_score = (
                2.0
                * min(
                    forward,
                    reverse,
                )
                / total_weight
            )

        else:

            reciprocity_score = 0.0

        # ---------------------------------------------------------
        # 3. INTERACTION CONCENTRATION SCORE
        # ---------------------------------------------------------
        #
        # Measures how strongly the two accounts focus their
        # interaction activity toward each other.
        #
        # concentration_a:
        #
        #     pair interaction
        #     ----------------
        #     total activity of A
        #
        # concentration_b:
        #
        #     pair interaction
        #     ----------------
        #     total activity of B
        #
        # The geometric mean prevents one highly concentrated account
        # from dominating the score if the other account interacts
        # broadly with many others.
        # ---------------------------------------------------------

        activity_a = account_activity.get(
            account_a,
            0.0,
        )

        activity_b = account_activity.get(
            account_b,
            0.0,
        )

        concentration_a = (
            total_weight
            / activity_a
            if activity_a > 0
            else 0.0
        )

        concentration_b = (
            total_weight
            / activity_b
            if activity_b > 0
            else 0.0
        )

        concentration_a = max(
            0.0,
            min(
                concentration_a,
                1.0,
            ),
        )

        concentration_b = max(
            0.0,
            min(
                concentration_b,
                1.0,
            ),
        )

        concentration_score = math.sqrt(
            concentration_a
            * concentration_b
        )

        # ---------------------------------------------------------
        # COMBINED NETWORK SCORE
        # ---------------------------------------------------------

        network_score = (
            NETWORK_VOLUME_WEIGHT
            * volume_score
            +
            NETWORK_RECIPROCITY_WEIGHT
            * reciprocity_score
            +
            NETWORK_CONCENTRATION_WEIGHT
            * concentration_score
        )

        network_score = max(
            0.0,
            min(
                network_score,
                1.0,
            ),
        )

        # ---------------------------------------------------------
        # REMOVE WEAK NETWORK-ONLY SIGNALS
        # ---------------------------------------------------------

        if (
            network_score
            < MIN_NETWORK_COORDINATION_SCORE
        ):

            rejected_scores += 1
            continue

        network_pairs[pair] = {
            "network_score": float(network_score),
            "volume_score": float(volume_score),
            "reciprocity_score": float(reciprocity_score),
            "concentration_score": float(concentration_score),
        }

    logger.info(
        "Rejected %d weak network coordination pairs "
        "below score threshold (%.2f)",
        rejected_scores,
        MIN_NETWORK_COORDINATION_SCORE,
    )

    logger.info(
        "Generated %d network coordination evidence pairs",
        len(network_pairs),
    )

    return network_pairs


# ---------------------------------------------------------------------
# DETERMINE COORDINATION TYPE
# ---------------------------------------------------------------------

def determine_coordination_type(
    content_score,
    temporal_score,
    network_score,
):
    """
    Return the evidence types supporting a pair.
    """

    evidence_types = []

    if content_score > MIN_EVIDENCE_SCORE:
        evidence_types.append(
            "content"
        )

    if temporal_score > MIN_EVIDENCE_SCORE:
        evidence_types.append(
            "temporal"
        )

    if network_score > MIN_EVIDENCE_SCORE:
        evidence_types.append(
            "network"
        )

    if not evidence_types:
        return "none"

    return "+".join(
        evidence_types
    )


# ---------------------------------------------------------------------
# COMPUTE FINAL PAIR SCORE
# ---------------------------------------------------------------------

def compute_final_score(
    content_score,
    temporal_score,
    network_score,
):
    """
    Compute an evidence-aware coordination score.

    Scoring principles:

    1. Missing evidence contributes zero.
    2. Network-only evidence remains conservative.
    3. Temporal-only evidence is treated as moderate support.
    4. Strong near-duplicate content receives meaningful standalone
       support.
    5. Independent evidence sources reinforce each other.
    6. Multi-source coordination should be stronger than isolated
       evidence without allowing weak signals to become strong merely
       because they are present together.
    """

    # -------------------------------------------------------------
    # NORMALIZE INPUT SCORES
    # -------------------------------------------------------------

    content_score = max(
        0.0,
        min(
            float(content_score),
            1.0,
        ),
    )

    temporal_score = max(
        0.0,
        min(
            float(temporal_score),
            1.0,
        ),
    )

    network_score = max(
        0.0,
        min(
            float(network_score),
            1.0,
        ),
    )

    # -------------------------------------------------------------
    # DETECT AVAILABLE EVIDENCE SOURCES
    # -------------------------------------------------------------

    evidence_types = []

    if content_score > MIN_EVIDENCE_SCORE:
        evidence_types.append(
            "content"
        )

    if temporal_score > MIN_EVIDENCE_SCORE:
        evidence_types.append(
            "temporal"
        )

    if network_score > MIN_EVIDENCE_SCORE:
        evidence_types.append(
            "network"
        )

    evidence_count = len(
        evidence_types
    )

    if evidence_count == 0:
        return 0.0

    # -------------------------------------------------------------
    # WEIGHTED BASE SCORE
    # -------------------------------------------------------------
    #
    # Missing evidence remains zero.
    #
    # We deliberately do not renormalize the weights because a pair
    # supported by one evidence source should not receive the same
    # baseline confidence as a pair supported by multiple independent
    # evidence sources.
    # -------------------------------------------------------------

    base_score = (
        CONTENT_WEIGHT
        * content_score
        +
        TEMPORAL_WEIGHT
        * temporal_score
        +
        NETWORK_WEIGHT
        * network_score
    )

    # -------------------------------------------------------------
    # SINGLE EVIDENCE SOURCE
    # -------------------------------------------------------------

    if evidence_count == 1:

        if "network" in evidence_types:

            final_score = (
                base_score
                * NETWORK_ONLY_FACTOR
            )

            # Network-only evidence should remain conservative even
            # when the structural interaction score is high.
            final_score = min(
                final_score,
                MAX_NETWORK_ONLY_FINAL_SCORE,
            )

        elif "temporal" in evidence_types:

            final_score = (
                base_score
                * TEMPORAL_ONLY_FACTOR
            )

        else:

            final_score = (
                base_score
                * CONTENT_ONLY_FACTOR
            )

    # -------------------------------------------------------------
    # TWO INDEPENDENT EVIDENCE SOURCES
    # -------------------------------------------------------------

    elif evidence_count == 2:

        # Independent evidence convergence is stronger than either
        # signal alone. Apply both reinforcement and a fixed
        # convergence bonus.
        final_score = (
            base_score
            * TWO_EVIDENCE_FACTOR
            + TWO_EVIDENCE_BONUS
        )

    # -------------------------------------------------------------
    # THREE INDEPENDENT EVIDENCE SOURCES
    # -------------------------------------------------------------

    else:

        # All three evidence sources provide the strongest support.
        final_score = (
            base_score
            * THREE_EVIDENCE_FACTOR
            + THREE_EVIDENCE_BONUS
        )

    # -------------------------------------------------------------
    # FINAL NORMALIZATION
    # -------------------------------------------------------------

    return max(
        0.0,
        min(
            final_score,
            1.0,
        ),
    )


# ---------------------------------------------------------------------
# BUILD ACCOUNT PAIRS
# ---------------------------------------------------------------------

def build_account_pairs():
    """
    Aggregate all available evidence into account_pairs.
    """

    conn = get_conn()

    try:

        logger.info(
            "Building account-level pair evidence"
        )

        content_pairs = load_content_pairs(
            conn
        )

        temporal_pairs = load_temporal_pairs(
            conn
        )

        network_pairs = load_network_pairs(
            conn
        )

        all_pairs = set()

        all_pairs.update(
            content_pairs.keys()
        )

        all_pairs.update(
            temporal_pairs.keys()
        )

        all_pairs.update(
            network_pairs.keys()
        )

        if not all_pairs:

            logger.warning(
                "No account pair evidence found"
            )

            return

        # ---------------------------------------------------------
        # REBUILD ACCOUNT_PAIRS
        # ---------------------------------------------------------

        conn.execute(
            """
            DELETE FROM account_pairs
            """
        )

        saved_pairs = []

        for (
            source_account_id,
            target_account_id,
        ) in sorted(all_pairs):

            pair = (
                source_account_id,
                target_account_id,
            )

            content_score = content_pairs.get(
                pair,
                0.0,
            )

            temporal_score = temporal_pairs.get(
                pair,
                0.0,
            )

            network_evidence = network_pairs.get(
                pair,
                {}
            )

            network_score = float(
                network_evidence.get(
                    "network_score",
                    0.0,
                )
            )

            network_volume_score = float(
                network_evidence.get(
                    "volume_score",
                    0.0,
                )
            )

            network_reciprocity_score = float(
                network_evidence.get(
                    "reciprocity_score",
                    0.0,
                )
            )

            network_concentration_score = float(
                network_evidence.get(
                    "concentration_score",
                    0.0,
                )
            )

            final_score = compute_final_score(
                content_score,
                temporal_score,
                network_score,
            )

            coordination_type = (
                determine_coordination_type(
                    content_score,
                    temporal_score,
                    network_score,
                )
            )

            saved_pairs.append(
                (
                    source_account_id,
                    target_account_id,
                    content_score,
                    temporal_score,
                    network_score,
                    network_volume_score,
                    network_reciprocity_score,
                    network_concentration_score,
                    final_score,
                    coordination_type,
                )
            )

        conn.executemany(
            """
            INSERT INTO account_pairs (
                source_account_id,
                target_account_id,
                content_score,
                temporal_score,
                network_score,
                network_volume_score,
                network_reciprocity_score,
                network_concentration_score,
                final_score,
                coordination_type
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            saved_pairs,
        )

        conn.commit()

        logger.info(
            "Saved %d account pairs",
            len(saved_pairs),
        )

        # ---------------------------------------------------------
        # SUMMARY
        # ---------------------------------------------------------

        content_count = sum(
            1
            for pair in saved_pairs
            if pair[2] > 0
        )

        temporal_count = sum(
            1
            for pair in saved_pairs
            if pair[3] > 0
        )

        network_count = sum(
            1
            for pair in saved_pairs
            if pair[4] > 0
        )

        multi_evidence_count = sum(
            1
            for pair in saved_pairs
            if sum(
                score > 0
                for score in pair[2:5]
            ) >= 2
        )

        print()
        print("=" * 100)
        print("ACCOUNT PAIR AGGREGATION SUMMARY")
        print("=" * 100)

        print()
        print(
            f"Total account pairs: {len(saved_pairs)}"
        )

        print(
            f"Pairs with content evidence : {content_count}"
        )

        print(
            f"Pairs with temporal evidence: {temporal_count}"
        )

        print(
            f"Pairs with network evidence : {network_count}"
        )

        print(
            "Pairs with multiple evidence types: "
            f"{multi_evidence_count}"
        )

        print()
        print("TOP 10 PAIRS")
        print("-" * 100)

        top_pairs = sorted(
            saved_pairs,
            key=lambda row: row[8],
            reverse=True,
        )[:10]

        for row in top_pairs:

            print(
                f"{row[0]} <-> {row[1]}"
            )

            print(
                f"  content       : {row[2]:.4f}"
            )

            print(
                f"  temporal      : {row[3]:.4f}"
            )

            print(
                f"  network       : {row[4]:.4f}"
            )

            print(
                f"    volume      : {row[5]:.4f}"
            )

            print(
                f"    reciprocity : {row[6]:.4f}"
            )

            print(
                f"    concentration: {row[7]:.4f}"
            )

            print(
                f"  final         : {row[8]:.4f}"
            )

            print(
                f"  type          : {row[9]}"
            )

            print()

    finally:

        conn.close()


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

if __name__ == "__main__":

    build_account_pairs()