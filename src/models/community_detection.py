"""
Community Detection for Coordinated Influence Analysis

Builds coordination communities from account-level pair evidence.

Pipeline:
    account_pairs
        ->
    weighted coordination graph
        ->
    Louvain community detection
        ->
    community strength calculation
        ->
    communities table

Important principle:

A direct interaction alone does not necessarily indicate coordinated
behavior. Therefore, the graph is built primarily from account pairs
with converging evidence sources such as:

    temporal + network
    content + temporal
    content + network
    content + temporal + network

A strong exact-content duplicate can also be included as a special
single-evidence case.

Weak network-only relationships are excluded to avoid interpreting
ordinary Reddit interaction patterns as coordination communities.
"""

from __future__ import annotations

import logging
from collections import defaultdict

import networkx as nx

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

# Minimum final coordination score for an account pair to become
# an edge in the coordination graph.
MIN_COMMUNITY_PAIR_SCORE = 0.30


# Strong single-evidence content pairs may enter the graph if their
# content similarity is extremely high.
STRONG_CONTENT_THRESHOLD = 0.95


# Ignore communities smaller than this.
MIN_COMMUNITY_SIZE = 2


# ---------------------------------------------------------------------
# PAIR ELIGIBILITY
# ---------------------------------------------------------------------

def count_evidence_sources(
    content_score,
    temporal_score,
    network_score,
):
    """
    Count how many evidence sources support an account pair.
    """

    scores = [
        content_score,
        temporal_score,
        network_score,
    ]

    return sum(
        score > 0
        for score in scores
    )


def is_pair_eligible(
    content_score,
    temporal_score,
    network_score,
    final_score,
):
    """
    Determine whether an account pair should become an edge in the
    coordination community graph.

    Primary rule:
        - Require at least two evidence sources.

    Exception:
        - Allow extremely strong content-only evidence.

    Network-only relationships are deliberately excluded because
    interaction structure alone is not sufficient evidence of
    coordinated behavior.
    """

    evidence_count = count_evidence_sources(
        content_score,
        temporal_score,
        network_score,
    )

    # -------------------------------------------------------------
    # MULTI-SOURCE EVIDENCE
    # -------------------------------------------------------------

    if evidence_count >= 2:

        return (
            final_score
            >= MIN_COMMUNITY_PAIR_SCORE
        )

    # -------------------------------------------------------------
    # STRONG CONTENT-ONLY EVIDENCE
    # -------------------------------------------------------------

    if (
        evidence_count == 1
        and content_score >= STRONG_CONTENT_THRESHOLD
    ):

        return True

    return False


# ---------------------------------------------------------------------
# LOAD ELIGIBLE ACCOUNT PAIRS
# ---------------------------------------------------------------------

def load_coordination_pairs(
    conn,
):
    """
    Load account pairs and retain only relationships suitable for
    community detection.
    """

    query = """
        SELECT
            source_account_id,
            target_account_id,
            content_score,
            temporal_score,
            network_score,
            final_score,
            coordination_type
        FROM account_pairs
        WHERE final_score > 0
    """

    rows = conn.execute(
        query
    ).fetchall()

    eligible_pairs = []

    rejected = 0

    for row in rows:

        source_account_id = str(
            row["source_account_id"]
        )

        target_account_id = str(
            row["target_account_id"]
        )

        content_score = float(
            row["content_score"]
            or 0.0
        )

        temporal_score = float(
            row["temporal_score"]
            or 0.0
        )

        network_score = float(
            row["network_score"]
            or 0.0
        )

        final_score = float(
            row["final_score"]
            or 0.0
        )

        if not is_pair_eligible(
            content_score,
            temporal_score,
            network_score,
            final_score,
        ):

            rejected += 1
            continue

        eligible_pairs.append(
            {
                "source_account_id": source_account_id,
                "target_account_id": target_account_id,
                "content_score": content_score,
                "temporal_score": temporal_score,
                "network_score": network_score,
                "final_score": final_score,
                "coordination_type": row[
                    "coordination_type"
                ],
            }
        )

    logger.info(
        "Loaded %d eligible coordination pairs",
        len(eligible_pairs),
    )

    logger.info(
        "Rejected %d weak or insufficient-evidence pairs",
        rejected,
    )

    return eligible_pairs


# ---------------------------------------------------------------------
# BUILD COORDINATION GRAPH
# ---------------------------------------------------------------------

def build_coordination_graph(
    pairs,
):
    """
    Build an undirected weighted graph.

    Each node:
        Reddit account

    Each edge:
        Evidence-supported coordination relationship

    Edge weight:
        final coordination score
    """

    graph = nx.Graph()

    for pair in pairs:

        source = pair[
            "source_account_id"
        ]

        target = pair[
            "target_account_id"
        ]

        weight = pair[
            "final_score"
        ]

        if source == target:

            continue

        graph.add_edge(
            source,
            target,
            weight=weight,
            content_score=pair[
                "content_score"
            ],
            temporal_score=pair[
                "temporal_score"
            ],
            network_score=pair[
                "network_score"
            ],
            coordination_type=pair[
                "coordination_type"
            ],
        )

    logger.info(
        "Coordination graph: %d accounts, %d relationships",
        graph.number_of_nodes(),
        graph.number_of_edges(),
    )

    return graph


# ---------------------------------------------------------------------
# DETECT COMMUNITIES
# ---------------------------------------------------------------------

def detect_communities(
    graph,
):
    """
    Detect coordination communities using Louvain community detection.

    Louvain identifies groups of accounts that are more strongly
    connected to each other than to the rest of the graph.
    """

    if graph.number_of_nodes() == 0:

        logger.warning(
            "Coordination graph is empty"
        )

        return []

    if graph.number_of_edges() == 0:

        logger.warning(
            "Coordination graph has no edges"
        )

        return []

    try:

        communities = nx.community.louvain_communities(
            graph,
            weight="weight",
            seed=42,
        )

    except AttributeError:

        logger.warning(
            "NetworkX Louvain implementation unavailable. "
            "Falling back to greedy modularity communities."
        )

        communities = list(
            nx.community.greedy_modularity_communities(
                graph,
                weight="weight",
            )
        )

    communities = [
        set(community)
        for community in communities
        if len(community)
        >= MIN_COMMUNITY_SIZE
    ]

    communities.sort(
        key=lambda community: (
            -len(community),
            sorted(community)[0],
        )
    )

    logger.info(
        "Detected %d coordination communities",
        len(communities),
    )

    return communities


# ---------------------------------------------------------------------
# COMMUNITY STRENGTH
# ---------------------------------------------------------------------

def calculate_community_strength(
    graph,
    community,
):
    """
    Calculate the coordination strength of one community.

    The strength is based on the average weight of internal
    coordination edges.

        community_strength =
            sum(internal edge weights)
            --------------------------
            number of internal edges

    Returns a value approximately in [0, 1].
    """

    subgraph = graph.subgraph(
        community
    )

    internal_edges = list(
        subgraph.edges(
            data=True
        )
    )

    if not internal_edges:

        return 0.0

    total_weight = sum(
        float(
            data.get(
                "weight",
                0.0,
            )
        )
        for _, _, data in internal_edges
    )

    strength = (
        total_weight
        / len(internal_edges)
    )

    return max(
        0.0,
        min(
            strength,
            1.0,
        ),
    )


# ---------------------------------------------------------------------
# ACCOUNT-LEVEL COORDINATION STRENGTH
# ---------------------------------------------------------------------

def calculate_account_strengths(
    graph,
    community,
):
    """
    Calculate coordination strength for every account inside a
    community.

    An account's score is the weighted average strength of its
    coordination relationships within that community.
    """

    community_set = set(
        community
    )

    strengths = {}

    for account_id in community:

        internal_weights = []

        for neighbor in graph.neighbors(
            account_id
        ):

            if neighbor not in community_set:

                continue

            edge_data = graph.get_edge_data(
                account_id,
                neighbor,
            )

            weight = float(
                edge_data.get(
                    "weight",
                    0.0,
                )
            )

            internal_weights.append(
                weight
            )

        if internal_weights:

            strength = (
                sum(internal_weights)
                / len(internal_weights)
            )

        else:

            strength = 0.0

        strengths[
            account_id
        ] = max(
            0.0,
            min(
                strength,
                1.0,
            ),
        )

    return strengths


# ---------------------------------------------------------------------
# SAVE COMMUNITIES
# ---------------------------------------------------------------------

def save_communities(
    graph,
    communities,
):
    """
    Save detected community assignments and coordination strengths.

    Existing PageRank and centrality values in the communities table
    are preserved.

    Only:
        community_id
        coordination_strength

    are updated here.
    """

    conn = get_conn()

    try:

        cursor = conn.cursor()

        # ---------------------------------------------------------
        # CLEAR OLD COMMUNITY ASSIGNMENTS
        # ---------------------------------------------------------

        cursor.execute(
            """
            UPDATE communities
            SET
                community_id = NULL,
                coordination_strength = NULL
            """
        )

        # ---------------------------------------------------------
        # SAVE NEW COMMUNITIES
        # ---------------------------------------------------------

        saved_accounts = 0

        for index, community in enumerate(
            communities,
            start=1,
        ):

            community_id = f"community_{index:03d}"

            community_strength = (
                calculate_community_strength(
                    graph,
                    community,
                )
            )

            account_strengths = (
                calculate_account_strengths(
                    graph,
                    community,
                )
            )

            logger.info(
                "%s | accounts=%d | "
                "community_strength=%.4f",
                community_id,
                len(community),
                community_strength,
            )

            for account_id in community:

                account_strength = (
                    account_strengths.get(
                        account_id,
                        community_strength,
                    )
                )

                cursor.execute(
                    """
                    INSERT INTO communities (
                        account_id,
                        community_id,
                        coordination_strength
                    )
                    VALUES (?, ?, ?)

                    ON CONFLICT(account_id)
                    DO UPDATE SET

                        community_id =
                            excluded.community_id,

                        coordination_strength =
                            excluded.coordination_strength
                    """,
                    (
                        account_id,
                        community_id,
                        account_strength,
                    ),
                )

                saved_accounts += 1

        conn.commit()

        logger.info(
            "Saved %d coordinated accounts",
            saved_accounts,
        )

    finally:

        conn.close()


# ---------------------------------------------------------------------
# PRINT SUMMARY
# ---------------------------------------------------------------------

def print_summary(
    graph,
    communities,
):
    """
    Print detected communities and their strongest relationships.
    """

    print()

    print(
        "=" * 100
    )

    print(
        "COORDINATION COMMUNITY DETECTION SUMMARY"
    )

    print(
        "=" * 100
    )

    print()

    print(
        f"Accounts in coordination graph : "
        f"{graph.number_of_nodes()}"
    )

    print(
        f"Coordination relationships     : "
        f"{graph.number_of_edges()}"
    )

    print(
        f"Communities detected           : "
        f"{len(communities)}"
    )

    print()

    if not communities:

        print(
            "No coordination communities were detected."
        )

        return

    print(
        "DETECTED COMMUNITIES"
    )

    print(
        "-" * 100
    )

    for index, community in enumerate(
        communities,
        start=1,
    ):

        community_id = (
            f"community_{index:03d}"
        )

        strength = (
            calculate_community_strength(
                graph,
                community,
            )
        )

        print()

        print(
            f"{community_id}"
        )

        print(
            f"  Accounts : {len(community)}"
        )

        print(
            f"  Strength : {strength:.4f}"
        )

        print(
            "  Members  : "
            + ", ".join(
                sorted(community)
            )
        )

        # Show internal edges.

        subgraph = graph.subgraph(
            community
        )

        edges = sorted(
            subgraph.edges(
                data=True
            ),
            key=lambda edge: float(
                edge[2].get(
                    "weight",
                    0.0,
                )
            ),
            reverse=True,
        )

        if edges:

            print(
                "  Strongest relationships:"
            )

            for (
                source,
                target,
                data,
            ) in edges[:5]:

                print(
                    f"    {source} <-> {target} | "
                    f"score="
                    f"{float(data['weight']):.4f} | "
                    f"type="
                    f"{data.get('coordination_type')}"
                )

    print()

    print(
        "=" * 100
    )


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():

    print()

    print(
        "=" * 100
    )

    print(
        "COORDINATION COMMUNITY DETECTION"
    )

    print(
        "=" * 100
    )

    conn = get_conn()

    try:

        pairs = load_coordination_pairs(
            conn
        )

    finally:

        conn.close()

    if not pairs:

        logger.warning(
            "No eligible coordination pairs found."
        )

        return

    graph = build_coordination_graph(
        pairs
    )

    communities = detect_communities(
        graph
    )

    save_communities(
        graph,
        communities,
    )

    print_summary(
        graph,
        communities,
    )


if __name__ == "__main__":

    main()