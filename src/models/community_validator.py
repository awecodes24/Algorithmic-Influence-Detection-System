"""
Community Validation for Coordinated Influence Analysis

Validates detected coordination communities by examining the
underlying account-pair evidence.

The validator does not attempt to prove that a community represents
malicious or intentional coordinated influence.

Instead, it classifies the strength and composition of the available
evidence.

Evidence considered:

    1. Content similarity
    2. Temporal synchronization
    3. Network coordination
    4. Multi-source evidence convergence
    5. Community structural density
    6. Community coordination strength

Output:

    - Community-level validation summary
    - Evidence composition
    - Confidence classification
    - Account-level coordination involvement

Important principle:

Detected communities represent evidence-supported candidate
coordination groups, not confirmed coordinated influence operations.
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

# Pair score considered meaningful inside an already detected
# coordination community.
MIN_MEANINGFUL_PAIR_SCORE = 0.30


# Confidence thresholds.

HIGH_CONFIDENCE_SCORE = 0.70
MODERATE_CONFIDENCE_SCORE = 0.40


# Strong exact or near-exact content duplication.

STRONG_CONTENT_THRESHOLD = 0.95


# ---------------------------------------------------------------------
# LOAD COMMUNITIES
# ---------------------------------------------------------------------

def load_communities(conn):
    """
    Load community assignments.

    Returns
    -------
    dict

        {
            community_id: {
                account_id,
                ...
            }
        }
    """

    query = """
        SELECT
            account_id,
            community_id,
            coordination_strength
        FROM communities
        WHERE community_id IS NOT NULL
    """

    rows = conn.execute(
        query
    ).fetchall()

    communities = defaultdict(set)

    for row in rows:

        account_id = str(
            row["account_id"]
        )

        community_id = str(
            row["community_id"]
        )

        communities[
            community_id
        ].add(
            account_id
        )

    logger.info(
        "Loaded %d communities",
        len(communities),
    )

    return dict(
        communities
    )


# ---------------------------------------------------------------------
# LOAD PAIR EVIDENCE
# ---------------------------------------------------------------------

def load_pair_evidence(conn):
    """
    Load all positive account-pair evidence.

    Returns a dictionary indexed by canonical account pairs.
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

    pairs = {}

    for row in rows:

        source = str(
            row["source_account_id"]
        )

        target = str(
            row["target_account_id"]
        )

        if source == target:

            continue

        pair = tuple(
            sorted(
                (
                    source,
                    target,
                )
            )
        )

        pairs[pair] = {
            "content_score": float(
                row["content_score"]
                or 0.0
            ),
            "temporal_score": float(
                row["temporal_score"]
                or 0.0
            ),
            "network_score": float(
                row["network_score"]
                or 0.0
            ),
            "final_score": float(
                row["final_score"]
                or 0.0
            ),
            "coordination_type": row[
                "coordination_type"
            ],
        }

    logger.info(
        "Loaded %d account-pair evidence records",
        len(pairs),
    )

    return pairs


# ---------------------------------------------------------------------
# BUILD COMMUNITY GRAPH
# ---------------------------------------------------------------------

def build_community_graph(
    accounts,
    pair_evidence,
):
    """
    Build an internal evidence graph for one community.
    """

    graph = nx.Graph()

    graph.add_nodes_from(
        accounts
    )

    account_list = sorted(
        accounts
    )

    for i, source in enumerate(
        account_list
    ):

        for target in account_list[
            i + 1:
        ]:

            pair = (
                source,
                target,
            )

            evidence = pair_evidence.get(
                pair
            )

            if evidence is None:

                continue

            if (
                evidence["final_score"]
                < MIN_MEANINGFUL_PAIR_SCORE
            ):

                continue

            graph.add_edge(
                source,
                target,
                **evidence,
            )

    return graph


# ---------------------------------------------------------------------
# EVIDENCE DISTRIBUTION
# ---------------------------------------------------------------------

def calculate_evidence_distribution(
    graph,
):
    """
    Count the evidence types represented by internal community edges.
    """

    distribution = {
        "content": 0,
        "temporal": 0,
        "network": 0,
        "multi_source": 0,
        "strong_content": 0,
    }

    for _, _, data in graph.edges(
        data=True
    ):

        evidence_count = 0

        content_score = float(
            data.get(
                "content_score",
                0.0,
            )
        )

        temporal_score = float(
            data.get(
                "temporal_score",
                0.0,
            )
        )

        network_score = float(
            data.get(
                "network_score",
                0.0,
            )
        )

        if content_score > 0:

            distribution[
                "content"
            ] += 1

            evidence_count += 1

            if (
                content_score
                >= STRONG_CONTENT_THRESHOLD
            ):

                distribution[
                    "strong_content"
                ] += 1

        if temporal_score > 0:

            distribution[
                "temporal"
            ] += 1

            evidence_count += 1

        if network_score > 0:

            distribution[
                "network"
            ] += 1

            evidence_count += 1

        if evidence_count >= 2:

            distribution[
                "multi_source"
            ] += 1

    return distribution


# ---------------------------------------------------------------------
# COMMUNITY METRICS
# ---------------------------------------------------------------------

def calculate_community_metrics(
    graph,
):
    """
    Calculate structural and evidence metrics for a community.
    """

    account_count = (
        graph.number_of_nodes()
    )

    edge_count = (
        graph.number_of_edges()
    )

    if edge_count == 0:

        return {
            "account_count": account_count,
            "edge_count": 0,
            "average_score": 0.0,
            "max_score": 0.0,
            "density": 0.0,
        }

    weights = [
        float(
            data.get(
                "final_score",
                0.0,
            )
        )
        for _, _, data
        in graph.edges(
            data=True
        )
    ]

    average_score = (
        sum(weights)
        / len(weights)
    )

    max_score = max(
        weights
    )

    density = nx.density(
        graph
    )

    return {
        "account_count": account_count,
        "edge_count": edge_count,
        "average_score": max(
            0.0,
            min(
                average_score,
                1.0,
            ),
        ),
        "max_score": max(
            0.0,
            min(
                max_score,
                1.0,
            ),
        ),
        "density": max(
            0.0,
            min(
                density,
                1.0,
            ),
        ),
    }


# ---------------------------------------------------------------------
# CONFIDENCE SCORE
# ---------------------------------------------------------------------

def calculate_validation_score(
    metrics,
    evidence_distribution,
):
    """
    Calculate a conservative community validation score.

    The score considers:

        - average coordination strength
        - graph density
        - evidence convergence
        - strong content duplication

    The score is not interpreted as probability.
    """

    average_score = metrics[
        "average_score"
    ]

    density = metrics[
        "density"
    ]

    edge_count = metrics[
        "edge_count"
    ]

    if edge_count == 0:

        return 0.0

    multi_source_ratio = (
        evidence_distribution[
            "multi_source"
        ]
        / edge_count
    )

    strong_content_ratio = (
        evidence_distribution[
            "strong_content"
        ]
        / edge_count
    )

    validation_score = (

        0.40
        * average_score

        +

        0.20
        * density

        +

        0.25
        * multi_source_ratio

        +

        0.15
        * strong_content_ratio
    )

    return max(
        0.0,
        min(
            validation_score,
            1.0,
        ),
    )


# ---------------------------------------------------------------------
# CONFIDENCE CLASSIFICATION
# ---------------------------------------------------------------------

def classify_confidence(
    validation_score,
    evidence_distribution,
):
    """
    Assign a conservative evidence confidence classification.

    High confidence requires either:

        - substantial multi-source convergence, or
        - strong content evidence combined with a high score.

    This prevents a dense interaction structure alone from being
    classified as high-confidence coordination.
    """

    multi_source = (
        evidence_distribution[
            "multi_source"
        ]
    )

    strong_content = (
        evidence_distribution[
            "strong_content"
        ]
    )

    if (
        validation_score
        >= HIGH_CONFIDENCE_SCORE
        and (
            multi_source > 0
            or strong_content > 0
        )
    ):

        return "high"

    if (
        validation_score
        >= MODERATE_CONFIDENCE_SCORE
        and (
            multi_source > 0
            or strong_content > 0
        )
    ):

        return "moderate"

    return "low"


# ---------------------------------------------------------------------
# ACCOUNT-LEVEL INVOLVEMENT
# ---------------------------------------------------------------------

def calculate_account_involvement(
    graph,
):
    """
    Calculate coordination involvement for each account.

    The score is based on the average strength of the account's
    evidence-supported relationships inside its community.
    """

    involvement = {}

    for account_id in graph.nodes():

        weights = []

        for neighbor in graph.neighbors(
            account_id
        ):

            data = graph.get_edge_data(
                account_id,
                neighbor,
            )

            weight = float(
                data.get(
                    "final_score",
                    0.0,
                )
            )

            weights.append(
                weight
            )

        if weights:

            score = (
                sum(weights)
                / len(weights)
            )

        else:

            score = 0.0

        involvement[
            account_id
        ] = max(
            0.0,
            min(
                score,
                1.0,
            ),
        )

    return involvement


# ---------------------------------------------------------------------
# VALIDATE COMMUNITIES
# ---------------------------------------------------------------------

def validate_communities(
    communities,
    pair_evidence,
):
    """
    Validate all detected coordination communities.
    """

    results = []

    for (
        community_id,
        accounts,
    ) in sorted(
        communities.items()
    ):

        graph = build_community_graph(
            accounts,
            pair_evidence,
        )

        metrics = (
            calculate_community_metrics(
                graph
            )
        )

        evidence_distribution = (
            calculate_evidence_distribution(
                graph
            )
        )

        validation_score = (
            calculate_validation_score(
                metrics,
                evidence_distribution,
            )
        )

        confidence = (
            classify_confidence(
                validation_score,
                evidence_distribution,
            )
        )

        account_involvement = (
            calculate_account_involvement(
                graph
            )
        )

        results.append(
            {
                "community_id": community_id,
                "accounts": sorted(
                    accounts
                ),
                "metrics": metrics,
                "evidence_distribution": (
                    evidence_distribution
                ),
                "validation_score": validation_score,
                "confidence": confidence,
                "account_involvement": (
                    account_involvement
                ),
            }
        )

    return results


# ---------------------------------------------------------------------
# PRINT SUMMARY
# ---------------------------------------------------------------------

def print_summary(
    results,
):
    """
    Print community validation results.
    """

    print()

    print(
        "=" * 100
    )

    print(
        "COORDINATION COMMUNITY VALIDATION"
    )

    print(
        "=" * 100
    )

    if not results:

        print()

        print(
            "No communities available for validation."
        )

        return

    for result in results:

        metrics = result[
            "metrics"
        ]

        evidence = result[
            "evidence_distribution"
        ]

        print()

        print(
            result[
                "community_id"
            ]
        )

        print(
            "-" * 100
        )

        print(
            f"Accounts           : "
            f"{metrics['account_count']}"
        )

        print(
            f"Relationships      : "
            f"{metrics['edge_count']}"
        )

        print(
            f"Average pair score : "
            f"{metrics['average_score']:.4f}"
        )

        print(
            f"Maximum pair score : "
            f"{metrics['max_score']:.4f}"
        )

        print(
            f"Graph density      : "
            f"{metrics['density']:.4f}"
        )

        print()

        print(
            "Evidence distribution:"
        )

        print(
            f"  Content          : "
            f"{evidence['content']}"
        )

        print(
            f"  Temporal         : "
            f"{evidence['temporal']}"
        )

        print(
            f"  Network          : "
            f"{evidence['network']}"
        )

        print(
            f"  Multi-source     : "
            f"{evidence['multi_source']}"
        )

        print(
            f"  Strong content   : "
            f"{evidence['strong_content']}"
        )

        print()

        print(
            f"Validation score   : "
            f"{result['validation_score']:.4f}"
        )

        print(
            f"Confidence         : "
            f"{result['confidence'].upper()}"
        )

        print()

        print(
            "Account involvement:"
        )

        sorted_accounts = sorted(
            result[
                "account_involvement"
            ].items(),
            key=lambda item: (
                item[1],
                item[0],
            ),
            reverse=True,
        )

        for (
            account_id,
            involvement,
        ) in sorted_accounts:

            print(
                f"  {account_id} | "
                f"involvement="
                f"{involvement:.4f}"
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
        "COMMUNITY VALIDATION"
    )

    print(
        "=" * 100
    )

    conn = get_conn()

    try:

        communities = load_communities(
            conn
        )

        pair_evidence = load_pair_evidence(
            conn
        )

    finally:

        conn.close()

    if not communities:

        logger.warning(
            "No detected communities available."
        )

        return

    results = validate_communities(
        communities,
        pair_evidence,
    )

    print_summary(
        results
    )


if __name__ == "__main__":

    main()