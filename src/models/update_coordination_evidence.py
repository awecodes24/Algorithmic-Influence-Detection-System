# src/models/update_coordination_evidence.py

from __future__ import annotations

import logging

from src.db import get_conn, init_db


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

STRONG_PAIR_THRESHOLD = 0.50
VERY_STRONG_PAIR_THRESHOLD = 0.65


# ---------------------------------------------------------------------
# COORDINATION AGGREGATION
# ---------------------------------------------------------------------


def get_account_coordination_summary(conn):
    """
    Aggregate account-level coordination evidence from account_pairs.

    Each account may appear as either:
        - source_account_id
        - target_account_id

    For every account we calculate:

        pair_count
        strong_pair_count
        very_strong_pair_count
        max_pair_score
        avg_pair_score
        evidence_types
    """

    cur = conn.cursor()

    cur.execute(
        """
        SELECT
            account_id,

            COUNT(*) AS pair_count,

            SUM(
                CASE
                    WHEN final_score >= ?
                    THEN 1
                    ELSE 0
                END
            ) AS strong_pair_count,

            SUM(
                CASE
                    WHEN final_score >= ?
                    THEN 1
                    ELSE 0
                END
            ) AS very_strong_pair_count,

            MAX(final_score) AS max_pair_score,

            AVG(final_score) AS avg_pair_score,

            MAX(content_score) AS max_content_score,

            MAX(temporal_score) AS max_temporal_score,

            MAX(network_score) AS max_network_score

        FROM
        (
            SELECT
                source_account_id AS account_id,
                content_score,
                temporal_score,
                network_score,
                final_score
            FROM account_pairs

            UNION ALL

            SELECT
                target_account_id AS account_id,
                content_score,
                temporal_score,
                network_score,
                final_score
            FROM account_pairs
        )

        GROUP BY account_id
        """,
        (
            STRONG_PAIR_THRESHOLD,
            VERY_STRONG_PAIR_THRESHOLD,
        ),
    )

    return cur.fetchall()


# ---------------------------------------------------------------------
# EVIDENCE STATUS
# ---------------------------------------------------------------------


def determine_evidence_status(
    pair_count: int,
    strong_pair_count: int,
    very_strong_pair_count: int,
    max_pair_score: float,
):
    """
    Determine how much direct coordination evidence exists.
    """

    if very_strong_pair_count >= 2:
        return "strong_support"

    if very_strong_pair_count >= 1:
        return "supported"

    if strong_pair_count >= 2:
        return "supported"

    if strong_pair_count == 1:
        return "weak_support"

    if pair_count >= 3 and max_pair_score >= 0.40:
        return "weak_support"

    return "no_direct_evidence"


# ---------------------------------------------------------------------
# CONFIDENCE LEVEL
# ---------------------------------------------------------------------


def determine_confidence_level(
    evidence_status: str,
    max_pair_score: float,
    pair_count: int,
):
    """
    Determine confidence in the coordination assessment.
    """

    if evidence_status == "strong_support" and max_pair_score >= 0.65:
        return "high"

    if evidence_status in {
        "strong_support",
        "supported",
    }:
        return "medium"

    if evidence_status == "weak_support" or pair_count >= 2:
        return "medium"

    return "low"


# ---------------------------------------------------------------------
# FINAL ACCOUNT ASSESSMENT
# ---------------------------------------------------------------------


def determine_assessment(
    tier: str,
    evidence_status: str,
):
    """
    Combine anomaly classification with pair-level coordination evidence.
    """

    if tier == "suspicious":
        if evidence_status == "strong_support":
            return "high_priority_coordinated_pattern"

        if evidence_status in {
            "supported",
            "weak_support",
        }:
            return "suspicious_with_coordination_evidence"

        return "suspicious"

    if tier == "organic":
        if evidence_status in {
            "strong_support",
            "supported",
        }:
            return "organic_with_coordination_pattern"

        return "likely_organic"

    return "unclassified"


# ---------------------------------------------------------------------
# UPDATE SCORES
# ---------------------------------------------------------------------


def update_coordination_evidence():
    """
    Update account-level evidence fields using account pair evidence.
    """

    init_db()

    conn = get_conn()
    cur = conn.cursor()

    summaries = get_account_coordination_summary(conn)

    logger.info(
        "Found %d accounts participating in coordination pairs",
        len(summaries),
    )

    updated = 0

    for row in summaries:
        account_id = row["account_id"]

        pair_count = int(row["pair_count"] or 0)

        strong_pair_count = int(row["strong_pair_count"] or 0)

        very_strong_pair_count = int(row["very_strong_pair_count"] or 0)

        max_pair_score = float(row["max_pair_score"] or 0.0)

        avg_pair_score = float(row["avg_pair_score"] or 0.0)

        evidence_status = determine_evidence_status(
            pair_count,
            strong_pair_count,
            very_strong_pair_count,
            max_pair_score,
        )

        confidence_level = determine_confidence_level(
            evidence_status,
            max_pair_score,
            pair_count,
        )

        cur.execute(
            """
            SELECT tier
            FROM scores
            WHERE account_id = ?
            """,
            (account_id,),
        )

        score_row = cur.fetchone()

        if score_row is None:
            continue

        tier = score_row["tier"]

        assessment = determine_assessment(
            tier,
            evidence_status,
        )

        cur.execute(
            """
            UPDATE scores

            SET
                confidence_level = ?,
                evidence_status = ?,
                assessment = ?

            WHERE account_id = ?
            """,
            (
                confidence_level,
                evidence_status,
                assessment,
                account_id,
            ),
        )

        updated += 1

        logger.debug(
            "Updated %s | pairs=%d | strong=%d | "
            "very_strong=%d | max=%.4f | avg=%.4f | "
            "evidence=%s | confidence=%s | assessment=%s",
            account_id,
            pair_count,
            strong_pair_count,
            very_strong_pair_count,
            max_pair_score,
            avg_pair_score,
            evidence_status,
            confidence_level,
            assessment,
        )

    conn.commit()

    logger.info(
        "Updated coordination evidence for %d accounts",
        updated,
    )

    conn.close()

    return updated


# ---------------------------------------------------------------------
# DIAGNOSTICS
# ---------------------------------------------------------------------


def print_summary():
    """
    Print a summary of the updated account assessments.
    """

    conn = get_conn()
    cur = conn.cursor()

    print()
    print("=" * 100)
    print("ACCOUNT-LEVEL COORDINATION EVIDENCE SUMMARY")
    print("=" * 100)

    cur.execute(
        """
        SELECT
            assessment,
            COUNT(*) AS count

        FROM scores

        GROUP BY assessment

        ORDER BY count DESC
        """
    )

    print()
    print("ASSESSMENT DISTRIBUTION")
    print("-" * 100)

    for row in cur.fetchall():
        assessment = row["assessment"] if row["assessment"] else "NULL"

        print(f"{assessment:<50} {row['count']}")

    cur.execute(
        """
        SELECT
            evidence_status,
            COUNT(*) AS count

        FROM scores

        GROUP BY evidence_status

        ORDER BY count DESC
        """
    )

    print()
    print("EVIDENCE STATUS DISTRIBUTION")
    print("-" * 100)

    for row in cur.fetchall():
        status = row["evidence_status"] if row["evidence_status"] else "NULL"

        print(f"{status:<50} {row['count']}")

    cur.execute(
        """
        SELECT
            s.account_id,
            a.username,
            s.tier,
            s.influence_score,
            s.evidence_status,
            s.confidence_level,
            s.assessment

        FROM scores s

        LEFT JOIN accounts a
            ON s.account_id = a.id

        WHERE s.assessment IN (
            'high_priority_coordinated_pattern',
            'suspicious_with_coordination_evidence'
        )

        ORDER BY
            CASE s.assessment
                WHEN 'high_priority_coordinated_pattern'
                    THEN 1
                ELSE 2
            END,
            s.influence_score DESC

        LIMIT 30
        """
    )

    rows = cur.fetchall()

    print()
    print("TOP PRIORITY ACCOUNTS")
    print("-" * 100)

    if not rows:
        print("No suspicious coordinated accounts found.")

    else:
        for row in rows:
            username = row["username"] if row["username"] else "unknown"

            print(
                f"{row['account_id']} | "
                f"{username:<20} | "
                f"tier={row['tier']:<12} | "
                f"influence={row['influence_score'] or 0:>6.2f} | "
                f"evidence={row['evidence_status']:<18} | "
                f"confidence={row['confidence_level']:<6} | "
                f"{row['assessment']}"
            )

    conn.close()

    print()
    print("=" * 100)


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------


def main():

    logger.info("Updating account-level coordination evidence")

    update_coordination_evidence()

    print_summary()


if __name__ == "__main__":
    main()
