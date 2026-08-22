"""
Evidence Inspector for Coordinated Influence Detection.

Provides an explainable view of why an account was assigned
a particular Influence Score and what coordination evidence
is associated with it.

Usage:

    python -m src.evidence_inspector ACCOUNT_ID

Example:

    python -m src.evidence_inspector 296029142755c020
"""

import sys
import logging

from src.db import get_conn


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def display_value(value, decimals=4):

    if value is None:
        return "N/A"

    try:
        return f"{float(value):.{decimals}f}"
    except (ValueError, TypeError):
        return str(value)


def print_section(title, width=100):

    print()
    print("━" * width)
    print(f"  {title}")
    print("━" * width)


# ---------------------------------------------------------------------
# ACCOUNT SUMMARY
# ---------------------------------------------------------------------

def get_account_summary(conn, account_id):

    query = """
    SELECT
        s.account_id,
        s.anomaly_score,
        s.coord_score,
        s.temporal_score,
        s.dup_score,
        s.network_score,
        s.influence_score,
        s.tier,
        s.confidence_level,
        s.evidence_status,
        s.scored_at,

        a.username,
        a.created_utc,
        a.comment_karma,
        a.link_karma,
        a.total_posts,
        a.total_comments

    FROM scores s

    LEFT JOIN accounts a
        ON s.account_id = a.id

    WHERE s.account_id = ?
    """

    return conn.execute(
        query,
        (account_id,),
    ).fetchone()


# ---------------------------------------------------------------------
# CONTENT SIMILARITY
# ---------------------------------------------------------------------

def get_content_similarity(conn, account_id):

    query = """
    SELECT
        source_account_id,
        target_account_id,
        similarity,
        method

    FROM content_similarity

    WHERE
        source_account_id = ?
        OR target_account_id = ?

    ORDER BY similarity DESC
    """

    return conn.execute(
        query,
        (account_id, account_id),
    ).fetchall()


# ---------------------------------------------------------------------
# TEMPORAL SIMILARITY
# ---------------------------------------------------------------------

def get_temporal_similarity(conn, account_id):

    query = """
    SELECT
        source_account_id,
        target_account_id,
        similarity,
        avg_time_diff

    FROM temporal_similarity

    WHERE
        source_account_id = ?
        OR target_account_id = ?

    ORDER BY similarity DESC,
             avg_time_diff ASC
    """

    return conn.execute(
        query,
        (account_id, account_id),
    ).fetchall()


# ---------------------------------------------------------------------
# COORDINATION EVENTS
# ---------------------------------------------------------------------

def get_coordination_events(conn, account_id):

    query = """
    SELECT
        id,
        source_account_id,
        target_account_id,
        source_post_id,
        target_post_id,
        event_type,
        similarity,
        event_time,
        created_at

    FROM coordination_events

    WHERE
        source_account_id = ?
        OR target_account_id = ?

    ORDER BY similarity DESC
    """

    return conn.execute(
        query,
        (account_id, account_id),
    ).fetchall()


# ---------------------------------------------------------------------
# ACCOUNT PAIRS
# ---------------------------------------------------------------------

def get_account_pairs(conn, account_id):

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

    WHERE
        source_account_id = ?
        OR target_account_id = ?

    ORDER BY final_score DESC
    """

    return conn.execute(
        query,
        (account_id, account_id),
    ).fetchall()


# ---------------------------------------------------------------------
# POSTS
# ---------------------------------------------------------------------

def get_recent_posts(conn, account_id, limit=10):

    query = """
    SELECT
        id,
        subreddit,
        title,
        text,
        score,
        created_utc,
        topic,
        sentiment

    FROM posts

    WHERE account_id = ?

    ORDER BY created_utc DESC

    LIMIT ?
    """

    return conn.execute(
        query,
        (account_id, limit),
    ).fetchall()


# ---------------------------------------------------------------------
# COMMENTS
# ---------------------------------------------------------------------

def get_recent_comments(conn, account_id, limit=10):

    query = """
    SELECT
        id,
        subreddit,
        post_id,
        parent_id,
        text,
        score,
        created_utc,
        topic,
        sentiment

    FROM comments

    WHERE account_id = ?

    ORDER BY created_utc DESC

    LIMIT ?
    """

    return conn.execute(
        query,
        (account_id, limit),
    ).fetchall()


# ---------------------------------------------------------------------
# RELATED ACCOUNT
# ---------------------------------------------------------------------

def related_account(account_id, row):

    source = row["source_account_id"]
    target = row["target_account_id"]

    if source == account_id:
        return target

    return source


# ---------------------------------------------------------------------
# DISPLAY SUMMARY
# ---------------------------------------------------------------------

def print_account_summary(row):

    print_section(
        "ACCOUNT SUMMARY"
    )

    print(f"Account ID       : {row['account_id']}")
    print(f"Username         : {row['username'] or 'N/A'}")

    print()

    print("CLASSIFICATION")

    print(
        f"  Influence Score : "
        f"{display_value(row['influence_score'], 2)}"
    )

    print(
        f"  Risk Tier       : "
        f"{row['tier'] or 'N/A'}"
    )

    print(
        f"  Confidence      : "
        f"{row['confidence_level'] or 'N/A'}"
    )

    print(
        f"  Evidence Status : "
        f"{row['evidence_status'] or 'N/A'}"
    )

    print()

    print("SIGNALS")

    signals = [
        ("Anomaly Score", row["anomaly_score"]),
        ("Behavior Score", row["coord_score"]),
        ("Temporal Score", row["temporal_score"]),
        ("Duplicate Score", row["dup_score"]),
        ("Network Score", row["network_score"]),
    ]

    for name, value in signals:

        print(
            f"  {name:<18}: "
            f"{display_value(value)}"
        )

    print()

    print("ACCOUNT ACTIVITY")

    print(
        f"  Total Posts     : "
        f"{row['total_posts'] or 0}"
    )

    print(
        f"  Total Comments  : "
        f"{row['total_comments'] or 0}"
    )

    print(
        f"  Comment Karma   : "
        f"{row['comment_karma'] or 0}"
    )

    print(
        f"  Link Karma      : "
        f"{row['link_karma'] or 0}"
    )


# ---------------------------------------------------------------------
# DISPLAY CONTENT EVIDENCE
# ---------------------------------------------------------------------

def print_content_evidence(account_id, rows):

    print_section(
        "CONTENT SIMILARITY EVIDENCE"
    )

    if not rows:

        print(
            "No direct content similarity evidence found."
        )

        return

    print(
        f"{'RELATED ACCOUNT':<22}"
        f"{'SIMILARITY':>15}"
        f"{'METHOD':>20}"
    )

    print("-" * 60)

    for row in rows:

        related = related_account(
            account_id,
            row,
        )

        print(
            f"{str(related)[:20]:<22}"
            f"{display_value(row['similarity']):>15}"
            f"{str(row['method'] or 'N/A'):>20}"
        )


# ---------------------------------------------------------------------
# DISPLAY TEMPORAL EVIDENCE
# ---------------------------------------------------------------------

def print_temporal_evidence(account_id, rows):

    print_section(
        "TEMPORAL SYNCHRONIZATION EVIDENCE"
    )

    if not rows:

        print(
            "No temporal similarity evidence found."
        )

        return

    print(
        f"{'RELATED ACCOUNT':<22}"
        f"{'SIMILARITY':>15}"
        f"{'AVG TIME DIFF (s)':>20}"
    )

    print("-" * 65)

    for row in rows:

        related = related_account(
            account_id,
            row,
        )

        print(
            f"{str(related)[:20]:<22}"
            f"{display_value(row['similarity']):>15}"
            f"{display_value(row['avg_time_diff'], 2):>20}"
        )


# ---------------------------------------------------------------------
# DISPLAY COORDINATION EVENTS
# ---------------------------------------------------------------------

def print_coordination_events(
    account_id,
    rows,
):

    print_section(
        "COORDINATION EVENTS"
    )

    if not rows:

        print(
            "No coordination events found."
        )

        return

    print(
        f"{'RELATED ACCOUNT':<22}"
        f"{'EVENT TYPE':<28}"
        f"{'EVIDENCE':>12}"
    )

    print("-" * 70)

    for row in rows:

        related = related_account(
            account_id,
            row,
        )

        print(
            f"{str(related)[:20]:<22}"
            f"{str(row['event_type'] or 'N/A'):<28}"
            f"{display_value(row['similarity']):>12}"
        )


# ---------------------------------------------------------------------
# DISPLAY ACCOUNT PAIRS
# ---------------------------------------------------------------------

def print_account_pairs(
    account_id,
    rows,
):

    print_section(
        "ACCOUNT PAIR ANALYSIS"
    )

    if not rows:

        print(
            "No account pair analysis found."
        )

        return

    print(
        f"{'RELATED ACCOUNT':<22}"
        f"{'CONTENT':>10}"
        f"{'TEMPORAL':>10}"
        f"{'NETWORK':>10}"
        f"{'FINAL':>10}"
    )

    print("-" * 75)

    for row in rows:

        related = related_account(
            account_id,
            row,
        )

        print(
            f"{str(related)[:20]:<22}"
            f"{display_value(row['content_score']):>10}"
            f"{display_value(row['temporal_score']):>10}"
            f"{display_value(row['network_score']):>10}"
            f"{display_value(row['final_score']):>10}"
        )


# ---------------------------------------------------------------------
# DISPLAY POSTS
# ---------------------------------------------------------------------

def print_recent_posts(rows):

    print_section(
        "RECENT POSTS"
    )

    if not rows:

        print(
            "No posts found."
        )

        return

    for index, row in enumerate(rows, start=1):

        print()

        print(
            f"[{index}] "
            f"Subreddit: {row['subreddit']}"
        )

        print(
            f"Title: {row['title'] or 'N/A'}"
        )

        text = row["text"] or ""

        if len(text) > 250:

            text = (
                text[:250]
                + "..."
            )

        print(
            f"Text: {text}"
        )

        print(
            f"Score: {row['score']}"
        )


# ---------------------------------------------------------------------
# DISPLAY COMMENTS
# ---------------------------------------------------------------------

def print_recent_comments(rows):

    print_section(
        "RECENT COMMENTS"
    )

    if not rows:

        print(
            "No comments found."
        )

        return

    for index, row in enumerate(rows, start=1):

        print()

        print(
            f"[{index}] "
            f"Subreddit: {row['subreddit']}"
        )

        text = row["text"] or ""

        if len(text) > 300:

            text = (
                text[:300]
                + "..."
            )

        print(
            f"Text: {text}"
        )

        print(
            f"Score: {row['score']}"
        )


# ---------------------------------------------------------------------
# FINAL INTERPRETATION
# ---------------------------------------------------------------------

def print_interpretation(summary):

    print_section(
        "INTERPRETATION"
    )

    print(
        "The Influence Score represents aggregated detection signals."
    )

    print()

    print(
        "Coordination evidence indicates measurable relationships "
        "between accounts, but should not independently be interpreted "
        "as proof of coordinated intent."
    )

    print()

    tier = summary["tier"]
    evidence = summary["evidence_status"]

    print(
        f"Current Risk Classification : {tier}"
    )

    print(
        f"Evidence Completeness       : {evidence}"
    )


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def inspect_account(account_id):

    conn = get_conn()

    try:

        summary = get_account_summary(
            conn,
            account_id,
        )

        if summary is None:

            print()

            print(
                f"Account not found: {account_id}"
            )

            return

        content_rows = get_content_similarity(
            conn,
            account_id,
        )

        temporal_rows = get_temporal_similarity(
            conn,
            account_id,
        )

        event_rows = get_coordination_events(
            conn,
            account_id,
        )

        pair_rows = get_account_pairs(
            conn,
            account_id,
        )

        posts = get_recent_posts(
            conn,
            account_id,
        )

        comments = get_recent_comments(
            conn,
            account_id,
        )

        print()

        print("█" * 100)

        print(
            "  COORDINATED INFLUENCE DETECTION"
        )

        print(
            "  ACCOUNT EVIDENCE INSPECTION REPORT"
        )

        print("█" * 100)

        print_account_summary(
            summary
        )

        print_content_evidence(
            account_id,
            content_rows,
        )

        print_temporal_evidence(
            account_id,
            temporal_rows,
        )

        print_coordination_events(
            account_id,
            event_rows,
        )

        print_account_pairs(
            account_id,
            pair_rows,
        )

        print_recent_posts(
            posts
        )

        print_recent_comments(
            comments
        )

        print_interpretation(
            summary
        )

        print()

        print("█" * 100)

    finally:

        conn.close()


if __name__ == "__main__":

    if len(sys.argv) != 2 or sys.argv[1] in [
        "-h",
        "--help",
    ]:

        print()

        print(
            "Usage:"
        )

        print()

        print(
            "python -m src.evidence_inspector ACCOUNT_ID"
        )

        print()

        print(
            "Examples:"
        )

        print()

        print(
            "python -m src.evidence_inspector "
            "2e2d9d61c8fe7c38"
        )

        print()

        print(
            "Options:"
        )

        print()

        print(
            "  -h, --help    Show this help message"
        )

        print()

        sys.exit(
            0
            if len(sys.argv) == 2
            else 1
        )

    account_id = sys.argv[1]

    inspect_account(
        account_id
    )