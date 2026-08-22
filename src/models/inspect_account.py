from __future__ import annotations

import argparse

from src.db import get_conn


LINE = "=" * 100
SMALL_LINE = "-" * 100


def print_section(title: str):
    print("\n" + LINE)
    print(f"  {title}")
    print(LINE)


def inspect_account(account_id: str):
    conn = get_conn()

    try:

        # ============================================================
        # ACCOUNT OVERVIEW
        # ============================================================

        print_section("ACCOUNT OVERVIEW")

        row = conn.execute(
            """
            SELECT
                a.id,
                a.username,
                a.created_utc,
                a.comment_karma,
                a.link_karma,
                a.total_posts,
                a.total_comments,

                s.anomaly_score,
                s.coord_score,
                s.temporal_score,
                s.dup_score,
                s.network_score,
                s.network_score_topic_scoped,
                s.influence_score,
                s.tier,
                s.cluster_id,
                s.evidence_status,
                s.confidence_level,
                s.assessment

            FROM accounts a

            LEFT JOIN scores s
                ON a.id = s.account_id

            WHERE a.id = ?
            """,
            (account_id,),
        ).fetchone()

        if row is None:
            print(f"\nAccount not found: {account_id}")
            return

        print(f"\nAccount ID       : {row['id']}")
        print(f"Username         : {row['username']}")
        print(f"Posts            : {row['total_posts']}")
        print(f"Comments         : {row['total_comments']}")
        print(f"Comment Karma    : {row['comment_karma']}")
        print(f"Link Karma       : {row['link_karma']}")

        print("\nDETECTION SCORES")
        print(SMALL_LINE)

        score_columns = [
            ("Anomaly Score", "anomaly_score"),
            ("Coordination Score", "coord_score"),
            ("Temporal Score", "temporal_score"),
            ("Duplicate Content Score", "dup_score"),
            ("Network Score", "network_score"),
            ("Topic Scoped Network Score", "network_score_topic_scoped"),
            ("Final Influence Score", "influence_score"),
        ]

        for label, column in score_columns:

            value = row[column]

            if value is None:
                value_text = "N/A"

            else:
                value_text = f"{float(value):.4f}"

            print(f"{label:<32}: {value_text}")

        print("\nCLASSIFICATION")
        print(SMALL_LINE)

        print(f"Tier              : {row['tier']}")
        print(f"Assessment        : {row['assessment']}")
        print(f"Evidence Status   : {row['evidence_status']}")
        print(f"Confidence Level  : {row['confidence_level']}")

        # ============================================================
        # DIRECT COORDINATION EVENTS
        # ============================================================

        print_section("DIRECT COORDINATION EVENTS")

        events = conn.execute(
            """
            SELECT
                source_account_id,
                target_account_id,
                source_post_id,
                target_post_id,
                event_type,
                similarity,
                event_time,
                source_content_type,
                target_content_type

            FROM coordination_events

            WHERE source_account_id = ?
               OR target_account_id = ?

            ORDER BY similarity DESC
            """,
            (account_id, account_id),
        ).fetchall()

        if not events:

            print("\nNo direct coordination events found.")

        else:

            print(f"\nTotal events: {len(events)}\n")

            for event in events[:20]:

                other_account = (
                    event["target_account_id"]
                    if event["source_account_id"] == account_id
                    else event["source_account_id"]
                )

                similarity = (
                    "N/A"
                    if event["similarity"] is None
                    else f"{float(event['similarity']):.4f}"
                )

                print(
                    f"{event['event_type']:<30} "
                    f"Other: {other_account} "
                    f"Score: {similarity}"
                )

        # ============================================================
        # ACCOUNT PAIRS
        # ============================================================

        print_section("ACCOUNT PAIR EVIDENCE")

        pairs = conn.execute(
            """
            SELECT
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

            FROM account_pairs

            WHERE source_account_id = ?
               OR target_account_id = ?

            ORDER BY final_score DESC
            """,
            (account_id, account_id),
        ).fetchall()

        if not pairs:

            print("\nNo account pair evidence found.")

        else:

            print(f"\nTotal relationships: {len(pairs)}\n")

            for pair in pairs[:20]:

                other_account = (
                    pair["target_account_id"]
                    if pair["source_account_id"] == account_id
                    else pair["source_account_id"]
                )

                print(
                    f"Partner: {other_account}"
                )

                print(
                    f"  Final Score      : "
                    f"{float(pair['final_score']):.4f}"
                )

                print(
                    f"  Type             : "
                    f"{pair['coordination_type']}"
                )

                for label, column in [
                    ("Content", "content_score"),
                    ("Temporal", "temporal_score"),
                    ("Network", "network_score"),
                    ("Network Volume", "network_volume_score"),
                    ("Network Reciprocity",
                     "network_reciprocity_score"),
                    ("Network Concentration",
                     "network_concentration_score"),
                ]:

                    value = pair[column]

                    value_text = (
                        "N/A"
                        if value is None
                        else f"{float(value):.4f}"
                    )

                    print(
                        f"  {label:<18}: "
                        f"{value_text}"
                    )

                print(SMALL_LINE)

        # ============================================================
        # COMMUNITY MEMBERSHIP
        # ============================================================

        print_section("COMMUNITY MEMBERSHIP")

        community = conn.execute(
            """
            SELECT
                community_id,
                centrality,
                pagerank,
                coordination_strength

            FROM communities

            WHERE account_id = ?
            """,
            (account_id,),
        ).fetchone()

        if community is None:

            print(
                "\nAccount is not part of a detected "
                "coordination community."
            )

        else:

            print(
                f"\nCommunity ID           : "
                f"{community['community_id']}"
            )

            print(
                f"Centrality             : "
                f"{float(community['centrality']):.4f}"
            )

            print(
                f"PageRank               : "
                f"{float(community['pagerank']):.4f}"
            )

            coordination_strength = community["coordination_strength"]

            if coordination_strength is None:
                coordination_strength_display = "N/A"
            else:
                coordination_strength_display = (
                    f"{float(coordination_strength):.4f}"
                )

            print(
                f"Coordination Strength  : "
                f"{coordination_strength_display}"
            )

            members = conn.execute(
                """
                SELECT
                    account_id,
                    coordination_strength

                FROM communities

                WHERE community_id = ?

                ORDER BY coordination_strength DESC
                """,
                (
                    community["community_id"],
                ),
            ).fetchall()

            print("\nCommunity Members:")
            print(SMALL_LINE)

            for member in members:

                marker = (
                    " <-- TARGET"
                    if member["account_id"] == account_id
                    else ""
                )

                print(
                    f"{member['account_id']:<25} "
                    f"Strength: "
                    f"{float(member['coordination_strength']):.4f}"
                    f"{marker}"
                )

        # ============================================================
        # NETWORK INTERACTIONS
        # ============================================================

        print_section("NETWORK INTERACTIONS")

        outgoing = conn.execute(
            """
            SELECT
                target_account_id,
                edge_type,
                weight

            FROM edges

            WHERE source_account_id = ?

            ORDER BY weight DESC
            LIMIT 20
            """,
            (account_id,),
        ).fetchall()

        incoming = conn.execute(
            """
            SELECT
                source_account_id,
                edge_type,
                weight

            FROM edges

            WHERE target_account_id = ?

            ORDER BY weight DESC
            LIMIT 20
            """,
            (account_id,),
        ).fetchall()

        print("\nOutgoing interactions:")

        if not outgoing:

            print("  None")

        else:

            for edge in outgoing:

                print(
                    f"  -> {edge['target_account_id']} "
                    f"| {edge['edge_type']} "
                    f"| weight={float(edge['weight']):.2f}"
                )

        print("\nIncoming interactions:")

        if not incoming:

            print("  None")

        else:

            for edge in incoming:

                print(
                    f"  <- {edge['source_account_id']} "
                    f"| {edge['edge_type']} "
                    f"| weight={float(edge['weight']):.2f}"
                )

        # ============================================================
        # FEATURE PROFILE
        # ============================================================

        print_section("BEHAVIORAL FEATURE PROFILE")

        features = conn.execute(
            """
            SELECT *

            FROM features

            WHERE account_id = ?
            """,
            (account_id,),
        ).fetchone()

        if features is None:

            print("\nNo feature profile found.")

        else:

            feature_names = [
                "age_days",
                "posts_per_day",
                "comments_per_day",
                "comment_ratio",
                "karma_score",
                "avg_score",
                "subreddit_count",
                "active_days",
                "hour_entropy",
                "duplicate_ratio",
                "avg_post_interval",
                "avg_comment_interval",
                "night_activity_ratio",
                "burstiness_score",
                "engagement_rate",
            ]

            for feature in feature_names:

                value = features[feature]

                if value is None:

                    value_text = "N/A"

                else:

                    value_text = f"{float(value):.4f}"

                print(
                    f"{feature:<28}: "
                    f"{value_text}"
                )

    finally:

        conn.close()


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Inspect all available evidence "
            "for a specific account."
        )
    )

    parser.add_argument(
        "account_id",
        help="Account ID to inspect",
    )

    args = parser.parse_args()

    inspect_account(
        args.account_id
    )


if __name__ == "__main__":

    main()