import pandas as pd

from src.db import get_conn


def main():

    conn = get_conn()

    try:

        pairs = pd.read_sql(
            """
            SELECT
                source_account_id,
                target_account_id,
                content_score,
                temporal_score,
                network_score,
                final_score,
                coordination_type
            FROM account_pairs
            """,
            conn,
        )

        events = pd.read_sql(
            """
            SELECT
                source_account_id,
                target_account_id,
                event_type,
                similarity,
                event_time
            FROM coordination_events
            """,
            conn,
        )

        scores = pd.read_sql(
            """
            SELECT
                account_id,
                influence_score,
                tier,
                confidence_level,
                evidence_status,
                assessment
            FROM scores
            """,
            conn,
        )

    finally:

        conn.close()

    print("\n" + "=" * 100)
    print("ACCOUNT PAIR AND COORDINATION EVIDENCE DIAGNOSTICS")
    print("=" * 100)

    # ---------------------------------------------------------
    # BASIC COUNTS
    # ---------------------------------------------------------

    print("\n" + "-" * 100)
    print("DATASET SUMMARY")
    print("-" * 100)

    print(f"Total account pairs          : {len(pairs)}")
    print(f"Total coordination events    : {len(events)}")

    if pairs.empty:

        print("\nNo account pairs found.")
        return

    unique_accounts = set(
        pairs["source_account_id"].dropna()
    ).union(
        set(
            pairs["target_account_id"].dropna()
        )
    )

    print(
        f"Unique accounts in pairs     : {len(unique_accounts)}"
    )

    # ---------------------------------------------------------
    # PAIR SCORE DISTRIBUTION
    # ---------------------------------------------------------

    print("\n" + "-" * 100)
    print("PAIR SCORE DISTRIBUTION")
    print("-" * 100)

    score_columns = [
        "content_score",
        "temporal_score",
        "network_score",
        "final_score",
    ]

    for column in score_columns:

        if column not in pairs.columns:

            continue

        series = pairs[column]

        available = int(
            series.notna().sum()
        )

        positive = int(
            (series > 0).sum()
        )

        print(f"\n{column}")

        print(f"  Available : {available}")
        print(f"  Positive  : {positive}")

        if available > 0:

            print(f"  Min       : {series.min():.4f}")
            print(f"  Mean      : {series.mean():.4f}")
            print(f"  Median    : {series.median():.4f}")
            print(f"  Max       : {series.max():.4f}")

    # ---------------------------------------------------------
    # COORDINATION TYPES
    # ---------------------------------------------------------

    print("\n" + "-" * 100)
    print("COORDINATION TYPE DISTRIBUTION")
    print("-" * 100)

    if (
        "coordination_type" in pairs.columns
    ):

        distribution = (
            pairs["coordination_type"]
            .fillna("unknown")
            .value_counts()
        )

        for name, count in distribution.items():

            percentage = (
                count / len(pairs) * 100
            )

            print(
                f"{str(name):<40}"
                f"{count:>6} "
                f"({percentage:.1f}%)"
            )

    # ---------------------------------------------------------
    # EVENT TYPE DISTRIBUTION
    # ---------------------------------------------------------

    print("\n" + "-" * 100)
    print("COORDINATION EVENT TYPES")
    print("-" * 100)

    if events.empty:

        print("No coordination events found.")

    else:

        event_distribution = (
            events["event_type"]
            .fillna("unknown")
            .value_counts()
        )

        for event_type, count in event_distribution.items():

            percentage = (
                count / len(events) * 100
            )

            print(
                f"{str(event_type):<40}"
                f"{count:>6} "
                f"({percentage:.1f}%)"
            )

    # ---------------------------------------------------------
    # EVENTS PER ACCOUNT PAIR
    # ---------------------------------------------------------

    print("\n" + "-" * 100)
    print("EVENT DENSITY PER ACCOUNT PAIR")
    print("-" * 100)

    if not events.empty:

        event_counts = (
            events
            .groupby(
                [
                    "source_account_id",
                    "target_account_id",
                ]
            )
            .size()
            .reset_index(
                name="event_count"
            )
        )

        print(
            f"Pairs with events: {len(event_counts)}"
        )

        print(
            f"Average events per active pair: "
            f"{event_counts['event_count'].mean():.2f}"
        )

        print(
            f"Maximum events for one pair: "
            f"{event_counts['event_count'].max()}"
        )

        print("\nTop pairs by number of events:\n")

        top_event_pairs = (
            event_counts
            .sort_values(
                "event_count",
                ascending=False,
            )
            .head(20)
        )

        print(
            top_event_pairs.to_string(
                index=False
            )
        )

    # ---------------------------------------------------------
    # TOP COORDINATION PAIRS
    # ---------------------------------------------------------

    print("\n" + "-" * 100)
    print("TOP 30 ACCOUNT PAIRS")
    print("-" * 100)

    top_pairs = (
        pairs
        .sort_values(
            "final_score",
            ascending=False,
        )
        .head(30)
    )

    display_columns = [
        "source_account_id",
        "target_account_id",
        "content_score",
        "temporal_score",
        "network_score",
        "final_score",
        "coordination_type",
    ]

    print(
        top_pairs[
            display_columns
        ].to_string(
            index=False
        )
    )

    # ---------------------------------------------------------
    # STRONG PAIRS
    # ---------------------------------------------------------

    print("\n" + "-" * 100)
    print("STRONG COORDINATION PAIRS")
    print("-" * 100)

    strong_pairs = (
        pairs[
            pairs["final_score"] >= 0.50
        ]
        .sort_values(
            "final_score",
            ascending=False,
        )
    )

    print(
        f"Pairs with final_score >= 0.50: "
        f"{len(strong_pairs)}"
    )

    if not strong_pairs.empty:

        print()

        print(
            strong_pairs[
                display_columns
            ]
            .head(50)
            .to_string(
                index=False
            )
        )

    # ---------------------------------------------------------
    # VERY STRONG PAIRS
    # ---------------------------------------------------------

    print("\n" + "-" * 100)
    print("VERY STRONG COORDINATION PAIRS")
    print("-" * 100)

    very_strong_pairs = (
        pairs[
            pairs["final_score"] >= 0.75
        ]
        .sort_values(
            "final_score",
            ascending=False,
        )
    )

    print(
        f"Pairs with final_score >= 0.75: "
        f"{len(very_strong_pairs)}"
    )

    # ---------------------------------------------------------
    # ACCOUNT PARTICIPATION
    # ---------------------------------------------------------

    print("\n" + "-" * 100)
    print("MOST CONNECTED ACCOUNTS")
    print("-" * 100)

    source_counts = (
        pairs["source_account_id"]
        .value_counts()
    )

    target_counts = (
        pairs["target_account_id"]
        .value_counts()
    )

    participation = (
        source_counts
        .add(
            target_counts,
            fill_value=0,
        )
        .sort_values(
            ascending=False
        )
        .reset_index()
    )

    participation.columns = [
        "account_id",
        "pair_count",
    ]

    account_summary = (
        participation
        .merge(
            scores,
            on="account_id",
            how="left",
        )
        .head(30)
    )

    print(
        account_summary[
            [
                "account_id",
                "pair_count",
                "influence_score",
                "tier",
                "confidence_level",
                "evidence_status",
                "assessment",
            ]
        ]
        .to_string(
            index=False
        )
    )

    # ---------------------------------------------------------
    # HIGH-RISK ACCOUNTS PARTICIPATING IN PAIRS
    # ---------------------------------------------------------

    print("\n" + "-" * 100)
    print("SUSPICIOUS ACCOUNTS PARTICIPATING IN COORDINATION PAIRS")
    print("-" * 100)

    suspicious_accounts = account_summary[
        account_summary["tier"]
        == "suspicious"
    ]

    print(
        f"Suspicious accounts among top connected accounts: "
        f"{len(suspicious_accounts)}"
    )

    if not suspicious_accounts.empty:

        print()

        print(
            suspicious_accounts.to_string(
                index=False
            )
        )

    # ---------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------

    print("\n" + "=" * 100)
    print("PAIR DIAGNOSTIC SUMMARY")
    print("=" * 100)

    print(
        f"Total accounts analysed      : "
        f"{len(scores)}"
    )

    print(
        f"Accounts in coordination pairs: "
        f"{len(unique_accounts)}"
    )

    print(
        f"Total account pairs          : "
        f"{len(pairs)}"
    )

    print(
        f"Coordination events          : "
        f"{len(events)}"
    )

    print(
        f"Strong pairs (>= 0.50)       : "
        f"{len(strong_pairs)}"
    )

    print(
        f"Very strong pairs (>= 0.75)  : "
        f"{len(very_strong_pairs)}"
    )

    print("\nInterpretation:")

    print(
        "Account-pair analysis is used to verify whether "
        "coordination signals form meaningful relationships "
        "between accounts."
    )

    print(
        "A high account-level anomaly score alone does not "
        "demonstrate coordination."
    )

    print(
        "Pairs supported by multiple event types or repeated "
        "events provide stronger evidence than isolated signals."
    )

    print("=" * 100 + "\n")


if __name__ == "__main__":

    main()