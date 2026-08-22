# src/score_diagnostics.py
#
# Diagnostic and validation utilities for the composite
# Influence Score pipeline.
#
# Examines:
#
#   - Individual signal distributions
#   - Signal availability and positive coverage
#   - Signal quantiles
#   - Risk tier distribution
#   - Coordination evidence distribution
#   - Confidence distribution
#   - Final assessment distribution
#   - Coordination-related accounts
#   - Assessment consistency
#   - High-priority accounts
#
# This script does not modify the database.

import pandas as pd

from src.db import get_conn


# ---------------------------------------------------------------------
# SIGNAL CONFIGURATION
# ---------------------------------------------------------------------

SIGNALS = [
    "anomaly_score",
    "coord_score",
    "temporal_score",
    "dup_score",
    "network_score",
]


# ---------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------

def print_section(title, width=100):

    print("\n" + "-" * width)

    print(title)

    print("-" * width)


def print_distribution(df, column, order=None):

    if column not in df.columns:

        print(
            f"{column} column not available."
        )

        return

    total = len(df)

    if total == 0:

        print(
            "No accounts available."
        )

        return

    if order is None:

        values = (
            df[column]
            .fillna("missing")
            .value_counts()
        )

    else:

        values = (
            df[column]
            .fillna("missing")
            .value_counts()
            .reindex(
                order,
                fill_value=0,
            )
        )

    for label, count in values.items():

        percentage = (
            count / total * 100
        )

        print(
            f"{str(label):<42}: "
            f"{int(count):>5} accounts "
            f"({percentage:.1f}%)"
        )


def safe_percentage(part, total):

    if total == 0:

        return 0.0

    return (
        part / total * 100
    )


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():

    conn = get_conn()

    query = """
        SELECT
            account_id,

            anomaly_score,
            coord_score,
            temporal_score,
            dup_score,
            network_score,

            influence_score,

            tier,
            confidence_level,
            evidence_status,
            assessment,

            scored_at

        FROM scores
    """

    try:

        df = pd.read_sql(
            query,
            conn,
        )

    finally:

        conn.close()

    # ---------------------------------------------------------
    # HEADER
    # ---------------------------------------------------------

    print("\n" + "=" * 100)

    print(
        "SCORE DIAGNOSTICS AND ASSESSMENT VALIDATION"
    )

    print("=" * 100)

    print(
        f"\nTotal accounts: {len(df)}"
    )

    if df.empty:

        print(
            "\nNo score data found."
        )

        print("\n" + "=" * 100)

        return

    # ---------------------------------------------------------
    # SIGNAL DISTRIBUTION
    # ---------------------------------------------------------

    print_section(
        "SIGNAL DISTRIBUTION"
    )

    for signal in SIGNALS:

        series = df[
            signal
        ]

        available = int(
            series.notna().sum()
        )

        missing = int(
            series.isna().sum()
        )

        zero = int(
            (
                series == 0
            ).sum()
        )

        positive = int(
            (
                series > 0
            ).sum()
        )

        print(
            f"\n{signal}"
        )

        print(
            f"  Available : {available}"
        )

        print(
            f"  Missing   : {missing}"
        )

        print(
            f"  Zero      : {zero}"
        )

        print(
            f"  Positive  : {positive}"
        )

        if available > 0:

            print(
                f"  Min       : "
                f"{series.min():.4f}"
            )

            print(
                f"  Mean      : "
                f"{series.mean():.4f}"
            )

            print(
                f"  Median    : "
                f"{series.median():.4f}"
            )

            print(
                f"  Max       : "
                f"{series.max():.4f}"
            )

    # ---------------------------------------------------------
    # SIGNAL AVAILABILITY
    # ---------------------------------------------------------

    print_section(
        "SIGNAL AVAILABILITY"
    )

    total_accounts = len(
        df
    )

    for signal in SIGNALS:

        available = int(
            df[
                signal
            ]
            .notna()
            .sum()
        )

        percentage = safe_percentage(
            available,
            total_accounts,
        )

        print(
            f"{signal:<20}: "
            f"{available:>5} accounts "
            f"({percentage:.1f}%)"
        )

    # ---------------------------------------------------------
    # POSITIVE SIGNAL COVERAGE
    # ---------------------------------------------------------

    print_section(
        "POSITIVE SIGNAL COVERAGE"
    )

    for signal in SIGNALS:

        count = int(
            (
                df[signal] > 0
            ).sum()
        )

        percentage = safe_percentage(
            count,
            total_accounts,
        )

        print(
            f"{signal:<20}: "
            f"{count:>5} accounts "
            f"({percentage:.1f}%)"
        )

    # ---------------------------------------------------------
    # SIGNAL QUANTILES
    # ---------------------------------------------------------

    print_section(
        "SIGNAL QUANTILES"
    )

    quantiles = [
        0.00,
        0.25,
        0.50,
        0.75,
        0.90,
        0.95,
        0.99,
        1.00,
    ]

    for signal in SIGNALS:

        series = (
            df[
                signal
            ]
            .dropna()
        )

        if series.empty:

            print(
                f"\n{signal}: "
                "No available values."
            )

            continue

        print(
            f"\n{signal}"
        )

        print(
            series.quantile(
                quantiles
            )
        )

    # ---------------------------------------------------------
    # INFLUENCE SCORE DISTRIBUTION
    # ---------------------------------------------------------

    print_section(
        "INFLUENCE SCORE DISTRIBUTION"
    )

    influence_series = (
        df[
            "influence_score"
        ]
        .dropna()
    )

    if influence_series.empty:

        print(
            "No influence scores available."
        )

    else:

        print(
            f"Available scores : "
            f"{len(influence_series)}"
        )

        print(
            f"Min              : "
            f"{influence_series.min():.2f}"
        )

        print(
            f"Mean             : "
            f"{influence_series.mean():.2f}"
        )

        print(
            f"Median           : "
            f"{influence_series.median():.2f}"
        )

        print(
            f"Max              : "
            f"{influence_series.max():.2f}"
        )

        print(
            "\nQuantiles:"
        )

        print(
            influence_series.quantile(
                quantiles
            )
        )

    # ---------------------------------------------------------
    # RISK TIER DISTRIBUTION
    # ---------------------------------------------------------

    print_section(
        "RISK TIER DISTRIBUTION"
    )

    tier_order = [

        "insufficient_data",
        "organic",
        "suspicious",
        "coordinated",

    ]

    print_distribution(
        df,
        "tier",
        tier_order,
    )

    # ---------------------------------------------------------
    # COORDINATION EVIDENCE DISTRIBUTION
    # ---------------------------------------------------------

    print_section(
        "COORDINATION EVIDENCE DISTRIBUTION"
    )

    evidence_order = [

        "strong_support",
        "supported",
        "weak_support",
        "no_direct_evidence",
        "insufficient_data",

    ]

    print_distribution(
        df,
        "evidence_status",
        evidence_order,
    )

    # ---------------------------------------------------------
    # CONFIDENCE LEVEL DISTRIBUTION
    # ---------------------------------------------------------

    print_section(
        "CONFIDENCE LEVEL DISTRIBUTION"
    )

    confidence_order = [

        "high",
        "medium",
        "low",
        "insufficient",

    ]

    print_distribution(
        df,
        "confidence_level",
        confidence_order,
    )

    # ---------------------------------------------------------
    # FINAL ASSESSMENT DISTRIBUTION
    # ---------------------------------------------------------

    print_section(
        "FINAL ASSESSMENT DISTRIBUTION"
    )

    assessment_order = [

        "insufficient_data",
        "likely_organic",
        "organic_with_coordination_pattern",
        "suspicious",
        "suspicious_with_coordination_evidence",
        "high_priority_coordinated_pattern",
        "likely_coordinated_influence",

    ]

    print_distribution(
        df,
        "assessment",
        assessment_order,
    )

    # ---------------------------------------------------------
    # TIER VS ASSESSMENT MATRIX
    # ---------------------------------------------------------

    print_section(
        "TIER VS FINAL ASSESSMENT"
    )

    matrix = pd.crosstab(

        df["tier"],

        df["assessment"],

        margins=True,

    )

    print(
        matrix.to_string()
    )

    # ---------------------------------------------------------
    # EVIDENCE STATUS VS ASSESSMENT MATRIX
    # ---------------------------------------------------------

    print_section(
        "COORDINATION EVIDENCE VS ASSESSMENT"
    )

    evidence_matrix = pd.crosstab(

        df[
            "evidence_status"
        ],

        df[
            "assessment"
        ],

        margins=True,

    )

    print(
        evidence_matrix.to_string()
    )

    # ---------------------------------------------------------
    # COORDINATION-RELATED ACCOUNTS
    # ---------------------------------------------------------

    print_section(
        "TOP COORDINATION-RELATED ACCOUNTS"
    )

    coordination_df = (
        df[
            (
                df["coord_score"] > 0
            )
            |
            (
                df["temporal_score"] > 0
            )
            |
            (
                df["dup_score"] > 0
            )
        ]
        .copy()
    )

    if coordination_df.empty:

        print(
            "No positive coordination-related "
            "signals found."
        )

    else:

        coordination_df[
            "coordination_strength"
        ] = (

            coordination_df[
                [
                    "coord_score",
                    "temporal_score",
                    "dup_score",
                ]
            ]

            .fillna(
                0
            )

            .max(
                axis=1
            )
        )

        top = (
            coordination_df
            .sort_values(
                [
                    "coordination_strength",
                    "influence_score",
                ],
                ascending=[
                    False,
                    False,
                ],
            )
            .head(
                30
            )
        )

        columns = [

            "account_id",

            "coord_score",
            "temporal_score",
            "dup_score",

            "network_score",

            "influence_score",

            "tier",
            "confidence_level",
            "evidence_status",
            "assessment",

        ]

        print(
            top[
                columns
            ]
            .to_string(
                index=False
            )
        )

    # ---------------------------------------------------------
    # SUSPICIOUS ACCOUNTS WITH COORDINATION EVIDENCE
    # ---------------------------------------------------------

    print_section(
        "SUSPICIOUS ACCOUNTS WITH COORDINATION EVIDENCE"
    )

    suspicious_evidence = (

        df[
            (
                df["assessment"]
                == "suspicious_with_coordination_evidence"
            )
        ]

        .sort_values(
            "influence_score",
            ascending=False,
        )

    )

    print(
        f"Accounts found: "
        f"{len(suspicious_evidence)}"
    )

    if not suspicious_evidence.empty:

        print()

        print(
            suspicious_evidence[
                [
                    "account_id",
                    "influence_score",
                    "tier",
                    "confidence_level",
                    "evidence_status",
                    "assessment",
                    "coord_score",
                    "temporal_score",
                    "dup_score",
                    "network_score",
                ]
            ]
            .head(30)
            .to_string(
                index=False
            )
        )

    # ---------------------------------------------------------
    # HIGH PRIORITY COORDINATED PATTERNS
    # ---------------------------------------------------------

    print_section(
        "HIGH PRIORITY COORDINATED PATTERNS"
    )

    high_priority = (

        df[
            (
                df["assessment"]
                == "high_priority_coordinated_pattern"
            )
        ]

        .sort_values(
            "influence_score",
            ascending=False,
        )

    )

    print(
        f"Accounts found: "
        f"{len(high_priority)}"
    )

    if not high_priority.empty:

        print()

        print(
            high_priority[
                [
                    "account_id",
                    "influence_score",
                    "tier",
                    "confidence_level",
                    "evidence_status",
                    "assessment",
                    "anomaly_score",
                    "coord_score",
                    "temporal_score",
                    "dup_score",
                    "network_score",
                ]
            ]
            .to_string(
                index=False
            )
        )

    # ---------------------------------------------------------
    # LIKELY COORDINATED INFLUENCE
    # ---------------------------------------------------------

    print_section(
        "LIKELY COORDINATED INFLUENCE"
    )

    likely_coordinated = (

        df[
            (
                df["assessment"]
                == "likely_coordinated_influence"
            )
        ]

        .sort_values(
            "influence_score",
            ascending=False,
        )

    )

    print(
        f"Accounts found: "
        f"{len(likely_coordinated)}"
    )

    if not likely_coordinated.empty:

        print()

        print(
            likely_coordinated[
                [
                    "account_id",
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
    # ASSESSMENT CONSISTENCY CHECKS
    # ---------------------------------------------------------

    print_section(
        "ASSESSMENT CONSISTENCY CHECKS"
    )

    issues = []

    # likely_organic should normally have organic tier

    inconsistent_organic = df[
        (
            df["assessment"]
            == "likely_organic"
        )
        &
        (
            df["tier"]
            != "organic"
        )
    ]

    if not inconsistent_organic.empty:

        issues.append(
            (
                "likely_organic with "
                "non-organic tier",
                len(inconsistent_organic),
            )
        )

    # suspicious should have suspicious tier

    inconsistent_suspicious = df[
        (
            df["assessment"]
            == "suspicious"
        )
        &
        (
            df["tier"]
            != "suspicious"
        )
    ]

    if not inconsistent_suspicious.empty:

        issues.append(
            (
                "suspicious assessment with "
                "non-suspicious tier",
                len(inconsistent_suspicious),
            )
        )

    # likely coordinated influence should have
    # direct coordination evidence

    inconsistent_coordinated = df[
        (
            df["assessment"]
            == "likely_coordinated_influence"
        )
        &
        (
            ~df[
                "evidence_status"
            ]
            .isin(
                [
                    "strong_support",
                    "supported",
                ]
            )
        )
    ]

    if not inconsistent_coordinated.empty:

        issues.append(
            (
                "likely coordinated influence "
                "without strong coordination evidence",
                len(inconsistent_coordinated),
            )
        )

    # insufficient_data should have insufficient tier

    inconsistent_insufficient = df[
        (
            df["assessment"]
            == "insufficient_data"
        )
        &
        (
            df["tier"]
            != "insufficient_data"
        )
    ]

    if not inconsistent_insufficient.empty:

        issues.append(
            (
                "insufficient assessment with "
                "non-insufficient tier",
                len(inconsistent_insufficient),
            )
        )

    if not issues:

        print(
            "PASS: No basic assessment consistency "
            "issues detected."
        )

    else:

        print(
            "WARNING: Potential assessment "
            "consistency issues detected:"
        )

        for description, count in issues:

            print(
                f"  {description}: "
                f"{count} accounts"
            )

    # ---------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------

    print_section(
        "DIAGNOSTIC SUMMARY"
    )

    total = len(df)

    suspicious_total = int(

        (
            df["tier"]
            == "suspicious"
        ).sum()

    )

    coordination_assessments = int(

        df[
            "assessment"
        ]
        .isin(
            [
                "organic_with_coordination_pattern",
                "suspicious_with_coordination_evidence",
                "high_priority_coordinated_pattern",
                "likely_coordinated_influence",
            ]
        )
        .sum()

    )

    high_priority_total = int(

        (
            df["assessment"]
            == "high_priority_coordinated_pattern"
        ).sum()

    )

    likely_coordinated_total = int(

        (
            df["assessment"]
            == "likely_coordinated_influence"
        ).sum()

    )

    print(
        f"Total accounts                        : "
        f"{total}"
    )

    print(
        f"Suspicious risk tier                   : "
        f"{suspicious_total}"
    )

    print(
        f"Accounts with coordination patterns    : "
        f"{coordination_assessments}"
    )

    print(
        f"High-priority coordinated patterns     : "
        f"{high_priority_total}"
    )

    print(
        f"Likely coordinated influence           : "
        f"{likely_coordinated_total}"
    )

    print(
        "\nInterpretation:"
    )

    print(
        "The diagnostic report separates "
        "overall influence risk from "
        "coordination evidence."
    )

    print(
        "A suspicious account is not "
        "automatically considered coordinated."
    )

    print(
        "Accounts classified as coordinated "
        "patterns should be treated as "
        "prioritized candidates for further "
        "investigation rather than definitive "
        "proof of coordinated intent."
    )

    print("\n" + "=" * 100)


# ---------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------

if __name__ == "__main__":

    main()