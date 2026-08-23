# src/pipeline/composite_score.py
#
# Combines all available detection signals into one composite
# Influence Score (0-100).
#
# Signals:
#
#   30% anomaly detection
#   25% behavioral coordination
#   20% temporal synchronization
#   15% content duplication
#   10% network evidence
#
# IMPORTANT:
#
# - Missing signals are not automatically treated as 0.
# - Available weights are normalized.
# - Low signal coverage applies a confidence calibration factor.
# - Behavioral clustering alone does not prove coordination.
# - Temporal synchronization alone does not prove intent.
# - Risk tier and coordination evidence are evaluated separately.
# - Direct evidence is stored separately in evidence_status.
# - Final assessment provides a human-readable interpretation.

import logging
from datetime import datetime, timezone

import pandas as pd

from src.config import TIERS
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
# COMPONENT CONFIGURATION
# ---------------------------------------------------------------------

COMPONENTS = {
    "anomaly_score": "anomaly",
    "coord_score": "coordination",
    "temporal_score": "temporal",
    "dup_score": "duplication",
    "network_score": "network",
}


# ---------------------------------------------------------------------
# COMPOSITE WEIGHTS
# ---------------------------------------------------------------------

COMPOSITE_WEIGHTS = {
    "anomaly": 0.30,
    "coordination": 0.25,
    "temporal": 0.20,
    "duplication": 0.15,
    "network": 0.10,
}


# ---------------------------------------------------------------------
# SIGNAL COVERAGE CALIBRATION
# ---------------------------------------------------------------------

SIGNAL_COVERAGE_FACTORS = {
    0: 0.00,
    1: 0.50,
    2: 0.70,
    3: 0.85,
    4: 0.95,
    5: 1.00,
}


# ---------------------------------------------------------------------
# DATA SUFFICIENCY RULES
# ---------------------------------------------------------------------

MIN_VALID_SIGNALS_FOR_TIER = 2

MIN_VALID_SIGNALS_FOR_COORDINATION = 3


# ---------------------------------------------------------------------
# COORDINATION THRESHOLDS
# ---------------------------------------------------------------------

COORDINATED_SCORE_THRESHOLD = 0.70

TEMPORAL_COORDINATION_THRESHOLD = 0.70

COORDINATED_INFLUENCE_THRESHOLD = 60.0

HIGH_PRIORITY_INFLUENCE_THRESHOLD = 45.0


# ---------------------------------------------------------------------
# LOAD DIRECT COORDINATION EVIDENCE
# ---------------------------------------------------------------------

def load_direct_coordination_evidence(conn):
    """
    Build mapping:

        account_id -> set of direct evidence sources

    Direct evidence comes from:

    1. Near-duplicate content
    2. Temporal synchronization
    3. Strong account-pair relationships
    """

    evidence_map = {}

    def add_evidence(account_id, source):

        if account_id not in evidence_map:
            evidence_map[account_id] = set()

        evidence_map[account_id].add(source)

    # ---------------------------------------------------------
    # 1. NEAR-DUPLICATE CONTENT
    # ---------------------------------------------------------

    rows = conn.execute(
        """
        SELECT DISTINCT
            source_account_id,
            target_account_id
        FROM coordination_events
        WHERE event_type = 'near_duplicate_content'
          AND similarity >= 0.80
        """
    ).fetchall()

    content_accounts = set()

    for row in rows:

        source = row["source_account_id"]
        target = row["target_account_id"]

        add_evidence(
            source,
            "near_duplicate_content",
        )

        add_evidence(
            target,
            "near_duplicate_content",
        )

        content_accounts.add(source)
        content_accounts.add(target)

    logger.info(
        "Direct evidence from near_duplicate_content : %d accounts",
        len(content_accounts),
    )

    # ---------------------------------------------------------
    # 2. STRONG TEMPORAL SYNCHRONIZATION
    # ---------------------------------------------------------

    temporal_threshold = float(TEMPORAL_COORDINATION_THRESHOLD)

    logger.info(
        "Using temporal coordination threshold: %.4f",
        temporal_threshold,
    )

    rows = conn.execute(
        """
        SELECT DISTINCT
            source_account_id,
            target_account_id
        FROM coordination_events
        WHERE event_type = 'temporal_synchronization'
          AND similarity >= ?
        """,
        (temporal_threshold,),
    ).fetchall()

    temporal_accounts = set()

    for row in rows:

        source = row["source_account_id"]
        target = row["target_account_id"]

        add_evidence(
            source,
            "temporal_synchronization",
        )

        add_evidence(
            target,
            "temporal_synchronization",
        )

        temporal_accounts.add(source)
        temporal_accounts.add(target)

    logger.info(
        "Direct evidence from temporal_synchronization: %d accounts",
        len(temporal_accounts),
    )

    # ---------------------------------------------------------
    # 3. ACCOUNT-PAIR EVIDENCE
    # ---------------------------------------------------------

    rows = conn.execute(
        """
        SELECT DISTINCT
            source_account_id,
            target_account_id
        FROM account_pairs
        WHERE final_score >= 0.50
        """
    ).fetchall()

    pair_accounts = set()

    for row in rows:

        source = row["source_account_id"]
        target = row["target_account_id"]

        add_evidence(
            source,
            "account_pair",
        )

        add_evidence(
            target,
            "account_pair",
        )

        pair_accounts.add(source)
        pair_accounts.add(target)

    logger.info(
        "Direct evidence from account_pairs         : %d accounts",
        len(pair_accounts),
    )

    logger.info(
        "Total accounts with direct coordination evidence: %d",
        len(evidence_map),
    )

    return evidence_map


# ---------------------------------------------------------------------
# LOAD SCORES
# ---------------------------------------------------------------------

def load_scores():
    """
    Load all available detection scores.
    """

    conn = get_conn()

    component_columns = ", ".join(
        COMPONENTS.keys()
    )

    query = f"""
        SELECT
            account_id,
            {component_columns}
        FROM scores
    """

    try:

        df = pd.read_sql(
            query,
            conn,
        )

    finally:

        conn.close()

    logger.info(
        "Loaded %d accounts from scores table",
        len(df),
    )

    return df


# ---------------------------------------------------------------------
# DETERMINE VALID SIGNALS
# ---------------------------------------------------------------------

def is_valid_signal(column, value):
    """
    Determine whether a signal should contribute to
    evidence coverage.
    """

    if pd.isna(value):
        return False

    if column == "dup_score" and float(value) <= 0:
        return False

    return True


# ---------------------------------------------------------------------
# COORDINATION STATUS
# ---------------------------------------------------------------------

def determine_coordination_status(
    signal_count,
    coord_score,
    temporal_score,
    evidence_sources,
):
    """
    Determine the strength of coordination evidence.

    Possible results:

        insufficient_data
        no_direct_evidence
        weak_support
        supported
        strong_support
    """

    if signal_count < MIN_VALID_SIGNALS_FOR_TIER:

        return "insufficient_data"

    evidence_count = len(
        evidence_sources
    )

    has_direct_evidence = (
        evidence_count > 0
    )

    strong_behavioral = (
        pd.notna(coord_score)
        and float(coord_score)
        >= COORDINATED_SCORE_THRESHOLD
    )

    strong_temporal = (
        pd.notna(temporal_score)
        and float(temporal_score)
        >= TEMPORAL_COORDINATION_THRESHOLD
    )

    strong_signal_count = sum(
        [
            strong_behavioral,
            strong_temporal,
        ]
    )

    # ---------------------------------------------------------
    # NO DIRECT EVIDENCE
    # ---------------------------------------------------------

    if not has_direct_evidence:

        if strong_signal_count > 0:
            return "weak_support"

        return "no_direct_evidence"

    # ---------------------------------------------------------
    # LIMITED SIGNAL COVERAGE
    # ---------------------------------------------------------

    if signal_count < MIN_VALID_SIGNALS_FOR_COORDINATION:

        return "weak_support"

    # ---------------------------------------------------------
    # THREE OR MORE INDEPENDENT EVIDENCE SOURCES
    # ---------------------------------------------------------

    if evidence_count >= 3:

        return "strong_support"


    # ---------------------------------------------------------
    # TWO INDEPENDENT EVIDENCE SOURCES
    # ---------------------------------------------------------

    if evidence_count == 2:

        return "supported"


    # ---------------------------------------------------------
    # ONE EVIDENCE SOURCE
    #
    # A single evidence channel is useful but should not by itself
    # establish supported coordination. A strong behavioral or temporal
    # score can increase analyst priority, but it does not make the
    # evidence independent.
    # ---------------------------------------------------------

    if evidence_count == 1:

        return "weak_support"


    return "no_direct_evidence"


# ---------------------------------------------------------------------
# DETERMINE FINAL ASSESSMENT
# ---------------------------------------------------------------------

def determine_assessment(
    influence_score,
    tier,
    evidence_status,
    confidence_level,
    signal_count,
    direct_evidence_count,
):
    """
    Generate the final human-readable assessment.

    Assessment categories:

        insufficient_data
        likely_organic
        organic_with_coordination_pattern
        suspicious
        suspicious_with_coordination_evidence
        high_priority_coordinated_pattern
        likely_coordinated_influence

    Important:

    The assessment is intentionally more descriptive than the
    numerical tier. Direct evidence alone does not automatically
    prove coordinated influence.
    """

    # ---------------------------------------------------------
    # INSUFFICIENT DATA
    # ---------------------------------------------------------

    if (
        pd.isna(influence_score)
        or signal_count < MIN_VALID_SIGNALS_FOR_TIER
        or tier == "insufficient_data"
    ):

        return "insufficient_data"

    score = float(influence_score)

    has_direct_evidence = (
        direct_evidence_count > 0
    )

    strong_evidence = (
        evidence_status == "strong_support"
    )

    supported_evidence = (
        evidence_status == "supported"
    )

    coordination_pattern = (
        strong_evidence
        or supported_evidence
    )

    # ---------------------------------------------------------
    # LIKELY COORDINATED INFLUENCE
    #
    # Requires:
    #
    # - High composite influence score
    # - Direct coordination evidence
    # - Strong evidence support
    # - At least medium confidence
    # ---------------------------------------------------------

    if (
        score >= COORDINATED_INFLUENCE_THRESHOLD
        and has_direct_evidence
        and strong_evidence
        and confidence_level
        in ["medium", "high"]
    ):

        return "likely_coordinated_influence"

    # ---------------------------------------------------------
    # HIGH PRIORITY COORDINATED PATTERN
    #
    # Strong pattern requiring analyst review.
    # This does not automatically claim proven intent.
    # ---------------------------------------------------------

    if (
        score >= HIGH_PRIORITY_INFLUENCE_THRESHOLD
        and direct_evidence_count >= 2
        and evidence_status
        in [
            "supported",
            "strong_support",
        ]
        and confidence_level
        in [
            "medium",
            "high",
        ]
    ):
        return "high_priority_coordinated_pattern"

    # ---------------------------------------------------------
    # ORGANIC WITH COORDINATION PATTERN
    #
    # Low overall influence risk, but some coordination-related
    # evidence exists.
    # ---------------------------------------------------------

    if tier == "organic":

        if (
            has_direct_evidence
            or evidence_status
            in [
                "weak_support",
                "supported",
                "strong_support",
            ]
        ):

            return "organic_with_coordination_pattern"

        return "likely_organic"

    # ---------------------------------------------------------
    # SUSPICIOUS WITH COORDINATION EVIDENCE
    # ---------------------------------------------------------

    if tier == "suspicious":

        if (
            has_direct_evidence
            or evidence_status
            in [
                "supported",
                "strong_support",
            ]
        ):

            return "suspicious_with_coordination_evidence"

        return "suspicious"

    # ---------------------------------------------------------
    # COORDINATED TIER
    # ---------------------------------------------------------

    if tier == "coordinated":

        if (
            has_direct_evidence
            and evidence_status
            in [
                "supported",
                "strong_support",
            ]
        ):

            return "likely_coordinated_influence"

        return "high_priority_coordinated_pattern"

    # ---------------------------------------------------------
    # SAFE FALLBACK
    # ---------------------------------------------------------

    if has_direct_evidence:

        return "suspicious_with_coordination_evidence"

    return "likely_organic"


# ---------------------------------------------------------------------
# COMPUTE COMPOSITE SCORE
# ---------------------------------------------------------------------

def compute_composite(
    df,
    evidence_map,
):
    """
    Calculate weighted composite influence scores.

    Missing signals are excluded from weight normalization.

    Low signal coverage applies a calibration factor.
    """

    df = df.copy()

    component_columns = list(
        COMPONENTS.keys()
    )

    # ---------------------------------------------------------
    # VALID SIGNAL COUNT
    # ---------------------------------------------------------

    def count_valid_signals(row):

        return sum(
            is_valid_signal(
                column,
                row[column],
            )

            for column in component_columns
        )

    df["valid_signal_count"] = (
        df.apply(
            count_valid_signals,
            axis=1,
        )
    )

    # ---------------------------------------------------------
    # EVIDENCE SOURCES
    # ---------------------------------------------------------

    df["evidence_sources"] = (
        df["account_id"]
        .apply(
            lambda account_id:
            evidence_map.get(
                account_id,
                set(),
            )
        )
    )

    df["direct_evidence_count"] = (
        df["evidence_sources"]
        .apply(len)
    )

    df[
        "has_direct_coordination_evidence"
    ] = (
        df[
            "direct_evidence_count"
        ]
        > 0
    )

    logger.info(
        "Accounts with direct coordination evidence: %d",
        int(
            df[
                "has_direct_coordination_evidence"
            ].sum()
        ),
    )

    # ---------------------------------------------------------
    # SIGNAL COVERAGE
    # ---------------------------------------------------------

    logger.info(
        "Signal coverage:"
    )

    coverage = (
        df[
            component_columns
        ]
        .notna()
        .mean()
    )

    for column, fraction in coverage.items():

        logger.info(
            "  %-18s %6.1f%%",
            column,
            fraction * 100,
        )

    # ---------------------------------------------------------
    # COMPUTE WEIGHTED SCORE
    # ---------------------------------------------------------

    influence_scores = []

    raw_scores = []

    coverage_factors = []

    for _, row in df.iterrows():

        valid_columns = [

            column

            for column in component_columns

            if is_valid_signal(
                column,
                row[column],
            )
        ]

        signal_count = len(
            valid_columns
        )

        if signal_count == 0:

            influence_scores.append(None)
            raw_scores.append(None)

            coverage_factors.append(
                SIGNAL_COVERAGE_FACTORS[0]
            )

            continue

        available_weight = sum(

            COMPOSITE_WEIGHTS[
                COMPONENTS[column]
            ]

            for column in valid_columns
        )

        weighted_score = sum(

            float(
                row[column]
            )

            * COMPOSITE_WEIGHTS[
                COMPONENTS[column]
            ]

            for column in valid_columns
        )

        normalized_score = (
            weighted_score
            / available_weight
        )

        raw_score = (
            normalized_score * 100
        )

        coverage_factor = (
            SIGNAL_COVERAGE_FACTORS.get(
                signal_count,
                1.00,
            )
        )

        calibrated_score = (
            raw_score
            * coverage_factor
        )

        raw_scores.append(
            round(
                raw_score,
                2,
            )
        )

        coverage_factors.append(
            coverage_factor
        )

        influence_scores.append(
            round(
                calibrated_score,
                2,
            )
        )

    df[
        "raw_influence_score"
    ] = raw_scores

    df[
        "coverage_factor"
    ] = coverage_factors

    df[
        "influence_score"
    ] = influence_scores

    # ---------------------------------------------------------
    # ASSIGN RISK TIERS
    # ---------------------------------------------------------

    tiers = []

    for _, row in df.iterrows():

        score = row[
            "influence_score"
        ]

        signal_count = int(
            row[
                "valid_signal_count"
            ]
        )

        has_direct_evidence = bool(
            row[
                "has_direct_coordination_evidence"
            ]
        )

        if pd.isna(score):

            tiers.append(
                "insufficient_data"
            )

            continue

        if signal_count < MIN_VALID_SIGNALS_FOR_TIER:

            tiers.append(
                "insufficient_data"
            )

            continue

        score = float(score)

        if score <= TIERS["organic"][1]:

            tiers.append(
                "organic"
            )

            continue

        if (
            score > COORDINATED_INFLUENCE_THRESHOLD
            and has_direct_evidence
        ):

            tiers.append(
                "coordinated"
            )

            continue

        tiers.append(
            "suspicious"
        )

    df["tier"] = tiers

    # ---------------------------------------------------------
    # COORDINATION EVIDENCE STATUS
    # ---------------------------------------------------------

    coordination_statuses = []

    for _, row in df.iterrows():

        status = (
            determine_coordination_status(

                signal_count=int(
                    row[
                        "valid_signal_count"
                    ]
                ),

                coord_score=row[
                    "coord_score"
                ],

                temporal_score=row[
                    "temporal_score"
                ],

                evidence_sources=row[
                    "evidence_sources"
                ],
            )
        )

        coordination_statuses.append(
            status
        )

    df[
        "evidence_status"
    ] = coordination_statuses

    # ---------------------------------------------------------
    # CONFIDENCE LEVEL
    # ---------------------------------------------------------

    confidence_levels = []

    for _, row in df.iterrows():

        signal_count = int(
            row[
                "valid_signal_count"
            ]
        )

        evidence_count = int(
            row[
                "direct_evidence_count"
            ]
        )

        if (
            signal_count >= 4
            and evidence_count >= 2
        ):

            confidence_levels.append(
                "high"
            )

        elif signal_count >= 4:

            confidence_levels.append(
                "high"
            )

        elif signal_count == 3:

            confidence_levels.append(
                "medium"
            )

        elif signal_count == 2:

            confidence_levels.append(
                "low"
            )

        else:

            confidence_levels.append(
                "insufficient"
            )

    df[
        "confidence_level"
    ] = confidence_levels

    # ---------------------------------------------------------
    # FINAL ASSESSMENT
    # ---------------------------------------------------------

    assessments = []

    for _, row in df.iterrows():

        assessment = determine_assessment(

            influence_score=row[
                "influence_score"
            ],

            tier=row[
                "tier"
            ],

            evidence_status=row[
                "evidence_status"
            ],

            confidence_level=row[
                "confidence_level"
            ],

            signal_count=int(
                row[
                    "valid_signal_count"
                ]
            ),

            direct_evidence_count=int(
                row[
                    "direct_evidence_count"
                ]
            ),
        )

        assessments.append(
            assessment
        )

    df["assessment"] = assessments

    # ---------------------------------------------------------
    # DATA SUFFICIENCY SUMMARY
    # ---------------------------------------------------------

    logger.info(
        "Data sufficiency summary:"
    )

    for count in sorted(
        df[
            "valid_signal_count"
        ].unique()
    ):

        accounts = int(
            (
                df[
                    "valid_signal_count"
                ]
                == count
            ).sum()
        )

        logger.info(
            "  %d valid signal(s): %d accounts",
            int(count),
            accounts,
        )

    # ---------------------------------------------------------
    # COVERAGE CALIBRATION SUMMARY
    # ---------------------------------------------------------

    logger.info(
        "Signal coverage calibration:"
    )

    for signal_count in sorted(
        df[
            "valid_signal_count"
        ].unique()
    ):

        factor = (
            SIGNAL_COVERAGE_FACTORS.get(
                int(signal_count),
                1.00,
            )
        )

        logger.info(
            "  %d signal(s) -> factor %.2f",
            int(signal_count),
            factor,
        )

    # ---------------------------------------------------------
    # COORDINATION EVIDENCE SUMMARY
    # ---------------------------------------------------------

    logger.info(
        "Coordination evidence summary:"
    )

    for status in [

        "strong_support",
        "supported",
        "weak_support",
        "no_direct_evidence",
        "insufficient_data",

    ]:

        count = int(
            (
                df[
                    "evidence_status"
                ]
                == status
            ).sum()
        )

        logger.info(
            "  %-20s %d accounts",
            status,
            count,
        )

    # ---------------------------------------------------------
    # FINAL ASSESSMENT SUMMARY
    # ---------------------------------------------------------

    logger.info(
        "Final assessment summary:"
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

    for assessment in assessment_order:

        count = int(
            (
                df["assessment"]
                == assessment
            ).sum()
        )

        logger.info(
            "  %-38s %d accounts",
            assessment,
            count,
        )

    return df.sort_values(
        "influence_score",
        ascending=False,
        na_position="last",
    )


# ---------------------------------------------------------------------
# SAVE RESULTS
# ---------------------------------------------------------------------

def save_composite(df):
    """
    Save composite influence scores and classifications.
    """

    conn = get_conn()

    cursor = conn.cursor()

    now = datetime.now(
        timezone.utc
    ).isoformat()

    saved = 0

    for _, row in df.iterrows():

        influence_score = (
            None
            if pd.isna(
                row["influence_score"]
            )
            else float(
                row["influence_score"]
            )
        )

        cursor.execute(
            """
            UPDATE scores
            SET
                influence_score = ?,
                tier = ?,
                confidence_level = ?,
                evidence_status = ?,
                assessment = ?,
                scored_at = ?
            WHERE account_id = ?
            """,
            (
                influence_score,
                row["tier"],
                row["confidence_level"],
                row["evidence_status"],
                row["assessment"],
                now,
                row["account_id"],
            ),
        )

        saved += 1

    conn.commit()

    conn.close()

    logger.info(
        "Saved composite results for %d accounts",
        saved,
    )


# ---------------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------------

def print_summary(df):

    print(
        "\n"
        + "━" * 110
    )

    print(
        "  COMPOSITE INFLUENCE SCORE SUMMARY"
    )

    print(
        "━" * 110
    )

    tier_order = [

        "insufficient_data",
        "organic",
        "suspicious",
        "coordinated",

    ]

    total_accounts = len(df)

    for tier in tier_order:

        count = int(
            (
                df["tier"] == tier
            ).sum()
        )

        percentage = (
            count / total_accounts * 100
            if total_accounts > 0
            else 0
        )

        print(
            f"  {tier:<22}: "
            f"{count:>5} accounts "
            f"({percentage:.1f}%)"
        )

    print(
        "━" * 110
    )

    # ---------------------------------------------------------
    # ASSESSMENT SUMMARY
    # ---------------------------------------------------------

    print(
        "\n"
        + "━" * 110
    )

    print(
        "  FINAL ASSESSMENT SUMMARY"
    )

    print(
        "━" * 110
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

    for assessment in assessment_order:

        count = int(
            (
                df["assessment"]
                == assessment
            ).sum()
        )

        percentage = (
            count / total_accounts * 100
            if total_accounts > 0
            else 0
        )

        print(
            f"  {assessment:<38}: "
            f"{count:>5} accounts "
            f"({percentage:.1f}%)"
        )

    print(
        "━" * 110
    )

    # ---------------------------------------------------------
    # DIRECT EVIDENCE
    # ---------------------------------------------------------

    print(
        "\n"
        + "━" * 110
    )

    print(
        "  DIRECT COORDINATION EVIDENCE"
    )

    print(
        "━" * 110
    )

    with_evidence = int(
        df[
            "has_direct_coordination_evidence"
        ].sum()
    )

    without_evidence = (
        len(df)
        - with_evidence
    )

    print(
        f"  With direct evidence    : "
        f"{with_evidence}"
    )

    print(
        f"  Without direct evidence : "
        f"{without_evidence}"
    )

    print(
        "━" * 110
    )

    # ---------------------------------------------------------
    # TOP 20
    # ---------------------------------------------------------

    print(
        "\n"
        + "━" * 210
    )

    print(
        "  TOP 20 ACCOUNTS"
    )

    print(
        "━" * 210
    )

    print(
        f"{'account_id':<22}"
        f"{'raw':>9}"
        f"{'factor':>9}"
        f"{'score':>10}"
        f"{'tier':>18}"
        f"{'assessment':>42}"
        f"{'evidence':>24}"
        f"{'signals':>9}"
        f"{'direct':>9}"
        f"{'A':>8}"
        f"{'B':>8}"
        f"{'T':>8}"
        f"{'D':>8}"
        f"{'N':>8}"
    )

    print(
        "-" * 210
    )

    top_accounts = df.head(20)

    def format_signal(value):

        if pd.isna(value):
            return "N/A"

        return f"{float(value):.2f}"

    for _, row in top_accounts.iterrows():

        direct = (
            "YES"
            if row[
                "has_direct_coordination_evidence"
            ]
            else "NO"
        )

        raw = (
            "N/A"
            if pd.isna(
                row["raw_influence_score"]
            )
            else f"{float(row['raw_influence_score']):.1f}"
        )

        score = (
            "N/A"
            if pd.isna(
                row["influence_score"]
            )
            else f"{float(row['influence_score']):.1f}"
        )

        print(
            f"{str(row['account_id']):<22}"
            f"{raw:>9}"
            f"{float(row['coverage_factor']):>9.2f}"
            f"{score:>10}"
            f"{str(row['tier']):>18}"
            f"{str(row['assessment']):>42}"
            f"{str(row['evidence_status']):>24}"
            f"{int(row['valid_signal_count']):>9}"
            f"{direct:>9}"
            f"{format_signal(row['anomaly_score']):>8}"
            f"{format_signal(row['coord_score']):>8}"
            f"{format_signal(row['temporal_score']):>8}"
            f"{format_signal(row['dup_score']):>8}"
            f"{format_signal(row['network_score']):>8}"
        )

    print(
        "━" * 210
    )


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():

    logger.info(
        "Computing composite Influence Score "
        "with evidence-aware coverage calibration"
    )

    # ---------------------------------------------------------
    # LOAD DIRECT EVIDENCE
    # ---------------------------------------------------------

    conn = get_conn()

    try:

        evidence_map = (
            load_direct_coordination_evidence(
                conn
            )
        )

    finally:

        conn.close()

    # ---------------------------------------------------------
    # LOAD SCORES
    # ---------------------------------------------------------

    df = load_scores()

    if df.empty:

        logger.warning(
            "No account scores found."
        )

        return

    # ---------------------------------------------------------
    # COMPUTE
    # ---------------------------------------------------------

    results = compute_composite(
        df,
        evidence_map,
    )

    # ---------------------------------------------------------
    # SAVE
    # ---------------------------------------------------------

    save_composite(
        results
    )

    # ---------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------

    print_summary(
        results
    )

    # ---------------------------------------------------------
    # INTERPRETATION
    # ---------------------------------------------------------

    logger.info(
        "INTERPRETATION:"
    )

    logger.info(
        "The Influence Score combines normalized "
        "detection signals with a signal coverage "
        "calibration factor."
    )

    logger.info(
        "Missing signals are not treated as zero."
    )

    logger.info(
        "Accounts with fewer independent signals "
        "receive lower score confidence through "
        "coverage calibration."
    )

    logger.info(
        "The final assessment combines overall risk, "
        "coordination evidence, and signal confidence."
    )

    logger.info(
        "Coordination evidence indicates patterns "
        "requiring interpretation and does not alone "
        "prove coordinated intent."
    )


# ---------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------

if __name__ == "__main__":

    main()