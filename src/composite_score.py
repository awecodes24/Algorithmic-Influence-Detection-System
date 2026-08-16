# src/composite_score.py
# Combines the four detection signals already written to `scores` into one
# composite Influence Score (0-100) + tier, per the proposal's Ch. 4
# formula: 40% anomaly + 40% coordination + 10% duplication + 10% network.
#
# Run this AFTER the four model scripts (reddit_isolation_forest.py,
# reddit_hdbscan.py, reddit_cosine_similarity.py, reddit_networkx.py) --
# order among those four doesn't matter, only that this one runs last.

import pandas as pd
from datetime import datetime, timezone

try:
    from src.db import get_conn
except ModuleNotFoundError:
    from db import get_conn

try:
    from src.config import WEIGHTS, TIERS
except ModuleNotFoundError:
    from config import WEIGHTS, TIERS

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# Maps each composite-score component to its scores-table column and its
# config.py weight key.
COMPONENTS = {
    'anomaly_score': 'anomaly',
    'coord_score':   'coordination',
    'dup_score':     'duplication',
    'network_score': 'network',
}


def load_scores():
    conn = get_conn()
    df = pd.read_sql(f"SELECT account_id, {', '.join(COMPONENTS)} FROM scores", conn)
    conn.close()
    logger.info(f"Loaded {len(df)} accounts from scores table")
    return df


def tier_for(score):
    """
    TIERS in config.py is written as integer cutoffs (organic 0-30,
    suspicious 31-60, coordinated 61-100), but influence_score is a
    continuous float -- a literal range check would leave gaps like
    30.45 matching neither "0-30" nor "31-60". Using each tier's upper
    bound as a threshold instead handles the continuous case correctly
    while keeping the exact cutoffs the proposal specifies.
    """
    if score <= TIERS['organic'][1]:
        return 'organic'
    elif score <= TIERS['suspicious'][1]:
        return 'suspicious'
    else:
        return 'coordinated'


def compute_composite(df):
    """
    Missing components (a model that hasn't run yet, or that dropped this
    particular account -- e.g. HDBSCAN's dropna on burstiness_score) are
    treated as 0, NOT re-weighted away across the remaining components: a
    component with no computed value means "no evidence from that
    detector for this account", which should contribute nothing rather
    than being inferred from whichever signals ARE present. An account
    needs at least ONE real signal to get scored at all -- an
    all-missing row is skipped, same spirit as reddit_preprocessor.py
    skipping accounts with zero activity.
    """
    has_any_signal = df[list(COMPONENTS)].notna().any(axis=1)
    n_missing_all = (~has_any_signal).sum()
    df = df[has_any_signal].copy()
    if n_missing_all:
        logger.warning(
            f"Skipped {n_missing_all} accounts with none of the four "
            f"signals computed yet (no model has scored them)"
        )

    coverage = df[list(COMPONENTS)].notna().mean()
    for col, frac in coverage.items():
        if frac < 1.0:
            logger.warning(
                f"{col}: only {frac * 100:.1f}% of accounts have this signal -- "
                f"missing values are treated as 0 for those accounts"
            )

    filled = df[list(COMPONENTS)].fillna(0.0)
    weighted = sum(filled[col] * WEIGHTS[key] for col, key in COMPONENTS.items())

    df['influence_score'] = (weighted * 100).round(2)
    df['tier'] = df['influence_score'].apply(tier_for)

    # Keep the raw components alongside for the summary print / dashboard,
    # filled the same way they were scored so the printed row matches
    # what was actually used to compute influence_score.
    for col in COMPONENTS:
        df[col] = filled[col]

    return df.sort_values('influence_score', ascending=False)


def save_composite(df):
    conn = get_conn()
    c = conn.cursor()
    now = datetime.now(timezone.utc).isoformat()

    # Plain UPDATE, not INSERT-or-update: every row here came FROM the
    # scores table via load_scores(), so a row already exists for each
    # one -- unlike the four model scripts, which may be writing a
    # column for an account for the first time.
    for _, row in df.iterrows():
        c.execute("""
            UPDATE scores
            SET influence_score = ?, tier = ?, scored_at = ?
            WHERE account_id = ?
        """, (float(row['influence_score']), row['tier'], now, row['account_id']))

    conn.commit()
    conn.close()
    logger.info(f"Saved influence_score + tier for {len(df)} accounts")


def print_summary(df):
    print(f"\n{'━'*60}")
    print("  TIER BREAKDOWN")
    print(f"{'━'*60}")
    for tier_name, (low, high) in TIERS.items():
        count = (df['tier'] == tier_name).sum()
        pct = 100 * count / len(df) if len(df) else 0
        print(f"  {tier_name:<12} ({low:>3}-{high:<3}): {count:>5} accounts ({pct:.1f}%)")
    print(f"{'━'*60}")

    print(f"\n{'━'*60}")
    print("  TOP 20 HIGHEST INFLUENCE SCORES")
    print(f"{'━'*60}")
    print(f"{'account_id':<20}{'score':>8}{'tier':>14}{'A':>7}{'C':>7}{'D':>7}{'N':>7}")
    print(f"{'-'*60}")
    for _, row in df.head(20).iterrows():
        print(
            f"{row['account_id'][:18]:<20}{row['influence_score']:>8.1f}{row['tier']:>14}"
            f"{row['anomaly_score']:>7.2f}{row['coord_score']:>7.2f}"
            f"{row['dup_score']:>7.2f}{row['network_score']:>7.2f}"
        )
    print(f"{'━'*60}\n")


if __name__ == "__main__":
    logger.info("Computing composite Influence Score\n")

    df = load_scores()
    df = compute_composite(df)
    save_composite(df)
    print_summary(df)

    logger.info(
        "NOTE: tiers (organic 0-30 / suspicious 31-60 / coordinated "
        "61-100) and weights (40/40/10/10) are exactly what the proposal "
        "specifies in Ch. 4. If the tier breakdown above looks skewed "
        "(e.g. nearly everyone lands in one tier), that's a signal to "
        "revisit the underlying model parameters (contamination, "
        "min_cluster_size, COSINE_THRESHOLD) -- not to change these "
        "weights to compensate."
    )
