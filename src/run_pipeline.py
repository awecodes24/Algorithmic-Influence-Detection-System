from __future__ import annotations

import subprocess
import sys

from src.config import BASE_DIR, ensure_runtime_ready
from src.db import init_db


# Dependency order, confirmed by reading each script's actual SQL
# (INSERT/UPDATE vs FROM) rather than assumed:
#
#   reddit_preprocessor        -> features
#   reddit_isolation_forest    -> scores.anomaly_score          (reads features)
#   reddit_hdbscan              -> scores.coord_score            (reads features)
#   reddit_cosine_similarity    -> scores.dup_score,
#                                   content_similarity,
#                                   coordination_events           (reads posts/comments)
#   reddit_content_coordination -> content_similarity             (reads posts/comments;
#                                   independent of cosine_similarity's own
#                                   content_similarity rows -- both write
#                                   to the same table from different logic)
#   reddit_temporal_coordination -> temporal_similarity,
#                                    coordination_events,
#                                    scores.temporal_score         (reads posts/comments)
#   reddit_networkx              -> edges, communities,
#                                    scores.network_score(_topic_scoped)
#                                    (reads accounts/comments)
#   build_account_pairs          -> account_pairs
#                                    (reads content_similarity, temporal_similarity, edges
#                                     -- MUST run after all three of the above)
#   composite_score               -> scores.influence_score, tier,
#                                     evidence_status, confidence_level, assessment
#                                     (reads account_pairs, coordination_events, scores
#                                      -- MUST run last: it's the only script whose
#                                      evidence_status/assessment logic sees all three
#                                      evidence channels -- content, temporal, AND
#                                      account_pairs -- combined. See note below.)
#
# update_coordination_evidence.py is DELIBERATELY NOT in this list. It
# recomputes evidence_status/confidence_level/assessment independently of
# composite_score.py, using only account_pairs (no visibility into
# coordination_events' content/temporal evidence), and would either be
# silently overwritten (if run before composite_score) or silently
# overwrite composite_score's more complete answer (if run after) --
# confirmed on real data: accounts exist where the two disagree between
# e.g. "suspicious" and "high_priority_coordinated_pattern" depending
# purely on which one ran last. Its pair-level aggregation
# (get_account_coordination_summary) is still useful as a standalone
# diagnostic -- run it manually with `python -m
# src.models.update_coordination_evidence` when you want to inspect
# account_pairs evidence in isolation, just don't wire it into the
# automated pipeline's writes to scores.
STEPS = [
    ("Feature engineering", "src.reddit_preprocessor"),
    ("Isolation Forest", "src.models.reddit_isolation_forest"),
    ("HDBSCAN clustering", "src.models.reddit_hdbscan"),
    ("Cosine similarity", "src.models.reddit_cosine_similarity"),
    ("Content coordination", "src.models.reddit_content_coordination"),
    ("Temporal coordination", "src.models.reddit_temporal_coordination"),
    ("NetworkX / PageRank", "src.models.reddit_networkx"),
    ("Account pair evidence", "src.models.build_account_pairs"),
    ("Composite Influence Score", "src.composite_score"),
]


def run_step(label, module):
    print("\n" + "=" * 64)
    print(f"  {label}")
    print(f"  python -m {module}")
    print("=" * 64)

    result = subprocess.run(
        [sys.executable, "-m", module],
        cwd=BASE_DIR,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"{label} failed with exit code {result.returncode}"
        )


def main():
    ensure_runtime_ready()
    init_db()

    for index, (label, module) in enumerate(STEPS, start=1):
        print(f"\nSTEP {index}/{len(STEPS)}")
        run_step(label, module)

    print("\n" + "=" * 64)
    print("  REDDIT ANALYSIS PIPELINE COMPLETE")
    print("=" * 64)
    print("Run: streamlit run src/dashboard.py")
    print(
        "\nOptional: python -m src.models.update_coordination_evidence\n"
        "  Standalone pair-level evidence diagnostic (account_pairs\n"
        "  only). Does NOT feed the dashboard or influence_score --\n"
        "  composite_score.py above already computed the fuller\n"
        "  evidence_status/assessment using content + temporal +\n"
        "  account_pairs combined. Running this after would overwrite\n"
        "  that with a narrower answer, so it's left out of the\n"
        "  automated steps above -- run it by hand only when you want\n"
        "  to inspect account_pairs evidence in isolation."
    )


if __name__ == "__main__":
    main()