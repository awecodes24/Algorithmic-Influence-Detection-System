# run_pipeline.py
# Convenience runner for everything AFTER data collection: feature
# engineering -> all four detection models -> composite score.
# Run this again every time you collect a fresh batch with collector.py.
#
# Data collection (collector.py) and the dashboard (dashboard.py) are
# deliberately NOT included here -- collection needs your live Apify
# token and judgment about timing/volume, and the dashboard is a server
# you leave running, not a one-shot step.
#
# Usage: python run_pipeline.py   (from the project root)

import subprocess
import sys

STEPS = [
    ("Feature engineering",          ["python3", "src/reddit_preprocessor.py"]),
    ("Isolation Forest",             ["python3", "src/models/reddit_isolation_forest.py"]),
    ("HDBSCAN clustering",           ["python3", "src/models/reddit_hdbscan.py"]),
    ("Cosine similarity",            ["python3", "src/models/reddit_cosine_similarity.py"]),
    ("NetworkX / PageRank",          ["python3", "src/models/reddit_networkx.py"]),
    ("Composite Influence Score",    ["python3", "src/composite_score.py"]),
]


def main():
    for i, (label, cmd) in enumerate(STEPS, start=1):
        print(f"\n{'=' * 60}")
        print(f"  STEP {i}/{len(STEPS)}: {label}")
        print(f"{'=' * 60}")

        result = subprocess.run(cmd)

        if result.returncode != 0:
            print(f"\n[STOPPED] '{label}' failed (exit code {result.returncode}).")
            print("Fix the error above before continuing -- later steps depend on this one's output.")
            sys.exit(1)

    print(f"\n{'=' * 60}")
    print("  ALL STEPS COMPLETE")
    print(f"{'=' * 60}")
    print("Run `streamlit run src/dashboard.py` to view the results.")


if __name__ == "__main__":
    main()
