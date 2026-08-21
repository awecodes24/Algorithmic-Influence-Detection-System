from __future__ import annotations

import subprocess
import sys

from src.config import BASE_DIR, ensure_runtime_ready
from src.db import init_db


STEPS = [
    ("Feature engineering", "src.reddit_preprocessor"),
    ("Isolation Forest", "src.models.reddit_isolation_forest"),
    ("HDBSCAN clustering", "src.models.reddit_hdbscan"),
    ("Cosine similarity", "src.models.reddit_cosine_similarity"),
    ("NetworkX / PageRank", "src.models.reddit_networkx"),
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


if __name__ == "__main__":
    main()