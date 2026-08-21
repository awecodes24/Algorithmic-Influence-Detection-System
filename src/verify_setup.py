from __future__ import annotations

import importlib

from src.config import (
    BASE_DIR,
    DB_PATH,
    BENCHMARK_DB_PATH,
    WEIGHTS,
    ensure_runtime_ready,
)

from src.db import get_conn, init_db


REQUIRED_MODULES = (
    "pandas",
    "numpy",
    "sklearn",
    "hdbscan",
    "networkx",
    "streamlit",
    "plotly",
    "dotenv",
)


def check_imports():
    missing = []

    for module in REQUIRED_MODULES:
        try:
            importlib.import_module(module)
        except ImportError:
            missing.append(module)

    return missing


def main():
    ensure_runtime_ready()

    print("Project root:")
    print(f"  {BASE_DIR}")

    print("\nDatabases:")
    print(f"  Reddit:    {DB_PATH}")
    print(f"  Benchmark: {BENCHMARK_DB_PATH}")

    print("\nInfluence Score weights:")
    print(f"  {WEIGHTS}")
    print(f"  Sum: {sum(WEIGHTS.values())}")

    missing = check_imports()

    if missing:
        print("\nMissing packages:")
        for package in missing:
            print(f"  - {package}")

        return 1

    init_db()

    conn = get_conn()

    try:
        tables = conn.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
            ORDER BY name
            """
        ).fetchall()

        print(f"\nReddit database tables: {len(tables)}")

        for row in tables:
            print(f"  ✓ {row[0]}")

    finally:
        conn.close()

    print("\nPhase 0 setup check: PASS")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())