from __future__ import annotations

import subprocess
import sys

from pathlib import Path


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

DATABASE = (
    PROJECT_ROOT
    / "data"
    / "benchmarks"
    / "cresci-2017.db"
)

FINAL_OUTPUT = (
    PROJECT_ROOT
    / "outputs"
    / "cresci"
    / "final"
)


def run(
    module: str,
    *arguments: str,
) -> None:

    command = [
        sys.executable,
        "-m",
        module,
        *arguments,
    ]

    print(
        "\n"
        + "=" * 75
    )

    print(
        "RUNNING:",
        " ".join(command),
    )

    print(
        "=" * 75
    )

    subprocess.run(
        command,
        check=True,
        cwd=PROJECT_ROOT,
    )


def main():

    database = str(
        DATABASE
    )

    # --------------------------------------------------
    # 1. Validate existing split
    # --------------------------------------------------

    run(
        "src.benchmarks.cresci.split",
        "--database",
        database,
    )

    # --------------------------------------------------
    # 2. Validate/build base features
    # --------------------------------------------------

    run(
        "src.benchmarks.cresci.features",
        "--database",
        database,
    )

    # --------------------------------------------------
    # 3. Build/validate temporal features
    # --------------------------------------------------

    run(
        "src.benchmarks.cresci.temporal",
        "--database",
        database,
        "--window",
        "5",
    )

    # --------------------------------------------------
    # 4. Build/validate final 42-feature table
    # --------------------------------------------------

    run(
        "src.benchmarks.cresci.final_features",
        "--database",
        database,
    )

    # --------------------------------------------------
    # 5. Train final model
    #
    # Explicit --overwrite is intentionally NOT used.
    # The first clean final run should be deliberate.
    # --------------------------------------------------

    model_path = (
        FINAL_OUTPUT
        / "cresci_final_model.joblib"
    )

    if not model_path.exists():

        run(
            "src.benchmarks.cresci.train",
            "--database",
            database,
        )

    else:

        print(
            "\nFinal model already exists:"
        )

        print(
            f"  {model_path}"
        )

        print(
            "Skipping retraining."
        )

        print(
            "Use train.py --overwrite to "
            "intentionally retrain."
        )

    # --------------------------------------------------
    # 6. Category evaluation
    # --------------------------------------------------

    run(
        "src.benchmarks.cresci.category_evaluation",
        "--database",
        database,
        "--model",
        str(
            model_path
        ),
        "--output-dir",
        str(
            FINAL_OUTPUT
        ),
        "--overwrite",
    )

    print(
        "\n"
        + "=" * 75
    )

    print(
        "CRESCI-2017 PIPELINE COMPLETE"
    )

    print(
        "=" * 75
    )

    print(
        f"Final output: {FINAL_OUTPUT}"
    )


if __name__ == "__main__":
    main()