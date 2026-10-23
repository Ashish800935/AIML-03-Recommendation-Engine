"""
run_phase2.py  --  Phase 2: Command-line entry point

Usage (from project root):
    python scripts/run_phase2.py              # quality checks + full EDA
    python scripts/run_phase2.py --quality-only   # only quality checks
    python scripts/run_phase2.py --eda-only        # only EDA charts

Outputs are saved to reports/.
"""

import argparse
import sys
from pathlib import Path

# Allow imports from src/ when running directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from quality_checks import run_all_quality_checks
from eda import run_full_eda
from data_loader import DATA_RAW_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 2 — Data Quality Checks and Exploratory Data Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/run_phase2.py\n"
            "  python scripts/run_phase2.py --quality-only\n"
            "  python scripts/run_phase2.py --eda-only\n"
            "  python scripts/run_phase2.py --data-dir data/raw\n"
        ),
    )
    parser.add_argument(
        "--data-dir", type=Path, default=None,
        help="Path to data/raw directory. Defaults to project-relative data/raw.",
    )
    parser.add_argument(
        "--quality-only", action="store_true",
        help="Run only data quality checks (skip EDA charts).",
    )
    parser.add_argument(
        "--eda-only", action="store_true",
        help="Run only EDA charts (skip quality checks).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.data_dir is not None:
        data_dir = (args.data_dir if args.data_dir.is_absolute()
                    else Path.cwd() / args.data_dir)
    else:
        data_dir = DATA_RAW_DIR

    print("\n" + "=" * 60)
    print("  AIML-03 Recommendation Engine — Phase 2")
    print("=" * 60)
    print(f"  Data directory : {data_dir}")

    if args.eda_only:
        run_full_eda(data_dir)
    elif args.quality_only:
        run_all_quality_checks(data_dir)
    else:
        # Default: run both
        run_all_quality_checks(data_dir)
        run_full_eda(data_dir)

    print("\n  All Phase 2 outputs saved to reports/")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
