"""
verify_dataset.py  —  Phase 1: Command-line entry point

Usage (from the project root):
    python scripts/verify_dataset.py
    python scripts/verify_dataset.py --rows 10
    python scripts/verify_dataset.py --data-dir path/to/custom/raw

This script calls src/data_loader.py to verify the MovieLens 25M dataset
without loading large files into memory.
"""

import argparse
import sys
from pathlib import Path

# Allow imports from src/ when running this script directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data_loader import run_full_verification, DATA_RAW_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 1 — MovieLens 25M Dataset Verification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/verify_dataset.py\n"
            "  python scripts/verify_dataset.py --rows 10\n"
            "  python scripts/verify_dataset.py --data-dir data/raw\n"
        ),
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=5,
        help="Number of sample rows to display per file (default: 5).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Path to the raw data directory. Defaults to data/raw/ relative to project root.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve data directory
    if args.data_dir is not None:
        # Support both absolute and project-relative paths
        data_dir = args.data_dir if args.data_dir.is_absolute() else Path.cwd() / args.data_dir
    else:
        data_dir = DATA_RAW_DIR

    print("\n" + "=" * 60)
    print("  AIML-03 Recommendation Engine — Phase 1 Verification")
    print("=" * 60)

    run_full_verification(data_dir=data_dir, sample_rows=args.rows)

    print("=" * 60)
    print("  Verification complete.  See output above for any issues.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
