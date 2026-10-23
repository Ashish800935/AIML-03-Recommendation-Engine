"""
data_loader.py  —  Phase 1: Dataset Verification Utilities

This module provides simple helper functions to:
  1. Check that the expected MovieLens CSV files exist on disk.
  2. Print their file sizes (without loading them fully).
  3. Read a small configurable sample and display basic statistics.
  4. Validate that the expected columns are present.

Nothing here loads the full dataset into memory.
"""

import os
from pathlib import Path
from typing import Optional
import pandas as pd


# ---------------------------------------------------------------------------
# Project-relative path configuration
# ---------------------------------------------------------------------------

# Resolve the project root as two levels up from this file:
#   src/data_loader.py  →  project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"

# Expected files and their required columns
EXPECTED_FILES: dict[str, list[str]] = {
    "ratings.csv":      ["userId", "movieId", "rating", "timestamp"],
    "movies.csv":       ["movieId", "title", "genres"],
    "tags.csv":         ["userId", "movieId", "tag", "timestamp"],
    "links.csv":        ["movieId", "imdbId", "tmdbId"],
    "genome-scores.csv":["movieId", "tagId", "relevance"],
    "genome-tags.csv":  ["tagId", "tag"],
}


# ---------------------------------------------------------------------------
# Helper: human-readable file size
# ---------------------------------------------------------------------------

def format_size(size_bytes: int) -> str:
    """Convert a byte count to a human-readable string (KB / MB / GB)."""
    if size_bytes >= 1_073_741_824:
        return f"{size_bytes / 1_073_741_824:.2f} GB"
    elif size_bytes >= 1_048_576:
        return f"{size_bytes / 1_048_576:.2f} MB"
    elif size_bytes >= 1_024:
        return f"{size_bytes / 1_024:.2f} KB"
    return f"{size_bytes} bytes"


# ---------------------------------------------------------------------------
# Step 1: Check file existence and sizes
# ---------------------------------------------------------------------------

def check_files(data_dir: Optional[Path] = None) -> dict[str, bool]:
    """
    Verify that every expected MovieLens CSV file exists in data_dir.

    Returns a dict mapping filename → True (found) / False (missing).
    Also prints a summary table to stdout.
    """
    if data_dir is None:
        data_dir = DATA_RAW_DIR

    print(f"\n{'='*60}")
    print(f"  Dataset Directory: {data_dir}")
    print(f"{'='*60}")
    print(f"  {'File':<25} {'Status':<10} {'Size'}")
    print(f"  {'-'*55}")

    results: dict[str, bool] = {}
    for filename in EXPECTED_FILES:
        filepath = data_dir / filename
        if filepath.exists():
            size = format_size(filepath.stat().st_size)
            print(f"  {'[OK] ' + filename:<25} {'FOUND':<10} {size}")
            results[filename] = True
        else:
            print(f"  {'[!!] ' + filename:<25} {'MISSING':<10} ---")
            results[filename] = False

    found = sum(results.values())
    total = len(results)
    print(f"\n  Summary: {found}/{total} files found.")
    print(f"{'='*60}\n")
    return results


# ---------------------------------------------------------------------------
# Step 2: Read a small sample and show basic statistics
# ---------------------------------------------------------------------------

def sample_csv(
    filename: str,
    data_dir: Optional[Path] = None,
    n_rows: int = 5,
) -> Optional[pd.DataFrame]:
    """
    Read only the first `n_rows` rows of a CSV file using pandas nrows.

    This is safe even for very large files (e.g. ratings.csv at ~650 MB)
    because pandas stops reading after n_rows without loading the rest.

    Returns a DataFrame, or None if the file is missing.
    """
    if data_dir is None:
        data_dir = DATA_RAW_DIR

    filepath = data_dir / filename
    if not filepath.exists():
        print(f"  [SKIP] '{filename}' not found - skipping sample.")
        return None

    df = pd.read_csv(filepath, nrows=n_rows)
    return df


def display_sample(
    filename: str,
    data_dir: Optional[Path] = None,
    n_rows: int = 5,
) -> Optional[pd.DataFrame]:
    """
    Sample a CSV file and print column names, sample rows,
    data types, and missing-value counts for the sampled rows.
    """
    df = sample_csv(filename, data_dir, n_rows)
    if df is None:
        return None

    print(f"\n{'-'*60}")
    print(f"  File      : {filename}  (showing first {n_rows} rows)")
    print(f"{'-'*60}")
    print(f"\n  Columns   : {list(df.columns)}")
    print(f"\n  Sample rows:")
    # Use to_string so the full row fits in the terminal
    print(df.to_string(index=False))
    print(f"\n  Data types:")
    for col, dtype in df.dtypes.items():
        print(f"    {col:<20} {dtype}")
    print(f"\n  Missing values (in {n_rows}-row sample):")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("    None found in this sample.")
    else:
        for col, count in missing.items():
            if count > 0:
                print(f"    {col}: {count} missing")
    print()
    return df


# ---------------------------------------------------------------------------
# Step 3: Validate column schemas
# ---------------------------------------------------------------------------

def validate_columns(
    filename: str,
    data_dir: Optional[Path] = None,
) -> bool:
    """
    Read only the header line of a CSV and check that all required
    columns are present.  Does NOT load any data rows.

    Returns True if schema is valid, False otherwise.
    """
    if data_dir is None:
        data_dir = DATA_RAW_DIR

    required_cols = EXPECTED_FILES.get(filename)
    if required_cols is None:
        print(f"  [WARN] '{filename}' is not in the expected-files list.")
        return False

    filepath = data_dir / filename
    if not filepath.exists():
        print(f"  [ERROR] '{filename}' not found — cannot validate columns.")
        return False

    # Read only the header (nrows=0 returns an empty DataFrame with column names)
    df_header = pd.read_csv(filepath, nrows=0)
    actual_cols = list(df_header.columns)
    missing_cols = [c for c in required_cols if c not in actual_cols]

    if missing_cols:
        print(
            f"  [SCHEMA ERROR] '{filename}': missing columns → {missing_cols}\n"
            f"  Found columns: {actual_cols}"
        )
        return False

    print(f"  [OK] '{filename}' schema is valid. Columns: {actual_cols}")
    return True


# ---------------------------------------------------------------------------
# Step 4: Run full verification for all expected files
# ---------------------------------------------------------------------------

def run_full_verification(
    data_dir: Optional[Path] = None,
    sample_rows: int = 5,
) -> None:
    """
    Orchestrate the complete Phase 1 verification:
      1. Check file existence and sizes.
      2. Validate column schemas (header only — no full load).
      3. Display a small sample for ratings.csv and movies.csv.
    """
    if data_dir is None:
        data_dir = DATA_RAW_DIR

    # 1. File existence check
    file_status = check_files(data_dir)

    # 2. Schema validation for every found file
    print("\n  Column Schema Validation")
    print(f"  {'-'*55}")
    all_schemas_ok = True
    for filename, found in file_status.items():
        if found:
            ok = validate_columns(filename, data_dir)
            if not ok:
                all_schemas_ok = False
        else:
            print(f"  [SKIP] '{filename}' not found - skipping schema check.")

    if all_schemas_ok:
        print("\n  [OK] All found files passed schema validation.\n")
    else:
        print("\n  [!!] Some files failed schema validation. See messages above.\n")

    # 3. Display samples for the two primary files
    print("\n  Sample Data Preview")
    for fname in ["ratings.csv", "movies.csv"]:
        if file_status.get(fname):
            display_sample(fname, data_dir, n_rows=sample_rows)
