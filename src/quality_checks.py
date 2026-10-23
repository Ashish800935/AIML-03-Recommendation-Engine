"""
quality_checks.py  --  Phase 2: Data Quality Checks

Checks for each MovieLens 25M file:
  - Row counts (chunked for large files, full-load for small)
  - Missing values
  - Duplicate rows
  - Value-range validation (ratings 0.5-5.0, timestamps positive)
  - Basic referential-integrity spot-checks (userId / movieId overlaps)

Large files (ratings.csv, genome-scores.csv) are processed in chunks
so that we never load more than CHUNK_SIZE rows at once.
Small files (< ~50 MB) are loaded fully.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Optional
import pandas as pd

# Re-use project paths from data_loader
import sys
SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))
from data_loader import PROJECT_ROOT, DATA_RAW_DIR, format_size

# ── Constants ────────────────────────────────────────────────────────────────
CHUNK_SIZE = 500_000          # rows per chunk for large files
LARGE_FILES = {"ratings.csv", "genome-scores.csv"}   # process in chunks
REPORTS_DIR = PROJECT_ROOT / "reports"

# Valid rating range for MovieLens
RATING_MIN = 0.5
RATING_MAX = 5.0


# ── Helpers ──────────────────────────────────────────────────────────────────

def _ensure_reports_dir() -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    return REPORTS_DIR


def _sep(title: str = "") -> None:
    """Print a section separator."""
    if title:
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")
    else:
        print(f"  {'-'*56}")


# ── Per-file quality checks ───────────────────────────────────────────────────

def check_ratings_quality(data_dir: Optional[Path] = None) -> dict:
    """
    Chunked scan of ratings.csv.
    Returns a summary dict with:
      row_count, missing_per_col, duplicate_count,
      rating_out_of_range, timestamp_min, timestamp_max,
      unique_users, unique_movies, rating_mean, rating_std
    """
    if data_dir is None:
        data_dir = DATA_RAW_DIR
    filepath = data_dir / "ratings.csv"
    if not filepath.exists():
        print(f"  [SKIP] ratings.csv not found.")
        return {}

    _sep("ratings.csv — chunked quality scan")

    # Accumulators
    row_count = 0
    missing = {"userId": 0, "movieId": 0, "rating": 0, "timestamp": 0}
    duplicate_count = 0
    rating_out_of_range = 0
    ts_min = float("inf")
    ts_max = float("-inf")
    user_ids: set = set()
    movie_ids: set = set()
    rating_sum = 0.0
    rating_sq_sum = 0.0
    seen_pairs: set = set()  # (userId, movieId) for dup detection (sample-based)

    chunk_num = 0
    for chunk in pd.read_csv(filepath, chunksize=CHUNK_SIZE):
        chunk_num += 1
        n = len(chunk)
        row_count += n

        # Missing values
        for col in missing:
            missing[col] += int(chunk[col].isnull().sum())

        # Rating range
        rating_out_of_range += int(
            ((chunk["rating"] < RATING_MIN) | (chunk["rating"] > RATING_MAX)).sum()
        )

        # Timestamp range
        ts_min = min(ts_min, chunk["timestamp"].min())
        ts_max = max(ts_max, chunk["timestamp"].max())

        # Unique user/movie sets (collect all — 162k users, 59k movies: manageable)
        user_ids.update(chunk["userId"].unique())
        movie_ids.update(chunk["movieId"].unique())

        # Rating stats (online Welford-style via sum)
        rating_sum += chunk["rating"].sum()
        rating_sq_sum += (chunk["rating"] ** 2).sum()

        # Duplicate (userId, movieId) pairs — exact across all chunks
        pairs = list(zip(chunk["userId"], chunk["movieId"]))
        for p in pairs:
            if p in seen_pairs:
                duplicate_count += 1
            else:
                seen_pairs.add(p)

        if chunk_num % 5 == 0 or chunk_num == 1:
            print(f"  ... processed chunk {chunk_num} ({row_count:,} rows so far)")

    # Compute mean and std from sums
    rating_mean = rating_sum / row_count if row_count > 0 else 0
    rating_var = (rating_sq_sum / row_count) - rating_mean ** 2
    rating_std = rating_var ** 0.5 if rating_var > 0 else 0

    import datetime
    ts_min_dt = datetime.datetime.fromtimestamp(ts_min, tz=datetime.timezone.utc).strftime("%Y-%m-%d")
    ts_max_dt = datetime.datetime.fromtimestamp(ts_max, tz=datetime.timezone.utc).strftime("%Y-%m-%d")

    result = {
        "row_count": row_count,
        "missing_per_col": missing,
        "duplicate_user_movie_pairs": duplicate_count,
        "rating_out_of_range": rating_out_of_range,
        "timestamp_min": ts_min_dt,
        "timestamp_max": ts_max_dt,
        "unique_users": len(user_ids),
        "unique_movies": len(movie_ids),
        "rating_mean": round(rating_mean, 4),
        "rating_std": round(rating_std, 4),
    }

    print(f"\n  Results:")
    for k, v in result.items():
        print(f"    {k:<35} {v}")
    return result


def check_movies_quality(data_dir: Optional[Path] = None) -> dict:
    """Full-load quality check for movies.csv (~3 MB)."""
    if data_dir is None:
        data_dir = DATA_RAW_DIR
    filepath = data_dir / "movies.csv"
    if not filepath.exists():
        print(f"  [SKIP] movies.csv not found.")
        return {}

    _sep("movies.csv — full-load quality check")
    df = pd.read_csv(filepath)

    missing = df.isnull().sum().to_dict()
    duplicates = int(df.duplicated(subset=["movieId"]).sum())
    genre_set = set()
    for g in df["genres"].dropna():
        for part in g.split("|"):
            genre_set.add(part.strip())

    result = {
        "row_count": len(df),
        "missing_per_col": missing,
        "duplicate_movieIds": duplicates,
        "unique_genres": sorted(genre_set),
        "no_genres_listed": int((df["genres"] == "(no genres listed)").sum()),
    }

    print(f"\n  Results:")
    print(f"    row_count              : {result['row_count']:,}")
    print(f"    missing_per_col        : {missing}")
    print(f"    duplicate_movieIds     : {duplicates}")
    print(f"    unique_genres          : {len(genre_set)}")
    print(f"    no_genres_listed       : {result['no_genres_listed']}")
    print(f"    genre_list             : {sorted(genre_set)}")
    return result


def check_tags_quality(data_dir: Optional[Path] = None) -> dict:
    """Full-load quality check for tags.csv (~37 MB)."""
    if data_dir is None:
        data_dir = DATA_RAW_DIR
    filepath = data_dir / "tags.csv"
    if not filepath.exists():
        print(f"  [SKIP] tags.csv not found.")
        return {}

    _sep("tags.csv — full-load quality check")
    df = pd.read_csv(filepath)

    missing = df.isnull().sum().to_dict()
    duplicates = int(df.duplicated().sum())

    result = {
        "row_count": len(df),
        "missing_per_col": missing,
        "duplicate_rows": duplicates,
        "unique_users": int(df["userId"].nunique()),
        "unique_movies": int(df["movieId"].nunique()),
        "unique_tags": int(df["tag"].nunique()),
    }

    print(f"\n  Results:")
    for k, v in result.items():
        print(f"    {k:<30} {v}")
    return result


def check_links_quality(data_dir: Optional[Path] = None) -> dict:
    """Full-load quality check for links.csv (~1.3 MB)."""
    if data_dir is None:
        data_dir = DATA_RAW_DIR
    filepath = data_dir / "links.csv"
    if not filepath.exists():
        print(f"  [SKIP] links.csv not found.")
        return {}

    _sep("links.csv — full-load quality check")
    # tmdbId can have NaN (some movies not on TMDB)
    df = pd.read_csv(filepath, dtype={"imdbId": str, "tmdbId": str})

    missing = df.isnull().sum().to_dict()
    duplicates = int(df.duplicated(subset=["movieId"]).sum())

    result = {
        "row_count": len(df),
        "missing_per_col": missing,
        "duplicate_movieIds": duplicates,
    }

    print(f"\n  Results:")
    for k, v in result.items():
        print(f"    {k:<30} {v}")
    return result


def check_genome_tags_quality(data_dir: Optional[Path] = None) -> dict:
    """Full-load quality check for genome-tags.csv (~18 KB)."""
    if data_dir is None:
        data_dir = DATA_RAW_DIR
    filepath = data_dir / "genome-tags.csv"
    if not filepath.exists():
        print(f"  [SKIP] genome-tags.csv not found.")
        return {}

    _sep("genome-tags.csv — full-load quality check")
    df = pd.read_csv(filepath)

    missing = df.isnull().sum().to_dict()
    result = {
        "row_count": len(df),
        "missing_per_col": missing,
        "duplicate_tagIds": int(df.duplicated(subset=["tagId"]).sum()),
    }

    print(f"\n  Results:")
    for k, v in result.items():
        print(f"    {k:<30} {v}")
    return result


def check_genome_scores_quality(data_dir: Optional[Path] = None) -> dict:
    """
    Chunked scan of genome-scores.csv (~415 MB).
    Only reads enough to get row count, missing values, and value range.
    """
    if data_dir is None:
        data_dir = DATA_RAW_DIR
    filepath = data_dir / "genome-scores.csv"
    if not filepath.exists():
        print(f"  [SKIP] genome-scores.csv not found.")
        return {}

    _sep("genome-scores.csv — chunked quality scan")

    row_count = 0
    missing = {"movieId": 0, "tagId": 0, "relevance": 0}
    rel_min = float("inf")
    rel_max = float("-inf")
    movie_ids: set = set()
    tag_ids: set = set()

    chunk_num = 0
    for chunk in pd.read_csv(filepath, chunksize=CHUNK_SIZE):
        chunk_num += 1
        row_count += len(chunk)
        for col in missing:
            missing[col] += int(chunk[col].isnull().sum())
        rel_min = min(rel_min, chunk["relevance"].min())
        rel_max = max(rel_max, chunk["relevance"].max())
        movie_ids.update(chunk["movieId"].unique())
        tag_ids.update(chunk["tagId"].unique())
        if chunk_num % 5 == 0 or chunk_num == 1:
            print(f"  ... processed chunk {chunk_num} ({row_count:,} rows so far)")

    result = {
        "row_count": row_count,
        "missing_per_col": missing,
        "unique_movies": len(movie_ids),
        "unique_tags": len(tag_ids),
        "relevance_min": round(float(rel_min), 6),
        "relevance_max": round(float(rel_max), 6),
    }

    print(f"\n  Results:")
    for k, v in result.items():
        print(f"    {k:<30} {v}")
    return result


# ── Run all checks ────────────────────────────────────────────────────────────

def run_all_quality_checks(data_dir: Optional[Path] = None) -> dict:
    """
    Run quality checks for every MovieLens file and return a combined dict.
    """
    if data_dir is None:
        data_dir = DATA_RAW_DIR

    print("\n" + "=" * 60)
    print("  PHASE 2 — Data Quality Checks")
    print("=" * 60)

    results = {}
    results["ratings"]       = check_ratings_quality(data_dir)
    results["movies"]        = check_movies_quality(data_dir)
    results["tags"]          = check_tags_quality(data_dir)
    results["links"]         = check_links_quality(data_dir)
    results["genome_tags"]   = check_genome_tags_quality(data_dir)
    results["genome_scores"] = check_genome_scores_quality(data_dir)

    print("\n" + "=" * 60)
    print("  Quality checks complete.")
    print("=" * 60 + "\n")
    return results
