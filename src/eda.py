"""
eda.py  --  Phase 2: Exploratory Data Analysis

Analyses performed:
  1. Rating distribution (histogram + stats)
  2. Ratings per user (log-scale histogram)
  3. Ratings per movie (log-scale histogram)
  4. Most-rated movies (top 20 bar chart)
  5. Rating trend over time (monthly average rating)
  6. Genre distribution (bar chart)
  7. Timestamp range / volume over years

All charts are saved to reports/ as PNG files.
All large files (ratings.csv) are read in chunks.
"""

from __future__ import annotations

import datetime
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — works without a display
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import sys
SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))
from data_loader import PROJECT_ROOT, DATA_RAW_DIR

REPORTS_DIR = PROJECT_ROOT / "reports"
CHUNK_SIZE = 500_000

# ── Helpers ───────────────────────────────────────────────────────────────────

def _ensure_reports() -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    return REPORTS_DIR


def _save(fig: plt.Figure, name: str) -> Path:
    out = _ensure_reports() / name
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVED] {out}")
    return out


def _timestamp_to_year_month(ts: pd.Series) -> pd.Series:
    """Convert Unix timestamps to YYYY-MM strings."""
    return pd.to_datetime(ts, unit="s").dt.to_period("M").astype(str)


# ── Analysis 1: Rating distribution ──────────────────────────────────────────

def analyse_rating_distribution(data_dir: Optional[Path] = None) -> dict:
    """
    Counts how many ratings fall in each half-star bucket (0.5 – 5.0).
    Uses chunked reading.  Returns a dict of {rating_value: count}.
    """
    if data_dir is None:
        data_dir = DATA_RAW_DIR
    filepath = data_dir / "ratings.csv"

    print("\n  [1/7] Rating distribution (chunked)...")
    counts: Counter = Counter()
    total_ratings = 0
    rating_sum = 0.0

    for chunk in pd.read_csv(filepath, chunksize=CHUNK_SIZE,
                             usecols=["rating"]):
        counts.update(chunk["rating"].value_counts().to_dict())
        total_ratings += len(chunk)
        rating_sum += chunk["rating"].sum()

    avg = rating_sum / total_ratings if total_ratings else 0
    sorted_counts = dict(sorted(counts.items()))

    # Plot
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = [str(k) for k in sorted_counts.keys()]
    values = list(sorted_counts.values())
    bars = ax.bar(labels, values, color="#4C72B0", edgecolor="white", width=0.6)
    ax.set_title(f"Rating Distribution (n={total_ratings:,}, mean={avg:.2f})",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Rating", fontsize=11)
    ax.set_ylabel("Number of Ratings", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{int(x):,}"))
    ax.axvline(x=labels.index(f"{round(avg*2)/2:.1f}"), color="red",
               linestyle="--", alpha=0.6, label=f"Mean ~ {avg:.2f}")
    ax.legend()
    fig.tight_layout()
    _save(fig, "01_rating_distribution.png")

    print(f"    total ratings : {total_ratings:,}")
    print(f"    mean rating   : {avg:.4f}")
    return {"counts": sorted_counts, "total": total_ratings, "mean": round(avg, 4)}


# ── Analysis 2 & 3: Ratings per user / movie ─────────────────────────────────

def analyse_user_movie_interactions(data_dir: Optional[Path] = None) -> dict:
    """
    Computes per-user and per-movie rating counts using chunked reading.
    Saves two log-scale histograms.
    """
    if data_dir is None:
        data_dir = DATA_RAW_DIR
    filepath = data_dir / "ratings.csv"

    print("\n  [2/7] User/movie interaction distributions (chunked)...")
    user_counts: Counter = Counter()
    movie_counts: Counter = Counter()

    for chunk in pd.read_csv(filepath, chunksize=CHUNK_SIZE,
                             usecols=["userId", "movieId"]):
        user_counts.update(chunk["userId"].value_counts().to_dict())
        movie_counts.update(chunk["movieId"].value_counts().to_dict())

    u_vals = list(user_counts.values())
    m_vals = list(movie_counts.values())

    # Ratings per user
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(u_vals, bins=100, color="#DD8452", edgecolor="white", log=True)
    ax.set_title(f"Ratings per User  (n={len(u_vals):,} unique users)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Number of Ratings", fontsize=11)
    ax.set_ylabel("Number of Users (log scale)", fontsize=11)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{int(x):,}"))
    fig.tight_layout()
    _save(fig, "02_ratings_per_user.png")

    # Ratings per movie
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(m_vals, bins=100, color="#55A868", edgecolor="white", log=True)
    ax.set_title(f"Ratings per Movie  (n={len(m_vals):,} unique movies)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Number of Ratings", fontsize=11)
    ax.set_ylabel("Number of Movies (log scale)", fontsize=11)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{int(x):,}"))
    fig.tight_layout()
    _save(fig, "03_ratings_per_movie.png")

    return {
        "unique_users": len(u_vals),
        "unique_movies": len(m_vals),
        "user_rating_min": min(u_vals),
        "user_rating_max": max(u_vals),
        "user_rating_median": int(np.median(u_vals)),
        "movie_rating_min": min(m_vals),
        "movie_rating_max": max(m_vals),
        "movie_rating_median": int(np.median(m_vals)),
    }


# ── Analysis 4: Top-rated movies ─────────────────────────────────────────────

def analyse_most_rated_movies(data_dir: Optional[Path] = None,
                               top_n: int = 20) -> pd.DataFrame:
    """
    Returns the top_n most-rated movies.  Joins with movies.csv for titles.
    Uses chunked reading for ratings.csv.
    """
    if data_dir is None:
        data_dir = DATA_RAW_DIR

    print(f"\n  [3/7] Top {top_n} most-rated movies (chunked)...")
    movie_counts: Counter = Counter()
    movie_sum: defaultdict = defaultdict(float)

    for chunk in pd.read_csv(data_dir / "ratings.csv", chunksize=CHUNK_SIZE,
                             usecols=["movieId", "rating"]):
        agg = chunk.groupby("movieId")["rating"].agg(["sum", "count"])
        for mid, s, c in agg.itertuples():
            movie_sum[mid] += s
            movie_counts[mid] += int(c)

    # Get top_n
    top_ids = [mid for mid, _ in movie_counts.most_common(top_n)]
    movies_df = pd.read_csv(data_dir / "movies.csv",
                            usecols=["movieId", "title"])
    id_to_title = dict(zip(movies_df["movieId"], movies_df["title"]))

    rows = []
    for mid in top_ids:
        s = movie_sum[mid]
        c = movie_counts[mid]
        rows.append({
            "movieId": mid,
            "title": id_to_title.get(mid, f"Movie {mid}"),
            "rating_count": c,
            "avg_rating": round(s / c, 3) if c else 0,
        })
    result_df = pd.DataFrame(rows).sort_values("rating_count", ascending=False)

    # Plot
    fig, ax = plt.subplots(figsize=(11, 7))
    colors = plt.cm.Blues_r(np.linspace(0.3, 0.9, top_n))
    ax.barh(result_df["title"][::-1], result_df["rating_count"][::-1],
            color=colors, edgecolor="white")
    ax.set_title(f"Top {top_n} Most-Rated Movies", fontsize=13, fontweight="bold")
    ax.set_xlabel("Number of Ratings", fontsize=11)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{int(x):,}"))
    fig.tight_layout()
    _save(fig, "04_top_rated_movies.png")

    print(result_df.to_string(index=False))
    return result_df


# ── Analysis 5: Rating trend over time ───────────────────────────────────────

def analyse_rating_trend(data_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Computes monthly average rating and monthly volume over the full dataset.
    Uses chunked reading.
    """
    if data_dir is None:
        data_dir = DATA_RAW_DIR

    print("\n  [4/7] Rating trend over time (chunked)...")
    monthly_sum: defaultdict = defaultdict(float)
    monthly_count: defaultdict = defaultdict(int)
    yearly_count: defaultdict = defaultdict(int)

    for chunk in pd.read_csv(data_dir / "ratings.csv", chunksize=CHUNK_SIZE,
                             usecols=["rating", "timestamp"]):
        dt_col = pd.to_datetime(chunk["timestamp"], unit="s")
        chunk["year_month"] = dt_col.dt.to_period("M").astype(str)
        chunk["year"] = dt_col.dt.year

        ym_agg = chunk.groupby("year_month")["rating"].agg(["sum", "count"])
        for ym, s, c in ym_agg.itertuples():
            monthly_sum[ym] += float(s)
            monthly_count[ym] += int(c)

        yr_counts = chunk["year"].value_counts()
        for yr, c in yr_counts.items():
            yearly_count[int(yr)] += int(c)

    # Build monthly df
    months = sorted(monthly_sum.keys())
    trend_df = pd.DataFrame({
        "year_month": months,
        "avg_rating": [monthly_sum[m] / monthly_count[m] for m in months],
        "rating_count": [monthly_count[m] for m in months],
    })

    # Plot 1: Monthly average rating
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)

    ax = axes[0]
    ax.plot(range(len(months)), trend_df["avg_rating"], color="#4C72B0",
            linewidth=1.5)
    ax.set_title("Monthly Average Rating Over Time", fontsize=12, fontweight="bold")
    ax.set_ylabel("Avg Rating", fontsize=10)
    ax.set_ylim(3.0, 4.5)
    step = max(1, len(months) // 12)
    ax.set_xticks(range(0, len(months), step))
    ax.set_xticklabels(months[::step], rotation=45, ha="right", fontsize=7)

    # Plot 2: Yearly volume
    ax2 = axes[1]
    yrs = sorted(yearly_count.keys())
    ax2.bar(yrs, [yearly_count[y] for y in yrs], color="#DD8452", edgecolor="white")
    ax2.set_title("Ratings Volume by Year", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Number of Ratings", fontsize=10)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{int(x/1e6):.0f}M" if x >= 1e6 else f"{int(x):,}"))
    ax2.set_xlabel("Year", fontsize=10)

    fig.tight_layout()
    _save(fig, "05_rating_trend_over_time.png")

    print(f"    Date range: {months[0]} to {months[-1]}")
    print(f"    Total months with data: {len(months)}")
    return trend_df


# ── Analysis 6: Genre distribution ───────────────────────────────────────────

def analyse_genre_distribution(data_dir: Optional[Path] = None) -> pd.Series:
    """
    Explodes the pipe-separated genres column and counts movies per genre.
    movies.csv is small (~3 MB) so we load it fully.
    """
    if data_dir is None:
        data_dir = DATA_RAW_DIR

    print("\n  [5/7] Genre distribution...")
    df = pd.read_csv(data_dir / "movies.csv")

    # Expand genres
    genre_series = (
        df["genres"]
        .dropna()
        .str.split("|")
        .explode()
        .str.strip()
    )
    genre_counts = genre_series.value_counts()
    genre_counts = genre_counts[genre_counts.index != "(no genres listed)"]

    fig, ax = plt.subplots(figsize=(11, 6))
    colors = plt.cm.tab20(np.linspace(0, 1, len(genre_counts)))
    ax.barh(genre_counts.index[::-1], genre_counts.values[::-1],
            color=colors[::-1], edgecolor="white")
    ax.set_title("Movie Genre Distribution (by number of movies)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Number of Movies", fontsize=11)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{int(x):,}"))
    fig.tight_layout()
    _save(fig, "06_genre_distribution.png")

    print(genre_counts.to_string())
    return genre_counts


# ── Analysis 7: Train/Val/Test split proposal ─────────────────────────────────

def propose_temporal_split(data_dir: Optional[Path] = None) -> dict:
    """
    Scans timestamp min/max (chunked) and proposes chronological split boundaries.

    Split strategy:
      - Train : interactions before the 80th percentile timestamp
      - Val   : 80th – 90th percentile
      - Test  : 90th percentile onward

    Why chronological? Using a random split would allow the model to train on
    future ratings of a user and predict past ones — temporal leakage.

    We do NOT execute the split here; we only compute and report the cutoff dates.
    """
    if data_dir is None:
        data_dir = DATA_RAW_DIR

    print("\n  [6/7] Temporal split proposal (chunked timestamp scan)...")

    # Collect all timestamps in a memory-efficient way using chunks
    chunks: list[np.ndarray] = []
    for chunk in pd.read_csv(data_dir / "ratings.csv", chunksize=CHUNK_SIZE,
                             usecols=["timestamp"]):
        chunks.append(chunk["timestamp"].to_numpy(dtype=np.int64))

    all_ts = np.concatenate(chunks)
    all_ts.sort()
    total = len(all_ts)

    def ts_to_str(ts: int | np.integer) -> str:
        return datetime.datetime.fromtimestamp(int(ts), tz=datetime.timezone.utc).strftime("%Y-%m-%d")

    p80_ts = int(all_ts[int(0.80 * total)])
    p90_ts = int(all_ts[int(0.90 * total)])

    result = {
        "total_ratings": total,
        "ts_min": ts_to_str(all_ts[0]),
        "ts_max": ts_to_str(all_ts[-1]),
        "train_cutoff": ts_to_str(p80_ts),
        "val_cutoff": ts_to_str(p90_ts),
        "train_unix": p80_ts,
        "val_unix": p90_ts,
        "train_pct": "~80%",
        "val_pct": "~10%",
        "test_pct": "~10%",
    }

    print(f"\n    Chronological Split Proposal:")
    print(f"    -----------------------------------------------")
    print(f"    Full date range  : {result['ts_min']}  to  {result['ts_max']}")
    print(f"    TRAIN  (<= {result['train_cutoff']})   ~80% of ratings")
    print(f"    VAL    ({result['train_cutoff']} - {result['val_cutoff']})  ~10%")
    print(f"    TEST   (> {result['val_cutoff']})   ~10%")
    print(f"    -----------------------------------------------")
    print(f"    Why chronological? A random split would let the model see")
    print(f"    future ratings during training, inflating evaluation metrics.")
    print(f"    Temporal ordering mimics real deployment conditions.")

    # Save split boundaries to a text file for reproducibility
    out = _ensure_reports() / "split_boundaries.txt"
    with open(out, "w") as f:
        for k, v in result.items():
            f.write(f"{k}: {v}\n")
    print(f"\n  [SAVED] {out}")

    return result


# ── Summary report ────────────────────────────────────────────────────────────

def save_summary_report(all_results: dict) -> Path:
    """Save a plain-text summary of all computed statistics to reports/."""
    out = _ensure_reports() / "phase2_summary.txt"
    with open(out, "w") as f:
        f.write("AIML-03 Recommendation Engine\n")
        f.write("Phase 2 — EDA Summary Report\n")
        f.write("=" * 60 + "\n\n")

        for section, data in all_results.items():
            f.write(f"\n[{section}]\n")
            if isinstance(data, dict):
                for k, v in data.items():
                    if k == "counts":   # rating distribution is too long
                        f.write(f"  {k}: (see chart 01_rating_distribution.png)\n")
                    else:
                        f.write(f"  {k}: {v}\n")
            elif isinstance(data, pd.DataFrame):
                f.write(data.to_string(index=False) + "\n")
            elif isinstance(data, pd.Series):
                f.write(data.to_string() + "\n")
            else:
                f.write(f"  {data}\n")

    print(f"\n  [SAVED] {out}")
    return out


# ── Orchestrator ──────────────────────────────────────────────────────────────

def run_full_eda(data_dir: Optional[Path] = None) -> dict:
    """
    Run all EDA analyses and save charts + summary to reports/.
    Returns a dict of all computed results.
    """
    if data_dir is None:
        data_dir = DATA_RAW_DIR

    print("\n" + "=" * 60)
    print("  PHASE 2 — Exploratory Data Analysis")
    print("=" * 60)

    results = {}
    results["rating_distribution"]    = analyse_rating_distribution(data_dir)
    results["user_movie_interactions"] = analyse_user_movie_interactions(data_dir)
    results["top_rated_movies"]        = analyse_most_rated_movies(data_dir)
    results["rating_trend"]            = analyse_rating_trend(data_dir)
    results["genre_distribution"]      = analyse_genre_distribution(data_dir)
    results["temporal_split"]          = propose_temporal_split(data_dir)

    save_summary_report(results)

    print("\n" + "=" * 60)
    print("  EDA complete. Charts saved to reports/")
    print("=" * 60 + "\n")
    return results
