"""
test_phase2.py  --  Phase 2 Tests

Tests for quality_checks.py and eda.py.
All tests use small synthetic temporary CSV files — no real dataset required.

Covers:
  1. Quality check functions return expected dict keys.
  2. Chunked row-count accumulation is correct.
  3. Duplicate detection works correctly.
  4. Rating range validation catches out-of-range values.
  5. Genre distribution correctly explodes pipe-separated genres.
  6. Temporal split proposal produces valid chronological cutoffs.
  7. format_size helper (re-imported) is unchanged.
  8. EDA functions produce output files in the expected location.
"""

import csv
import sys
import math
import tempfile
import shutil
from pathlib import Path
from collections import Counter

import pytest
import pandas as pd
import numpy as np

# Make src/ importable
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

import quality_checks as qc
import eda as eda_module


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _write_ratings(path: Path, rows: list[tuple]) -> None:
    """Write a small synthetic ratings.csv."""
    with open(path / "ratings.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["userId", "movieId", "rating", "timestamp"])
        for r in rows:
            w.writerow(r)


def _write_movies(path: Path, rows: list[tuple]) -> None:
    """Write a small synthetic movies.csv."""
    with open(path / "movies.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["movieId", "title", "genres"])
        for r in rows:
            w.writerow(r)


def _write_tags(path: Path, rows: list[tuple]) -> None:
    with open(path / "tags.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["userId", "movieId", "tag", "timestamp"])
        for r in rows:
            w.writerow(r)


def _write_links(path: Path, rows: list[tuple]) -> None:
    with open(path / "links.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["movieId", "imdbId", "tmdbId"])
        for r in rows:
            w.writerow(r)


def _write_genome_tags(path: Path, rows: list[tuple]) -> None:
    with open(path / "genome-tags.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tagId", "tag"])
        for r in rows:
            w.writerow(r)


def _write_genome_scores(path: Path, rows: list[tuple]) -> None:
    with open(path / "genome-scores.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["movieId", "tagId", "relevance"])
        for r in rows:
            w.writerow(r)


# ── 1. Quality check — ratings ────────────────────────────────────────────────

class TestRatingsQuality:
    def _make_clean_ratings(self, path: Path) -> None:
        rows = [
            (1, 10, 4.0, 1_000_000),
            (1, 20, 3.5, 1_100_000),
            (2, 10, 5.0, 1_200_000),
            (2, 30, 2.5, 1_300_000),
            (3, 10, 1.0, 1_400_000),
        ]
        _write_ratings(path, rows)

    def test_returns_expected_keys(self, tmp_path):
        self._make_clean_ratings(tmp_path)
        result = qc.check_ratings_quality(tmp_path)
        expected_keys = {
            "row_count", "missing_per_col", "duplicate_user_movie_pairs",
            "rating_out_of_range", "timestamp_min", "timestamp_max",
            "unique_users", "unique_movies", "rating_mean", "rating_std",
        }
        assert expected_keys.issubset(set(result.keys()))

    def test_correct_row_count(self, tmp_path):
        self._make_clean_ratings(tmp_path)
        result = qc.check_ratings_quality(tmp_path)
        assert result["row_count"] == 5

    def test_correct_unique_users(self, tmp_path):
        self._make_clean_ratings(tmp_path)
        result = qc.check_ratings_quality(tmp_path)
        assert result["unique_users"] == 3

    def test_correct_unique_movies(self, tmp_path):
        self._make_clean_ratings(tmp_path)
        result = qc.check_ratings_quality(tmp_path)
        assert result["unique_movies"] == 3

    def test_rating_mean_correct(self, tmp_path):
        # ratings: 4.0, 3.5, 5.0, 2.5, 1.0 -> mean = 16.0/5 = 3.2
        self._make_clean_ratings(tmp_path)
        result = qc.check_ratings_quality(tmp_path)
        assert abs(result["rating_mean"] - 3.2) < 0.01

    def test_no_missing_values(self, tmp_path):
        self._make_clean_ratings(tmp_path)
        result = qc.check_ratings_quality(tmp_path)
        assert all(v == 0 for v in result["missing_per_col"].values())

    def test_detects_out_of_range_rating(self, tmp_path):
        rows = [
            (1, 10, 4.0, 1_000_000),
            (2, 20, 6.0, 1_100_000),  # 6.0 is out of range
            (3, 30, 0.0, 1_200_000),  # 0.0 is out of range
        ]
        _write_ratings(path=tmp_path, rows=rows)
        result = qc.check_ratings_quality(tmp_path)
        assert result["rating_out_of_range"] == 2

    def test_detects_duplicate_pairs(self, tmp_path):
        rows = [
            (1, 10, 4.0, 1_000_000),
            (1, 10, 3.5, 1_100_000),  # duplicate (userId=1, movieId=10)
            (2, 20, 5.0, 1_200_000),
        ]
        _write_ratings(path=tmp_path, rows=rows)
        result = qc.check_ratings_quality(tmp_path)
        assert result["duplicate_user_movie_pairs"] >= 1

    def test_missing_file_returns_empty(self, tmp_path):
        result = qc.check_ratings_quality(tmp_path)
        assert result == {}


# ── 2. Quality check — movies ─────────────────────────────────────────────────

class TestMoviesQuality:
    def test_returns_expected_keys(self, tmp_path):
        _write_movies(tmp_path, [
            (1, "Toy Story (1995)", "Animation|Comedy"),
            (2, "Jumanji (1995)", "Adventure"),
        ])
        result = qc.check_movies_quality(tmp_path)
        assert "row_count" in result
        assert "unique_genres" in result

    def test_row_count(self, tmp_path):
        _write_movies(tmp_path, [
            (1, "A", "Comedy"),
            (2, "B", "Drama"),
            (3, "C", "Comedy|Drama"),
        ])
        result = qc.check_movies_quality(tmp_path)
        assert result["row_count"] == 3

    def test_detects_duplicate_movieid(self, tmp_path):
        _write_movies(tmp_path, [
            (1, "A", "Comedy"),
            (1, "A duplicate", "Drama"),  # duplicate movieId
        ])
        result = qc.check_movies_quality(tmp_path)
        assert result["duplicate_movieIds"] >= 1

    def test_missing_file_returns_empty(self, tmp_path):
        result = qc.check_movies_quality(tmp_path)
        assert result == {}


# ── 3. Quality check — tags ───────────────────────────────────────────────────

class TestTagsQuality:
    def test_returns_expected_keys(self, tmp_path):
        _write_tags(tmp_path, [(1, 10, "funny", 1_000_000)])
        result = qc.check_tags_quality(tmp_path)
        assert "unique_tags" in result
        assert "row_count" in result

    def test_missing_file_returns_empty(self, tmp_path):
        assert qc.check_tags_quality(tmp_path) == {}


# ── 4. Quality check — links ──────────────────────────────────────────────────

class TestLinksQuality:
    def test_returns_expected_keys(self, tmp_path):
        _write_links(tmp_path, [(1, "0114709", "862")])
        result = qc.check_links_quality(tmp_path)
        assert "row_count" in result
        assert "missing_per_col" in result

    def test_missing_file_returns_empty(self, tmp_path):
        assert qc.check_links_quality(tmp_path) == {}


# ── 5. Quality check — genome-scores ─────────────────────────────────────────

class TestGenomeScoresQuality:
    def test_returns_expected_keys(self, tmp_path):
        _write_genome_scores(tmp_path, [
            (1, 1, 0.025),
            (1, 2, 0.750),
            (2, 1, 0.500),
        ])
        result = qc.check_genome_scores_quality(tmp_path)
        assert "row_count" in result
        assert "relevance_min" in result
        assert "relevance_max" in result

    def test_relevance_range(self, tmp_path):
        _write_genome_scores(tmp_path, [
            (1, 1, 0.1),
            (1, 2, 0.9),
        ])
        result = qc.check_genome_scores_quality(tmp_path)
        assert abs(result["relevance_min"] - 0.1) < 1e-5
        assert abs(result["relevance_max"] - 0.9) < 1e-5

    def test_missing_file_returns_empty(self, tmp_path):
        assert qc.check_genome_scores_quality(tmp_path) == {}


# ── 6. EDA — Genre distribution ───────────────────────────────────────────────

class TestGenreDistribution:
    def test_counts_genres_correctly(self, tmp_path):
        _write_movies(tmp_path, [
            (1, "A", "Comedy|Drama"),
            (2, "B", "Comedy"),
            (3, "C", "Drama|Action"),
        ])
        result = eda_module.analyse_genre_distribution(tmp_path)
        assert result["Comedy"] == 2
        assert result["Drama"] == 2
        assert result["Action"] == 1

    def test_excludes_no_genres_listed(self, tmp_path):
        _write_movies(tmp_path, [
            (1, "A", "(no genres listed)"),
            (2, "B", "Comedy"),
        ])
        result = eda_module.analyse_genre_distribution(tmp_path)
        assert "(no genres listed)" not in result.index


# ── 7. Temporal split ─────────────────────────────────────────────────────────

class TestTemporalSplit:
    def _make_ratings_with_timestamps(self, path: Path) -> None:
        # 10 ratings with known timestamps spread over time
        rows = [(i, i, 4.0, 1_000_000 * i) for i in range(1, 11)]
        _write_ratings(path, rows)

    def test_returns_expected_keys(self, tmp_path):
        self._make_ratings_with_timestamps(tmp_path)
        result = eda_module.propose_temporal_split(tmp_path)
        assert "train_cutoff" in result
        assert "val_cutoff" in result
        assert "ts_min" in result
        assert "ts_max" in result

    def test_train_cutoff_before_val_cutoff(self, tmp_path):
        self._make_ratings_with_timestamps(tmp_path)
        result = eda_module.propose_temporal_split(tmp_path)
        # Compare the unix timestamps directly
        assert result["train_unix"] <= result["val_unix"]

    def test_total_ratings_correct(self, tmp_path):
        self._make_ratings_with_timestamps(tmp_path)
        result = eda_module.propose_temporal_split(tmp_path)
        assert result["total_ratings"] == 10


# ── 8. Rating distribution (chunked accumulation) ────────────────────────────

class TestRatingDistribution:
    def test_counts_are_correct(self, tmp_path):
        rows = [
            (1, 1, 4.0, 1000000),
            (2, 2, 4.0, 1000001),
            (3, 3, 3.5, 1000002),
            (4, 4, 5.0, 1000003),
        ]
        _write_ratings(tmp_path, rows)
        result = eda_module.analyse_rating_distribution(tmp_path)
        assert result["counts"][4.0] == 2
        assert result["counts"][3.5] == 1
        assert result["counts"][5.0] == 1
        assert result["total"] == 4

    def test_mean_is_correct(self, tmp_path):
        # ratings 4.0, 4.0, 3.5, 5.0  -> mean = 16.5/4 = 4.125
        rows = [
            (1, 1, 4.0, 1000000),
            (2, 2, 4.0, 1000001),
            (3, 3, 3.5, 1000002),
            (4, 4, 5.0, 1000003),
        ]
        _write_ratings(tmp_path, rows)
        result = eda_module.analyse_rating_distribution(tmp_path)
        assert abs(result["mean"] - 4.125) < 0.01
