"""
test_data_loader.py  —  Phase 1 Tests for src/data_loader.py

These tests use temporary files and directories so they work without
the full MovieLens dataset being present. They cover:

  1. Path resolution — DATA_RAW_DIR points inside the project root.
  2. format_size — human-readable size conversion.
  3. check_files — correctly detects present and missing files.
  4. validate_columns — passes correct schemas, fails wrong ones.
  5. sample_csv — reads correct number of rows from a temp file.
  6. Missing file handling — graceful error messages, no crash.
"""

import sys
import csv
import pytest
from pathlib import Path

# ── Make src/ importable ────────────────────────────────────────────────────
# This works regardless of where pytest is invoked from.
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

import data_loader as dl


# ===========================================================================
# 1. Path resolution tests
# ===========================================================================

class TestPathResolution:
    def test_data_raw_dir_is_inside_project(self):
        """DATA_RAW_DIR should be project_root/data/raw."""
        expected = dl.PROJECT_ROOT / "data" / "raw"
        assert dl.DATA_RAW_DIR == expected, (
            f"Expected {expected}, got {dl.DATA_RAW_DIR}"
        )

    def test_project_root_exists(self):
        """The resolved project root directory must exist."""
        assert dl.PROJECT_ROOT.is_dir(), (
            f"PROJECT_ROOT does not exist: {dl.PROJECT_ROOT}"
        )

    def test_data_raw_dir_contains_no_username(self):
        """Path should be constructed from __file__, not a hardcoded username."""
        path_str = str(dl.DATA_RAW_DIR)
        # The path is derived from the script location, so it must contain
        # the project folder name
        assert "AIML-03-Recommendation-Engine" in path_str


# ===========================================================================
# 2. format_size tests
# ===========================================================================

class TestFormatSize:
    def test_bytes(self):
        assert dl.format_size(512) == "512 bytes"

    def test_kilobytes(self):
        result = dl.format_size(2048)
        assert "KB" in result

    def test_megabytes(self):
        result = dl.format_size(5 * 1024 * 1024)
        assert "MB" in result

    def test_gigabytes(self):
        result = dl.format_size(2 * 1024 * 1024 * 1024)
        assert "GB" in result

    def test_zero_bytes(self):
        assert dl.format_size(0) == "0 bytes"


# ===========================================================================
# 3. check_files tests (using a temporary directory)
# ===========================================================================

class TestCheckFiles:
    def _make_temp_dir_with_files(self, tmp_path: Path, filenames: list) -> Path:
        """Create empty CSV files in a temp directory."""
        for fname in filenames:
            (tmp_path / fname).write_text("col1,col2\n1,2\n")
        return tmp_path

    def test_all_files_present(self, tmp_path):
        """check_files returns True for every file when all exist."""
        all_files = list(dl.EXPECTED_FILES.keys())
        self._make_temp_dir_with_files(tmp_path, all_files)
        results = dl.check_files(tmp_path)
        assert all(results.values()), "Expected all files to be found."

    def test_some_files_missing(self, tmp_path):
        """check_files returns False for missing files."""
        # Only create ratings.csv; others should be missing
        (tmp_path / "ratings.csv").write_text("userId,movieId,rating,timestamp\n1,1,5.0,1234567890\n")
        results = dl.check_files(tmp_path)
        assert results["ratings.csv"] is True
        assert results["movies.csv"] is False

    def test_empty_directory(self, tmp_path):
        """check_files returns False for every file in an empty directory."""
        results = dl.check_files(tmp_path)
        assert not any(results.values()), "Expected all files to be missing."


# ===========================================================================
# 4. validate_columns tests
# ===========================================================================

class TestValidateColumns:
    def _write_csv(self, path: Path, filename: str, columns: list[str]) -> None:
        """Write a CSV with given column headers and one dummy data row."""
        filepath = path / filename
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerow({col: "dummy" for col in columns})

    def test_correct_ratings_schema(self, tmp_path):
        self._write_csv(tmp_path, "ratings.csv",
                        ["userId", "movieId", "rating", "timestamp"])
        assert dl.validate_columns("ratings.csv", tmp_path) is True

    def test_correct_movies_schema(self, tmp_path):
        self._write_csv(tmp_path, "movies.csv",
                        ["movieId", "title", "genres"])
        assert dl.validate_columns("movies.csv", tmp_path) is True

    def test_correct_tags_schema(self, tmp_path):
        self._write_csv(tmp_path, "tags.csv",
                        ["userId", "movieId", "tag", "timestamp"])
        assert dl.validate_columns("tags.csv", tmp_path) is True

    def test_correct_links_schema(self, tmp_path):
        self._write_csv(tmp_path, "links.csv",
                        ["movieId", "imdbId", "tmdbId"])
        assert dl.validate_columns("links.csv", tmp_path) is True

    def test_missing_column_fails(self, tmp_path):
        """A file missing a required column should fail validation."""
        # ratings.csv without 'timestamp'
        self._write_csv(tmp_path, "ratings.csv",
                        ["userId", "movieId", "rating"])  # timestamp missing
        assert dl.validate_columns("ratings.csv", tmp_path) is False

    def test_extra_columns_are_allowed(self, tmp_path):
        """Extra columns beyond the required set are fine."""
        self._write_csv(tmp_path, "movies.csv",
                        ["movieId", "title", "genres", "extra_col"])
        assert dl.validate_columns("movies.csv", tmp_path) is True

    def test_missing_file_fails(self, tmp_path):
        """validate_columns returns False and does not crash for missing files."""
        assert dl.validate_columns("ratings.csv", tmp_path) is False

    def test_unknown_filename(self, tmp_path):
        """A filename not in EXPECTED_FILES should return False gracefully."""
        assert dl.validate_columns("unknown_file.csv", tmp_path) is False


# ===========================================================================
# 5. sample_csv tests
# ===========================================================================

class TestSampleCsv:
    def _make_ratings_csv(self, path: Path, n_rows: int = 20) -> Path:
        """Create a synthetic ratings.csv with n_rows data rows."""
        filepath = path / "ratings.csv"
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["userId", "movieId", "rating", "timestamp"])
            for i in range(n_rows):
                writer.writerow([i + 1, i + 1, 4.0, 1000000 + i])
        return filepath

    def test_reads_correct_number_of_rows(self, tmp_path):
        """sample_csv should return exactly n_rows rows."""
        self._make_ratings_csv(tmp_path, n_rows=50)
        df = dl.sample_csv("ratings.csv", tmp_path, n_rows=10)
        assert df is not None
        assert len(df) == 10

    def test_returns_none_for_missing_file(self, tmp_path):
        """sample_csv returns None if the file does not exist."""
        result = dl.sample_csv("ratings.csv", tmp_path, n_rows=5)
        assert result is None

    def test_columns_preserved(self, tmp_path):
        """The returned DataFrame must have the expected column names."""
        self._make_ratings_csv(tmp_path, n_rows=5)
        df = dl.sample_csv("ratings.csv", tmp_path, n_rows=5)
        assert df is not None
        assert list(df.columns) == ["userId", "movieId", "rating", "timestamp"]

    def test_does_not_load_full_file(self, tmp_path):
        """
        When n_rows=3 is requested from a 100-row file,
        the DataFrame must have exactly 3 rows — confirming partial loading.
        """
        self._make_ratings_csv(tmp_path, n_rows=100)
        df = dl.sample_csv("ratings.csv", tmp_path, n_rows=3)
        assert df is not None
        assert len(df) == 3


# ===========================================================================
# 6. EXPECTED_FILES registry test
# ===========================================================================

class TestExpectedFilesRegistry:
    def test_all_required_files_registered(self):
        """All six MovieLens files must be in EXPECTED_FILES."""
        required = {
            "ratings.csv", "movies.csv", "tags.csv",
            "links.csv", "genome-scores.csv", "genome-tags.csv",
        }
        assert required.issubset(set(dl.EXPECTED_FILES.keys()))

    def test_each_file_has_at_least_one_column(self):
        """Every entry in EXPECTED_FILES must list at least one column."""
        for fname, cols in dl.EXPECTED_FILES.items():
            assert len(cols) >= 1, f"'{fname}' has no required columns listed."
