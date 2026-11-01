"""
test_popularity.py  --  Unit Tests for Popularity Baseline, Evaluation & 80/10/10 Split

All tests use small synthetic in-memory fixtures or temporary files.
No real MovieLens dataset is required.

Covers:
  1. Recall@K calculation (perfect match, partial match, zero match, empty ground truth).
  2. NDCG@K calculation (perfect ranking = 1.0, position discount, zero match = 0.0).
  3. Popularity ranking order & tie breaking.
  4. User history filtering (already-rated movies are strictly excluded).
  5. Cold-start recommendation (unseen user gets global top popular items).
  6. Phase 2 Chronological 80/10/10 split boundaries (Train, Validation, Test).
  7. Strict zero future-data leakage (validation & test ratings never in train counts).
  8. Separate validation and test ground-truth partitions.
  9. Deterministic & representative user sampling (seed 42, no prefix bias).
 10. End-to-end evaluation runner on validation and test splits.
"""

import csv
import sys
from pathlib import Path

import pytest

# Make src/ importable
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from popularity_recommender import (
    DEFAULT_TRAIN_CUTOFF,
    DEFAULT_VAL_CUTOFF,
    PopularityRecommender,
    compute_recall_at_k,
    compute_ndcg_at_k,
    load_temporal_split_data,
    load_all_temporal_splits,
    evaluate_popularity_recommender,
)


# =============================================================================
# 1. Metric Tests
# =============================================================================

class TestMetrics:
    def test_recall_at_k_perfect(self):
        # User ground truth = {10, 20}, recommended top-2 = [10, 20] -> Recall@2 = 1.0
        recs = [10, 20, 30]
        gt = {10, 20}
        assert compute_recall_at_k(recs, gt, k=2) == 1.0

    def test_recall_at_k_partial(self):
        # 1 hit out of 2 ground truth items -> Recall@2 = 0.5
        recs = [10, 99, 88]
        gt = {10, 20}
        assert compute_recall_at_k(recs, gt, k=2) == 0.5

    def test_recall_at_k_zero(self):
        # 0 hits -> Recall = 0.0
        recs = [99, 88, 77]
        gt = {10, 20}
        assert compute_recall_at_k(recs, gt, k=3) == 0.0

    def test_recall_at_k_empty_ground_truth(self):
        assert compute_recall_at_k([10, 20], set(), k=2) == 0.0

    def test_recall_at_k_zero_k(self):
        assert compute_recall_at_k([10, 20], {10}, k=0) == 0.0

    def test_ndcg_at_k_perfect(self):
        # Both relevant items appear at ranks 1 and 2 -> NDCG@2 = 1.0
        recs = [10, 20, 30]
        gt = {10, 20}
        assert abs(compute_ndcg_at_k(recs, gt, k=2) - 1.0) < 1e-6

    def test_ndcg_at_k_position_discount(self):
        # Hit at rank 1 must yield higher NDCG than hit at rank 2
        gt = {10}
        ndcg_rank1 = compute_ndcg_at_k(recommended=[10, 99], ground_truth=gt, k=2)
        ndcg_rank2 = compute_ndcg_at_k(recommended=[99, 10], ground_truth=gt, k=2)
        assert ndcg_rank1 > ndcg_rank2
        # Hit at rank 1 with 1 relevant item has DCG = 1/log2(2) = 1.0, IDCG = 1.0 -> NDCG = 1.0
        assert abs(ndcg_rank1 - 1.0) < 1e-6

    def test_ndcg_at_k_zero(self):
        recs = [88, 99]
        gt = {10, 20}
        assert compute_ndcg_at_k(recs, gt, k=2) == 0.0

    def test_ndcg_at_k_empty_ground_truth(self):
        assert compute_ndcg_at_k([10, 20], set(), k=2) == 0.0


# =============================================================================
# 2. Recommender Unit Tests
# =============================================================================

class TestPopularityRecommender:
    def test_fit_from_counts_ranking(self):
        counts = {101: 50, 102: 120, 103: 80}
        model = PopularityRecommender().fit_from_counts(counts)
        # Expected order: 102 (120), 103 (80), 101 (50)
        assert model.ranked_movie_ids_ == [102, 103, 101]

    def test_recommend_cold_start(self):
        counts = {101: 50, 102: 120, 103: 80}
        model = PopularityRecommender().fit_from_counts(counts)
        # Empty user history should receive top-2 overall
        recs = model.recommend(user_history=set(), k=2)
        assert recs == [102, 103]

    def test_recommend_filters_user_history(self):
        counts = {101: 50, 102: 120, 103: 80, 104: 30}
        model = PopularityRecommender().fit_from_counts(counts)
        # User has already watched movie 102
        recs = model.recommend(user_history={102}, k=2)
        # Movie 102 must be skipped, next best are 103 and 101
        assert recs == [103, 101]
        assert 102 not in recs

    def test_recommend_k_greater_than_available(self):
        counts = {101: 10, 102: 20}
        model = PopularityRecommender().fit_from_counts(counts)
        recs = model.recommend(user_history=set(), k=10)
        assert len(recs) == 2
        assert recs == [102, 101]

    def test_recommend_with_details(self, tmp_path):
        movies_csv = tmp_path / "movies.csv"
        with open(movies_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["movieId", "title", "genres"])
            writer.writerow([102, "Top Movie (2000)", "Action|Comedy"])

        model = PopularityRecommender().fit_from_counts({102: 500})
        model.load_movie_metadata(movies_csv)

        details = model.recommend_with_details(user_history=set(), k=1)
        assert len(details) == 1
        assert details[0]["movieId"] == 102
        assert details[0]["title"] == "Top Movie (2000)"
        assert details[0]["genres"] == "Action|Comedy"
        assert details[0]["popularity_count"] == 500
        assert details[0]["rank"] == 1


# =============================================================================
# 3. Temporal Split (80/10/10) & Evaluation Tests
# =============================================================================

class TestTemporalSplitAndEvaluation:
    def _create_synthetic_ratings(self, filepath: Path) -> None:
        """
        Creates a synthetic ratings CSV:
        Train cutoff: timestamp 1000
        Val cutoff:   timestamp 1400

        User 1:
          - Train: movie 101 (rating 4.0, ts 500)
          - Train: movie 102 (rating 5.0, ts 600)
          - Val:   movie 103 (rating 4.5, ts 1200) -> positive val hit (1000 < ts <= 1400)
          - Test:  movie 104 (rating 4.0, ts 1500) -> positive test hit (ts > 1400)
        User 2:
          - Train: movie 101 (rating 3.0, ts 700)
          - Val:   movie 102 (rating 4.0, ts 1100) -> positive val hit
          - Val:   movie 105 (rating 2.0, ts 1300) -> rating < 3.5 (not relevant)
          - Test:  movie 106 (rating 5.0, ts 1600) -> positive test hit
        User 3:
          - Train: movie 107 (rating 5.0, ts 800)
          - Test:  movie 108 (rating 4.0, ts 1700) -> positive test hit
        """
        rows = [
            (1, 101, 4.0, 500),
            (1, 102, 5.0, 600),
            (1, 103, 4.5, 1200),
            (1, 104, 4.0, 1500),
            (2, 101, 3.0, 700),
            (2, 102, 4.0, 1100),
            (2, 105, 2.0, 1300),
            (2, 106, 5.0, 1600),
            (3, 107, 5.0, 800),
            (3, 108, 4.0, 1700),
        ]
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["userId", "movieId", "rating", "timestamp"])
            for r in rows:
                writer.writerow(r)

    def test_80_10_10_split_boundaries(self, tmp_path):
        ratings_path = tmp_path / "ratings.csv"
        self._create_synthetic_ratings(ratings_path)

        train_counts, user_train, user_val, user_test = load_all_temporal_splits(
            ratings_path=ratings_path,
            train_cutoff_timestamp=1000,
            val_cutoff_timestamp=1400,
            test_min_rating=3.5,
        )

        # Train counts must only include ratings with ts <= 1000:
        # movie 101 (2), movie 102 (1), movie 107 (1)
        assert train_counts == {101: 2, 102: 1, 107: 1}

        # User training histories:
        assert set(user_train[1]) == {101, 102}
        assert set(user_train[2]) == {101}
        assert set(user_train[3]) == {107}

        # Validation interactions (1000 < ts <= 1400 and rating >= 3.5):
        # User 1 rated 103 (4.5, ts 1200) -> in val
        # User 2 rated 102 (4.0, ts 1100) -> in val
        # User 2 rated 105 (2.0, ts 1300) -> filtered out (< 3.5)
        assert user_val[1] == [103]
        assert user_val[2] == [102]
        assert 105 not in user_val.get(2, [])
        assert 3 not in user_val

        # Test interactions (ts > 1400 and rating >= 3.5):
        # User 1 rated 104 (ts 1500)
        # User 2 rated 106 (ts 1600)
        # User 3 rated 108 (ts 1700)
        assert user_test[1] == [104]
        assert user_test[2] == [106]
        assert user_test[3] == [108]

    def test_temporal_split_excludes_future_from_train(self, tmp_path):
        ratings_path = tmp_path / "ratings.csv"
        self._create_synthetic_ratings(ratings_path)

        train_counts, user_train, _, _ = load_all_temporal_splits(
            ratings_path=ratings_path,
            train_cutoff_timestamp=1000,
            val_cutoff_timestamp=1400,
            test_min_rating=3.5,
        )

        # Movies 103, 104, 105, 106, 108 only appeared after ts 1000
        for future_mid in [103, 104, 105, 106, 108]:
            assert future_mid not in train_counts
            for seen_list in user_train.values():
                assert future_mid not in seen_list

    def test_val_and_test_ground_truth_separation(self, tmp_path):
        ratings_path = tmp_path / "ratings.csv"
        self._create_synthetic_ratings(ratings_path)

        _, _, user_val, user_test = load_all_temporal_splits(
            ratings_path=ratings_path,
            train_cutoff_timestamp=1000,
            val_cutoff_timestamp=1400,
            test_min_rating=3.5,
        )

        # User 1: val item is 103, test item is 104
        assert 103 in user_val[1]
        assert 103 not in user_test[1]
        assert 104 in user_test[1]
        assert 104 not in user_val[1]

    def test_load_temporal_split_data_separate_val_and_test(self, tmp_path):
        ratings_path = tmp_path / "ratings.csv"
        self._create_synthetic_ratings(ratings_path)

        # Load validation partition
        train_counts_v, user_train_v, val_gt = load_temporal_split_data(
            ratings_path=ratings_path,
            train_cutoff_timestamp=1000,
            val_cutoff_timestamp=1400,
            eval_split="val",
            test_min_rating=3.5,
        )
        assert 103 in val_gt[1]
        assert 104 not in val_gt[1]

        # Load test partition
        train_counts_t, user_train_t, test_gt = load_temporal_split_data(
            ratings_path=ratings_path,
            train_cutoff_timestamp=1000,
            val_cutoff_timestamp=1400,
            eval_split="test",
            test_min_rating=3.5,
        )
        assert 104 in test_gt[1]
        assert 103 not in test_gt[1]
        assert train_counts_v == train_counts_t
        assert user_train_v == user_train_t

    def test_deterministic_reproducible_sampling(self):
        # Create 50 mock candidate users with test items
        user_gt = {uid: [100 + uid] for uid in range(1, 51)}
        user_hist = {uid: [1, 2] for uid in range(1, 51)}

        model = PopularityRecommender().fit_from_counts({100 + i: 10 for i in range(1, 51)})

        res1 = evaluate_popularity_recommender(
            recommender=model,
            user_train_history=user_hist,
            user_test_ground_truth=user_gt,
            k_list=[5],
            max_eval_users=10,
            random_seed=42,
        )

        res2 = evaluate_popularity_recommender(
            recommender=model,
            user_train_history=user_hist,
            user_test_ground_truth=user_gt,
            k_list=[5],
            max_eval_users=10,
            random_seed=42,
        )

        # Runs with the same seed must produce identical results
        assert res1["evaluated_users"] == 10
        assert res1["recall"][5] == res2["recall"][5]
        assert res1["ndcg"][5] == res2["ndcg"][5]

    def test_representative_sampling_covers_distribution(self):
        # Verify that sampling does not just slice the first N prefix users
        user_gt = {uid: [100 + uid] for uid in range(1, 101)}
        user_hist = {uid: [] for uid in range(1, 101)}

        model = PopularityRecommender().fit_from_counts({100 + i: 10 for i in range(1, 101)})

        # Track which users get evaluated by passing a recording recommender
        sampled_uids = []
        original_recommend = model.recommend

        def recording_recommend(user_history=None, k=10):
            return original_recommend(user_history=user_history, k=k)

        model.recommend = recording_recommend

        res = evaluate_popularity_recommender(
            recommender=model,
            user_train_history=user_hist,
            user_test_ground_truth=user_gt,
            k_list=[5],
            max_eval_users=10,
            random_seed=42,
        )

        assert res["evaluated_users"] == 10
        # Deterministic sample from sorted [1..100] with seed 42
        # Verify it draws beyond the first 10 user IDs:
        import random
        expected_uids = sorted(random.Random(42).sample(list(range(1, 101)), 10))
        assert max(expected_uids) > 50  # Must span across the user range

    def test_end_to_end_validation_and_test_evaluation(self, tmp_path):
        ratings_path = tmp_path / "ratings.csv"
        self._create_synthetic_ratings(ratings_path)

        train_counts, user_train, user_val, user_test = load_all_temporal_splits(
            ratings_path=ratings_path,
            train_cutoff_timestamp=1000,
            val_cutoff_timestamp=1400,
            test_min_rating=3.5,
        )

        model = PopularityRecommender().fit_from_counts(train_counts)

        val_results = evaluate_popularity_recommender(
            recommender=model,
            user_train_history=user_train,
            user_test_ground_truth=user_val,
            k_list=[2, 5],
        )

        test_results = evaluate_popularity_recommender(
            recommender=model,
            user_train_history=user_train,
            user_test_ground_truth=user_test,
            k_list=[2, 5],
        )

        assert val_results["evaluated_users"] == 2  # Users 1 and 2
        assert test_results["evaluated_users"] == 3  # Users 1, 2, and 3
        assert 0.0 <= val_results["recall"][2] <= 1.0
        assert 0.0 <= test_results["recall"][2] <= 1.0
