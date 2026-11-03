"""
test_ranking_model.py  --  Unit Tests for Phase 5B Stage 2 Ranking Model

All tests use small synthetic in-memory fixtures or temporary files.
No real MovieLens dataset is required.

Covers:
  1. Precision@K metric calculation.
  2. Feature extraction (Item-CF, popularity, Bayesian smoothed ratings, genre affinity).
  3. Logistic ranking model weight fitting and score prediction.
  4. Candidate reranking order and history exclusion (already rated items strictly excluded).
  5. Cold-start user handling (unseen users fall back to global averages).
  6. Cold-start item handling (unseen items fall back to global smoothed mean).
  7. Data leakage safeguards (validation/test timestamps never enter feature stats).
  8. End-to-end evaluation runner (NDCG@K, Recall@K, Precision@K).
"""

import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

# Make src/ importable
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from popularity_recommender import (
    DEFAULT_TRAIN_CUTOFF,
    DEFAULT_VAL_CUTOFF,
    PopularityRecommender,
    compute_recall_at_k,
    compute_ndcg_at_k,
)
from item_collaborative_filtering import (
    ItemCollaborativeFiltering,
)
from candidate_retrieval import (
    CandidateRetriever,
)
from ranking_model import (
    RankingFeatureExtractor,
    RankingModel,
    compute_precision_at_k,
    evaluate_ranking_model,
)


# =============================================================================
# 1. Precision@K Metric Tests
# =============================================================================

class TestPrecisionMetric:
    def test_precision_at_k_perfect(self):
        # 2 hits out of top 2 -> Precision@2 = 1.0
        recs = [10, 20, 30]
        gt = {10, 20}
        assert compute_precision_at_k(recs, gt, k=2) == 1.0

    def test_precision_at_k_partial(self):
        # 1 hit out of top 2 -> Precision@2 = 0.5
        recs = [10, 99, 88]
        gt = {10, 20}
        assert compute_precision_at_k(recs, gt, k=2) == 0.5

    def test_precision_at_k_zero(self):
        recs = [99, 88, 77]
        gt = {10, 20}
        assert compute_precision_at_k(recs, gt, k=3) == 0.0

    def test_precision_at_k_empty_ground_truth(self):
        assert compute_precision_at_k([10, 20], set(), k=2) == 0.0

    def test_precision_at_k_zero_k(self):
        assert compute_precision_at_k([10, 20], {10}, k=0) == 0.0


# =============================================================================
# 2. Feature Extractor Tests
# =============================================================================

class TestRankingFeatureExtractor:
    @pytest.fixture
    def synthetic_extractor(self):
        ext = RankingFeatureExtractor(smoothing_weight=10.0)
        item_counts = Counter({101: 50, 102: 10, 103: 0})
        item_sums = Counter({101: 200.0, 102: 30.0, 103: 0.0})
        user_counts = {1: 20, 2: 5}
        user_sums = {1: 80.0, 2: 22.5}
        user_genres = {1: Counter({"Action": 10, "Sci-Fi": 5})}
        movie_genres = {
            101: {"Action", "Adventure"},
            102: {"Drama"},
            103: {"Action", "Sci-Fi"},
        }
        ext.fit_from_training_stats(
            item_counts=item_counts,
            item_sums=item_sums,
            user_counts=user_counts,
            user_sums=user_sums,
            user_genres=user_genres,
            movie_genres=movie_genres,
        )
        return ext

    def test_movie_avg_rating_bayesian_smoothing(self, synthetic_extractor):
        # Movie 101: (200 + 10 * 3.833) / (50 + 10)
        avg_101 = synthetic_extractor.get_movie_avg_rating(101)
        assert 3.5 <= avg_101 <= 4.0

        # Unseen movie 999 falls back to global mean
        avg_unseen = synthetic_extractor.get_movie_avg_rating(999)
        assert abs(avg_unseen - synthetic_extractor.global_mean_rating) < 1e-5

    def test_user_avg_rating(self, synthetic_extractor):
        # User 1: 80.0 / 20 = 4.0
        assert synthetic_extractor.get_user_avg_rating(1) == 4.0
        # User 2: 22.5 / 5 = 4.5
        assert synthetic_extractor.get_user_avg_rating(2) == 4.5
        # Unseen user falls back to global mean
        assert synthetic_extractor.get_user_avg_rating(999) == synthetic_extractor.global_mean_rating

    def test_extract_features_single_dimensions_and_values(self, synthetic_extractor):
        feat = synthetic_extractor.extract_features_single(
            user_id=1,
            movie_id=101,
            cf_score=0.85,
        )
        assert len(feat) == 8
        assert feat[0] == 0.85  # cf_score
        assert feat[1] > 0.0   # log_pop
        assert feat[4] == 4.0   # user_avg
        assert feat[6] > 0.0   # genre affinity (Action overlap)

    def test_extract_candidate_matrix_shape(self, synthetic_extractor):
        mat = synthetic_extractor.extract_candidate_matrix(
            user_id=1,
            candidate_ids=[101, 102, 103],
            cf_scores={101: 0.9, 103: 0.5},
        )
        assert mat.shape == (3, 8)


# =============================================================================
# 3. Model Weight Fitting & Prediction Tests
# =============================================================================

class TestRankingModel:
    def test_fit_weights_and_predict(self):
        model = RankingModel(l2_reg=0.1)

        # 4 samples with 8 features
        # Positive samples have high cf_score (feature 0) and high avg rating (feature 2)
        X = np.array([
            [1.0, 2.0, 4.5, 2.0, 4.0, 1.5, 0.8, 2.0],
            [0.9, 2.1, 4.2, 2.1, 4.0, 1.5, 0.7, 1.0],
            [0.1, 0.5, 2.0, 0.5, 4.0, 1.5, 0.0, 0.0],
            [0.0, 0.3, 1.8, 0.3, 4.0, 1.5, 0.0, 0.0],
        ], dtype=np.float32)
        y = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)

        model.fit_weights(X, y)
        assert model.is_fitted_

        scores = model.predict_scores(X)
        assert len(scores) == 4
        # Positive instances must have higher predicted probabilities than negative instances
        assert scores[0] > scores[2]
        assert scores[1] > scores[3]
        # All probabilities bounded in [0, 1]
        assert np.all((scores >= 0.0) & (scores <= 1.0))


# =============================================================================
# 4. Candidate Reranking & History Exclusion Tests
# =============================================================================

class TestCandidateReranking:
    @pytest.fixture
    def fitted_ranker(self):
        ext = RankingFeatureExtractor()
        # Item 101 is high quality, Item 102 is low quality, Item 103 is medium
        item_counts = Counter({101: 100, 102: 10, 103: 50})
        item_sums = Counter({101: 450.0, 102: 20.0, 103: 175.0})
        user_counts = {1: 10}
        user_sums = {1: 40.0}
        user_genres = {1: Counter({"Drama": 5})}
        movie_genres = {101: {"Drama"}, 102: {"Comedy"}, 103: {"Drama"}}

        ext.fit_from_training_stats(
            item_counts=item_counts,
            item_sums=item_sums,
            user_counts=user_counts,
            user_sums=user_sums,
            user_genres=user_genres,
            movie_genres=movie_genres,
        )

        model = RankingModel(feature_extractor=ext, l2_reg=0.1)

        # Train with positive on 101, negative on 102
        X = np.array([
            [1.0, 2.0, 4.5, 2.0, 4.0, 1.0, 1.0, 1.0],
            [0.0, 1.0, 2.0, 1.0, 4.0, 1.0, 0.0, 0.0],
        ], dtype=np.float32)
        y = np.array([1.0, 0.0], dtype=np.float32)
        model.fit_weights(X, y)
        return model

    def test_reranks_high_quality_candidate_top(self, fitted_ranker):
        # Candidate pool is [102, 101] (102 was erroneously first from retrieval)
        ranked = fitted_ranker.rank_candidates(
            user_id=1,
            candidate_ids=[102, 101],
            k=2,
        )
        # Reranker should place 101 first
        assert ranked[0] == 101
        assert ranked[1] == 102

    def test_history_exclusion_strictly_enforced(self, fitted_ranker):
        # User already watched 101 in training
        user_history = [101]
        ranked = fitted_ranker.rank_candidates(
            user_id=1,
            candidate_ids=[101, 102, 103],
            user_history=user_history,
            k=3,
        )
        assert 101 not in ranked
        assert len(ranked) == 2

    def test_rank_with_details_structure(self, fitted_ranker):
        details = fitted_ranker.rank_with_details(
            user_id=1,
            candidate_ids=[101, 102],
            k=2,
        )
        assert len(details) == 2
        assert details[0]["rank"] == 1
        assert details[0]["movieId"] in (101, 102)
        assert "movie_avg_rating" in details[0]
        assert "popularity_count" in details[0]


# =============================================================================
# 5. Cold-Start Handling Tests
# =============================================================================

class TestColdStartHandling:
    def test_cold_start_unseen_user(self):
        ext = RankingFeatureExtractor()
        ext.fit_from_training_stats(
            item_counts=Counter({101: 50, 102: 40}),
            item_sums=Counter({101: 200.0, 102: 120.0}),
            user_counts={},
            user_sums={},
        )
        model = RankingModel(feature_extractor=ext)
        model.weights_ = np.ones(8, dtype=np.float32)
        model.scaler_mean_ = np.zeros(8, dtype=np.float32)
        model.scaler_std_ = np.ones(8, dtype=np.float32)
        model.is_fitted_ = True

        # Unseen user 999
        ranked = model.rank_candidates(
            user_id=999,
            candidate_ids=[102, 101],
            k=2,
        )
        assert len(ranked) == 2
        # Movie 101 has higher rating (4.0 vs 3.0), so it should rank first
        assert ranked[0] == 101

    def test_cold_start_unseen_item(self):
        ext = RankingFeatureExtractor()
        ext.fit_from_training_stats(
            item_counts=Counter(),
            item_sums=Counter(),
            user_counts={1: 5},
            user_sums={1: 20.0},
        )
        feat = ext.extract_features_single(user_id=1, movie_id=999)
        # Should gracefully use global averages
        assert feat[2] == ext.global_mean_rating


# =============================================================================
# 6. Data Leakage Safeguards
# =============================================================================

class TestDataLeakageSafeguards:
    @staticmethod
    def _create_synthetic_ratings(file_path: Path):
        rows = [
            # Training interactions (ts <= 1000)
            {"userId": 1, "movieId": 101, "rating": 5.0, "timestamp": 800},
            {"userId": 1, "movieId": 102, "rating": 4.0, "timestamp": 850},
            {"userId": 2, "movieId": 101, "rating": 4.5, "timestamp": 900},

            # Validation interactions (1000 < ts <= 1400)
            {"userId": 1, "movieId": 999, "rating": 5.0, "timestamp": 1200},

            # Test interactions (ts > 1400)
            {"userId": 2, "movieId": 888, "rating": 5.0, "timestamp": 1500},
        ]
        with open(file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["userId", "movieId", "rating", "timestamp"])
            writer.writeheader()
            writer.writerows(rows)

    def test_feature_extractor_excludes_validation_and_test(self, tmp_path):
        ratings_path = tmp_path / "ratings.csv"
        self._create_synthetic_ratings(ratings_path)

        ext = RankingFeatureExtractor()
        ext.fit(
            ratings_path=ratings_path,
            train_cutoff_timestamp=1000,
            chunksize=100,
        )

        # Movies 999 (val) and 888 (test) must NEVER be in training item statistics
        assert 999 not in ext.item_counts_
        assert 888 not in ext.item_counts_
        assert 999 not in ext.item_rating_sums_
        assert 888 not in ext.item_rating_sums_

        # Training items 101 and 102 must be present
        assert 101 in ext.item_counts_
        assert 102 in ext.item_counts_


# =============================================================================
# 7. End-to-End Evaluation Runner Tests
# =============================================================================

class TestEvaluateRankingModel:
    def test_evaluate_ranking_model_metrics(self):
        # Set up a synthetic retriever and ranking model
        pop_counts = {101: 50, 102: 40, 103: 30, 104: 20}
        pop_sums = {101: 200.0, 102: 160.0, 103: 90.0, 104: 60.0}
        pop_model = PopularityRecommender().fit_from_counts(pop_counts)

        mat = sp.csr_matrix([[5.0, 4.0], [4.0, 5.0]], dtype=np.float32)
        cf_model = ItemCollaborativeFiltering(top_k_neighbors=5, shrinkage=0.0)
        cf_model.fit_from_matrix(mat, item_ids=[101, 102], popularity_recommender=pop_model)

        retriever = CandidateRetriever(
            item_cf=cf_model,
            popularity_recommender=pop_model,
            strategy="waterfall",
        )

        ext = RankingFeatureExtractor()
        ext.fit_from_training_stats(
            item_counts=pop_counts,
            item_sums=pop_sums,
            user_counts={1: 10},
            user_sums={1: 40.0},
        )

        ranker = RankingModel(feature_extractor=ext, candidate_retriever=retriever)
        ranker.weights_ = np.array([1.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        ranker.scaler_mean_ = np.zeros(8, dtype=np.float32)
        ranker.scaler_std_ = np.ones(8, dtype=np.float32)
        ranker.is_fitted_ = True

        user_train = {1: [101]}
        user_val = {1: [102]}  # 102 is ground truth

        metrics = evaluate_ranking_model(
            model=ranker,
            candidate_retriever=retriever,
            user_train_history=user_train,
            user_eval_ground_truth=user_val,
            candidate_pool_size=5,
            k_list=[2, 5],
        )

        assert metrics["evaluated_users"] == 1
        assert 0.0 <= metrics["recall"][2] <= 1.0
        assert 0.0 <= metrics["precision"][2] <= 1.0
        assert 0.0 <= metrics["ndcg"][2] <= 1.0

