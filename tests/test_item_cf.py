"""
test_item_cf.py  --  Unit Tests for Phase 4A Item-Based Collaborative Filtering

All tests use small synthetic in-memory fixtures or temporary files.
No real MovieLens dataset is required.

Covers:
  1. Cosine similarity calculation (identical items, orthogonal items, shrinkage effect).
  2. Top-K neighbor retention and minimum similarity thresholding.
  3. Personalized recommendation ranking based on user interaction history.
  4. User history exclusion (already-rated items are never recommended).
  5. Cold-start fallback handling (unseen users get popularity baseline).
  6. Backfill handling (fills shortfall from popularity when Item-CF yields < K items).
  7. Strict zero future-data leakage (validation & test timestamps never in training matrix).
  8. End-to-end evaluation runner (Recall@K, NDCG@K, deterministic sampling).
"""

import csv
import sys
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
    load_all_temporal_splits,
)
from item_collaborative_filtering import (
    ItemCollaborativeFiltering,
    evaluate_item_cf,
)


# =============================================================================
# 1. Similarity Calculation Tests
# =============================================================================

class TestItemSimilarity:
    def test_identical_item_vectors_similarity_one(self):
        # 2 items, 3 users. Both items have identical ratings from users: [5.0, 4.0, 3.0]
        # Item-user matrix: shape (2, 3)
        mat = sp.csr_matrix(
            [
                [5.0, 4.0, 3.0],
                [5.0, 4.0, 3.0],
            ],
            dtype=np.float32,
        )
        model = ItemCollaborativeFiltering(top_k_neighbors=10, shrinkage=0.0)
        model.fit_from_matrix(mat, item_ids=[101, 102])

        # Neighbors of 101 should include 102 with similarity ~ 1.0
        neighbors_101 = model.item_neighbors_.get(101, [])
        assert len(neighbors_101) == 1
        neighbor_id, sim = neighbors_101[0]
        assert neighbor_id == 102
        assert abs(sim - 1.0) < 1e-5

    def test_orthogonal_item_vectors_zero_similarity(self):
        # Item 101 rated only by User 0; Item 102 rated only by User 1
        mat = sp.csr_matrix(
            [
                [5.0, 0.0],
                [0.0, 4.0],
            ],
            dtype=np.float32,
        )
        model = ItemCollaborativeFiltering(top_k_neighbors=10, shrinkage=0.0)
        model.fit_from_matrix(mat, item_ids=[101, 102])

        # Dot product is 0 -> orthogonal items should have no neighbors
        assert model.item_neighbors_.get(101, []) == []
        assert model.item_neighbors_.get(102, []) == []

    def test_shrinkage_penalizes_similarity(self):
        mat = sp.csr_matrix(
            [
                [4.0, 4.0],
                [4.0, 4.0],
            ],
            dtype=np.float32,
        )
        # Without shrinkage: sim = 1.0
        m_no_shrink = ItemCollaborativeFiltering(top_k_neighbors=10, shrinkage=0.0)
        m_no_shrink.fit_from_matrix(mat, item_ids=[101, 102])
        sim_no_shrink = m_no_shrink.item_neighbors_[101][0][1]

        # With shrinkage = 5.0: denominator becomes (norm + 5.0), reducing similarity
        m_shrink = ItemCollaborativeFiltering(top_k_neighbors=10, shrinkage=5.0)
        m_shrink.fit_from_matrix(mat, item_ids=[101, 102])
        sim_shrink = m_shrink.item_neighbors_[101][0][1]

        assert sim_no_shrink > sim_shrink
        assert sim_shrink > 0.0

    def test_top_k_neighbors_capping(self):
        # 1 target item, 5 candidate items
        # Candidate items have varying overlap with target
        # Target item 100 has rating [1, 1, 1, 1]
        mat = sp.csr_matrix(
            [
                [1.0, 1.0, 1.0, 1.0],  # item 100
                [1.0, 0.0, 0.0, 0.0],  # item 101
                [1.0, 1.0, 0.0, 0.0],  # item 102
                [1.0, 1.0, 1.0, 0.0],  # item 103
                [1.0, 1.0, 1.0, 1.0],  # item 104
            ],
            dtype=np.float32,
        )
        # Set top_k_neighbors = 2
        model = ItemCollaborativeFiltering(top_k_neighbors=2, shrinkage=0.0)
        model.fit_from_matrix(mat, item_ids=[100, 101, 102, 103, 104])

        neighbors = model.item_neighbors_.get(100, [])
        assert len(neighbors) == 2
        # The top 2 neighbors should be 104 and 103 (highest cosine similarity)
        neighbor_ids = [mid for mid, _ in neighbors]
        assert neighbor_ids == [104, 103]


# =============================================================================
# 2. Personalized Recommendation & Ranking Tests
# =============================================================================

class TestPersonalizedRanking:
    @pytest.fixture
    def small_model(self):
        # Synthetic universe of 4 movies:
        # 101 (Action A), 102 (Action B), 201 (Romance A), 202 (Romance B)
        # Users 0, 1 love Action (101, 102)
        # Users 2, 3 love Romance (201, 202)
        mat = sp.csr_matrix(
            [
                [5.0, 4.0, 0.0, 0.0],  # 101
                [4.0, 5.0, 0.0, 0.0],  # 102
                [0.0, 0.0, 5.0, 4.0],  # 201
                [0.0, 0.0, 4.0, 5.0],  # 202
            ],
            dtype=np.float32,
        )
        pop_counts = {101: 2, 102: 2, 201: 2, 202: 2}
        pop_rec = PopularityRecommender().fit_from_counts(pop_counts)

        model = ItemCollaborativeFiltering(top_k_neighbors=5, shrinkage=0.0)
        model.fit_from_matrix(mat, item_ids=[101, 102, 201, 202], popularity_recommender=pop_rec)
        return model

    def test_user_history_ranks_similar_genre_top(self, small_model):
        # User who rated Action movie 101 should receive Action movie 102 as top recommendation
        recs = small_model.recommend(user_history=[101], k=2)
        assert len(recs) >= 1
        assert recs[0] == 102

        # User who rated Romance movie 201 should receive Romance movie 202 as top recommendation
        recs_romance = small_model.recommend(user_history=[201], k=2)
        assert len(recs_romance) >= 1
        assert recs_romance[0] == 202

    def test_weighted_ratings_influence_ranking(self):
        # User has rated item 101 with 5.0 and item 201 with 1.0
        # 101 is similar to 102; 201 is similar to 202
        mat = sp.csr_matrix(
            [
                [1.0, 1.0, 0.0],  # 101
                [1.0, 1.0, 0.0],  # 102
                [0.0, 1.0, 1.0],  # 201
                [0.0, 1.0, 1.0],  # 202
            ],
            dtype=np.float32,
        )
        model = ItemCollaborativeFiltering(top_k_neighbors=5, shrinkage=0.0)
        model.fit_from_matrix(mat, item_ids=[101, 102, 201, 202])

        # Pass user_ratings weight: 101 gets weight 5.0, 201 gets weight 1.0
        recs = model.recommend(
            user_history=[101, 201],
            user_ratings={101: 5.0, 201: 1.0},
            k=2,
        )
        # 102 score = sim(101, 102)*5.0 = 5.0
        # 202 score = sim(201, 202)*1.0 = 1.0
        # Hence 102 must be ranked first
        assert recs == [102, 202]


# =============================================================================
# 3. History Exclusion & Deduplication Tests
# =============================================================================

class TestHistoryExclusion:
    def test_already_rated_items_are_never_recommended(self):
        mat = sp.csr_matrix(
            [
                [5.0, 5.0],  # 101
                [5.0, 5.0],  # 102
                [5.0, 4.0],  # 103
            ],
            dtype=np.float32,
        )
        model = ItemCollaborativeFiltering(top_k_neighbors=5, shrinkage=0.0)
        model.fit_from_matrix(mat, item_ids=[101, 102, 103])

        # User has already rated 101 and 102
        user_history = [101, 102]
        recs = model.recommend(user_history=user_history, k=5)

        for mid in user_history:
            assert mid not in recs
        assert recs == [103]

    def test_recommendation_list_has_no_duplicates(self):
        mat = sp.csr_matrix(
            [
                [5.0, 5.0, 0.0],  # 101
                [5.0, 5.0, 0.0],  # 102
                [0.0, 5.0, 5.0],  # 103
            ],
            dtype=np.float32,
        )
        pop_counts = {101: 10, 102: 8, 103: 6, 104: 4, 105: 2}
        pop_rec = PopularityRecommender().fit_from_counts(pop_counts)

        model = ItemCollaborativeFiltering(top_k_neighbors=5, shrinkage=0.0)
        model.fit_from_matrix(mat, item_ids=[101, 102, 103], popularity_recommender=pop_rec)

        recs = model.recommend(user_history=[101], k=5)
        # Unique check
        assert len(recs) == len(set(recs))


# =============================================================================
# 4. Cold-Start Fallback & Backfill Tests
# =============================================================================

class TestColdStartAndBackfill:
    def test_cold_start_user_falls_back_to_popularity(self):
        pop_counts = {101: 100, 102: 80, 103: 60}
        pop_rec = PopularityRecommender().fit_from_counts(pop_counts)

        model = ItemCollaborativeFiltering(top_k_neighbors=5)
        model.popularity_recommender_ = pop_rec

        # User with empty history
        recs_empty = model.recommend(user_history=[], k=3)
        assert recs_empty == [101, 102, 103]

        # User with None history
        recs_none = model.recommend(user_history=None, k=2)
        assert recs_none == [101, 102]

    def test_cold_start_without_popularity_returns_empty(self):
        model = ItemCollaborativeFiltering(top_k_neighbors=5)
        model.popularity_recommender_ = None
        recs = model.recommend(user_history=[], k=5)
        assert recs == []

    def test_backfill_fills_shortfall_from_popularity(self):
        # 101 only has 1 neighbor: 102
        mat = sp.csr_matrix(
            [
                [5.0, 5.0],  # 101
                [5.0, 5.0],  # 102
            ],
            dtype=np.float32,
        )
        pop_counts = {101: 50, 102: 40, 201: 30, 202: 20, 203: 10}
        pop_rec = PopularityRecommender().fit_from_counts(pop_counts)

        model = ItemCollaborativeFiltering(top_k_neighbors=5, shrinkage=0.0)
        model.fit_from_matrix(mat, item_ids=[101, 102], popularity_recommender=pop_rec)

        # Asking for K=4 recommendations when Item-CF only has 1 candidate (102)
        recs = model.recommend(user_history=[101], k=4)
        assert len(recs) == 4
        # First item must be personalized (102)
        assert recs[0] == 102
        # Remaining 3 must be popular items not in user_history: [201, 202, 203]
        assert recs[1:] == [201, 202, 203]
        assert 101 not in recs


# =============================================================================
# 5. Temporal Leakage & Split Boundary Tests
# =============================================================================

class TestTemporalLeakage:
    @staticmethod
    def _create_synthetic_ratings(file_path: Path):
        """
        Creates a synthetic ratings file with clear timestamp cutoffs:
          Train:      ts <= 1000
          Validation: 1000 < ts <= 1400
          Test:       ts > 1400
        """
        rows = [
            # Training interactions (ts <= 1000)
            {"userId": 1, "movieId": 101, "rating": 5.0, "timestamp": 800},
            {"userId": 1, "movieId": 102, "rating": 4.0, "timestamp": 850},
            {"userId": 2, "movieId": 101, "rating": 4.5, "timestamp": 900},
            {"userId": 2, "movieId": 102, "rating": 4.0, "timestamp": 950},
            {"userId": 3, "movieId": 101, "rating": 3.0, "timestamp": 1000},
            {"userId": 3, "movieId": 103, "rating": 5.0, "timestamp": 990},

            # Validation interactions (1000 < ts <= 1400)
            # Future movie 999 only appears in validation
            {"userId": 1, "movieId": 999, "rating": 5.0, "timestamp": 1200},
            {"userId": 2, "movieId": 103, "rating": 4.5, "timestamp": 1250},

            # Test interactions (ts > 1400)
            # Future movie 888 only appears in test
            {"userId": 1, "movieId": 888, "rating": 5.0, "timestamp": 1500},
            {"userId": 2, "movieId": 888, "rating": 4.0, "timestamp": 1600},
        ]
        with open(file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["userId", "movieId", "rating", "timestamp"])
            writer.writeheader()
            writer.writerows(rows)

    def test_fit_excludes_future_timestamps(self, tmp_path):
        ratings_path = tmp_path / "ratings.csv"
        self._create_synthetic_ratings(ratings_path)

        model = ItemCollaborativeFiltering(
            top_k_neighbors=5,
            min_item_ratings=1,
            shrinkage=0.0,
        )
        model.fit(
            ratings_path=ratings_path,
            train_cutoff_timestamp=1000,
            chunksize=100,
        )

        # Movies 999 and 888 only appear AFTER timestamp 1000
        # They MUST NOT be in item counts or neighbors
        assert 999 not in model.item_counts_
        assert 888 not in model.item_counts_
        assert 999 not in model.item_neighbors_
        assert 888 not in model.item_neighbors_

        # Training items 101, 102, 103 must be present
        assert 101 in model.item_counts_
        assert 102 in model.item_counts_
        assert 103 in model.item_counts_

    def test_item_similarity_only_computed_from_train_co_ratings(self, tmp_path):
        ratings_path = tmp_path / "ratings.csv"
        self._create_synthetic_ratings(ratings_path)

        model = ItemCollaborativeFiltering(
            top_k_neighbors=5,
            min_item_ratings=1,
            shrinkage=0.0,
        )
        model.fit(
            ratings_path=ratings_path,
            train_cutoff_timestamp=1000,
            chunksize=100,
        )

        # In train (ts <= 1000), Users 1 and 2 co-rated 101 and 102
        # Neighbors of 102 should be 101
        neighbors_102 = [mid for mid, _ in model.item_neighbors_.get(102, [])]
        assert 101 in neighbors_102
        # Future movies must not be neighbors
        assert 999 not in neighbors_102
        assert 888 not in neighbors_102


# =============================================================================
# 6. Evaluation Function Tests
# =============================================================================

class TestEvaluateItemCF:
    def test_evaluation_metrics_and_cutoffs(self):
        mat = sp.csr_matrix(
            [
                [5.0, 5.0],  # 101
                [5.0, 5.0],  # 102
                [1.0, 0.0],  # 103
            ],
            dtype=np.float32,
        )
        pop_counts = {101: 10, 102: 8, 103: 6}
        pop_rec = PopularityRecommender().fit_from_counts(pop_counts)

        model = ItemCollaborativeFiltering(top_k_neighbors=5, shrinkage=0.0)
        model.fit_from_matrix(mat, item_ids=[101, 102, 103], popularity_recommender=pop_rec)

        # User 1 rated 101 in train; test ground truth is 102 (perfect hit)
        user_train = {1: [101]}
        user_test = {1: [102]}

        metrics = evaluate_item_cf(
            model=model,
            user_train_history=user_train,
            user_test_ground_truth=user_test,
            k_list=[1, 5],
        )

        assert metrics["evaluated_users"] == 1
        assert metrics["recall"][1] == 1.0
        assert metrics["recall"][5] == 1.0
        assert metrics["ndcg"][1] == 1.0
        assert metrics["ndcg"][5] == 1.0

    def test_evaluation_deterministic_sampling(self):
        # 20 synthetic users
        user_train = {u: [101] for u in range(1, 21)}
        user_test = {u: [102] for u in range(1, 21)}

        mat = sp.csr_matrix([[5.0], [5.0]], dtype=np.float32)
        model = ItemCollaborativeFiltering(top_k_neighbors=5)
        model.fit_from_matrix(mat, item_ids=[101, 102])

        res1 = evaluate_item_cf(
            model=model,
            user_train_history=user_train,
            user_test_ground_truth=user_test,
            k_list=[5],
            max_eval_users=5,
            random_seed=42,
        )
        res2 = evaluate_item_cf(
            model=model,
            user_train_history=user_train,
            user_test_ground_truth=user_test,
            k_list=[5],
            max_eval_users=5,
            random_seed=42,
        )

        assert res1["evaluated_users"] == 5
        assert res1["recall"][5] == res2["recall"][5]
        assert res1["ndcg"][5] == res2["ndcg"][5]

    def test_evaluation_empty_ground_truth(self):
        model = ItemCollaborativeFiltering()
        metrics = evaluate_item_cf(
            model=model,
            user_train_history={1: [101]},
            user_test_ground_truth={1: []},
            k_list=[5],
        )
        assert metrics["evaluated_users"] == 0
        assert metrics["recall"][5] == 0.0
        assert metrics["ndcg"][5] == 0.0
