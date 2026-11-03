"""
test_candidate_retrieval.py  --  Unit Tests for Phase 5A Candidate Retrieval

All tests use small synthetic in-memory fixtures or temporary files.
No real MovieLens dataset is required.

Covers:
  1. Multi-source candidate merging (Item-CF + Popularity).
  2. Candidate deduplication (no repeated movie IDs in retrieved candidates).
  3. User history exclusion (already-rated movies are strictly excluded).
  4. Cold-start graceful handling (unseen users get popularity candidates).
  5. Configurable candidate limits (K=5, 50, 100, 200).
  6. Deterministic generation & reproducibility.
  7. Strict zero future-data leakage (validation & test items never retrieved).
  8. Retrieval strategies: Waterfall, Blend, Popularity-only, Item-CF-only.
  9. Evaluation runner (Recall@K, HitRate@K, NDCG@K, deterministic sampling).
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
)
from item_collaborative_filtering import (
    ItemCollaborativeFiltering,
)
from candidate_retrieval import (
    CandidateRetriever,
    evaluate_candidate_retrieval,
    compute_hit_rate_at_k,
)


@pytest.fixture
def synthetic_retriever():
    """
    Creates a small synthetic retriever with known items:
      - Item 101 has neighbor 102 (sim=0.9) and 103 (sim=0.7)
      - Item 201 has neighbor 202 (sim=0.85)
      - Popularity rankings: 301 (count=100), 302 (count=90), 303 (count=80), 102 (count=70), 103 (count=60)
    """
    mat = sp.csr_matrix(
        [
            [5.0, 5.0, 0.0],  # 101
            [5.0, 4.0, 0.0],  # 102
            [4.0, 5.0, 0.0],  # 103
            [0.0, 0.0, 5.0],  # 201
            [0.0, 0.0, 4.5],  # 202
        ],
        dtype=np.float32,
    )
    pop_counts = {
        301: 100,
        302: 90,
        303: 80,
        102: 70,
        103: 60,
        101: 50,
        201: 40,
        202: 30,
        304: 20,
        305: 10,
    }
    pop_model = PopularityRecommender().fit_from_counts(pop_counts)

    cf_model = ItemCollaborativeFiltering(top_k_neighbors=5, shrinkage=0.0)
    cf_model.fit_from_matrix(mat, item_ids=[101, 102, 103, 201, 202], popularity_recommender=pop_model)

    retriever = CandidateRetriever(
        item_cf=cf_model,
        popularity_recommender=pop_model,
        strategy="waterfall",
    )
    return retriever


# =============================================================================
# 1. Multi-Source Merging & Deduplication Tests
# =============================================================================

class TestMergingAndDeduplication:
    def test_merges_item_cf_and_popularity_without_duplicates(self, synthetic_retriever):
        # User rated 101.
        # Item-CF produces neighbors: 102, 103 (2 items).
        # We ask for K=5 candidates.
        # Candidates 1-2 must come from Item-CF: [102, 103].
        # Candidates 3-5 must come from Popularity: [301, 302, 303].
        # Note: 102 and 103 are also in the popularity list, but MUST NOT be duplicated.
        cands = synthetic_retriever.retrieve_candidates(user_history=[101], k=5)

        assert len(cands) == 5
        # No duplicates
        assert len(cands) == len(set(cands))
        # Personalized items first in waterfall
        assert cands[0] == 102
        assert cands[1] == 103
        # Popularity items backfilled without 102 or 103 repeated
        assert cands[2:] == [301, 302, 303]

    def test_candidate_details_track_source(self, synthetic_retriever):
        details = synthetic_retriever.retrieve_candidates_with_details(user_history=[101], k=4)
        assert len(details) == 4

        # First 2 should be item_cf
        assert details[0]["movieId"] == 102
        assert details[0]["source"] == "item_cf"
        assert details[1]["movieId"] == 103
        assert details[1]["source"] == "item_cf"

        # Next 2 should be popularity
        assert details[2]["movieId"] == 301
        assert details[2]["source"] == "popularity"
        assert details[3]["movieId"] == 302
        assert details[3]["source"] == "popularity"


# =============================================================================
# 2. History Exclusion Tests
# =============================================================================

class TestHistoryExclusion:
    def test_user_history_is_strictly_excluded(self, synthetic_retriever):
        # User rated 101, 102, and 301
        user_history = [101, 102, 301]
        cands = synthetic_retriever.retrieve_candidates(user_history=user_history, k=5)

        for watched in user_history:
            assert watched not in cands

    def test_details_also_exclude_history(self, synthetic_retriever):
        user_history = [101, 301]
        details = synthetic_retriever.retrieve_candidates_with_details(user_history=user_history, k=5)
        cand_ids = [d["movieId"] for d in details]

        assert 101 not in cand_ids
        assert 301 not in cand_ids


# =============================================================================
# 3. Cold-Start Fallback Tests
# =============================================================================

class TestColdStart:
    def test_cold_start_empty_history_gets_popularity(self, synthetic_retriever):
        # User with empty list []
        cands = synthetic_retriever.retrieve_candidates(user_history=[], k=4)
        assert len(cands) == 4
        # Should be top 4 popular items: 301, 302, 303, 102
        assert cands == [301, 302, 303, 102]

    def test_cold_start_none_history_gets_popularity(self, synthetic_retriever):
        cands = synthetic_retriever.retrieve_candidates(user_history=None, k=3)
        assert cands == [301, 302, 303]

    def test_cold_start_details_mark_source_popularity(self, synthetic_retriever):
        details = synthetic_retriever.retrieve_candidates_with_details(user_history=[], k=3)
        for d in details:
            assert d["source"] == "popularity"


# =============================================================================
# 4. Candidate Limits (K=50, 100, 200, custom) Tests
# =============================================================================

class TestCandidateLimits:
    def test_respects_custom_candidate_count(self, synthetic_retriever):
        for k in [1, 3, 5, 8]:
            cands = synthetic_retriever.retrieve_candidates(user_history=[101], k=k)
            assert len(cands) == k

    def test_handles_k_greater_than_catalogue(self, synthetic_retriever):
        # Total catalogue is 10 items. User rated 1 item -> 9 remaining unrated items.
        cands = synthetic_retriever.retrieve_candidates(user_history=[101], k=200)
        assert len(cands) == 9
        assert 101 not in cands

    def test_zero_or_negative_k_returns_empty(self, synthetic_retriever):
        assert synthetic_retriever.retrieve_candidates(user_history=[101], k=0) == []
        assert synthetic_retriever.retrieve_candidates(user_history=[101], k=-5) == []


# =============================================================================
# 5. Deterministic Generation & Reproducibility Tests
# =============================================================================

class TestReproducibility:
    def test_deterministic_candidate_order(self, synthetic_retriever):
        res1 = synthetic_retriever.retrieve_candidates(user_history=[101, 201], k=6)
        res2 = synthetic_retriever.retrieve_candidates(user_history=[101, 201], k=6)
        assert res1 == res2


# =============================================================================
# 6. Retrieval Strategies Tests (Waterfall, Blend, Item-CF Only, Pop Only)
# =============================================================================

class TestStrategies:
    def test_blend_strategy_respects_quota(self, synthetic_retriever):
        # Blend strategy: 50% CF, 50% Popularity
        synthetic_retriever.strategy = "blend"
        synthetic_retriever.cf_ratio = 0.5

        # K=4 -> 2 CF candidates (102, 103), 2 Pop candidates (301, 302)
        cands = synthetic_retriever.retrieve_candidates(user_history=[101], k=4)
        assert len(cands) == 4
        assert cands[:2] == [102, 103]
        assert cands[2:] == [301, 302]

    def test_popularity_only_strategy(self, synthetic_retriever):
        synthetic_retriever.strategy = "popularity_only"
        cands = synthetic_retriever.retrieve_candidates(user_history=[101], k=3)
        # Should return top 3 popular items excluding 101
        assert cands == [301, 302, 303]

    def test_item_cf_only_strategy(self, synthetic_retriever):
        synthetic_retriever.strategy = "item_cf_only"
        # User 101 has 2 neighbors (102, 103)
        cands = synthetic_retriever.retrieve_candidates(user_history=[101], k=10)
        # Item-CF only returns available CF candidates without backfill
        assert cands == [102, 103]


# =============================================================================
# 7. Temporal Leakage Tests
# =============================================================================

class TestTemporalLeakage:
    @staticmethod
    def _create_synthetic_ratings(file_path: Path):
        rows = [
            # Training interactions (ts <= 1000)
            {"userId": 1, "movieId": 101, "rating": 5.0, "timestamp": 800},
            {"userId": 1, "movieId": 102, "rating": 4.0, "timestamp": 850},
            {"userId": 2, "movieId": 101, "rating": 4.5, "timestamp": 900},
            {"userId": 2, "movieId": 102, "rating": 4.0, "timestamp": 950},
            {"userId": 3, "movieId": 101, "rating": 3.0, "timestamp": 1000},
            {"userId": 3, "movieId": 103, "rating": 5.0, "timestamp": 990},

            # Validation interactions (1000 < ts <= 1400)
            # Future movie 999 only in validation
            {"userId": 1, "movieId": 999, "rating": 5.0, "timestamp": 1200},

            # Test interactions (ts > 1400)
            # Future movie 888 only in test
            {"userId": 1, "movieId": 888, "rating": 5.0, "timestamp": 1500},
        ]
        with open(file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["userId", "movieId", "rating", "timestamp"])
            writer.writeheader()
            writer.writerows(rows)

    def test_fit_excludes_future_items_from_candidates(self, tmp_path):
        ratings_path = tmp_path / "ratings.csv"
        self._create_synthetic_ratings(ratings_path)

        retriever = CandidateRetriever(strategy="waterfall")
        retriever.fit(
            ratings_path=ratings_path,
            train_cutoff_timestamp=1000,
            chunksize=100,
            top_k_neighbors=5,
            min_item_ratings=1,
            shrinkage=0.0,
        )

        cands = retriever.retrieve_candidates(user_history=[101], k=10)

        # Future movies 999 and 888 must NEVER appear in candidate retrieval
        assert 999 not in cands
        assert 888 not in cands


# =============================================================================
# 8. Evaluation Function Tests (Recall@K, HitRate@K, NDCG@K)
# =============================================================================

class TestCandidateEvaluation:
    def test_compute_hit_rate_at_k(self):
        assert compute_hit_rate_at_k([10, 20, 30], {20}, k=2) == 1.0
        assert compute_hit_rate_at_k([10, 20, 30], {99}, k=3) == 0.0
        assert compute_hit_rate_at_k([10, 20, 30], set(), k=2) == 0.0
        assert compute_hit_rate_at_k([10, 20, 30], {10}, k=0) == 0.0

    def test_evaluate_candidate_retrieval(self, synthetic_retriever):
        user_train = {1: [101]}
        user_test = {1: [102, 301]}  # User likes 102 and 301 in test

        # For user 1, retrieved top-5 are [102, 103, 301, 302, 303]
        # At K=1: rec=[102], hits={102} -> Recall=0.5, HitRate=1.0
        # At K=5: rec has both 102 and 301 -> Recall=1.0, HitRate=1.0
        metrics = evaluate_candidate_retrieval(
            retriever=synthetic_retriever,
            user_train_history=user_train,
            user_test_ground_truth=user_test,
            k_list=[1, 5],
        )

        assert metrics["evaluated_users"] == 1
        assert metrics["recall"][1] == 0.5
        assert metrics["recall"][5] == 1.0
        assert metrics["hit_rate"][1] == 1.0
        assert metrics["hit_rate"][5] == 1.0

    def test_evaluation_deterministic_sampling(self, synthetic_retriever):
        user_train = {u: [101] for u in range(1, 21)}
        user_test = {u: [102] for u in range(1, 21)}

        res1 = evaluate_candidate_retrieval(
            retriever=synthetic_retriever,
            user_train_history=user_train,
            user_test_ground_truth=user_test,
            k_list=[5],
            max_eval_users=5,
            random_seed=42,
        )
        res2 = evaluate_candidate_retrieval(
            retriever=synthetic_retriever,
            user_train_history=user_train,
            user_test_ground_truth=user_test,
            k_list=[5],
            max_eval_users=5,
            random_seed=42,
        )

        assert res1["evaluated_users"] == 5
        assert res1["recall"][5] == res2["recall"][5]
        assert res1["hit_rate"][5] == res2["hit_rate"][5]
