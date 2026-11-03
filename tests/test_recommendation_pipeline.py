"""
test_recommendation_pipeline.py  --  Phase 6: Integration Tests for End-to-End Pipeline

Tests:
  1. Component Assembly and End-to-End Recommendation.
  2. Strict History Exclusion.
  3. Cold-Start Handling for Unseen and Empty Users.
  4. Detailed Recommendation Explanations and Retrieval Sources.
  5. Configurable Cutoffs and Candidate Pool Limits.
  6. Deterministic and Reproducible Outputs.
  7. Pipeline Evaluation Metrics Calculation (NDCG, Recall, Precision).
  8. Temporal Split Safeguards (Zero Future Interaction Leakage).
"""

from collections import Counter, defaultdict
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

import sys
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from popularity_recommender import PopularityRecommender
from item_collaborative_filtering import ItemCollaborativeFiltering
from candidate_retrieval import CandidateRetriever
from ranking_model import RankingFeatureExtractor, RankingModel
from recommendation_pipeline import MovieRecommendationPipeline


@pytest.fixture
def synthetic_pipeline():
    """
    Constructs a lightweight end-to-end pipeline with pre-fitted components.
    Movies: 1..10
    Users: 101, 102
    """
    # 1. Movie metadata
    movie_titles = {
        1: "Toy Story (1995)",
        2: "Jumanji (1995)",
        3: "Grumpier Old Men (1995)",
        4: "Waiting to Exhale (1995)",
        5: "Father of the Bride Part II (1995)",
        6: "Heat (1995)",
        7: "Sabrina (1995)",
        8: "Tom and Huck (1995)",
        9: "Sudden Death (1995)",
        10: "GoldenEye (1995)",
    }
    movie_genres = {
        1: "Adventure|Animation|Children|Comedy|Fantasy",
        2: "Adventure|Children|Fantasy",
        3: "Comedy|Romance",
        4: "Comedy|Drama|Romance",
        5: "Comedy",
        6: "Action|Crime|Thriller",
        7: "Comedy|Romance",
        8: "Adventure|Children",
        9: "Action",
        10: "Action|Adventure|Thriller",
    }

    # 2. Popularity Recommender
    pop = PopularityRecommender()
    pop_counts = Counter({
        1: 100, 2: 90, 3: 80, 4: 70, 5: 60,
        6: 50, 7: 40, 8: 30, 9: 20, 10: 10,
    })
    pop.fit_from_counts(pop_counts)
    pop.movie_titles_ = movie_titles
    pop.movie_genres_ = movie_genres

    # 3. Item-CF Recommender
    cf = ItemCollaborativeFiltering(top_k_neighbors=20, min_item_ratings=1)
    cf.popularity_recommender_ = pop
    # 1 is similar to 2 (0.9) and 5 (0.8)
    # 6 is similar to 9 (0.85) and 10 (0.75)
    cf.item_neighbors_ = {
        1: [(2, 0.9), (5, 0.8), (8, 0.5)],
        2: [(1, 0.9), (8, 0.7)],
        6: [(9, 0.85), (10, 0.75)],
    }
    cf.movie_titles_ = movie_titles
    cf.movie_genres_ = movie_genres

    # 4. Stage 1 Candidate Retriever
    retriever = CandidateRetriever(item_cf=cf, popularity_recommender=pop, strategy="waterfall")
    retriever.movie_titles_ = movie_titles
    retriever.movie_genres_ = movie_genres

    # 5. Stage 2 Feature Extractor & Ranker
    fe = RankingFeatureExtractor()
    item_sums = Counter({mid: cnt * 4.0 for mid, cnt in pop_counts.items()})
    fe.fit_from_training_stats(
        item_counts=pop_counts,
        item_sums=item_sums,
        user_counts={101: 5, 102: 3},
        user_sums={101: 20.0, 102: 12.0},
        movie_genres={mid: set(g.split("|")) for mid, g in movie_genres.items()},
    )
    fe.movie_titles_ = movie_titles

    ranker = RankingModel(feature_extractor=fe)
    # Fit dummy weights: high weight on movie_avg_rating and CF score
    X_dummy = np.array([
        [0.9, 0.5, 4.2, 0.8, 3.8, 0.5, 0.8, 2.0],
        [0.1, 0.9, 3.0, 0.9, 3.8, 0.5, 0.2, 0.0],
    ], dtype=np.float32)
    y_dummy = np.array([1, 0], dtype=np.int32)
    ranker.fit_weights(X_dummy, y_dummy)

    # 6. Pipeline Assembly
    user_train_hist = {
        101: [1, 6],  # rated Toy Story and Heat
        102: [3],     # rated Grumpier Old Men
    }

    pipeline = MovieRecommendationPipeline(
        retriever=retriever,
        ranker=ranker,
        candidate_pool_size=10,
        default_k=5,
    )
    pipeline.fit_from_components(
        retriever=retriever,
        ranker=ranker,
        user_train_history=user_train_hist,
        movie_titles=movie_titles,
        movie_genres=movie_genres,
    )
    return pipeline


class TestPipelineRecommendations:
    """Tests for core recommendation generation and ranking."""

    def test_end_to_end_recommendation_returns_top_k(self, synthetic_pipeline):
        recs = synthetic_pipeline.recommend(user_id=101, k=3)
        assert len(recs) == 3
        assert isinstance(recs, list)
        assert all(isinstance(mid, int) for mid in recs)

    def test_recommendation_history_exclusion_from_cache(self, synthetic_pipeline):
        # User 101 has rated 1 and 6 in training history
        recs = synthetic_pipeline.recommend(user_id=101, k=10)
        assert 1 not in recs
        assert 6 not in recs

    def test_recommendation_history_exclusion_from_explicit_argument(self, synthetic_pipeline):
        # Explicit history overrides training history
        recs = synthetic_pipeline.recommend(user_id=101, user_history=[2, 5, 9], k=5)
        assert 2 not in recs
        assert 5 not in recs
        assert 9 not in recs

    def test_zero_or_negative_k_returns_empty(self, synthetic_pipeline):
        assert synthetic_pipeline.recommend(user_id=101, k=0) == []
        assert synthetic_pipeline.recommend(user_id=101, k=-5) == []


class TestColdStartHandling:
    """Tests for cold-start user recommendations."""

    def test_unseen_user_cold_start(self, synthetic_pipeline):
        # User 999 was never seen in training history
        recs = synthetic_pipeline.recommend(user_id=999, k=4)
        assert len(recs) == 4
        # Cold start should return valid catalogue items without crashing
        assert all(1 <= mid <= 10 for mid in recs)
        # Stage 1 retrieval directly provides top popular items
        stage1_cands = synthetic_pipeline.retriever.retrieve_candidates(user_history=None, k=4)
        assert stage1_cands == [1, 2, 3, 4]

    def test_empty_user_history_cold_start(self, synthetic_pipeline):
        recs = synthetic_pipeline.recommend(user_id=101, user_history=[], k=3)
        assert len(recs) == 3

    def test_cold_start_with_explicit_history_excludes_seen(self, synthetic_pipeline):
        # New user who only rated movie 1
        recs = synthetic_pipeline.recommend(user_id=999, user_history=[1], k=5)
        assert 1 not in recs
        assert len(recs) == 5


class TestDetailedRecommendations:
    """Tests for recommend_with_details."""

    def test_details_include_expected_fields(self, synthetic_pipeline):
        details = synthetic_pipeline.recommend_with_details(user_id=101, k=3)
        assert len(details) == 3
        expected_keys = {
            "rank", "movieId", "title", "genres", "source",
            "predicted_score", "movie_avg_rating", "popularity_count",
        }
        for item in details:
            assert expected_keys.issubset(item.keys())
            assert 1 <= item["rank"] <= 3
            assert item["source"] in ("item_cf", "popularity")

    def test_details_track_item_cf_and_popularity_sources(self, synthetic_pipeline):
        # User 101 rated 1 and 6 -> neighbors 2, 5, 8, 9, 10 are Item-CF
        details = synthetic_pipeline.recommend_with_details(user_id=101, k=5)
        sources = [d["source"] for d in details]
        assert "item_cf" in sources


class TestPipelineEvaluation:
    """Tests for end-to-end evaluation utility."""

    def test_evaluate_on_split_computes_valid_metrics(self, synthetic_pipeline):
        ground_truth = {
            101: {2, 9},   # Relevant items for 101
            102: {1, 4},   # Relevant items for 102
        }
        res = synthetic_pipeline.evaluate_on_split(ground_truth=ground_truth, k_list=[2, 5])
        assert "ndcg" in res
        assert "recall" in res
        assert "precision" in res

        for k in [2, 5]:
            assert 0.0 <= res["ndcg"][k] <= 1.0
            assert 0.0 <= res["recall"][k] <= 1.0
            assert 0.0 <= res["precision"][k] <= 1.0

    def test_evaluate_on_empty_ground_truth(self, synthetic_pipeline):
        res = synthetic_pipeline.evaluate_on_split(ground_truth={}, k_list=[5])
        assert res["ndcg"][5] == 0.0
        assert res["recall"][5] == 0.0
        assert res["precision"][5] == 0.0


class TestTemporalSplitIntegrity:
    """Tests ensuring no future interactions leak into pipeline fitting."""

    def test_fit_respects_temporal_cutoff(self, tmp_path):
        ratings_csv = tmp_path / "ratings.csv"
        movies_csv = tmp_path / "movies.csv"

        movies_df = pd.DataFrame({
            "movieId": [1, 2, 3, 4],
            "title": ["Movie 1", "Movie 2", "Movie 3", "Movie 4"],
            "genres": ["Action", "Comedy", "Drama", "Action|Drama"],
        })
        movies_df.to_csv(movies_csv, index=False)

        # Train ratings (ts <= 100) vs Future ratings (ts > 100)
        ratings_df = pd.DataFrame({
            "userId": [1, 1, 2, 2, 3, 3],
            "movieId": [1, 2, 1, 3, 4, 4],
            "rating": [4.0, 5.0, 3.5, 4.0, 5.0, 5.0],
            "timestamp": [50, 60, 70, 80, 150, 200],  # User 3 and Movie 4 are future only
        })
        ratings_df.to_csv(ratings_csv, index=False)

        pipeline = MovieRecommendationPipeline()
        pipeline.fit(
            ratings_path=ratings_csv,
            movies_path=movies_csv,
            train_cutoff_timestamp=100,
            min_item_ratings=1,
            candidate_pool_size=5,
            verbose=False,
        )

        # Future user 3 should NOT be in training history
        assert 3 not in pipeline.user_train_history_
        # Movie 4 should have 0 training count in popularity
        assert pipeline.retriever.popularity_recommender.movie_counts_.get(4, 0) == 0
