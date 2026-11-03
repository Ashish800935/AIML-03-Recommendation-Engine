"""
recommendation_pipeline.py  --  Phase 6: End-to-End Recommendation Pipeline

Integrates Stage 1 (Candidate Retrieval) and Stage 2 (Pointwise Ranking)
into a unified end-to-end recommendation engine for MovieLens 25M.

Architecture:
  User History / User ID
            │
            ▼
  Stage 1: Multi-Source Candidate Retrieval (Phase 5A)
    - Item-Based Collaborative Filtering (personalized channel)
    - Popularity Baseline (exploration, coverage, cold-start fallback)
    - Deduplication & History Exclusion
    - Outputs: N candidates (e.g., N=100) + CF affinity scores
            │
            ▼
  Stage 2: Pointwise Feature Extraction & Ranking Model (Phase 5B)
    - Extracts 8 features (CF score, popularity, movie avg rating, movie count,
      user avg rating, user count, genre affinity, genre overlap)
    - Evaluates calibrated probability / ranking score
    - Outputs: Top-K reranked recommendations (e.g., K=10)

Key Properties:
  - Supports active users with full training history.
  - Supports cold-start users (empty or unseen history) via popularity fallback.
  - Strictly excludes movies already rated by the user.
  - Preserves chronological train/validation/test boundaries (zero data leakage).
  - Memory-efficient streaming ingestion suitable for 8 GB RAM.
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

import sys
SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

from popularity_recommender import (
    DEFAULT_TRAIN_CUTOFF,
    DEFAULT_VAL_CUTOFF,
    PopularityRecommender,
    compute_recall_at_k,
    compute_ndcg_at_k,
)
from item_collaborative_filtering import ItemCollaborativeFiltering
from candidate_retrieval import CandidateRetriever
from ranking_model import (
    RankingFeatureExtractor,
    RankingModel,
    compute_precision_at_k,
    train_ranking_model_from_history,
)


class MovieRecommendationPipeline:
    """
    End-to-end two-stage recommendation pipeline.
    
    Combines:
      - Stage 1: CandidateRetriever (Item-CF + Popularity Waterfall/Blend)
      - Stage 2: RankingModel (Pointwise Logistic Ranker with 8 features)
    """

    def __init__(
        self,
        retriever: Optional[CandidateRetriever] = None,
        ranker: Optional[RankingModel] = None,
        candidate_pool_size: int = 100,
        default_k: int = 10,
    ) -> None:
        """
        Parameters:
            retriever: Pre-fitted CandidateRetriever instance.
            ranker: Pre-fitted RankingModel instance.
            candidate_pool_size: Number of candidates generated in Stage 1 (default: 100).
            default_k: Default number of final recommendations returned (default: 10).
        """
        self.retriever = retriever
        self.ranker = ranker
        self.candidate_pool_size = max(1, candidate_pool_size)
        self.default_k = max(1, default_k)

        # User training history cache: {userId: [movieId, ...]}
        self.user_train_history_: dict[int, list[int]] = {}

        # Movie metadata cache: {movieId: title}, {movieId: genres}
        self.movie_titles_: dict[int, str] = {}
        self.movie_genres_: dict[int, str] = {}

    def fit_from_components(
        self,
        retriever: CandidateRetriever,
        ranker: RankingModel,
        user_train_history: Optional[dict[int, list[int]]] = None,
        movie_titles: Optional[dict[int, str]] = None,
        movie_genres: Optional[dict[int, str]] = None,
    ) -> MovieRecommendationPipeline:
        """
        Attach pre-fitted Stage 1 and Stage 2 models directly.
        Ideal for unit testing or when components are trained separately.
        """
        self.retriever = retriever
        self.ranker = ranker

        if user_train_history is not None:
            self.user_train_history_ = user_train_history

        if movie_titles is not None:
            self.movie_titles_.update(movie_titles)
        elif retriever.movie_titles_:
            self.movie_titles_.update(retriever.movie_titles_)

        if movie_genres is not None:
            self.movie_genres_.update(movie_genres)
        elif retriever.movie_genres_:
            self.movie_genres_.update(retriever.movie_genres_)

        return self

    def fit(
        self,
        ratings_path: Union[str, Path],
        movies_path: Optional[Union[str, Path]] = None,
        train_cutoff_timestamp: int = DEFAULT_TRAIN_CUTOFF,
        chunksize: int = 1_000_000,
        min_item_ratings: int = 20,
        candidate_pool_size: int = 100,
        max_ranker_train_users: int = 2000,
        verbose: bool = True,
    ) -> MovieRecommendationPipeline:
        """
        Fit the complete two-stage pipeline strictly on training data (timestamp <= cutoff).

        Steps:
          1. Ingest training ratings to build user interaction histories and popularity counts.
          2. Fit ItemCollaborativeFiltering model on training data.
          3. Construct CandidateRetriever (waterfall strategy).
          4. Extract ranking feature statistics from training ratings.
          5. Train Stage 2 RankingModel weights on retrieved candidates.
        """
        ratings_path = Path(ratings_path)
        self.candidate_pool_size = candidate_pool_size

        if verbose:
            print("[Pipeline Step 1/5] Ingesting training interactions...")

        # Track user training histories and movie counts
        user_train_history: dict[int, list[int]] = defaultdict(list)
        movie_counts: Counter = Counter()

        for chunk in pd.read_csv(
            ratings_path,
            chunksize=chunksize,
            usecols=["userId", "movieId", "timestamp"],
        ):
            train_mask = chunk["timestamp"] <= train_cutoff_timestamp
            train_chunk = chunk[train_mask]
            if train_chunk.empty:
                continue

            for uid, mid in zip(train_chunk["userId"].to_numpy(), train_chunk["movieId"].to_numpy()):
                user_train_history[int(uid)].append(int(mid))
                movie_counts[int(mid)] += 1

        self.user_train_history_ = dict(user_train_history)

        if verbose:
            print(f"  Ingested {len(self.user_train_history_):,} unique training users.")
            print("[Pipeline Step 2/5] Fitting Stage 1 Popularity & Item-CF Recommenders...")

        # 1. Popularity Recommender
        pop_recommender = PopularityRecommender()
        pop_recommender.fit_from_counts(movie_counts)
        if movies_path:
            pop_recommender.load_movie_metadata(movies_path)

        # 2. Item-CF Recommender
        cf_recommender = ItemCollaborativeFiltering(
            top_k_neighbors=20,
            min_item_ratings=min_item_ratings,
            shrinkage=10.0,
        )
        cf_recommender.fit(
            ratings_path=ratings_path,
            train_cutoff_timestamp=train_cutoff_timestamp,
            chunksize=chunksize,
            popularity_recommender=pop_recommender,
        )
        if movies_path:
            cf_recommender.load_movie_metadata(movies_path)

        # 3. Stage 1 Candidate Retriever
        self.retriever = CandidateRetriever(
            item_cf=cf_recommender,
            popularity_recommender=pop_recommender,
            strategy="waterfall",
        )
        if movies_path:
            self.retriever.load_movie_metadata(movies_path)

        if verbose:
            print("[Pipeline Step 3/5] Extracting Stage 2 Ranking Features from training data...")

        # 4. Feature Extractor
        feature_extractor = RankingFeatureExtractor()
        feature_extractor.fit(
            ratings_path=ratings_path,
            movies_path=movies_path,
            train_cutoff_timestamp=train_cutoff_timestamp,
            chunksize=chunksize,
        )

        if verbose:
            print("[Pipeline Step 4/5] Training Stage 2 Pointwise Logistic Ranker...")

        # 5. Ranking Model
        ranker = RankingModel(feature_extractor=feature_extractor, l2_reg=1.0)
        train_ranking_model_from_history(
            model=ranker,
            user_train_history=self.user_train_history_,
            retriever=self.retriever,
            max_train_users=max_ranker_train_users,
            candidate_pool_size=min(50, candidate_pool_size),
        )
        self.ranker = ranker

        # Synchronize metadata caches
        if movies_path:
            self.load_movie_metadata(movies_path)

        if verbose:
            print("[Pipeline Step 5/5] Pipeline ready for inference.")

        return self

    def load_movie_metadata(self, movies_path: Union[str, Path]) -> None:
        """Load movie titles and genres for user-facing displays."""
        movies_path = Path(movies_path)
        if not movies_path.exists():
            return

        df = pd.read_csv(movies_path, usecols=["movieId", "title", "genres"])
        for _, row in df.iterrows():
            mid = int(row["movieId"])
            self.movie_titles_[mid] = str(row["title"])
            self.movie_genres_[mid] = str(row["genres"])

    def recommend(
        self,
        user_id: int,
        user_history: Optional[Union[set[int], list[int]]] = None,
        user_ratings: Optional[dict[int, float]] = None,
        k: Optional[int] = None,
        candidate_k: Optional[int] = None,
    ) -> list[int]:
        """
        Generate Top-K recommendations for a user through the full two-stage pipeline.

        Parameters:
            user_id: Target user identifier.
            user_history: Optional set/list of movieIds previously rated by the user.
                          If None, look up in self.user_train_history_.
                          Movies in this history are strictly excluded.
            user_ratings: Optional dict of {movieId: rating} for weighting CF interactions.
            k: Number of final recommendations to return (default: self.default_k).
            candidate_k: Number of candidates to retrieve in Stage 1 (default: self.candidate_pool_size).

        Returns:
            List of top-K recommended movieIds ordered by ranking score.
        """
        final_k = k if k is not None else self.default_k
        cand_k = candidate_k if candidate_k is not None else self.candidate_pool_size
        cand_k = max(cand_k, final_k)

        if final_k <= 0 or self.retriever is None:
            return []

        # 1. Resolve user history (lookup training interactions if not explicitly provided)
        if user_history is None:
            resolved_history: set[int] = set(self.user_train_history_.get(user_id, []))
        else:
            resolved_history = set(user_history)

        # 2. Stage 1: Candidate Retrieval (with cold-start handling and history exclusion)
        candidates = self.retriever.retrieve_candidates(
            user_history=resolved_history,
            user_ratings=user_ratings,
            k=cand_k,
        )

        if not candidates:
            return []

        # If Stage 2 Ranker is not fitted, return Stage 1 candidate order directly
        if self.ranker is None:
            return candidates[:final_k]

        # 3. Retrieve raw Item-CF candidate scores for ranking feature modulation
        cf_raw = dict(
            self.retriever._get_raw_item_cf_candidates(
                resolved_history,
                user_ratings=user_ratings,
                limit=cand_k * 2,
            )
        ) if resolved_history else {}

        # 4. Stage 2: Rerank candidates with rich features
        ranked_movie_ids = self.ranker.rank_candidates(
            user_id=user_id,
            candidate_ids=candidates,
            user_history=resolved_history,
            cf_scores=cf_raw,
            k=final_k,
        )

        return ranked_movie_ids

    def recommend_with_details(
        self,
        user_id: int,
        user_history: Optional[Union[set[int], list[int]]] = None,
        user_ratings: Optional[dict[int, float]] = None,
        k: Optional[int] = None,
        candidate_k: Optional[int] = None,
    ) -> list[dict]:
        """
        Generate Top-K recommendations along with detailed explanation metadata:
          rank, movieId, title, genres, retrieval_source, predicted_score,
          movie_avg_rating, popularity_count.
        """
        final_k = k if k is not None else self.default_k
        cand_k = candidate_k if candidate_k is not None else self.candidate_pool_size
        cand_k = max(cand_k, final_k)

        if final_k <= 0 or self.retriever is None:
            return []

        # 1. Resolve history
        if user_history is None:
            resolved_history: set[int] = set(self.user_train_history_.get(user_id, []))
        else:
            resolved_history = set(user_history)

        # 2. Stage 1: Candidates
        candidates = self.retriever.retrieve_candidates(
            user_history=resolved_history,
            user_ratings=user_ratings,
            k=cand_k,
        )

        if not candidates:
            return []

        # 3. Item-CF scores and source mapping
        cf_raw = dict(
            self.retriever._get_raw_item_cf_candidates(
                resolved_history,
                user_ratings=user_ratings,
                limit=cand_k * 2,
            )
        ) if resolved_history else {}

        # 4. Stage 2: Rerank and compute scores
        if self.ranker is not None and self.ranker.is_fitted_:
            # Build feature matrix
            user_g_counter = Counter(
                g for mid in resolved_history
                for g in self.ranker.feature_extractor.movie_genres_.get(mid, ())
            ) if resolved_history else None

            X = self.ranker.feature_extractor.extract_candidate_matrix(
                user_id=user_id,
                candidate_ids=candidates,
                cf_scores=cf_raw,
                user_favored_genres=user_g_counter,
            )
            scores = self.ranker.predict_scores(X)

            ranked_pairs = sorted(
                zip(candidates, scores),
                key=lambda x: (-x[1], x[0]),
            )[:final_k]
        else:
            # Fallback: maintain retriever order with unit score
            ranked_pairs = [(mid, 1.0) for mid in candidates[:final_k]]

        # 5. Format detailed results
        results: list[dict] = []
        for rank, (mid, score) in enumerate(ranked_pairs, start=1):
            source = "item_cf" if mid in cf_raw else "popularity"
            title = self.movie_titles_.get(mid, f"Movie {mid}")
            genres = self.movie_genres_.get(mid, "Unknown")

            # Extract quality stats if available
            avg_rating = 0.0
            pop_count = 0
            if self.ranker is not None:
                avg_rating = self.ranker.feature_extractor.get_movie_avg_rating(mid)
                pop_count = self.ranker.feature_extractor.item_counts_.get(mid, 0)
            elif self.retriever.popularity_recommender is not None:
                pop_count = self.retriever.popularity_recommender.movie_counts_.get(mid, 0)

            results.append({
                "rank": rank,
                "movieId": mid,
                "title": title,
                "genres": genres,
                "source": source,
                "predicted_score": round(float(score), 4),
                "movie_avg_rating": round(float(avg_rating), 2) if avg_rating else None,
                "popularity_count": pop_count,
            })

        return results

    def evaluate_on_split(
        self,
        ground_truth: dict[int, set[int]],
        k_list: list[int] = [5, 10, 20],
        candidate_k: Optional[int] = None,
    ) -> dict[str, dict[int, float]]:
        """
        Evaluate the complete end-to-end pipeline across a set of evaluation users.

        Returns dictionary of:
          - 'ndcg': {k: score}
          - 'recall': {k: score}
          - 'precision': {k: score}
        """
        cand_k = candidate_k or self.candidate_pool_size
        max_k = max(k_list)

        ndcg_accum: dict[int, float] = {k: 0.0 for k in k_list}
        recall_accum: dict[int, float] = {k: 0.0 for k in k_list}
        prec_accum: dict[int, float] = {k: 0.0 for k in k_list}
        user_count = 0

        for uid, target_items in ground_truth.items():
            if not target_items:
                continue

            recs = self.recommend(
                user_id=uid,
                k=max_k,
                candidate_k=cand_k,
            )

            for k in k_list:
                ndcg_accum[k] += compute_ndcg_at_k(recs, target_items, k=k)
                recall_accum[k] += compute_recall_at_k(recs, target_items, k=k)
                prec_accum[k] += compute_precision_at_k(recs, target_items, k=k)

            user_count += 1

        if user_count == 0:
            return {
                "ndcg": {k: 0.0 for k in k_list},
                "recall": {k: 0.0 for k in k_list},
                "precision": {k: 0.0 for k in k_list},
            }

        return {
            "ndcg": {k: round(ndcg_accum[k] / user_count, 4) for k in k_list},
            "recall": {k: round(recall_accum[k] / user_count, 4) for k in k_list},
            "precision": {k: round(prec_accum[k] / user_count, 4) for k in k_list},
        }
