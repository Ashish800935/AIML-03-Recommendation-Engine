"""
candidate_retrieval.py  --  Phase 5A: Multi-Source Candidate Retrieval Module

Candidate Retrieval (Stage 1 of a two-stage recommendation system):
From a catalogue of tens of thousands of movies, rapidly retrieve a shortlist
of N candidate movies (e.g., N in [50, 100, 200]) per user with high Recall.

Combines two complementary channels:
  1. Personalized Item-Based Collaborative Filtering (Item-CF):
     - Exploitation channel.
     - Recommends items similar to movies the user rated in training history.
  2. Popularity Baseline:
     - Exploration and coverage channel.
     - Recommends globally trending/popular items.
     - Acts as an automatic fallback for cold-start users and backfill when
       personalized candidate counts are insufficient.

Key Properties:
  - Deduplication: Merged candidate lists never contain duplicate movie IDs.
  - History Exclusion: Movies already rated in training data are strictly excluded.
  - Cold-Start Graceful Fallback: Unseen users receive top popular candidates.
  - Deterministic: Stable sorting ensures 100% reproducible candidate lists.
  - Temporal Split Integrity: Fitted exclusively on training data (timestamp <= cutoff).
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional, Union

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
    load_all_temporal_splits,
)
from item_collaborative_filtering import (
    ItemCollaborativeFiltering,
)


class CandidateRetriever:
    """
    Multi-source candidate retrieval module for two-stage recommendation engines.

    Retrieves top-N candidate movies for a user by combining personalized Item-CF
    recommendations with popularity-based fallback and backfill.
    """

    def __init__(
        self,
        item_cf: Optional[ItemCollaborativeFiltering] = None,
        popularity_recommender: Optional[PopularityRecommender] = None,
        strategy: str = "waterfall",
        cf_ratio: float = 0.8,
    ) -> None:
        """
        Parameters:
            item_cf: Fitted ItemCollaborativeFiltering model (personalized channel).
            popularity_recommender: Fitted PopularityRecommender model (popularity & fallback channel).
            strategy: Candidate generation strategy:
                      - 'waterfall': Prioritize all available Item-CF candidates up to K;
                                     backfill remaining shortfall with Popularity.
                      - 'blend': Allocate quotas (cf_ratio * K to Item-CF, remainder to Popularity),
                                 backfilling if either channel produces fewer than quota.
                      - 'item_cf_only': Only retrieve from Item-CF (falls back to popularity on cold-start).
                      - 'popularity_only': Pure popularity candidate retrieval.
            cf_ratio: Quota fraction for Item-CF in 'blend' strategy (default: 0.8 = 80% CF, 20% Pop).
        """
        self.item_cf = item_cf
        self.popularity_recommender = popularity_recommender
        self.strategy = strategy.lower()
        self.cf_ratio = max(0.0, min(1.0, cf_ratio))

        # Movie metadata cache for display
        self.movie_titles_: dict[int, str] = {}
        self.movie_genres_: dict[int, str] = {}

    def fit_from_models(
        self,
        item_cf: ItemCollaborativeFiltering,
        popularity_recommender: PopularityRecommender,
    ) -> CandidateRetriever:
        """Attach pre-fitted models directly."""
        self.item_cf = item_cf
        self.popularity_recommender = popularity_recommender
        self.movie_titles_.update(item_cf.movie_titles_)
        self.movie_genres_.update(item_cf.movie_genres_)
        return self

    def fit(
        self,
        ratings_path: Union[str, Path],
        train_cutoff_timestamp: int = DEFAULT_TRAIN_CUTOFF,
        chunksize: int = 1_000_000,
        top_k_neighbors: int = 50,
        min_item_ratings: int = 50,
        shrinkage: float = 5.0,
    ) -> CandidateRetriever:
        """
        Fit both Popularity and Item-CF models strictly on training ratings (timestamp <= cutoff).

        Parameters:
            ratings_path: Path to MovieLens ratings.csv.
            train_cutoff_timestamp: Training cutoff timestamp (Phase 2 80% boundary).
            chunksize: Chunk size for streaming CSV ingestion.
            top_k_neighbors: Max neighbors to store per item in Item-CF.
            min_item_ratings: Minimum training ratings for item eligibility in Item-CF.
            shrinkage: Shrinkage parameter for cosine similarity in Item-CF.
        """
        ratings_path = Path(ratings_path)

        # 1. Fit Popularity Baseline
        pop_model = PopularityRecommender()
        pop_model.fit(
            ratings_path=ratings_path,
            train_cutoff_timestamp=train_cutoff_timestamp,
            chunksize=chunksize,
        )
        self.popularity_recommender = pop_model

        # 2. Fit Item-Based Collaborative Filtering
        cf_model = ItemCollaborativeFiltering(
            top_k_neighbors=top_k_neighbors,
            min_item_ratings=min_item_ratings,
            shrinkage=shrinkage,
        )
        cf_model.fit(
            ratings_path=ratings_path,
            train_cutoff_timestamp=train_cutoff_timestamp,
            chunksize=chunksize,
            popularity_recommender=pop_model,
        )
        self.item_cf = cf_model

        return self

    def load_movie_metadata(self, movies_path: Union[str, Path]) -> None:
        """Load movie titles and genres from movies.csv for display."""
        movies_path = Path(movies_path)
        if not movies_path.exists():
            return

        df = pd.read_csv(movies_path, usecols=["movieId", "title", "genres"])
        for _, row in df.iterrows():
            mid = int(row["movieId"])
            self.movie_titles_[mid] = str(row["title"])
            self.movie_genres_[mid] = str(row["genres"])

        if self.item_cf is not None:
            self.item_cf.load_movie_metadata(movies_path)
        if self.popularity_recommender is not None:
            self.popularity_recommender.load_movie_metadata(movies_path)

    def _get_raw_item_cf_candidates(
        self,
        user_history: set[int],
        user_ratings: Optional[dict[int, float]] = None,
        limit: int = 200,
    ) -> list[tuple[int, float]]:
        """
        Retrieve raw Item-CF candidate items and scores without backfill.
        Returns list of (movieId, score) sorted descending by score.
        """
        if self.item_cf is None or not user_history:
            return []

        candidate_scores: defaultdict[int, float] = defaultdict(float)

        for rated_item in user_history:
            neighbors = self.item_cf.item_neighbors_.get(rated_item, [])
            weight = user_ratings.get(rated_item, 1.0) if user_ratings else 1.0
            for neighbor_mid, sim_score in neighbors:
                if neighbor_mid not in user_history:
                    candidate_scores[neighbor_mid] += sim_score * weight

        # Sort descending by score, tie-break by movieId ascending
        sorted_candidates = sorted(
            candidate_scores.items(),
            key=lambda x: (-x[1], x[0]),
        )
        return sorted_candidates[:limit]

    def _get_raw_popularity_candidates(
        self,
        exclude_set: set[int],
        limit: int = 200,
    ) -> list[int]:
        """
        Retrieve top popular items excluding already selected/rated items.
        """
        if self.popularity_recommender is None or limit <= 0:
            return []
        return self.popularity_recommender.recommend(user_history=exclude_set, k=limit)

    def retrieve_candidates(
        self,
        user_history: Optional[Union[set[int], list[int]]] = None,
        user_ratings: Optional[dict[int, float]] = None,
        k: int = 100,
    ) -> list[int]:
        """
        Retrieve top-K candidate movie IDs for a user.

        Parameters:
            user_history: Items the user already rated in the training set (strictly excluded).
            user_ratings: Optional dict of {movieId: rating} for weighting Item-CF neighbors.
            k: Number of candidates to retrieve (e.g., 50, 100, 200).

        Returns:
            List of unique movieIds of length up to k.
        """
        if k <= 0:
            return []

        seen_set = set(user_history) if user_history else set()

        # Pure popularity only
        if self.strategy == "popularity_only":
            return self._get_raw_popularity_candidates(seen_set, limit=k)

        # Cold-start fallback: user has no training history
        if not seen_set:
            return self._get_raw_popularity_candidates(seen_set, limit=k)

        # Personalized candidate generation
        candidates: list[int] = []
        selected_set: set[int] = set(seen_set)

        if self.strategy == "waterfall":
            # 1. Fill as many candidates as possible from Item-CF (up to K)
            cf_candidates = self._get_raw_item_cf_candidates(seen_set, user_ratings, limit=k)
            for mid, _ in cf_candidates:
                candidates.append(mid)
                selected_set.add(mid)

            # 2. Backfill remaining slots from Popularity
            if len(candidates) < k:
                shortfall = k - len(candidates)
                pop_candidates = self._get_raw_popularity_candidates(selected_set, limit=shortfall)
                candidates.extend(pop_candidates)

        elif self.strategy == "blend":
            # Quota allocation
            cf_quota = int(round(k * self.cf_ratio))
            pop_quota = k - cf_quota

            # Retrieve from Item-CF
            cf_candidates = self._get_raw_item_cf_candidates(seen_set, user_ratings, limit=cf_quota)
            for mid, _ in cf_candidates:
                candidates.append(mid)
                selected_set.add(mid)

            # If Item-CF fell short of its quota, add shortfall to popularity quota
            actual_pop_needed = pop_quota + (cf_quota - len(candidates))
            pop_candidates = self._get_raw_popularity_candidates(selected_set, limit=actual_pop_needed)
            for mid in pop_candidates:
                candidates.append(mid)
                selected_set.add(mid)

            # If still short, backfill again
            if len(candidates) < k:
                shortfall = k - len(candidates)
                extra_pop = self._get_raw_popularity_candidates(selected_set, limit=shortfall)
                candidates.extend(extra_pop)

        elif self.strategy == "item_cf_only":
            cf_candidates = self._get_raw_item_cf_candidates(seen_set, user_ratings, limit=k)
            for mid, _ in cf_candidates:
                candidates.append(mid)

        else:
            # Fallback to waterfall if unknown strategy
            cf_candidates = self._get_raw_item_cf_candidates(seen_set, user_ratings, limit=k)
            for mid, _ in cf_candidates:
                candidates.append(mid)
                selected_set.add(mid)
            if len(candidates) < k:
                pop_candidates = self._get_raw_popularity_candidates(selected_set, limit=k - len(candidates))
                candidates.extend(pop_candidates)

        return candidates[:k]

    def retrieve_candidates_with_details(
        self,
        user_history: Optional[Union[set[int], list[int]]] = None,
        user_ratings: Optional[dict[int, float]] = None,
        k: int = 100,
    ) -> list[dict]:
        """
        Retrieve top-K candidates with rank, metadata, and retrieval channel source.
        """
        seen_set = set(user_history) if user_history else set()
        cands = self.retrieve_candidates(user_history=user_history, user_ratings=user_ratings, k=k)

        # Determine source of each candidate
        cf_raw = dict(self._get_raw_item_cf_candidates(seen_set, user_ratings, limit=k * 2)) if seen_set else {}

        results: list[dict] = []
        for rank, mid in enumerate(cands, start=1):
            if mid in cf_raw:
                source = "item_cf"
                score = round(cf_raw[mid], 4)
            else:
                source = "popularity"
                score = float(self.popularity_recommender.movie_counts_.get(mid, 0)) if self.popularity_recommender else 0.0

            results.append({
                "rank": rank,
                "movieId": mid,
                "title": self.movie_titles_.get(mid, f"Movie {mid}"),
                "genres": self.movie_genres_.get(mid, "Unknown"),
                "source": source,
                "score": score,
            })
        return results


# =============================================================================
# Candidate Retrieval Evaluation Function
# =============================================================================

def compute_hit_rate_at_k(
    recommended: list[int],
    ground_truth: set[int],
    k: int,
) -> float:
    """
    Compute HitRate@K for candidate retrieval: 1.0 if at least one relevant item
    is in the candidate shortlist of length K, else 0.0.
    """
    if not ground_truth or k <= 0:
        return 0.0
    return 1.0 if any(item in ground_truth for item in recommended[:k]) else 0.0


def evaluate_candidate_retrieval(
    retriever: CandidateRetriever,
    user_train_history: dict[int, list[int]],
    user_test_ground_truth: dict[int, list[int]],
    k_list: Optional[list[int]] = None,
    max_eval_users: Optional[int] = None,
    random_seed: int = 42,
) -> dict:
    """
    Evaluate candidate retrieval on test users across high candidate cutoffs (e.g. 50, 100, 200).

    Computes:
      - Recall@K: What fraction of a user's relevant test items were successfully retrieved.
      - HitRate@K: Fraction of users for whom at least one relevant item was in the candidate set.
      - NDCG@K: Ranking quality discount of candidates.

    Parameters:
        retriever: Fitted CandidateRetriever.
        user_train_history: dict mapping userId -> list of training movieIds.
        user_test_ground_truth: dict mapping userId -> list of test movieIds.
        k_list: List of candidate pool sizes to evaluate (default: [50, 100, 200]).
        max_eval_users: Optional user sample limit for fast benchmarking.
        random_seed: Seed for deterministic sampling.

    Returns:
        Dictionary containing evaluated_users, recall, hit_rate, and ndcg per K.
    """
    if k_list is None:
        k_list = [50, 100, 200]

    max_k = max(k_list)

    recall_sums = {k: 0.0 for k in k_list}
    hit_rate_sums = {k: 0.0 for k in k_list}
    ndcg_sums = {k: 0.0 for k in k_list}

    # Candidate users: users with at least one relevant test item
    candidate_users = sorted(
        uid for uid, test_items in user_test_ground_truth.items()
        if len(test_items) > 0
    )

    if max_eval_users is not None and 0 < max_eval_users < len(candidate_users):
        rng = random.Random(random_seed)
        candidate_users = sorted(rng.sample(candidate_users, max_eval_users))

    evaluated_users = 0
    for uid in candidate_users:
        gt_set = set(user_test_ground_truth[uid])
        if not gt_set:
            continue

        train_history_list = user_train_history.get(uid, [])

        # Retrieve up to max_k candidates
        candidates = retriever.retrieve_candidates(user_history=train_history_list, k=max_k)

        for k in k_list:
            recall_sums[k] += compute_recall_at_k(candidates, gt_set, k)
            hit_rate_sums[k] += compute_hit_rate_at_k(candidates, gt_set, k)
            ndcg_sums[k] += compute_ndcg_at_k(candidates, gt_set, k)

        evaluated_users += 1

    if evaluated_users == 0:
        return {
            "evaluated_users": 0,
            "recall": {k: 0.0 for k in k_list},
            "hit_rate": {k: 0.0 for k in k_list},
            "ndcg": {k: 0.0 for k in k_list},
        }

    return {
        "evaluated_users": evaluated_users,
        "recall": {k: round(recall_sums[k] / evaluated_users, 5) for k in k_list},
        "hit_rate": {k: round(hit_rate_sums[k] / evaluated_users, 5) for k in k_list},
        "ndcg": {k: round(ndcg_sums[k] / evaluated_users, 5) for k in k_list},
    }
