"""
popularity_recommender.py  --  Popularity-Based Recommendation Baseline

A clean, beginner-friendly popularity recommendation engine and evaluation module.
Designed for 4th-year B.Tech / ML internship students.

Key Concepts:
1. Popularity Baseline:
   - A non-personalized baseline that ranks items by their interaction volume in the training set.
   - Any personalized model (Two-Tower, Neural Matrix Factorization, etc.) MUST beat
     this baseline to prove that personalization adds genuine value.

2. Temporal Split (Zero Future Leakage):
   - Training interactions: ratings with timestamp <= train_cutoff.
   - Test interactions: ratings with timestamp > train_cutoff.
   - We NEVER use test-period ratings when computing popularity ranks.

3. User History Filtering:
   - When generating recommendations for user u, we filter out all movies user u
     has already rated during the training period.
   - For new/cold-start users (empty history), we return the global top-K popular movies.

4. Ranking Evaluation Metrics:
   - Recall@K: What fraction of relevant test items were retrieved in top-K?
   - NDCG@K: Did the relevant items appear near the top of the recommendation list?
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional, Union

import pandas as pd


# =============================================================================
# 1. Evaluation Metrics: Recall@K and NDCG@K
# =============================================================================

def compute_recall_at_k(
    recommended: list[int],
    ground_truth: set[int],
    k: int,
) -> float:
    """
    Compute Recall@K for a single user.

    Formula:
        Recall@K = |Top-K Recommendations ∩ Ground Truth| / |Ground Truth|

    Parameters:
        recommended: Ordered list of recommended movieIds.
        ground_truth: Set of relevant movieIds the user actually rated in the test period.
        k: Cutoff rank (e.g., 5, 10, 20).

    Returns:
        Recall score in [0.0, 1.0]. Returns 0.0 if ground_truth is empty.
    """
    if not ground_truth or k <= 0:
        return 0.0

    k_recommended = recommended[:k]
    hits = len(set(k_recommended).intersection(ground_truth))
    return hits / len(ground_truth)


def compute_ndcg_at_k(
    recommended: list[int],
    ground_truth: set[int],
    k: int,
) -> float:
    """
    Compute Normalized Discounted Cumulative Gain (NDCG@K) for a single user.

    Uses binary relevance (1 if recommended item is in ground truth, else 0)
    with standard logarithmic position discount:
        DCG@K = sum_{i=1}^K [ rel_i / log2(i + 1) ]
        IDCG@K = sum_{j=1}^{min(K, |Ground Truth|)} [ 1 / log2(j + 1) ]
        NDCG@K = DCG@K / IDCG@K

    Parameters:
        recommended: Ordered list of recommended movieIds.
        ground_truth: Set of relevant movieIds the user actually rated in the test period.
        k: Cutoff rank (e.g., 5, 10, 20).

    Returns:
        NDCG score in [0.0, 1.0]. Returns 0.0 if ground_truth is empty or DCG is 0.
    """
    if not ground_truth or k <= 0:
        return 0.0

    k_recommended = recommended[:k]

    # Calculate Discounted Cumulative Gain (DCG)
    dcg = 0.0
    for rank_idx, item_id in enumerate(k_recommended):
        if item_id in ground_truth:
            # rank_idx 0 -> rank 1 -> log2(2) = 1.0
            # rank_idx 1 -> rank 2 -> log2(3) = 1.585
            dcg += 1.0 / math.log2(rank_idx + 2)

    if dcg == 0.0:
        return 0.0

    # Calculate Ideal DCG (IDCG): all possible hits placed at the top ranks
    ideal_hits = min(k, len(ground_truth))
    idcg = sum(1.0 / math.log2(j + 2) for j in range(ideal_hits))

    return dcg / idcg if idcg > 0.0 else 0.0


# =============================================================================
# 2. Popularity Recommender Class
# =============================================================================

class PopularityRecommender:
    """
    A baseline recommender that recommends the most popular movies from training data,
    excluding items the user has already rated.
    """

    def __init__(self, min_rating: float = 0.0) -> None:
        """
        Parameters:
            min_rating: Minimum rating threshold for an interaction to count towards
                        popularity (default 0.0 = count all ratings).
        """
        self.min_rating = min_rating
        self.movie_counts_: Counter = Counter()
        self.ranked_movie_ids_: list[int] = []
        self.movie_titles_: dict[int, str] = {}
        self.movie_genres_: dict[int, str] = {}

    def load_movie_metadata(self, movies_path: Union[str, Path]) -> None:
        """
        Load movie titles and genres from movies.csv for human-readable recommendations.
        """
        movies_path = Path(movies_path)
        if not movies_path.exists():
            return

        df = pd.read_csv(movies_path, usecols=["movieId", "title", "genres"])
        for _, row in df.iterrows():
            mid = int(row["movieId"])
            self.movie_titles_[mid] = str(row["title"])
            self.movie_genres_[mid] = str(row["genres"])

    def fit_from_counts(self, movie_counts: dict[int, int] | Counter) -> PopularityRecommender:
        """
        Fit directly from pre-computed {movieId: count} dictionary or Counter.
        Useful for testing or when counts are already computed.
        """
        self.movie_counts_ = Counter(movie_counts)
        # Sort movieIds descending by count; break ties by movieId ascending for deterministic order
        self.ranked_movie_ids_ = [
            mid for mid, _ in sorted(
                self.movie_counts_.items(),
                key=lambda x: (-x[1], x[0])
            )
        ]
        return self

    def fit(
        self,
        ratings_path: Union[str, Path],
        train_cutoff_timestamp: Optional[int] = None,
        chunksize: int = 1_000_000,
    ) -> PopularityRecommender:
        """
        Fit the popularity baseline from ratings.csv in chunked memory-safe mode.

        Parameters:
            ratings_path: Path to ratings.csv.
            train_cutoff_timestamp: Unix timestamp cutoff. Only interactions with
                                    timestamp <= cutoff are used. If None, uses all rows.
            chunksize: Chunk size for pandas read_csv to avoid high memory consumption.
        """
        ratings_path = Path(ratings_path)
        counts: Counter = Counter()

        cols = ["movieId", "rating", "timestamp"] if train_cutoff_timestamp is not None else ["movieId", "rating"]

        for chunk in pd.read_csv(ratings_path, chunksize=chunksize, usecols=cols):
            if train_cutoff_timestamp is not None:
                chunk = chunk[chunk["timestamp"] <= train_cutoff_timestamp]

            if self.min_rating > 0.0:
                chunk = chunk[chunk["rating"] >= self.min_rating]

            if not chunk.empty:
                counts.update(chunk["movieId"].value_counts().to_dict())

        return self.fit_from_counts(counts)

    def recommend(
        self,
        user_history: Optional[set[int]] = None,
        k: int = 10,
    ) -> list[int]:
        """
        Recommend top-K movies for a user, strictly excluding movies in user_history.

        Parameters:
            user_history: Set of movieIds the user has already rated in training data.
                          If None or empty, returns top-K globally popular movies.
            k: Number of recommendations to return.

        Returns:
            List of recommended movieIds of length at most k.
        """
        if k <= 0:
            return []

        if not user_history:
            return self.ranked_movie_ids_[:k]

        recommendations: list[int] = []
        for mid in self.ranked_movie_ids_:
            if mid not in user_history:
                recommendations.append(mid)
                if len(recommendations) == k:
                    break

        return recommendations

    def recommend_with_details(
        self,
        user_history: Optional[set[int]] = None,
        k: int = 10,
    ) -> list[dict]:
        """
        Recommend top-K movies with rank, title, genres, and training popularity count.
        """
        recs = self.recommend(user_history=user_history, k=k)
        results = []
        for rank, mid in enumerate(recs, start=1):
            results.append({
                "rank": rank,
                "movieId": mid,
                "title": self.movie_titles_.get(mid, f"Movie {mid}"),
                "genres": self.movie_genres_.get(mid, "Unknown"),
                "popularity_count": self.movie_counts_.get(mid, 0),
            })
        return results

    def get_top_popular_movies(self, k: int = 10) -> list[dict]:
        """Return the top-K globally most popular movies overall."""
        return self.recommend_with_details(user_history=None, k=k)


# =============================================================================
# 3. Chunked Temporal Data Preparation & Evaluation (80/10/10 Split)
# =============================================================================

# Phase 2 Chronological 80/10/10 Split Boundaries (MovieLens 25M)
# 80% Train cutoff: 2016-06-25 06:50:00 UTC
DEFAULT_TRAIN_CUTOFF = 1466837400
# 90% Val cutoff:   2018-01-04 14:45:08 UTC
DEFAULT_VAL_CUTOFF = 1515033908


def load_all_temporal_splits(
    ratings_path: Union[str, Path],
    train_cutoff_timestamp: int = DEFAULT_TRAIN_CUTOFF,
    val_cutoff_timestamp: int = DEFAULT_VAL_CUTOFF,
    test_min_rating: float = 3.5,
    chunksize: int = 1_000_000,
) -> tuple[Counter, dict[int, list[int]], dict[int, list[int]], dict[int, list[int]]]:
    """
    Scans ratings.csv in chunks and separates interactions into:
      1. Train interactions: timestamp <= train_cutoff_timestamp (~80% of data)
      2. Validation interactions: train_cutoff_timestamp < timestamp <= val_cutoff_timestamp (~10% of data)
      3. Test interactions: timestamp > val_cutoff_timestamp (~10% of data)

    Strict zero-leakage guarantee:
      - train_movie_counts and user_train_history are computed ONLY from training rows.
      - Validation and test ratings never influence item popularity ranking or seen-item filtering.

    Parameters:
        ratings_path: Path to ratings.csv.
        train_cutoff_timestamp: Split timestamp for training set.
        val_cutoff_timestamp: Split timestamp for validation set.
        test_min_rating: Minimum rating for an item in val/test to be considered relevant (default 3.5).
        chunksize: Rows per chunk for memory safety.

    Returns:
        train_movie_counts: Counter of {movieId: count} in training period.
        user_train_history: dict mapping userId -> list of movieIds rated in training period.
        user_val_ground_truth: dict mapping userId -> list of relevant movieIds in validation period.
        user_test_ground_truth: dict mapping userId -> list of relevant movieIds in test period.
    """
    ratings_path = Path(ratings_path)
    train_movie_counts: Counter = Counter()
    user_train_history: defaultdict[int, list[int]] = defaultdict(list)
    user_val_ground_truth: defaultdict[int, list[int]] = defaultdict(list)
    user_test_ground_truth: defaultdict[int, list[int]] = defaultdict(list)

    for chunk in pd.read_csv(
        ratings_path,
        chunksize=chunksize,
        usecols=["userId", "movieId", "rating", "timestamp"],
    ):
        # 1. Training rows (strictly <= train_cutoff_timestamp)
        train_mask = chunk["timestamp"] <= train_cutoff_timestamp
        train_chunk = chunk[train_mask]

        if not train_chunk.empty:
            train_movie_counts.update(train_chunk["movieId"].value_counts().to_dict())
            for uid, mids in train_chunk.groupby("userId")["movieId"]:
                user_train_history[uid].extend(mids.tolist())

        # 2. Validation rows (train_cutoff_timestamp < timestamp <= val_cutoff_timestamp)
        val_mask = (
            (chunk["timestamp"] > train_cutoff_timestamp)
            & (chunk["timestamp"] <= val_cutoff_timestamp)
            & (chunk["rating"] >= test_min_rating)
        )
        val_chunk = chunk[val_mask]
        if not val_chunk.empty:
            for uid, mids in val_chunk.groupby("userId")["movieId"]:
                user_val_ground_truth[uid].extend(mids.tolist())

        # 3. Test rows (strictly > val_cutoff_timestamp)
        test_mask = (chunk["timestamp"] > val_cutoff_timestamp) & (chunk["rating"] >= test_min_rating)
        test_chunk = chunk[test_mask]
        if not test_chunk.empty:
            for uid, mids in test_chunk.groupby("userId")["movieId"]:
                user_test_ground_truth[uid].extend(mids.tolist())

    return (
        train_movie_counts,
        dict(user_train_history),
        dict(user_val_ground_truth),
        dict(user_test_ground_truth),
    )


def load_temporal_split_data(
    ratings_path: Union[str, Path],
    train_cutoff_timestamp: int = DEFAULT_TRAIN_CUTOFF,
    val_cutoff_timestamp: Optional[int] = None,
    eval_split: str = "test",
    test_min_rating: float = 3.5,
    chunksize: int = 1_000_000,
) -> tuple[Counter, dict[int, list[int]], dict[int, list[int]]]:
    """
    Scans ratings.csv in chunks and separates interactions into training popularity
    and evaluation ground truth for either the 'test' or 'val' partition.

    Parameters:
        ratings_path: Path to ratings.csv.
        train_cutoff_timestamp: Training cutoff timestamp (ratings <= cutoff are TRAIN).
        val_cutoff_timestamp: Validation cutoff timestamp. If None and train_cutoff is default,
                              defaults to DEFAULT_VAL_CUTOFF (1515033908).
        eval_split: Evaluation partition to return: 'test' (default) or 'val'/'validation'.
        test_min_rating: Minimum rating for an item to be considered a relevant test interaction.
        chunksize: Rows per chunk.

    Returns:
        train_movie_counts: Counter of {movieId: count} in training period.
        user_train_history: dict mapping userId -> list of movieIds rated in training period.
        user_eval_ground_truth: dict mapping userId -> list of relevant movieIds in the requested eval period.
    """
    eval_split_norm = eval_split.strip().lower()
    if eval_split_norm not in ("test", "val", "validation"):
        raise ValueError(f"eval_split must be 'test' or 'val', got '{eval_split}'")

    # If val_cutoff_timestamp not provided and using default train cutoff, apply default val cutoff
    if val_cutoff_timestamp is None and train_cutoff_timestamp == DEFAULT_TRAIN_CUTOFF:
        val_cutoff_timestamp = DEFAULT_VAL_CUTOFF

    if val_cutoff_timestamp is not None:
        train_counts, user_train, user_val, user_test = load_all_temporal_splits(
            ratings_path=ratings_path,
            train_cutoff_timestamp=train_cutoff_timestamp,
            val_cutoff_timestamp=val_cutoff_timestamp,
            test_min_rating=test_min_rating,
            chunksize=chunksize,
        )
        eval_gt = user_val if eval_split_norm in ("val", "validation") else user_test
        return train_counts, user_train, eval_gt

    # Fallback when val_cutoff is not specified (e.g. 2-way split where test is > train_cutoff)
    if eval_split_norm in ("val", "validation"):
        raise ValueError("val_cutoff_timestamp must be provided to evaluate on validation split.")

    ratings_path = Path(ratings_path)
    train_movie_counts: Counter = Counter()
    user_train_history: defaultdict[int, list[int]] = defaultdict(list)
    user_test_ground_truth: defaultdict[int, list[int]] = defaultdict(list)

    for chunk in pd.read_csv(
        ratings_path,
        chunksize=chunksize,
        usecols=["userId", "movieId", "rating", "timestamp"],
    ):
        train_mask = chunk["timestamp"] <= train_cutoff_timestamp
        train_chunk = chunk[train_mask]
        if not train_chunk.empty:
            train_movie_counts.update(train_chunk["movieId"].value_counts().to_dict())
            for uid, mids in train_chunk.groupby("userId")["movieId"]:
                user_train_history[uid].extend(mids.tolist())

        test_mask = (chunk["timestamp"] > train_cutoff_timestamp) & (chunk["rating"] >= test_min_rating)
        test_chunk = chunk[test_mask]
        if not test_chunk.empty:
            for uid, mids in test_chunk.groupby("userId")["movieId"]:
                user_test_ground_truth[uid].extend(mids.tolist())

    return train_movie_counts, dict(user_train_history), dict(user_test_ground_truth)


def evaluate_popularity_recommender(
    recommender: PopularityRecommender,
    user_train_history: dict[int, list[int]],
    user_test_ground_truth: dict[int, list[int]],
    k_list: Optional[list[int]] = None,
    max_eval_users: Optional[int] = None,
    random_seed: int = 42,
) -> dict:
    """
    Evaluate the popularity recommender on test users across multiple K cutoffs.

    Sampling is deterministic and representative:
      - Candidate users are sorted by userId.
      - If max_eval_users is set, users are sampled uniformly using random.Random(random_seed)
        across the full candidate pool rather than taking the first N sequential users.

    Parameters:
        recommender: Fitted PopularityRecommender instance.
        user_train_history: dict mapping userId -> list/set of training movieIds.
        user_test_ground_truth: dict mapping userId -> list/set of eval ground-truth movieIds.
        k_list: List of K values to evaluate (default [5, 10, 20]).
        max_eval_users: Optional limit on the number of users to evaluate.
        random_seed: Random seed for deterministic sampling (default: 42).

    Returns:
        Dictionary with:
          - evaluated_users: int
          - recall: dict of {k: mean_recall}
          - ndcg: dict of {k: mean_ndcg}
    """
    if k_list is None:
        k_list = [5, 10, 20]

    max_k = max(k_list)

    recall_sums = {k: 0.0 for k in k_list}
    ndcg_sums = {k: 0.0 for k in k_list}

    # Candidate users: users who have at least one relevant item in the evaluation set
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

        train_history_set = set(user_train_history.get(uid, []))

        # Recommend up to max_k items excluding training history
        recs = recommender.recommend(user_history=train_history_set, k=max_k)

        for k in k_list:
            recall_sums[k] += compute_recall_at_k(recs, gt_set, k)
            ndcg_sums[k] += compute_ndcg_at_k(recs, gt_set, k)

        evaluated_users += 1

    if evaluated_users == 0:
        return {
            "evaluated_users": 0,
            "recall": {k: 0.0 for k in k_list},
            "ndcg": {k: 0.0 for k in k_list},
        }

    return {
        "evaluated_users": evaluated_users,
        "recall": {k: round(recall_sums[k] / evaluated_users, 5) for k in k_list},
        "ndcg": {k: round(ndcg_sums[k] / evaluated_users, 5) for k in k_list},
    }
