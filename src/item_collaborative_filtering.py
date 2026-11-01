"""
item_collaborative_filtering.py  --  Phase 4A: Item-Based Collaborative Filtering

An item-item collaborative filtering (Item-CF) recommendation engine.
Designed for 4th-year B.Tech / ML internship students.

Core Principles:
1. Item-to-Item Similarity:
   - Items are represented by user interaction/rating vectors.
   - Two movies are similar if users rate/consume them in similar patterns.
   - Uses Cosine Similarity with an optional shrinkage factor to avoid small-sample bias:
       sim(i, j) = (r_i . r_j) / (||r_i||_2 * ||r_j||_2 + shrinkage)

2. Personalized Scoring:
   - For user u with rated items I_u and ratings r_{ui}, candidate movie j is scored by:
       score(u, j) = sum_{i in I_u} sim(i, j) * r_{ui}
   - Items already rated by user u are strictly excluded.

3. Graceful Cold-Start & Backfill:
   - Unseen/cold-start users fall back to the Popularity Baseline.
   - If personalized scoring produces fewer than K candidates, remaining slots
     are backfilled with the most popular unrated movies.

4. Zero Future-Data Leakage:
   - Item similarities and user histories are computed EXCLUSIVELY on training data
     (timestamp <= train_cutoff_timestamp = 1466837400).
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp

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


class ItemCollaborativeFiltering:
    """
    Item-Based Collaborative Filtering Recommender.

    Builds an item-item similarity graph from training interactions and generates
    personalized top-K movie recommendations for each user.
    """

    def __init__(
        self,
        top_k_neighbors: int = 50,
        min_item_ratings: int = 20,
        shrinkage: float = 0.0,
        min_similarity: float = 0.0,
    ) -> None:
        """
        Parameters:
            top_k_neighbors: Maximum number of most similar neighbors to store per item.
            min_item_ratings: Minimum training ratings an item must have to be included
                              in the similarity matrix (improves speed & statistical reliability).
            shrinkage: Shrinkage parameter lambda added to denominator of cosine similarity.
            min_similarity: Minimum cosine similarity threshold for two items to be neighbors.
        """
        self.top_k_neighbors = top_k_neighbors
        self.min_item_ratings = min_item_ratings
        self.shrinkage = shrinkage
        self.min_similarity = min_similarity

        # Item mappings
        self.item_neighbors_: dict[int, list[tuple[int, float]]] = {}
        self.item_counts_: Counter = Counter()
        self.movie_titles_: dict[int, str] = {}
        self.movie_genres_: dict[int, str] = {}

        # Fallback popularity recommender
        self.popularity_recommender_: Optional[PopularityRecommender] = None

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

        if self.popularity_recommender_ is not None:
            self.popularity_recommender_.load_movie_metadata(movies_path)

    def fit_from_matrix(
        self,
        item_user_matrix: sp.csr_matrix,
        item_ids: list[int],
        popularity_recommender: Optional[PopularityRecommender] = None,
    ) -> ItemCollaborativeFiltering:
        """
        Compute item-item cosine similarities directly from an item-user sparse matrix.
        Useful for unit testing with synthetic matrices and fast in-memory execution.

        Parameters:
            item_user_matrix: Sparse matrix of shape (n_items, n_users) with rating values.
            item_ids: List of movieIds corresponding to rows in item_user_matrix.
            popularity_recommender: Fitted popularity baseline for cold-start fallback.
        """
        n_items = len(item_ids)
        if n_items == 0 or item_user_matrix.nnz == 0:
            self.item_neighbors_ = {}
            self.popularity_recommender_ = popularity_recommender
            return self

        # 1. Row-normalize item vectors (L2 norm)
        # Compute row L2 norms: ||r_i||_2
        row_sq = item_user_matrix.multiply(item_user_matrix).sum(axis=1)
        norms = np.sqrt(np.asarray(row_sq).flatten())
        if self.shrinkage > 0.0:
            denom = norms + self.shrinkage
        else:
            denom = norms
        denom[denom == 0.0] = 1.0

        inv_norms = sp.diags(1.0 / denom)
        normalized_matrix = inv_norms @ item_user_matrix

        # 2. Compute Cosine Similarity Matrix: Sim = Norm_Matrix @ Norm_Matrix.T
        sim_matrix = (normalized_matrix @ normalized_matrix.T).tocsr()

        # 3. Extract Top-K neighbors per item
        self.item_neighbors_ = {}
        for i, mid in enumerate(item_ids):
            row_start = sim_matrix.indptr[i]
            row_end = sim_matrix.indptr[i + 1]
            col_indices = sim_matrix.indices[row_start:row_end]
            scores = sim_matrix.data[row_start:row_end]

            # Exclude self-similarity and scores below min_similarity
            valid_mask = (col_indices != i) & (scores > self.min_similarity)
            col_indices = col_indices[valid_mask]
            scores = scores[valid_mask]

            if len(scores) == 0:
                self.item_neighbors_[mid] = []
                continue

            # Select top neighbors
            if len(scores) > self.top_k_neighbors:
                top_idx = np.argpartition(scores, -self.top_k_neighbors)[-self.top_k_neighbors:]
                sorted_top = top_idx[np.argsort(-scores[top_idx])]
                neighbors = [
                    (item_ids[col_indices[idx]], float(scores[idx]))
                    for idx in sorted_top
                ]
            else:
                sorted_idx = np.argsort(-scores)
                neighbors = [
                    (item_ids[col_indices[idx]], float(scores[idx]))
                    for idx in sorted_idx
                ]

            self.item_neighbors_[mid] = neighbors

        self.popularity_recommender_ = popularity_recommender
        return self

    def fit(
        self,
        ratings_path: Union[str, Path],
        train_cutoff_timestamp: int = DEFAULT_TRAIN_CUTOFF,
        chunksize: int = 1_000_000,
        popularity_recommender: Optional[PopularityRecommender] = None,
    ) -> ItemCollaborativeFiltering:
        """
        Fit Item-Based Collaborative Filtering model on training interactions from ratings.csv.
        Streams data in memory-safe chunks and computes item similarities.

        Parameters:
            ratings_path: Path to MovieLens ratings.csv.
            train_cutoff_timestamp: Timestamp boundary for training data (ratings <= cutoff).
            chunksize: Rows per chunk for streaming.
            popularity_recommender: Pre-fitted PopularityRecommender, or None to fit one automatically.
        """
        ratings_path = Path(ratings_path)

        # Step 1: Count movie frequencies in training set to identify candidate items
        movie_counts: Counter = Counter()
        for chunk in pd.read_csv(
            ratings_path,
            chunksize=chunksize,
            usecols=["movieId", "timestamp"],
        ):
            train_chunk = chunk[chunk["timestamp"] <= train_cutoff_timestamp]
            if not train_chunk.empty:
                movie_counts.update(train_chunk["movieId"].value_counts().to_dict())

        self.item_counts_ = movie_counts

        # Fit fallback popularity recommender if not provided
        if popularity_recommender is None:
            self.popularity_recommender_ = PopularityRecommender().fit_from_counts(movie_counts)
        else:
            self.popularity_recommender_ = popularity_recommender

        # Filter movies meeting min_item_ratings threshold
        eligible_movies = [
            mid for mid, cnt in movie_counts.items()
            if cnt >= self.min_item_ratings
        ]
        eligible_movie_set = set(eligible_movies)
        mid_to_row = {mid: idx for idx, mid in enumerate(eligible_movies)}

        if not eligible_movies:
            self.item_neighbors_ = {}
            return self

        # Step 2: Accumulate user ratings for eligible movies into COO lists
        user_to_col: dict[int, int] = {}
        row_indices: list[np.ndarray] = []
        col_indices: list[np.ndarray] = []
        data_values: list[np.ndarray] = []

        for chunk in pd.read_csv(
            ratings_path,
            chunksize=chunksize,
            usecols=["userId", "movieId", "rating", "timestamp"],
        ):
            # Strict training mask (<= cutoff)
            train_mask = (chunk["timestamp"] <= train_cutoff_timestamp) & (chunk["movieId"].isin(eligible_movie_set))
            train_chunk = chunk[train_mask]
            if train_chunk.empty:
                continue

            uids = train_chunk["userId"].to_numpy()
            mids = train_chunk["movieId"].to_numpy()
            ratings = train_chunk["rating"].to_numpy(dtype=np.float32)

            # Map userIds to contiguous column indices
            chunk_col = np.empty(len(uids), dtype=np.int32)
            for i, uid in enumerate(uids):
                if uid not in user_to_col:
                    user_to_col[uid] = len(user_to_col)
                chunk_col[i] = user_to_col[uid]

            chunk_row = np.array([mid_to_row[m] for m in mids], dtype=np.int32)

            row_indices.append(chunk_row)
            col_indices.append(chunk_col)
            data_values.append(ratings)

        if not row_indices:
            self.item_neighbors_ = {}
            return self

        all_rows = np.concatenate(row_indices)
        all_cols = np.concatenate(col_indices)
        all_data = np.concatenate(data_values)

        n_items = len(eligible_movies)
        n_users = len(user_to_col)

        item_user_matrix = sp.csr_matrix(
            (all_data, (all_rows, all_cols)),
            shape=(n_items, n_users),
            dtype=np.float32,
        )

        # Step 3: Compute similarities and extract top neighbors
        return self.fit_from_matrix(
            item_user_matrix=item_user_matrix,
            item_ids=eligible_movies,
            popularity_recommender=self.popularity_recommender_,
        )

    def recommend(
        self,
        user_history: Optional[Union[set[int], list[int]]] = None,
        user_ratings: Optional[dict[int, float]] = None,
        k: int = 10,
    ) -> list[int]:
        """
        Generate top-K recommendations for a user based on their training history.

        Parameters:
            user_history: Items the user already rated in the training set (excluded from recs).
            user_ratings: Optional dictionary of {movieId: rating} to weight item similarities.
                          If None, each watched movie has weight = 1.0.
            k: Number of recommendations to return.

        Returns:
            List of recommended movieIds of length up to k.
        """
        if k <= 0:
            return []

        seen_set = set(user_history) if user_history else set()

        # Cold-start fallback: user has no training history
        if not seen_set:
            if self.popularity_recommender_ is not None:
                return self.popularity_recommender_.recommend(user_history=seen_set, k=k)
            return []

        # Candidate item scoring
        candidate_scores: defaultdict[int, float] = defaultdict(float)

        for rated_item in seen_set:
            neighbors = self.item_neighbors_.get(rated_item, [])
            weight = user_ratings.get(rated_item, 1.0) if user_ratings else 1.0
            for neighbor_mid, sim_score in neighbors:
                if neighbor_mid not in seen_set:
                    candidate_scores[neighbor_mid] += sim_score * weight

        # Sort candidate items descending by aggregated score
        recommended: list[int] = [
            mid for mid, _ in sorted(
                candidate_scores.items(),
                key=lambda x: (-x[1], x[0]),
            )[:k]
        ]

        # Graceful backfill: if Item-CF produced fewer than K items, backfill with popularity
        if len(recommended) < k and self.popularity_recommender_ is not None:
            backfill_needed = k - len(recommended)
            already_selected = seen_set.union(recommended)
            pop_recs = self.popularity_recommender_.recommend(
                user_history=already_selected,
                k=backfill_needed,
            )
            recommended.extend(pop_recs)

        return recommended[:k]

    def recommend_with_details(
        self,
        user_history: Optional[Union[set[int], list[int]]] = None,
        user_ratings: Optional[dict[int, float]] = None,
        k: int = 10,
    ) -> list[dict]:
        """
        Generate top-K recommendations with rank, title, genres, and metadata.
        """
        recs = self.recommend(user_history=user_history, user_ratings=user_ratings, k=k)
        results = []
        for rank, mid in enumerate(recs, start=1):
            results.append({
                "rank": rank,
                "movieId": mid,
                "title": self.movie_titles_.get(mid, f"Movie {mid}"),
                "genres": self.movie_genres_.get(mid, "Unknown"),
                "neighbor_count": len(self.item_neighbors_.get(mid, [])),
            })
        return results

    def get_similar_items(self, movie_id: int, top_n: int = 10) -> list[dict]:
        """Return the top_n most similar movies to movie_id."""
        neighbors = self.item_neighbors_.get(movie_id, [])[:top_n]
        return [
            {
                "movieId": mid,
                "title": self.movie_titles_.get(mid, f"Movie {mid}"),
                "genres": self.movie_genres_.get(mid, "Unknown"),
                "similarity": round(score, 4),
            }
            for mid, score in neighbors
        ]


# =============================================================================
# Evaluation Function
# =============================================================================

def evaluate_item_cf(
    model: ItemCollaborativeFiltering,
    user_train_history: dict[int, list[int]],
    user_test_ground_truth: dict[int, list[int]],
    k_list: Optional[list[int]] = None,
    max_eval_users: Optional[int] = None,
    random_seed: int = 42,
) -> dict:
    """
    Evaluate Item-Based Collaborative Filtering on test users across K cutoffs.

    Parameters:
        model: Fitted ItemCollaborativeFiltering model.
        user_train_history: dict mapping userId -> list of movieIds rated in training set.
        user_test_ground_truth: dict mapping userId -> list of relevant movieIds in test set.
        k_list: List of K values to evaluate (default [5, 10, 20]).
        max_eval_users: Optional limit on the number of users to evaluate.
        random_seed: Seed for deterministic sampling.

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

        # Recommend up to max_k items excluding user's training history
        recs = model.recommend(user_history=train_history_list, k=max_k)

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
