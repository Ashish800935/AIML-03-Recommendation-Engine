"""
recommend.py  --  Phase 6: End-to-End Recommendation CLI

Usage:
    # 1. Recommend top 10 movies for an existing user (e.g. user 100):
    python scripts/recommend.py --user-id 100

    # 2. Recommend top 5 movies with 50 Stage 1 candidates:
    python scripts/recommend.py --user-id 100 --k 5 --candidate-k 50

    # 3. Recommend for a new/cold-start user with interactive history:
    python scripts/recommend.py --history 1,2,260 --k 10

    # 4. Pure cold-start recommendation (no history):
    python scripts/recommend.py --user-id 999999 --cold-start

    # 5. Run end-to-end validation evaluation across sample users:
    python scripts/recommend.py --eval --sample-users 100
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Allow imports from src/
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from data_loader import DATA_RAW_DIR
from popularity_recommender import (
    DEFAULT_TRAIN_CUTOFF,
    DEFAULT_VAL_CUTOFF,
    load_all_temporal_splits,
)
from recommendation_pipeline import MovieRecommendationPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 6: End-to-End Movie Recommendation Pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--user-id", type=int, default=1,
        help="Target user ID to generate recommendations for (default: 1).",
    )
    parser.add_argument(
        "--k", type=int, default=10,
        help="Number of recommendations to return (default: 10).",
    )
    parser.add_argument(
        "--candidate-k", type=int, default=100,
        help="Number of Stage 1 candidates to retrieve (default: 100).",
    )
    parser.add_argument(
        "--history", type=str, default="",
        help="Comma-separated movie IDs representing custom user history (e.g. '1,2,260').",
    )
    parser.add_argument(
        "--cold-start", action="store_true",
        help="Simulate a cold-start user with zero past interactions.",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=DATA_RAW_DIR,
        help="Path to data/raw directory containing ratings.csv and movies.csv.",
    )
    parser.add_argument(
        "--eval", action="store_true",
        help="Run end-to-end evaluation benchmark on the validation split.",
    )
    parser.add_argument(
        "--sample-users", type=int, default=100,
        help="Number of validation users for evaluation benchmark (default: 100).",
    )
    parser.add_argument(
        "--train-users", type=int, default=1000,
        help="Number of users to train the Stage 2 ranker on (default: 1000).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ratings_path = args.data_dir / "ratings.csv"
    movies_path = args.data_dir / "movies.csv"

    if not ratings_path.exists():
        print(f"Error: {ratings_path} not found.")
        sys.exit(1)
    if not movies_path.exists():
        print(f"Error: {movies_path} not found.")
        sys.exit(1)

    print("=" * 78)
    print("  AIML-03 Recommendation Engine  --  Phase 6: Two-Stage Recommendation Pipeline")
    print("=" * 78)
    print(f"  Dataset directory     : {args.data_dir.resolve()}")
    print(f"  Stage 1 Candidates    : {args.candidate_k}")
    print(f"  Top-K Recommendations : {args.k}")
    print(f"  Target User ID        : {args.user_id}")
    print("=" * 78)

    t0 = time.time()
    pipeline = MovieRecommendationPipeline(
        candidate_pool_size=args.candidate_k,
        default_k=args.k,
    )

    pipeline.fit(
        ratings_path=ratings_path,
        movies_path=movies_path,
        train_cutoff_timestamp=DEFAULT_TRAIN_CUTOFF,
        candidate_pool_size=args.candidate_k,
        max_ranker_train_users=args.train_users,
        verbose=True,
    )
    print(f"\nPipeline fitted in {time.time() - t0:.2f} seconds.")

    # -------------------------------------------------------------------------
    # Evaluation Mode
    # -------------------------------------------------------------------------
    if args.eval:
        print("\n" + "=" * 78)
        print(f"  END-TO-END VALIDATION BENCHMARK ({args.sample_users} users, seed=42)")
        print("=" * 78)

        _, val_gt, _, _ = load_all_temporal_splits(
            ratings_path=ratings_path,
            sample_val_users=args.sample_users,
            sample_test_users=0,
            random_seed=42,
        )

        t_eval = time.time()
        metrics = pipeline.evaluate_on_split(
            ground_truth=val_gt,
            k_list=[5, 10, 20],
            candidate_k=args.candidate_k,
        )
        eval_duration = time.time() - t_eval

        print(f"\n  Evaluation completed across {len(val_gt)} users in {eval_duration:.2f}s.")
        print("-" * 78)
        print(f"  {'Metric':<15} | {'@5':<12} | {'@10':<12} | {'@20':<12}")
        print("-" * 78)
        print(f"  {'NDCG':<15} | {metrics['ndcg'][5]:<12.4f} | {metrics['ndcg'][10]:<12.4f} | {metrics['ndcg'][20]:<12.4f}")
        print(f"  {'Recall':<15} | {metrics['recall'][5]:<12.4f} | {metrics['recall'][10]:<12.4f} | {metrics['recall'][20]:<12.4f}")
        print(f"  {'Precision':<15} | {metrics['precision'][5]:<12.4f} | {metrics['precision'][10]:<12.4f} | {metrics['precision'][20]:<12.4f}")
        print("-" * 78)
        return

    # -------------------------------------------------------------------------
    # Single User Recommendation Mode
    # -------------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("  USER RECOMMENDATION RESULTS")
    print("=" * 78)

    # Determine user history
    if args.cold_start:
        user_history = []
        user_type = "Cold-Start User (simulated empty history)"
    elif args.history:
        try:
            user_history = [int(m.strip()) for m in args.history.split(",") if m.strip()]
            user_type = f"Ad-hoc Profile ({len(user_history)} provided movies)"
        except ValueError:
            print(f"Error: Invalid history format '{args.history}'. Expected comma-separated integers.")
            sys.exit(1)
    else:
        user_history = pipeline.user_train_history_.get(args.user_id, [])
        if user_history:
            user_type = f"Existing User ({len(user_history)} training interactions)"
        else:
            user_type = "Unseen Cold-Start User (no training interactions found)"

    print(f"  User Profile: User {args.user_id} -- {user_type}")
    if user_history and len(user_history) <= 5:
        sample_titles = [pipeline.movie_titles_.get(m, f"Movie {m}") for m in user_history]
        print(f"  History Sample: {', '.join(sample_titles)}")
    elif user_history:
        sample_titles = [pipeline.movie_titles_.get(m, f"Movie {m}") for m in user_history[:5]]
        print(f"  History Sample (first 5): {', '.join(sample_titles)} ... (+{len(user_history)-5} more)")

    print("-" * 78)

    recs = pipeline.recommend_with_details(
        user_id=args.user_id,
        user_history=user_history,
        k=args.k,
        candidate_k=args.candidate_k,
    )

    if not recs:
        print("  No recommendations could be generated.")
        return

    # Print table
    header = f"  {'Rank':<4} | {'ID':<6} | {'Title':<32} | {'Genres':<22} | {'Source':<10} | {'Score':<6} | {'Rating':<6}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for item in recs:
        title = item["title"][:30] + ".." if len(item["title"]) > 32 else item["title"]
        genres = item["genres"][:20] + ".." if len(item["genres"]) > 22 else item["genres"]
        rating = f"{item['movie_avg_rating']:.2f}" if item["movie_avg_rating"] is not None else "N/A"
        print(
            f"  {item['rank']:<4} | "
            f"{item['movieId']:<6} | "
            f"{title:<32} | "
            f"{genres:<22} | "
            f"{item['source']:<10} | "
            f"{item['predicted_score']:<6.4f} | "
            f"{rating:<6}"
        )

    print("=" * 78)


if __name__ == "__main__":
    main()

