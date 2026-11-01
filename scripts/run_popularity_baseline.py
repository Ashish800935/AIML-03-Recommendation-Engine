"""
run_popularity_baseline.py  --  Popularity Baseline CLI Runner & Evaluation

Usage:
    python scripts/run_popularity_baseline.py                  # Evaluates on Test split (default)
    python scripts/run_popularity_baseline.py --split val      # Evaluates on Validation split
    python scripts/run_popularity_baseline.py --split both     # Evaluates on both Validation and Test splits
    python scripts/run_popularity_baseline.py --sample-users 2000
    python scripts/run_popularity_baseline.py --eval-all
    python scripts/run_popularity_baseline.py --min-rating 3.0 --k 5,10,20,50
    python scripts/run_popularity_baseline.py --help

Designed for 4th-year B.Tech / ML internship students:
- Phase 2 80/10/10 Chronological Split:
    Train:      timestamp <= 1466837400 (2016-06-25 UTC, ~80%)
    Validation: 1466837400 < timestamp <= 1515033908 (2018-01-04 UTC, ~10%)
    Test:       timestamp > 1515033908 (~10%)
- Zero future leakage: Model fits ONLY on ratings before the training cutoff.
- Memory safe: Streams MovieLens 25M in 1,000,000-row chunks.
- Representative sampling: Sorts candidate users and draws deterministically using seed 42.
- Standard top-K recommendation metrics: Recall@K and NDCG@K.
"""

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
    PopularityRecommender,
    load_all_temporal_splits,
    evaluate_popularity_recommender,
    compute_recall_at_k,
    compute_ndcg_at_k,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Popularity Recommendation Baseline and Recall@K / NDCG@K Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-dir", type=Path, default=DATA_RAW_DIR,
        help="Path to data/raw directory containing ratings.csv and movies.csv.",
    )
    parser.add_argument(
        "--train-cutoff", type=int, default=DEFAULT_TRAIN_CUTOFF,
        help="Unix timestamp cutoff for training set (default: 1466837400 = 2016-06-25).",
    )
    parser.add_argument(
        "--val-cutoff", type=int, default=DEFAULT_VAL_CUTOFF,
        help="Unix timestamp cutoff for validation set (default: 1515033908 = 2018-01-04).",
    )
    parser.add_argument(
        "--split", choices=["test", "val", "both"], default="test",
        help="Evaluation split: 'test' (ts > val_cutoff), 'val' (train_cutoff < ts <= val_cutoff), or 'both' (default: 'test').",
    )
    parser.add_argument(
        "--min-rating", type=float, default=3.5,
        help="Minimum rating threshold for test items to be considered relevant (default: 3.5).",
    )
    parser.add_argument(
        "--k", type=str, default="5,10,20",
        help="Comma-separated list of K cutoffs for evaluation (default: '5,10,20').",
    )
    parser.add_argument(
        "--sample-users", type=int, default=5000,
        help="Number of test users to evaluate for fast benchmarking (default: 5000). Use --eval-all for all.",
    )
    parser.add_argument(
        "--eval-all", action="store_true",
        help="Evaluate all available test users instead of a sample.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for deterministic user sampling (default: 42).",
    )
    return parser.parse_args()


def format_ts(ts: int) -> str:
    """Format Unix timestamp as YYYY-MM-DD UTC."""
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def print_results_table(title: str, results: dict, k_list: list[int]) -> None:
    """Helper to display formatted evaluation results."""
    print("\n" + "=" * 50)
    print(f"  {title}")
    print("=" * 50)
    print(f"  Evaluated users: {results['evaluated_users']:,}")
    print("  " + "-" * 45)
    print(f"  {'Metric':<12} {'K':<6} {'Score':<10} {'Percentage':<10}")
    print("  " + "-" * 45)
    for k in k_list:
        rec_score = results['recall'][k]
        print(f"  {'Recall':<12} @{k:<5} {rec_score:<10.5f} {rec_score * 100:.2f}%")
    print("  " + "-" * 45)
    for k in k_list:
        ndcg_score = results['ndcg'][k]
        print(f"  {'NDCG':<12} @{k:<5} {ndcg_score:<10.5f} {ndcg_score * 100:.2f}%")
    print("=" * 50)


def main() -> None:
    args = parse_args()

    k_list = [int(x.strip()) for x in args.k.split(",") if x.strip()]
    data_dir = args.data_dir.resolve()
    ratings_path = data_dir / "ratings.csv"
    movies_path = data_dir / "movies.csv"

    if not ratings_path.exists():
        print(f"\n[ERROR] File not found: {ratings_path}")
        print("Please check that MovieLens 25M ratings.csv is placed in data/raw/")
        sys.exit(1)

    print("\n" + "=" * 75)
    print("  AIML-03 Recommendation Engine  --  Popularity Baseline & Evaluation")
    print("=" * 75)
    print(f"  Dataset directory   : {data_dir}")
    print(f"  Train cutoff TS     : {args.train_cutoff} ({format_ts(args.train_cutoff)}) [80%]")
    print(f"  Val cutoff TS       : {args.val_cutoff} ({format_ts(args.val_cutoff)}) [90%]")
    print(f"  Target eval split   : {args.split.upper()}")
    print(f"  Relevance threshold : rating >= {args.min_rating}")
    print(f"  Evaluation K values : {k_list}")
    print(f"  User sampling mode  : {'ALL test users' if args.eval_all else f'Sample of {args.sample_users} users (seed={args.seed})'}")
    print("=" * 75)

    # -------------------------------------------------------------------------
    # Step 1: Initialize Recommender and load movie metadata
    # -------------------------------------------------------------------------
    print("\n[Step 1/5] Loading movie metadata (titles & genres)...")
    recommender = PopularityRecommender()
    if movies_path.exists():
        recommender.load_movie_metadata(movies_path)
        print(f"  Loaded metadata for {len(recommender.movie_titles_):,} movies from movies.csv")
    else:
        print("  Warning: movies.csv not found; movie titles will display as IDs.")

    # -------------------------------------------------------------------------
    # Step 2: Temporal 80/10/10 Ingestion (Single Chunked Pass, Memory-Safe)
    # -------------------------------------------------------------------------
    print("\n[Step 2/5] Ingesting ratings.csv with Phase 2 chronological 80/10/10 split...")
    t0 = time.time()
    train_counts, user_train_history, user_val_gt, user_test_gt = load_all_temporal_splits(
        ratings_path=ratings_path,
        train_cutoff_timestamp=args.train_cutoff,
        val_cutoff_timestamp=args.val_cutoff,
        test_min_rating=args.min_rating,
        chunksize=1_000_000,
    )
    t_ingest = time.time() - t0

    total_train_ratings = sum(train_counts.values())
    total_val_interactions = sum(len(items) for items in user_val_gt.values())
    total_test_interactions = sum(len(items) for items in user_test_gt.values())

    print(f"  Completed in {t_ingest:.2f} seconds.")
    print(f"  [Train]      ratings ingested : {total_train_ratings:,} (<= {format_ts(args.train_cutoff)})")
    print(f"               unique movies    : {len(train_counts):,}")
    print(f"  [Validation] eligible users   : {len(user_val_gt):,} ({total_val_interactions:,} ratings in 80-90% window)")
    print(f"  [Test]       eligible users   : {len(user_test_gt):,} ({total_test_interactions:,} ratings in >90% window)")

    # -------------------------------------------------------------------------
    # Step 3: Fit Popularity Model (Strictly on Training Data)
    # -------------------------------------------------------------------------
    print("\n[Step 3/5] Fitting Popularity Baseline on training interactions...")
    recommender.fit_from_counts(train_counts)

    print("\n  Top 10 Most Popular Movies (Training Set):")
    print("  " + "-" * 66)
    print(f"  {'Rank':<5} {'MovieID':<9} {'Ratings':<10} {'Title':<40}")
    print("  " + "-" * 66)
    top10 = recommender.get_top_popular_movies(k=10)
    for m in top10:
        title = m['title'][:38] + ".." if len(m['title']) > 38 else m['title']
        print(f"  #{m['rank']:<4} {m['movieId']:<9} {m['popularity_count']:<10,} {title:<40}")
    print("  " + "-" * 66)

    # -------------------------------------------------------------------------
    # Step 4: Sample Recommendation Walkthrough for a User
    # -------------------------------------------------------------------------
    active_eval_gt = user_val_gt if args.split == "val" else user_test_gt
    split_label = "Validation" if args.split == "val" else "Test"
    print(f"\n[Step 4/5] Demonstration: Inspecting recommendations for a sample {split_label} user...")

    sample_uid = None
    for uid in active_eval_gt:
        if len(user_train_history.get(uid, [])) >= 5 and len(active_eval_gt[uid]) >= 2:
            sample_uid = uid
            break

    if sample_uid is not None:
        user_seen = set(user_train_history.get(sample_uid, []))
        user_gt = set(active_eval_gt[sample_uid])
        user_recs = recommender.recommend_with_details(user_history=user_seen, k=5)
        rec_ids = [m["movieId"] for m in user_recs]

        recall_5 = compute_recall_at_k(rec_ids, user_gt, k=5)
        ndcg_5 = compute_ndcg_at_k(rec_ids, user_gt, k=5)

        print(f"  Sample User ID: {sample_uid}")
        print(f"  - Training ratings count : {len(user_seen)} movies already rated (excluded from recommendations)")
        print(f"  - {split_label} ratings       : {len(user_gt)} relevant movies actually rated in {split_label.lower()} window")
        print("  - Top 5 Recommended Movies (Excluding seen movies):")
        for m in user_recs:
            hit_str = "[HIT!]" if m["movieId"] in user_gt else "      "
            title = m['title'][:36] + ".." if len(m['title']) > 36 else m['title']
            print(f"    {hit_str} Rank {m['rank']}: {title} ({m['genres']})")
        print(f"  - User Metrics @ 5: Recall@5 = {recall_5:.4f}, NDCG@5 = {ndcg_5:.4f}")

    # -------------------------------------------------------------------------
    # Step 5: Quantitative Evaluation Across Evaluation Users
    # -------------------------------------------------------------------------
    max_eval = None if args.eval_all else args.sample_users
    eval_desc = "ALL eligible users" if args.eval_all else f"{args.sample_users:,} deterministically sampled users (seed={args.seed})"

    if args.split in ("val", "both"):
        print(f"\n[Step 5/5] Evaluating Popularity Baseline on VALIDATION split ({eval_desc})...")
        t_start = time.time()
        val_results = evaluate_popularity_recommender(
            recommender=recommender,
            user_train_history=user_train_history,
            user_test_ground_truth=user_val_gt,
            k_list=k_list,
            max_eval_users=max_eval,
            random_seed=args.seed,
        )
        print(f"  Evaluation finished in {time.time() - t_start:.2f} seconds.")
        print_results_table("Popularity Baseline — VALIDATION Split (80–90% window)", val_results, k_list)

    if args.split in ("test", "both"):
        print(f"\n[Step 5/5] Evaluating Popularity Baseline on TEST split ({eval_desc})...")
        t_start = time.time()
        test_results = evaluate_popularity_recommender(
            recommender=recommender,
            user_train_history=user_train_history,
            user_test_ground_truth=user_test_gt,
            k_list=k_list,
            max_eval_users=max_eval,
            random_seed=args.seed,
        )
        print(f"  Evaluation finished in {time.time() - t_start:.2f} seconds.")
        print_results_table("Popularity Baseline — FINAL TEST Split (>90% window)", test_results, k_list)

    # -------------------------------------------------------------------------
    # Educational Summary for 4th-Year B.Tech Students
    # -------------------------------------------------------------------------
    print("\n" + "-" * 75)
    print("  KEY TAKEAWAYS FOR ML INTERVIEW / VIVA:")
    print("-" * 75)
    print("  1. Phase 2 Chronological 80/10/10 Split:")
    print("     - Train: ratings <= 2016-06-25 (80% of data). Model learns ONLY here.")
    print("     - Validation: 2016-06-25 to 2018-01-04 (10%). For hyperparameter tuning.")
    print("     - Test: ratings > 2018-01-04 (10%). For final benchmark comparison.")
    print("     - Zero Future Leakage: Future ratings never influence popularity ranks.")
    print()
    print("  2. Deterministic & Representative Sampling:")
    print("     - Candidate users are sorted by userId, then sampled with seed 42.")
    print("     - Avoids prefix bias (only evaluating early users) while guaranteeing")
    print("       100% reproducible metrics across runs.")
    print()
    print("  3. Benchmark Role in Two-Stage Recommender Systems:")
    print("     - Global popularity is the simplest non-personalized baseline.")
    print("     - Phase 4 Two-Tower retrieval and ranking models MUST outperform")
    print("       these validation and test numbers to justify neural architecture.")
    print("-" * 75 + "\n")


if __name__ == "__main__":
    main()
