"""
run_candidate_retrieval.py  --  Phase 5A: Candidate Retrieval Runner & Benchmark

Usage:
    python scripts/run_candidate_retrieval.py                          # Evaluates on final Test split (default)
    python scripts/run_candidate_retrieval.py --split val              # Evaluates on Validation split
    python scripts/run_candidate_retrieval.py --split both             # Evaluates on both splits
    python scripts/run_candidate_retrieval.py --sample-users 5000      # Fast benchmarking on 5000 users (seed 42)
    python scripts/run_candidate_retrieval.py --eval-all               # Evaluates on all available test users
    python scripts/run_candidate_retrieval.py --strategy blend         # Evaluates quota blend (80% CF, 20% Pop)
    python scripts/run_candidate_retrieval.py --help

Key Features:
  - Stage 1 Candidate Retrieval evaluation for Two-Stage Recommender Systems.
  - Multi-source candidate generation: Item-Based CF + Popularity fallback/backfill.
  - Compares candidate retrieval Recall@50, Recall@100, Recall@200 against individual baselines.
  - Evaluates HitRate@K (fraction of users where >= 1 relevant item was retrieved in candidates).
  - Memory safe for 8 GB RAM laptops (streaming ingestion + sparse matrix calculations).
  - Strictly zero future-data leakage (models fitted only on timestamp <= 1466837400).
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
from item_collaborative_filtering import (
    ItemCollaborativeFiltering,
    evaluate_item_cf,
)
from candidate_retrieval import (
    CandidateRetriever,
    evaluate_candidate_retrieval,
    compute_hit_rate_at_k,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase 5A Candidate Retrieval Evaluation and Benchmark Baselines",
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
        "--k", type=str, default="50,100,200",
        help="Comma-separated list of candidate pool sizes K for evaluation (default: '50,100,200').",
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
        "--strategy", choices=["waterfall", "blend", "item_cf_only", "popularity_only"], default="waterfall",
        help="Candidate retrieval strategy (default: 'waterfall').",
    )
    parser.add_argument(
        "--cf-ratio", type=float, default=0.8,
        help="Item-CF candidate quota ratio for 'blend' strategy (default: 0.8 = 80%% CF, 20%% Pop).",
    )
    parser.add_argument(
        "--top-neighbors", type=int, default=50,
        help="Number of most similar neighbors to store per item in Item-CF (default: 50).",
    )
    parser.add_argument(
        "--min-item-ratings", type=int, default=50,
        help="Minimum training ratings for a movie to be included in Item-CF (default: 50).",
    )
    parser.add_argument(
        "--shrinkage", type=float, default=5.0,
        help="Shrinkage factor added to cosine denominator to penalize small overlaps (default: 5.0).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for deterministic user sampling (default: 42).",
    )
    return parser.parse_args()


def format_ts(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def evaluate_retrieval_split(
    split_name: str,
    user_train_history: dict[int, list[int]],
    eval_ground_truth: dict[int, list[int]],
    retriever: CandidateRetriever,
    pop_model: PopularityRecommender,
    item_cf: ItemCollaborativeFiltering,
    k_list: list[int],
    sample_users: int,
    eval_all: bool,
    seed: int,
) -> None:
    max_eval = None if eval_all else sample_users

    print("\n" + "=" * 78)
    print(f"  BENCHMARK EVALUATION: {split_name.upper()} SPLIT (Candidate Retrieval Stage 1)")
    print("=" * 78)

    # 1. Evaluate Pure Popularity Baseline
    print(f"\n[1/3] Evaluating Popularity Baseline at K={k_list}...")
    t0 = time.time()
    pop_metrics = evaluate_popularity_recommender(
        recommender=pop_model,
        user_train_history=user_train_history,
        user_test_ground_truth=eval_ground_truth,
        k_list=k_list,
        max_eval_users=max_eval,
        random_seed=seed,
    )
    t_pop = time.time() - t0
    print(f"      Completed in {t_pop:.2f}s across {pop_metrics['evaluated_users']:,} users.")

    # 2. Evaluate Pure Item-CF Baseline
    print(f"\n[2/3] Evaluating Item-Based Collaborative Filtering at K={k_list}...")
    t0 = time.time()
    cf_metrics = evaluate_item_cf(
        model=item_cf,
        user_train_history=user_train_history,
        user_test_ground_truth=eval_ground_truth,
        k_list=k_list,
        max_eval_users=max_eval,
        random_seed=seed,
    )
    t_cf = time.time() - t0
    print(f"      Completed in {t_cf:.2f}s across {cf_metrics['evaluated_users']:,} users.")

    # 3. Evaluate Multi-Source Candidate Retrieval
    print(f"\n[3/3] Evaluating Multi-Source Candidate Retrieval (Strategy: {retriever.strategy.upper()}) at K={k_list}...")
    t0 = time.time()
    cand_metrics = evaluate_candidate_retrieval(
        retriever=retriever,
        user_train_history=user_train_history,
        user_test_ground_truth=eval_ground_truth,
        k_list=k_list,
        max_eval_users=max_eval,
        random_seed=seed,
    )
    t_cand = time.time() - t0
    print(f"      Completed in {t_cand:.2f}s across {cand_metrics['evaluated_users']:,} users.")

    # Print Side-by-Side Comparison Table
    print("\n" + "-" * 78)
    print(f"  CANDIDATE RETRIEVAL BENCHMARK RESULTS ({split_name.upper()} SPLIT)")
    print("-" * 78)

    # Header
    col_w = 14
    header_rec = "  Model / Retrieval Method       " + "".join(f"| Recall@{k:<{col_w-8}}" for k in k_list)
    print(header_rec)
    print("  " + "-" * 31 + ("+" + "-" * (col_w - 1)) * len(k_list))

    # Popularity row
    pop_row = "  Popularity Baseline            "
    for k in k_list:
        pop_row += f"| {pop_metrics['recall'].get(k, 0.0):.4f}{' ' * (col_w - 8)}"
    print(pop_row)

    # Item-CF row
    cf_row = "  Item-Based CF                  "
    for k in k_list:
        cf_row += f"| {cf_metrics['recall'].get(k, 0.0):.4f}{' ' * (col_w - 8)}"
    print(cf_row)

    # Candidate Retrieval row
    cand_label = f"  Candidate Retrieval ({retriever.strategy[:9]})"
    cand_row = f"{cand_label:<33}"
    for k in k_list:
        cand_row += f"| {cand_metrics['recall'].get(k, 0.0):.4f}{' ' * (col_w - 8)}"
    print(cand_row)
    print("  " + "-" * 74)

    # HitRate Table (Crucial for Stage 1 candidate generation)
    print("\n  CANDIDATE COVERAGE: Hit Rate (Fraction of users with >= 1 relevant candidate)")
    header_hit = "  Model / Retrieval Method       " + "".join(f"| HitRate@{k:<{col_w-9}}" for k in k_list)
    print(header_hit)
    print("  " + "-" * 31 + ("+" + "-" * (col_w - 1)) * len(k_list))

    cand_hit_row = f"{cand_label:<33}"
    for k in k_list:
        cand_hit_row += f"| {cand_metrics['hit_rate'].get(k, 0.0)*100:.2f}%{' ' * (col_w - 9)}"
    print(cand_hit_row)
    print("  " + "-" * 74)


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

    print("\n" + "=" * 78)
    print("  AIML-03 Recommendation Engine  --  Phase 5A: Multi-Source Candidate Retrieval")
    print("=" * 78)
    print(f"  Dataset directory     : {data_dir}")
    print(f"  Train cutoff TS       : {args.train_cutoff} ({format_ts(args.train_cutoff)}) [80%]")
    print(f"  Val cutoff TS         : {args.val_cutoff} ({format_ts(args.val_cutoff)}) [90%]")
    print(f"  Target eval split     : {args.split.upper()}")
    print(f"  Candidate sizes (K)   : {k_list}")
    print(f"  Retrieval strategy    : {args.strategy.upper()}" + (f" (CF quota: {args.cf_ratio:.0%})" if args.strategy == "blend" else ""))
    print(f"  User sampling mode    : {'ALL users' if args.eval_all else f'{args.sample_users:,} users (seed={args.seed})'}")
    print("=" * 78)

    # -------------------------------------------------------------------------
    # Step 1: Initialize Models & Metadata
    # -------------------------------------------------------------------------
    print("\n[Step 1/5] Initializing models and loading movie metadata...")
    item_cf = ItemCollaborativeFiltering(
        top_k_neighbors=args.top_neighbors,
        min_item_ratings=args.min_item_ratings,
        shrinkage=args.shrinkage,
    )
    pop_model = PopularityRecommender()

    if movies_path.exists():
        item_cf.load_movie_metadata(movies_path)
        pop_model.load_movie_metadata(movies_path)
        print(f"  Loaded metadata for {len(item_cf.movie_titles_):,} movies from movies.csv")

    # -------------------------------------------------------------------------
    # Step 2: Temporal Ingestion (Single Chunked Pass, Memory-Safe)
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
    # Step 3: Fit Models Strictly on Training Data
    # -------------------------------------------------------------------------
    print("\n[Step 3/5] Fitting Popularity Baseline & Item-Item Collaborative Filtering...")
    pop_model.fit_from_counts(train_counts)

    t_fit_start = time.time()
    item_cf.fit(
        ratings_path=ratings_path,
        train_cutoff_timestamp=args.train_cutoff,
        chunksize=1_000_000,
        popularity_recommender=pop_model,
    )
    t_fit = time.time() - t_fit_start
    print(f"  Item-CF fitted in {t_fit:.2f} seconds.")

    # Assemble Candidate Retriever
    retriever = CandidateRetriever(
        item_cf=item_cf,
        popularity_recommender=pop_model,
        strategy=args.strategy,
        cf_ratio=args.cf_ratio,
    )
    retriever.movie_titles_.update(item_cf.movie_titles_)
    retriever.movie_genres_.update(item_cf.movie_genres_)

    # -------------------------------------------------------------------------
    # Step 4: Sample Candidate Retrieval Demonstrations
    # -------------------------------------------------------------------------
    print("\n[Step 4/5] Candidate Retrieval Demonstrations (Top 10 of 100 Candidates)...")
    active_uids = [u for u, h in user_train_history.items() if len(h) >= 5]
    demo_uid = active_uids[0] if active_uids else 1
    demo_history = user_train_history.get(demo_uid, [])

    print(f"\n  Active User {demo_uid} (Watched {len(demo_history)} movies in training):")
    sample_rated = demo_history[:3]
    for mid in sample_rated:
        print(f"    - [{mid}] {item_cf.movie_titles_.get(mid, f'Movie {mid}')}")

    demo_cands = retriever.retrieve_candidates_with_details(user_history=demo_history, k=10)
    print(f"\n  Retrieved Candidate Shortlist for User {demo_uid}:")
    for cand in demo_cands:
        print(f"    Rank {cand['rank']:2d}: [{cand['movieId']}] {cand['title'][:40]:<40} | Source: {cand['source']:<10} | Score: {cand['score']}")

    # Cold Start Demo
    print(f"\n  Cold-Start User (No Training History):")
    cold_cands = retriever.retrieve_candidates_with_details(user_history=[], k=5)
    for cand in cold_cands:
        print(f"    Rank {cand['rank']:2d}: [{cand['movieId']}] {cand['title'][:40]:<40} | Source: {cand['source']:<10} | Score: {cand['score']}")

    # -------------------------------------------------------------------------
    # Step 5: Benchmark Evaluation on Target Split(s)
    # -------------------------------------------------------------------------
    print("\n[Step 5/5] Running Candidate Retrieval Benchmark Evaluation...")

    if args.split in ("val", "both"):
        evaluate_retrieval_split(
            split_name="validation",
            user_train_history=user_train_history,
            eval_ground_truth=user_val_gt,
            retriever=retriever,
            pop_model=pop_model,
            item_cf=item_cf,
            k_list=k_list,
            sample_users=args.sample_users,
            eval_all=args.eval_all,
            seed=args.seed,
        )

    if args.split in ("test", "both"):
        evaluate_retrieval_split(
            split_name="test",
            user_train_history=user_train_history,
            eval_ground_truth=user_test_gt,
            retriever=retriever,
            pop_model=pop_model,
            item_cf=item_cf,
            k_list=k_list,
            sample_users=args.sample_users,
            eval_all=args.eval_all,
            seed=args.seed,
        )

    print("\n" + "=" * 78)
    print("  Phase 5A Candidate Retrieval Evaluation Complete.")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
