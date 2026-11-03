"""
run_ranking_model.py  --  Phase 5B: Ranking Model Evaluation and Benchmark

Usage:
    python scripts/run_ranking_model.py                          # Evaluates on Validation split (default)
    python scripts/run_ranking_model.py --split test             # Evaluates on final Test split
    python scripts/run_ranking_model.py --split both             # Evaluates on both splits
    python scripts/run_ranking_model.py --candidate-pool 100     # Stage 1 candidate pool size
    python scripts/run_ranking_model.py --sample-users 5000      # Fast benchmark on 5,000 users (seed 42)
    python scripts/run_ranking_model.py --help

Key Features:
  - Stage 2 Ranking Model for Two-Stage Recommender Systems.
  - Takes candidate shortlist from Phase 5A and reranks them using training-set features.
  - Features: Item-CF score, popularity, movie avg rating, user avg rating, genre affinity.
  - Evaluates NDCG@10, Recall@10, and Precision@10 side-by-side against Popularity and Item-CF.
  - Memory safe for 8 GB RAM laptops (< 450 MB RAM total).
  - Strictly zero future-data leakage (trained and engineered on timestamp <= 1466837400).
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
)
from item_collaborative_filtering import (
    ItemCollaborativeFiltering,
    evaluate_item_cf,
)
from candidate_retrieval import (
    CandidateRetriever,
)
from ranking_model import (
    RankingFeatureExtractor,
    RankingModel,
    train_ranking_model_from_history,
    evaluate_ranking_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase 5B Ranking Model Evaluation and Benchmark against Baselines",
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
        "--split", choices=["val", "test", "both"], default="val",
        help="Evaluation split: 'val' (default for parameter tuning) or 'test' (final benchmark).",
    )
    parser.add_argument(
        "--min-rating", type=float, default=3.5,
        help="Minimum rating threshold for test items to be considered relevant (default: 3.5).",
    )
    parser.add_argument(
        "--k", type=str, default="5,10,20",
        help="Comma-separated list of recommendation cutoffs K (default: '5,10,20').",
    )
    parser.add_argument(
        "--candidate-pool", type=int, default=100,
        help="Stage 1 candidate pool size retrieved per user before reranking (default: 100).",
    )
    parser.add_argument(
        "--sample-users", type=int, default=5000,
        help="Number of evaluation users to sample for benchmarking (default: 5000). Use --eval-all for all.",
    )
    parser.add_argument(
        "--eval-all", action="store_true",
        help="Evaluate all available evaluation users instead of a sample.",
    )
    parser.add_argument(
        "--train-users", type=int, default=3000,
        help="Number of training users sampled to fit ranking model weights (default: 3000).",
    )
    parser.add_argument(
        "--l2-reg", type=float, default=1.0,
        help="L2 regularization penalty for logistic ranker (default: 1.0).",
    )
    parser.add_argument(
        "--top-neighbors", type=int, default=50,
        help="Item-CF neighbor count (default: 50).",
    )
    parser.add_argument(
        "--min-item-ratings", type=int, default=50,
        help="Minimum ratings for item eligibility in Item-CF (default: 50).",
    )
    parser.add_argument(
        "--shrinkage", type=float, default=5.0,
        help="Item-CF shrinkage factor (default: 5.0).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for deterministic user sampling (default: 42).",
    )
    return parser.parse_args()


def format_ts(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def evaluate_split(
    split_name: str,
    user_train_history: dict[int, list[int]],
    eval_ground_truth: dict[int, list[int]],
    pop_model: PopularityRecommender,
    item_cf: ItemCollaborativeFiltering,
    retriever: CandidateRetriever,
    ranking_model: RankingModel,
    candidate_pool_size: int,
    k_list: list[int],
    sample_users: int,
    eval_all: bool,
    seed: int,
) -> None:
    max_eval = None if eval_all else sample_users

    print("\n" + "=" * 78)
    print(f"  BENCHMARK EVALUATION: {split_name.upper()} SPLIT (Top-K Ranking Stage 2)")
    print("=" * 78)

    # 1. Popularity Baseline
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

    # 2. Item-CF Baseline
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

    # 3. Phase 5B Ranking Model
    print(f"\n[3/3] Evaluating Phase 5B Ranking Model (Candidate Pool: {candidate_pool_size}) at K={k_list}...")
    t0 = time.time()
    rank_metrics = evaluate_ranking_model(
        model=ranking_model,
        candidate_retriever=retriever,
        user_train_history=user_train_history,
        user_eval_ground_truth=eval_ground_truth,
        candidate_pool_size=candidate_pool_size,
        k_list=k_list,
        max_eval_users=max_eval,
        random_seed=seed,
    )
    t_rank = time.time() - t0
    print(f"      Completed in {t_rank:.2f}s across {rank_metrics['evaluated_users']:,} users.")

    # Print Comparative Benchmark Tables
    col_w = 14
    print("\n" + "-" * 78)
    print(f"  RANKING BENCHMARK RESULTS ({split_name.upper()} SPLIT)")
    print("-" * 78)

    # NDCG Table
    print(f"  NDCG@K (Ranking Quality & Position Discount):")
    header_ndcg = "  Model                          " + "".join(f"| NDCG@{k:<{col_w-7}}" for k in k_list)
    print(header_ndcg)
    print("  " + "-" * 31 + ("+" + "-" * (col_w - 1)) * len(k_list))

    row_pop = "  Popularity Baseline            " + "".join(f"| {pop_metrics['ndcg'].get(k, 0.0):.4f}{' ' * (col_w - 8)}" for k in k_list)
    row_cf = "  Item-Based CF                  " + "".join(f"| {cf_metrics['ndcg'].get(k, 0.0):.4f}{' ' * (col_w - 8)}" for k in k_list)
    row_rank = "  Phase 5B Ranking Model         " + "".join(f"| {rank_metrics['ndcg'].get(k, 0.0):.4f}{' ' * (col_w - 8)}" for k in k_list)
    print(row_pop)
    print(row_cf)
    print(row_rank)
    print("  " + "-" * 74)

    # Recall Table
    print(f"\n  Recall@K (Relevant Item Coverage):")
    header_rec = "  Model                          " + "".join(f"| Recall@{k:<{col_w-9}}" for k in k_list)
    print(header_rec)
    print("  " + "-" * 31 + ("+" + "-" * (col_w - 1)) * len(k_list))

    row_pop_r = "  Popularity Baseline            " + "".join(f"| {pop_metrics['recall'].get(k, 0.0):.4f}{' ' * (col_w - 8)}" for k in k_list)
    row_cf_r = "  Item-Based CF                  " + "".join(f"| {cf_metrics['recall'].get(k, 0.0):.4f}{' ' * (col_w - 8)}" for k in k_list)
    row_rank_r = "  Phase 5B Ranking Model         " + "".join(f"| {rank_metrics['recall'].get(k, 0.0):.4f}{' ' * (col_w - 8)}" for k in k_list)
    print(row_pop_r)
    print(row_cf_r)
    print(row_rank_r)
    print("  " + "-" * 74)

    # Precision Table
    print(f"\n  Precision@K (Density of Relevant Items in Top-K):")
    header_prec = "  Model                          " + "".join(f"| Prec@{k:<{col_w-7}}" for k in k_list)
    print(header_prec)
    print("  " + "-" * 31 + ("+" + "-" * (col_w - 1)) * len(k_list))

    row_rank_p = "  Phase 5B Ranking Model         " + "".join(f"| {rank_metrics['precision'].get(k, 0.0):.4f}{' ' * (col_w - 8)}" for k in k_list)
    print(row_rank_p)
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
    print("  AIML-03 Recommendation Engine  --  Phase 5B: Stage 2 Ranking Model")
    print("=" * 78)
    print(f"  Dataset directory     : {data_dir}")
    print(f"  Train cutoff TS       : {args.train_cutoff} ({format_ts(args.train_cutoff)}) [80%]")
    print(f"  Val cutoff TS         : {args.val_cutoff} ({format_ts(args.val_cutoff)}) [90%]")
    print(f"  Target eval split     : {args.split.upper()}")
    print(f"  Candidate pool size   : {args.candidate_pool} candidates from Stage 1")
    print(f"  Recommendation cutoffs: {k_list}")
    print(f"  L2 Regularization     : {args.l2_reg}")
    print(f"  User sampling mode    : {'ALL users' if args.eval_all else f'{args.sample_users:,} users (seed={args.seed})'}")
    print("=" * 78)

    # Step 1: Temporal Ingestion
    print("\n[Step 1/5] Ingesting ratings.csv with Phase 2 chronological 80/10/10 split...")
    t0 = time.time()
    train_counts, user_train_history, user_val_gt, user_test_gt = load_all_temporal_splits(
        ratings_path=ratings_path,
        train_cutoff_timestamp=args.train_cutoff,
        val_cutoff_timestamp=args.val_cutoff,
        test_min_rating=args.min_rating,
        chunksize=1_000_000,
    )
    t_ingest = time.time() - t0
    print(f"  Completed in {t_ingest:.2f} seconds.")
    print(f"  [Train]      ratings ingested : {sum(train_counts.values()):,}")
    print(f"  [Validation] eligible users   : {len(user_val_gt):,}")
    print(f"  [Test]       eligible users   : {len(user_test_gt):,}")

    # Step 2: Fit Stage 1 Models (Item-CF + Popularity)
    print("\n[Step 2/5] Fitting Stage 1 Candidate Retrieval (Item-CF + Popularity)...")
    pop_model = PopularityRecommender().fit_from_counts(train_counts)
    if movies_path.exists():
        pop_model.load_movie_metadata(movies_path)

    item_cf = ItemCollaborativeFiltering(
        top_k_neighbors=args.top_neighbors,
        min_item_ratings=args.min_item_ratings,
        shrinkage=args.shrinkage,
    )
    if movies_path.exists():
        item_cf.load_movie_metadata(movies_path)

    item_cf.fit(
        ratings_path=ratings_path,
        train_cutoff_timestamp=args.train_cutoff,
        chunksize=1_000_000,
        popularity_recommender=pop_model,
    )

    retriever = CandidateRetriever(
        item_cf=item_cf,
        popularity_recommender=pop_model,
        strategy="waterfall",
    )
    retriever.movie_titles_.update(item_cf.movie_titles_)
    retriever.movie_genres_.update(item_cf.movie_genres_)

    # Step 3: Extract Ranking Features from Training Data
    print("\n[Step 3/5] Extracting feature statistics from training ratings...")
    t_feat_start = time.time()
    feature_extractor = RankingFeatureExtractor()
    if movies_path.exists():
        feature_extractor.load_movie_metadata(movies_path)

    # Use training ratings to populate user and item statistics
    # To be fast and memory-safe, stream training chunks for user/item sums
    feature_extractor.fit(
        ratings_path=ratings_path,
        movies_path=movies_path if movies_path.exists() else None,
        train_cutoff_timestamp=args.train_cutoff,
        chunksize=1_000_000,
    )
    t_feat = time.time() - t_feat_start
    print(f"  Feature statistics extracted in {t_feat:.2f} seconds.")
    print(f"  Global training rating mean: {feature_extractor.global_mean_rating:.3f}")

    # Step 4: Fit Stage 2 Ranking Model
    print("\n[Step 4/5] Training Stage 2 Pointwise Logistic Ranker from training history...")
    t_train_start = time.time()
    ranking_model = RankingModel(
        feature_extractor=feature_extractor,
        candidate_retriever=retriever,
        l2_reg=args.l2_reg,
    )
    train_ranking_model_from_history(
        model=ranking_model,
        user_train_history=user_train_history,
        retriever=retriever,
        max_train_users=args.train_users,
        candidate_pool_size=50,
        random_seed=args.seed,
    )
    t_train = time.time() - t_train_start
    print(f"  Ranker weights trained in {t_train:.2f} seconds.")
    print("\n  Learned Feature Weights:")
    for fname, w in zip(RankingFeatureExtractor.FEATURE_NAMES, ranking_model.weights_):
        print(f"    - {fname:<22}: {w:+.4f}")
    print(f"    - {'bias (intercept)':<22}: {ranking_model.bias_:+.4f}")

    # Step 5: Benchmark Evaluation
    print("\n[Step 5/5] Running Benchmark Evaluation...")
    if args.split in ("val", "both"):
        evaluate_split(
            split_name="validation",
            user_train_history=user_train_history,
            eval_ground_truth=user_val_gt,
            pop_model=pop_model,
            item_cf=item_cf,
            retriever=retriever,
            ranking_model=ranking_model,
            candidate_pool_size=args.candidate_pool,
            k_list=k_list,
            sample_users=args.sample_users,
            eval_all=args.eval_all,
            seed=args.seed,
        )

    if args.split in ("test", "both"):
        evaluate_split(
            split_name="test",
            user_train_history=user_train_history,
            eval_ground_truth=user_test_gt,
            pop_model=pop_model,
            item_cf=item_cf,
            retriever=retriever,
            ranking_model=ranking_model,
            candidate_pool_size=args.candidate_pool,
            k_list=k_list,
            sample_users=args.sample_users,
            eval_all=args.eval_all,
            seed=args.seed,
        )

    print("\n" + "=" * 78)
    print("  Phase 5B Ranking Model Evaluation Complete.")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()

