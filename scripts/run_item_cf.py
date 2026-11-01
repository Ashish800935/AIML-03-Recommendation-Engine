"""
run_item_cf.py  --  Phase 4A: Item-Based Collaborative Filtering Runner & Benchmark

Usage:
    python scripts/run_item_cf.py                          # Evaluates on final Test split (default)
    python scripts/run_item_cf.py --split val              # Evaluates on Validation split
    python scripts/run_item_cf.py --split both             # Evaluates on both Validation and Test splits
    python scripts/run_item_cf.py --sample-users 2000
    python scripts/run_item_cf.py --eval-all
    python scripts/run_item_cf.py --top-neighbors 100 --min-item-ratings 50
    python scripts/run_item_cf.py --help

Key Features:
- Reuses Phase 2 chronological 80/10/10 split boundaries:
    Train: timestamp <= 1466837400 (2016-06-25 UTC, 80%)
    Val:   1466837400 < timestamp <= 1515033908 (2018-01-04 UTC, 10%)
    Test:  timestamp > 1515033908 (10%)
- Item similarity computed via Cosine Similarity on training interactions only.
- Compares Item-CF directly against the Popularity Baseline side-by-side.
- Memory safe for 8 GB RAM laptops (chunked streaming + sparse matrix computation).
- Deterministic and reproducible user sampling (seed 42).
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase 4A Item-Based Collaborative Filtering and Benchmark Against Popularity",
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
        "--top-neighbors", type=int, default=50,
        help="Number of most similar neighbors to store per item (default: 50).",
    )
    parser.add_argument(
        "--min-item-ratings", type=int, default=50,
        help="Minimum training ratings for a movie to be included in item-item graph (default: 50).",
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
    """Format Unix timestamp as YYYY-MM-DD UTC."""
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def print_comparison_table(
    split_name: str,
    pop_results: dict,
    cf_results: dict,
    k_list: list[int],
) -> None:
    """Helper to display formatted side-by-side comparison."""
    print("\n" + "=" * 78)
    print(f"  Benchmark Comparison: Popularity Baseline vs. Item-CF [{split_name}]")
    print("=" * 78)
    print(f"  Evaluated Users: {cf_results['evaluated_users']:,} (same sampled pool)")
    print("  " + "-" * 74)
    print(f"  {'Metric':<10} {'K':<5} {'Popularity':<14} {'Item-CF':<14} {'Gain':<12} {'Rel. Improvement':<16}")
    print("  " + "-" * 74)

    for k in k_list:
        p_val = pop_results['recall'][k]
        c_val = cf_results['recall'][k]
        gain = c_val - p_val
        rel = (gain / p_val * 100.0) if p_val > 0 else 0.0
        sign = "+" if gain >= 0 else ""
        print(f"  {'Recall':<10} @{k:<4} {p_val:<14.5f} {c_val:<14.5f} {sign}{gain:<11.5f} {sign}{rel:.2f}%")

    print("  " + "-" * 74)

    for k in k_list:
        p_val = pop_results['ndcg'][k]
        c_val = cf_results['ndcg'][k]
        gain = c_val - p_val
        rel = (gain / p_val * 100.0) if p_val > 0 else 0.0
        sign = "+" if gain >= 0 else ""
        print(f"  {'NDCG':<10} @{k:<4} {p_val:<14.5f} {c_val:<14.5f} {sign}{gain:<11.5f} {sign}{rel:.2f}%")

    print("=" * 78)


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
    print("  AIML-03 Recommendation Engine  --  Phase 4A: Item-Based Collaborative Filtering")
    print("=" * 78)
    print(f"  Dataset directory     : {data_dir}")
    print(f"  Train cutoff TS       : {args.train_cutoff} ({format_ts(args.train_cutoff)}) [80%]")
    print(f"  Val cutoff TS         : {args.val_cutoff} ({format_ts(args.val_cutoff)}) [90%]")
    print(f"  Target eval split     : {args.split.upper()}")
    print(f"  Min item ratings      : >= {args.min_item_ratings} ratings in training set")
    print(f"  Neighbors stored (M)  : {args.top_neighbors}")
    print(f"  Shrinkage lambda      : {args.shrinkage}")
    print(f"  Evaluation K values   : {k_list}")
    print(f"  User sampling mode    : {'ALL users' if args.eval_all else f'{args.sample_users:,} users (seed={args.seed})'}")
    print("=" * 78)

    # -------------------------------------------------------------------------
    # Step 1: Initialize Models & Metadata
    # -------------------------------------------------------------------------
    print("\n[Step 1/6] Initializing models and loading movie metadata...")
    item_cf = ItemCollaborativeFiltering(
        top_k_neighbors=args.top_neighbors,
        min_item_ratings=args.min_item_ratings,
        shrinkage=args.shrinkage,
    )
    if movies_path.exists():
        item_cf.load_movie_metadata(movies_path)
        print(f"  Loaded metadata for {len(item_cf.movie_titles_):,} movies from movies.csv")
    else:
        print("  Warning: movies.csv not found; movie titles will display as IDs.")

    # -------------------------------------------------------------------------
    # Step 2: Temporal Ingestion (Single Chunked Pass, Memory-Safe)
    # -------------------------------------------------------------------------
    print("\n[Step 2/6] Ingesting ratings.csv with Phase 2 chronological 80/10/10 split...")
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
    # Step 3: Fit Popularity Baseline (Fallback) & Item-Item CF Model
    # -------------------------------------------------------------------------
    print("\n[Step 3/6] Fitting Popularity Baseline & Item-Item Collaborative Filtering...")
    pop_model = PopularityRecommender().fit_from_counts(train_counts)
    if movies_path.exists():
        pop_model.load_movie_metadata(movies_path)

    t_fit_start = time.time()
    item_cf.fit(
        ratings_path=ratings_path,
        train_cutoff_timestamp=args.train_cutoff,
        chunksize=1_000_000,
        popularity_recommender=pop_model,
    )
    t_fit = time.time() - t_fit_start

    print(f"  Item-CF fitted in {t_fit:.2f} seconds.")
    print(f"  Movies with computed neighbors: {len(item_cf.item_neighbors_):,} (>= {args.min_item_ratings} ratings)")

    # -------------------------------------------------------------------------
    # Step 4: Inspect Learned Item Similarities (Sanity Check)
    # -------------------------------------------------------------------------
    print("\n[Step 4/6] Sanity Check: Inspecting Top Similar Movies for Landmark Titles...")
    landmark_mids = [1, 260, 296, 356]  # Toy Story, Star Wars IV, Pulp Fiction, Forrest Gump
    for l_mid in landmark_mids:
        if l_mid in item_cf.item_neighbors_:
            l_title = item_cf.movie_titles_.get(l_mid, f"Movie {l_mid}")
            print(f"\n  Top 3 Movies Similar to '{l_title}':")
            similar = item_cf.get_similar_items(l_mid, top_n=3)
            for rank, s in enumerate(similar, 1):
                print(f"    {rank}. {s['title']} (Cosine Sim: {s['similarity']:.4f})")

    # -------------------------------------------------------------------------
    # Step 5: User Demonstration & Walkthrough
    # -------------------------------------------------------------------------
    active_eval_gt = user_val_gt if args.split == "val" else user_test_gt
    split_label = "Validation" if args.split == "val" else "Test"
    print(f"\n[Step 5/6] Demonstration: Comparing Recommendations for a Sample {split_label} User...")

    sample_uid = None
    for uid in active_eval_gt:
        if len(user_train_history.get(uid, [])) >= 8 and len(active_eval_gt[uid]) >= 2:
            sample_uid = uid
            break

    if sample_uid is not None:
        user_seen = user_train_history.get(sample_uid, [])
        user_gt = set(active_eval_gt[sample_uid])

        # Get Item-CF recommendations
        cf_recs = item_cf.recommend_with_details(user_history=user_seen, k=5)
        cf_ids = [m["movieId"] for m in cf_recs]
        cf_rec_5 = compute_recall_at_k(cf_ids, user_gt, k=5)
        cf_ndcg_5 = compute_ndcg_at_k(cf_ids, user_gt, k=5)

        # Get Popularity recommendations
        pop_recs = pop_model.recommend_with_details(user_history=set(user_seen), k=5)
        pop_ids = [m["movieId"] for m in pop_recs]
        pop_rec_5 = compute_recall_at_k(pop_ids, user_gt, k=5)
        pop_ndcg_5 = compute_ndcg_at_k(pop_ids, user_gt, k=5)

        print(f"  Sample User ID: {sample_uid}")
        print(f"  - Training history: {len(user_seen)} movies already rated (excluded from recommendations)")
        print(f"  - {split_label} ground truth: {len(user_gt)} relevant movies actually rated after cutoff")

        print("\n  Personalized Item-CF Recommendations:")
        for m in cf_recs:
            hit = "[HIT!]" if m["movieId"] in user_gt else "      "
            print(f"    {hit} Rank {m['rank']}: {m['title']} ({m['genres']})")
        print(f"    Metrics @ 5: Recall@5 = {cf_rec_5:.4f}, NDCG@5 = {cf_ndcg_5:.4f}")

        print("\n  Global Popularity Baseline Recommendations:")
        for m in pop_recs:
            hit = "[HIT!]" if m["movieId"] in user_gt else "      "
            print(f"    {hit} Rank {m['rank']}: {m['title']} ({m['genres']})")
        print(f"    Metrics @ 5: Recall@5 = {pop_rec_5:.4f}, NDCG@5 = {pop_ndcg_5:.4f}")

    # -------------------------------------------------------------------------
    # Step 6: Quantitative Side-by-Side Evaluation
    # -------------------------------------------------------------------------
    max_eval = None if args.eval_all else args.sample_users
    eval_desc = "ALL eligible users" if args.eval_all else f"{args.sample_users:,} deterministically sampled users (seed={args.seed})"

    if args.split in ("val", "both"):
        print(f"\n[Step 6/6] Benchmarking on VALIDATION split ({eval_desc})...")
        t_eval = time.time()
        val_cf = evaluate_item_cf(
            model=item_cf,
            user_train_history=user_train_history,
            user_test_ground_truth=user_val_gt,
            k_list=k_list,
            max_eval_users=max_eval,
            random_seed=args.seed,
        )
        val_pop = evaluate_popularity_recommender(
            recommender=pop_model,
            user_train_history=user_train_history,
            user_test_ground_truth=user_val_gt,
            k_list=k_list,
            max_eval_users=max_eval,
            random_seed=args.seed,
        )
        print(f"  Evaluation finished in {time.time() - t_eval:.2f} seconds.")
        print_comparison_table("VALIDATION Split (80–90% window)", val_pop, val_cf, k_list)

    if args.split in ("test", "both"):
        print(f"\n[Step 6/6] Benchmarking on FINAL TEST split ({eval_desc})...")
        t_eval = time.time()
        test_cf = evaluate_item_cf(
            model=item_cf,
            user_train_history=user_train_history,
            user_test_ground_truth=user_test_gt,
            k_list=k_list,
            max_eval_users=max_eval,
            random_seed=args.seed,
        )
        test_pop = evaluate_popularity_recommender(
            recommender=pop_model,
            user_train_history=user_train_history,
            user_test_ground_truth=user_test_gt,
            k_list=k_list,
            max_eval_users=max_eval,
            random_seed=args.seed,
        )
        print(f"  Evaluation finished in {time.time() - t_eval:.2f} seconds.")
        print_comparison_table("FINAL TEST Split (>90% window)", test_pop, test_cf, k_list)

    # -------------------------------------------------------------------------
    # Educational Takeaways for B.Tech Viva / Interviews
    # -------------------------------------------------------------------------
    print("\n" + "-" * 78)
    print("  KEY TAKEAWAYS FOR ML INTERVIEW / VIVA:")
    print("-" * 78)
    print("  1. Why Item-Based Collaborative Filtering (Item-CF)?")
    print("     - User-based CF scales poorly because users constantly change and")
    print("       the user space is huge (162k+ users).")
    print("     - Item relationships are static, stable, and have clear semantics.")
    print("     - Precomputing the top-M item neighbors makes online recommendation")
    print("       O(1) lookups per user interaction, ideal for web scale.")
    print()
    print("  2. Cosine Similarity with Shrinkage:")
    print("     - Cosine similarity measures the angle between item rating vectors.")
    print("     - Adding a shrinkage term lambda penalizes items with few co-raters,")
    print("       preventing obscure pairs from dominating similarity ranks.")
    print()
    print("  3. Personalization Outperforms Global Popularity:")
    print("     - Item-CF tailors suggestions to what the user actually watched,")
    print("       delivering substantial gains in Recall@K and NDCG@K.")
    print("     - Cold-start fallback ensures robust coverage when history is missing.")
    print("-" * 78 + "\n")


if __name__ == "__main__":
    main()
