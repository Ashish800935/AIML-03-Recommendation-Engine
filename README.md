# AIML-03 — Large-Scale Personalized Recommendation Engine

> **Internship Project** | Built with TensorFlow Recommenders · FAISS · Apache Spark · Redis · Airflow · Triton Inference Server

---

## 🎯 Project Purpose

This project builds a production-grade, two-stage personalized recommendation system modelled after architectures used at Netflix and Amazon.  
The goal is to go from raw interaction data all the way to a served, continuously retrained recommendation API.

---

## 🏗️ Two-Stage Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│              Stage 1 — Candidate Retrieval           │
│                                                     │
│   User interactions  ──►  Two-Tower Model           │
│   (ratings, tags)         (user embed + item embed) │
│                                ▼                    │
│                      FAISS Index (ANN search)       │
│                    → Top-K candidate items          │
└──────────────────────────┬──────────────────────────┘
                           │  Top-K candidates
┌──────────────────────────▼──────────────────────────┐
│              Stage 2 — Ranking                      │
│                                                     │
│   Candidates + rich features  ──►  Ranking Model    │
│   (genome tags, metadata)         (DNN / LambdaRank)│
│                                ▼                    │
│                     Final ranked list               │
└──────────────────────────┬──────────────────────────┘
                           │
          ┌────────────────▼────────────────┐
          │    Supporting Infrastructure     │
          │  Redis Cache  |  Cold-Start       │
          │  Airflow DAG  |  Triton Serving   │
          │  Spark Feature Pipeline           │
          └──────────────────────────────────┘
```

**Evaluation:** Offline — Recall@K, NDCG@K.  Online — A/B test design.

---

## 📦 Dataset — MovieLens 25M

| File | Description | Size |
|------|-------------|------|
| `ratings.csv` | 25M user-movie ratings | ~647 MB |
| `movies.csv` | Movie titles and genres | ~3 MB |
| `tags.csv` | User-applied tags | ~37 MB |
| `links.csv` | IMDB / TMDB IDs | ~1.3 MB |
| `genome-scores.csv` | Tag relevance scores per movie | ~415 MB |
| `genome-tags.csv` | Tag vocabulary | ~18 KB |

**Download:** [https://grouplens.org/datasets/movielens/25m/](https://grouplens.org/datasets/movielens/25m/)

After downloading, extract all files to `data/raw/`.  
**⚠️ Never commit these files to Git.** They are covered by `.gitignore`.

---

## 📁 Project Structure

```
AIML-03-Recommendation-Engine/
│
├── data/
│   ├── raw/            ← MovieLens CSV files (Git-ignored)
│   └── processed/      ← Spark/Pandas outputs (Git-ignored)
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py     ← Phase 1: dataset verification utilities
│   ├── quality_checks.py  ← Phase 2: chunked data quality validation
│   └── eda.py             ← Phase 2: exploratory data analysis & plotting
│
├── scripts/
│   ├── verify_dataset.py  ← CLI: Phase 1 dataset verification
│   └── run_phase2.py      ← CLI: Phase 2 quality checks & EDA runner
│
├── tests/
│   ├── test_data_loader.py ← Phase 1 unit tests (25 tests)
│   └── test_phase2.py      ← Phase 2 unit tests (27 tests)
│
├── notebooks/          ← Jupyter exploratory notebooks
├── models/             ← Trained model artifacts (Git-ignored)
├── reports/            ← Generated EDA charts & summary reports (Git-ignored)
├── docs/               ← Architecture docs
├── app/                ← Serving API (future phases)
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🐍 Python Environment Setup (Windows PowerShell)

```powershell
# 1. Navigate to the project root
cd "C:\Users\ASUS\OneDrive\Documents\Desktop\AIML-03-Recommendation-Engine"

# 2. Create a virtual environment
python -m venv .venv

# 3. Activate it
.venv\Scripts\Activate.ps1

# 4. Install dependencies
pip install -r requirements.txt
```

> **Note:** If PowerShell blocks script execution, run:  
> `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

---

## ▶️ Running Dataset Verification (Phase 1)

```powershell
# From the project root, with .venv activated:
python scripts/verify_dataset.py

# Show 10 sample rows instead of the default 5:
python scripts/verify_dataset.py --rows 10

# Point to a custom data directory:
python scripts/verify_dataset.py --data-dir data/raw
```

The script will:
1. List all expected files and their sizes.
2. Validate column schemas (reads only the header — safe for 650 MB files).
3. Display sample rows for `ratings.csv` and `movies.csv`.

---

## 📊 Running Data Quality & EDA (Phase 2)

```powershell
# Run both quality checks and full EDA:
python scripts/run_phase2.py

# Run only quality checks:
python scripts/run_phase2.py --quality-only

# Run only EDA (charts and distributions):
python scripts/run_phase2.py --eda-only
```

Generated outputs in `reports/`:
- `01_rating_distribution.png` — Distribution across half-star rating buckets (mean: 3.53)
- `02_ratings_per_user.png` — Log-scale distribution of interactions per user
- `03_ratings_per_movie.png` — Log-scale distribution of interactions per movie
- `04_top_rated_movies.png` — Top 20 most-rated movies (Forrest Gump, Shawshank Redemption, etc.)
- `05_rating_trend_over_time.png` — Monthly average rating and yearly interaction volume
- `06_genre_distribution.png` — Frequency of each movie genre
- `split_boundaries.txt` — Chronological cutoffs for 80/10/10 temporal split
- `phase2_summary.txt` — Full textual statistics summary

---

## 🧪 Running the Tests

```powershell
# From the project root, with .venv activated:
pytest tests/ -v
```

All 52 tests run against lightweight synthetic data — full 25M CSVs are **not required** for testing.

---

## 🔒 Dataset Files and Git

Dataset files and generated report artifacts are excluded from version control via `.gitignore`:

```
data/raw/
data/processed/
models/
reports/*.png
reports/*.txt
```

If you accidentally staged a dataset file, use:
```powershell
git rm --cached data/raw/ratings.csv
```
Then commit the removal before pushing.

---

## ✅ What Phase 1 & 2 Verify

| Component | Status | Findings / Metrics |
|-----------|--------|---------------------|
| Dataset files present | ✅ | All 6 files present in `data/raw/` |
| Column schemas | ✅ | Exactly match MovieLens 25M specification |
| Memory safety | ✅ | Chunked iteration (500k rows/chunk) avoids OOM |
| Data quality checks | ✅ | Zero duplicate (user, movie) pairs; zero out-of-range ratings |
| Missing values | ✅ | `ratings.csv` 0 missing; `tags.csv` 16 missing tags; `links.csv` 107 missing tmdbIds |
| Unique entities | ✅ | 162,541 users, 59,047 movies, 25,000,095 total ratings |
| Temporal split | ✅ | Train <= 2016-06-25 (80%), Val 2016-06-25–2018-01-04 (10%), Test > 2018-01-04 (10%) |
| Test suite | ✅ | 52 / 52 unit tests passing |

## 🚧 What Has NOT Been Implemented Yet

- Two-tower model (TensorFlow Recommenders)
- FAISS index construction
- Ranking model
- Apache Spark feature pipeline
- Redis caching / cold-start handling
- Triton Inference Server deployment
- Airflow retraining DAG
- Online A/B testing
- Online serving API

---

## 📋 Phase Roadmap

| Phase | Description | Status |
|-------|-------------|--------|
| **Phase 1** | Project setup & dataset verification | ✅ Complete |
| **Phase 2** | Data quality checks & Exploratory Data Analysis (EDA) | ✅ Complete |
| Phase 3 | Two-tower candidate retrieval (TFRS + FAISS) | 🔜 Next |
| Phase 4 | Ranking model (features & DNN reranking) | 🔜 |
| Phase 5 | Redis caching & cold-start handling | 🔜 |
| Phase 6 | Triton model serving & Airflow retraining | 🔜 |
| Phase 7 | Offline evaluation (Recall@K, NDCG@K) & Online A/B test design | 🔜 |
