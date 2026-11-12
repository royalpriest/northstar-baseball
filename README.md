# NorthStar Baseball

NorthStar is a baseball analytics project that turns a full season of MLB Statcast pitch data into
context-aware hitter diagnostics (Selection, Contact and Damage) and serves them through a FastAPI service
and a Next.js web app, with dedicated views for the Toronto Blue Jays and Canadian-born MLB players. You can
search any 2025 hitter, read their profile next to established measures such as wOBA, xwOBA and RE24, and
compare two to four hitters side by side.

**Live site:** https://northstar-baseball.vercel.app

## Key result

- The 2025 regular season was processed from **742,080 pitches into 190,987 plate appearances**, with each
  plate appearance's starting base/out state reconstructed from its first pitch, not its last.
- Scoring-probability models were trained only on information available before each plate appearance, on a
  chronological split: 131,045 plate appearances for training, 31,877 for validation, and a September holdout
  of 28,065.
- **Neither logistic regression nor XGBoost meaningfully beat a simple 24-state base/out baseline.** On the
  holdout, the context logistic model reached ROC AUC **0.711** against the baseline's **0.707**, but its
  log loss (0.5076 vs 0.5071) and Brier score (0.1665 vs 0.1663) were slightly worse. XGBoost reached
  0.709 / 0.5089 / 0.1669.
- That negative result is reported rather than hidden because it is the finding: base state and outs already
  carry almost all of the signal, and XGBoost's feature importance was dominated by them. Instead of adding
  model complexity, the project changed its target to player-level run value. The later hitter research
  followed the same rule, and no composite "NorthStar score" is published, because none survived
  out-of-time, reliability and incremental-prediction tests against established metrics.

The reasoning behind each model is in [`docs/methodology.md`](docs/methodology.md), and every experiment
with its command and results is in [`docs/experiments.md`](docs/experiments.md).

## Stack

| Layer | Tools |
| --- | --- |
| Data and research | Python 3.12+, pandas, Polars, DuckDB, Parquet, pybaseball (Baseball Savant / Statcast) |
| Modelling | scikit-learn, XGBoost, statsmodels, SHAP, matplotlib |
| API | FastAPI, SQLAlchemy, PostgreSQL (CSV fallback for local development) |
| Web | Next.js 16, React 19, TypeScript |
| Delivery | Docker, GitHub Actions CI, Vercel Services |
| Quality | pytest, Ruff, ESLint, Node test runner |

## Run it

### Install

```bash
python -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .\.venv\Scripts\Activate.ps1
make install                     # pip install -e ".[dev]"
```

Make targets:

| Target | What it does |
| --- | --- |
| `make install` | Install the package with development dependencies |
| `make test` | Run the Python test suite |
| `make lint` | Run Ruff over `src`, `tests` and `scripts` |
| `make baseline` | Download the 2025 regular season and build the baseline dataset |

### Run the API and web app

The API and web app run from the derived hitter-profile CSV already in `reports/tables/`, so no data
download is needed.

```bash
uvicorn northstar.api.app:app --reload        # API on http://127.0.0.1:8000, docs at /docs
```

```bash
cd web
npm install
npm run dev                                   # http://localhost:3000
```

To serve the API from PostgreSQL instead, start the database with Docker Compose, load the profiles, and
point the API at it:

```bash
docker compose up -d postgres
python scripts/load_profiles_to_db.py
export NORTHSTAR_DATABASE_URL="postgresql+psycopg://northstar:northstar@localhost:5432/northstar"
uvicorn northstar.api.app:app --reload
```

The API can also be built and run as a container:

```bash
docker build -f Dockerfile.api -t northstar-api .
docker run -p 8000:8000 -e NORTHSTAR_DATABASE_URL="$NORTHSTAR_DATABASE_URL" northstar-api
```

Settings are documented in [`.env.example`](.env.example) and [`web/.env.example`](web/.env.example).
Deployment is described in [`docs/production.md`](docs/production.md).

### Run the tests

```bash
make test          # 126 Python tests
make lint
```

```bash
cd web
npm test           # frontend tests (Node.js 22.6+)
npm run lint
npm run build
```

CI runs the Python checks, the web checks, and an API container build on every push to `main` and on pull
requests.

## Get the data

Raw Statcast data is not stored in this repository. It is downloaded from Baseball Savant through
pybaseball by the existing scripts and cached locally under `data/` (Git-ignored).

```bash
# Full 2025 regular season (same as `make baseline`)
python scripts/build_baseline.py --start 2025-03-18 --end 2025-09-28

# 2024 season for external validation, cached separately
python scripts/build_baseline.py --season 2024 --start 2024-03-20 --end 2024-09-30
```

The download runs in seven-day chunks saved as Parquet, so an interrupted run resumes where it stopped.
It writes:

```text
data/raw/statcast/                              # cached pitch-level chunks
data/processed/plate_appearances_2025.parquet
data/processed/run_expectancy_2025.csv
data/processed/scoring_probability_2025.csv
```

Every other research step reads that cache. The scripts in `scripts/` (`train_*`, `validate_*`, `audit_*`,
`build_*`) reproduce each experiment, and each one's commands and outputs are listed in
[`docs/experiments.md`](docs/experiments.md). Their derived summaries are committed in `reports/tables/` and
`reports/figures/`.

## Project layout

```text
northstar-baseball/
├── src/northstar/
│   ├── data/          # Statcast download and MLB player lookup
│   ├── features/      # plate-appearance reconstruction, context and pitch features
│   ├── metrics/       # run expectancy, player value, decision and execution metrics
│   ├── models/        # baselines, scoring probability, PA outcome, pitch models
│   ├── validation/    # reliability, cross-fitting, external-season and incremental tests
│   ├── profiles/      # hitter profile, comparison and Canada layer
│   ├── db/            # SQLAlchemy models, loader and SQL-backed store
│   └── api/           # FastAPI app and CSV-backed store
├── scripts/           # command-line entry points for every pipeline step
├── tests/             # pytest suite
├── reports/
│   ├── tables/        # derived CSV summaries (player-level, no raw pitch rows)
│   └── figures/       # charts produced by the scripts
├── web/               # Next.js app
├── docs/              # methodology, experiments, production and roadmap
├── data/              # local data cache (Git-ignored)
├── app.py             # Vercel FastAPI entry point
├── Dockerfile.api
├── compose.yaml       # local PostgreSQL
└── vercel.json
```

## Data sources

- MLB Baseball Savant / Statcast, retrieved through pybaseball
- Chadwick player register, via pybaseball, for player names
- MLB Stats API for player birthplaces (Canadian-born filter)

---

© Royalpriest Babalola. All rights reserved. Shared for viewing.
