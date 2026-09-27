from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from northstar.data.statcast import StatcastPull, download_statcast, read_statcast
from northstar.features.plate_appearances import build_plate_appearances
from northstar.metrics.run_expectancy import add_re24, build_run_expectancy_table
from northstar.metrics.scoring_probability import build_scoring_probability_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build NorthStar's Phase 1 run-expectancy baseline.")
    parser.add_argument("--start", default="2025-03-18", help="YYYY-MM-DD")
    parser.add_argument("--end", default="2025-09-28", help="YYYY-MM-DD")
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help=(
            "Statcast cache directory. Defaults to data/raw/statcast for 2025 "
            "and data/raw/statcast_<season> for other seasons."
        ),
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed"),
    )
    parser.add_argument("--chunk-days", type=int, default=7)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.season < 2008:
        raise ValueError("--season must be >= 2008 for Statcast-era workflows")

    raw_dir = args.raw_dir
    if raw_dir is None:
        raw_dir = (
            Path("data/raw/statcast")
            if args.season == 2025
            else Path(f"data/raw/statcast_{args.season}")
        )

    processed_dir = args.processed_dir
    processed_dir.mkdir(parents=True, exist_ok=True)

    pull = StatcastPull(
        start_date=date.fromisoformat(args.start),
        end_date=date.fromisoformat(args.end),
        output_dir=raw_dir,
        chunk_days=args.chunk_days,
    )

    paths = download_statcast(pull, overwrite=args.overwrite)
    pitches = read_statcast(paths)
    plate_appearances = build_plate_appearances(pitches)

    expectancy = build_run_expectancy_table(plate_appearances, min_samples=25)
    plate_appearances = add_re24(plate_appearances, expectancy)
    score_probability = build_scoring_probability_table(plate_appearances)

    plate_appearances.to_parquet(
        processed_dir / f"plate_appearances_{args.season}.parquet",
        index=False,
    )
    expectancy.to_csv(
        processed_dir / f"run_expectancy_{args.season}.csv",
        index=False,
    )
    score_probability.to_csv(
        processed_dir / f"scoring_probability_{args.season}.csv",
        index=False,
    )

    print(f"Season: {args.season}")
    print(f"Raw cache: {raw_dir}")
    print(f"Pitches: {len(pitches):,}")
    print(f"Plate appearances: {len(plate_appearances):,}")
    print("\nRun expectancy table:")
    print(expectancy.to_string(index=False))


if __name__ == "__main__":
    main()
