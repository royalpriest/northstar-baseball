from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from northstar.db.database import (
    create_engine_from_url,
    create_schema,
    session_factory,
)
from northstar.db.load import replace_hitter_profiles


DEFAULT_PROFILE = Path(
    "reports/tables/hitter_profile_with_canada_2025.csv"
)
DEFAULT_DATABASE_URL = (
    "postgresql+psycopg://northstar:northstar@localhost:5432/northstar"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load the canonical NorthStar hitter profile into SQL."
    )
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument(
        "--database-url",
        default=os.getenv(
            "NORTHSTAR_DATABASE_URL",
            DEFAULT_DATABASE_URL,
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.profile.exists():
        raise FileNotFoundError(
            f"Missing enriched hitter profile: {args.profile}. "
            "Run build_canada_layer.py first."
        )

    frame = pd.read_csv(args.profile)
    engine = create_engine_from_url(args.database_url)
    create_schema(engine)
    sessions = session_factory(engine)

    with sessions() as session:
        inserted = replace_hitter_profiles(
            session,
            frame,
            season=args.season,
        )

    print()
    print("NorthStar SQL profile load V1")
    print(f"Season:   {args.season}")
    print(f"Profiles: {inserted:,}")
    print(f"Source:   {args.profile}")
    print("Database load completed.")


if __name__ == "__main__":
    main()
