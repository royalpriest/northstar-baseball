from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class StatcastPull:
    start_date: date
    end_date: date
    output_dir: Path
    chunk_days: int = 7


def _date_chunks(start: date, end: date, chunk_days: int) -> list[tuple[date, date]]:
    if end < start:
        raise ValueError("end_date must be on or after start_date")
    if chunk_days < 1:
        raise ValueError("chunk_days must be >= 1")

    chunks: list[tuple[date, date]] = []
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + timedelta(days=chunk_days - 1), end)
        chunks.append((cursor, chunk_end))
        cursor = chunk_end + timedelta(days=1)
    return chunks


def download_statcast(pull: StatcastPull, overwrite: bool = False) -> list[Path]:
    """Download Statcast pitch-level data in small date chunks.

    pybaseball returns a pandas DataFrame with one row per pitch. Chunks are
    written to Parquet so a failed/restarted season pull does not repeat work.
    """
    from pybaseball import statcast

    pull.output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    for start, end in _date_chunks(pull.start_date, pull.end_date, pull.chunk_days):
        output = pull.output_dir / f"statcast_{start.isoformat()}_{end.isoformat()}.parquet"
        paths.append(output)
        if output.exists() and not overwrite:
            continue

        frame = statcast(
            start_dt=start.isoformat(),
            end_dt=end.isoformat(),
            verbose=False,
            parallel=True,
        )
        if frame.empty:
            continue

        frame.to_parquet(output, index=False)

    return paths


def read_statcast(
    paths: list[Path],
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Read cached Statcast chunks, optionally projecting only needed columns.

    Column projection happens inside the Parquet reader, which avoids loading the
    entire wide Statcast schema into memory when a downstream task only needs a
    subset of fields.
    """
    existing = [path for path in paths if path.exists()]
    if not existing:
        raise FileNotFoundError("No Statcast Parquet files were found.")

    frames = [
        pd.read_parquet(path, columns=columns)
        for path in existing
    ]
    return pd.concat(frames, ignore_index=True, copy=False)
