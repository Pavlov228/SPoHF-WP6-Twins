"""``wp6-blue-irrigation``: inspect and (later) run the blue irrigation twin.

Reads whatever database ``WP6_TSDB_URL`` points at, so the same command works
locally and against prod through a port-forward. It only reads.

    wp6-blue-irrigation fuse [--start YYYY-MM-DD] [--end YYYY-MM-DD] [--csv PATH]
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from wp6_data.blue.irrigation.config import load_config
from wp6_data.blue.irrigation.moisture import load_fused, quality_table


def _day(value: str) -> date:
    return date.fromisoformat(value)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="wp6-blue-irrigation", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    fuse = sub.add_parser("fuse", help="fused soil moisture and per-probe quality")
    fuse.add_argument("--start", type=_day, help="first day shown (default: 30 days ago)")
    fuse.add_argument("--end", type=_day, help="last day shown (default: today)")
    fuse.add_argument("--csv", type=Path, help="write the hourly fused stream here")
    return parser


async def _fuse(args: argparse.Namespace) -> int:
    from wp6_data.blue.provider import BlueSensorProvider
    from wp6_data.config import Settings
    from wp6_data.db.pool import close_pool, init_pool

    today = datetime.now(UTC).date()
    end_day = args.end or today
    start_day = args.start or end_day - timedelta(days=30)
    start = datetime.combine(start_day, time.min, tzinfo=UTC)
    end = datetime.combine(end_day, time.max, tzinfo=UTC)

    await init_pool(Settings().tsdb_url)
    try:
        result = await load_fused(BlueSensorProvider(), load_config(), now=datetime.now(UTC))
    finally:
        await close_pool()

    window = result.fused.loc[start:end]
    coverage = result.coverage.loc[start:end]
    print(f"Fused soil moisture {start_day} .. {end_day} ({len(window)} h)")
    print(f"  fused: median {window.median():.2f}, min {window.min():.2f}, max {window.max():.2f}")
    print(f"  coverage: median {coverage.median():.0f} probes, "
          f"{int((coverage == 0).sum())} h with none")
    print()
    print(quality_table(result, start, end).to_string())

    if args.csv:
        frame = pd.DataFrame({
            "fused": result.fused, "low": result.low, "high": result.high,
            "coverage": result.coverage,
        }).loc[start:end]
        frame.to_csv(args.csv, index_label="time")
        print(f"\nwrote {len(frame)} rows to {args.csv}")
    return 0


def main() -> None:
    load_dotenv()
    args = _parser().parse_args()
    if args.command == "fuse":
        sys.exit(asyncio.run(_fuse(args)))
