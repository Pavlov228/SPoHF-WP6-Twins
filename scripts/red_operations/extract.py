"""Pull every LetsGrow column from local wp6_red into one wide 5-minute frame.

Columns are `device_short:tag` (e.g. `climate:lg_temp_greenhouse`), index is
UTC. Cumulative running totals keep their raw (climbing) values; callers decide
how to difference them.

usage: python extract.py <out.parquet>   (reads WP6_RED_TSDB_URL, or .env in the cwd)

LetsGrow history reaches back to 2023-02; a local database synced with the
default start lacks it. Backfill first, e.g.
  WP6_ENDPOINTS=letsgrow-data WP6_SYNC_MODE=full WP6_SYNC_START=2023-02-01 uv run wp6-data-sync
"""

import os
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import psycopg


def dsn() -> str:
    if url := os.environ.get("WP6_RED_TSDB_URL"):
        return url
    env = Path(".env")
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("WP6_RED_TSDB_URL="):
                return line.split("=", 1)[1].strip().strip('"')
    raise SystemExit("set WP6_RED_TSDB_URL (or run from the repo root with a .env)")


def months(start: date, end: date):
    y, m = start.year, start.month
    while date(y, m, 1) < end:
        ny, nm = (y + 1, 1) if m == 12 else (y, m + 1)
        yield date(y, m, 1), date(ny, nm, 1)
        y, m = ny, nm


SQL = """
select time_bucket('5 minutes', time) t,
       replace(replace(device_name, 'letsgrow-', ''), '-b4', '') || ':' || sensor_tag k,
       avg(value) v
from readings
where source = 'letsgrow' and time >= %s and time < %s
group by 1, 2
"""


def main() -> None:
    out = sys.argv[1]
    parts = []
    with psycopg.connect(dsn()) as conn:
        end = (pd.Timestamp.today() + pd.offsets.MonthBegin(1)).date()
        for a, b in months(date(2023, 1, 1), end):
            rows = conn.execute(SQL, (a, b)).fetchall()
            if not rows:
                continue
            df = pd.DataFrame(rows, columns=["t", "k", "v"])
            wide = df.pivot_table(index="t", columns="k", values="v", aggfunc="mean")
            parts.append(wide.astype("float32"))
            print(a, len(wide), flush=True)
    frame = pd.concat(parts).sort_index()
    frame.index = pd.DatetimeIndex(frame.index).tz_convert("UTC")
    frame.to_parquet(out)
    print(frame.shape, frame.index.min(), frame.index.max())


if __name__ == "__main__":
    main()
