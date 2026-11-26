"""Day-ahead electricity prices (energy-charts.info, CC BY 4.0) as an hourly series.

Red is at Neurath (Grevenbroich, Germany), so the bidding zone is DE-LU.
"""

import json
import os
import sys
import urllib.request
from pathlib import Path

import pandas as pd

DATA = Path(os.environ.get("RED_OPS_DATA", Path(__file__).resolve().parent / "data"))
ZONE = "DE-LU"
URL = "https://api.energy-charts.info/price?bzn={zone}&start={y}-01-01T00:00Z&end={y1}-01-01T00:00Z"


def _file(y: int) -> Path:
    return DATA / f"epex_{ZONE.lower().replace('-', '')}_{y}.json"


def fetch(years: list[int]) -> None:
    """Download the zone's day-ahead prices per year into DATA (public, CC BY 4.0)."""
    DATA.mkdir(parents=True, exist_ok=True)
    for y in years:
        with urllib.request.urlopen(URL.format(zone=ZONE, y=y, y1=y + 1), timeout=120) as r:
            _file(y).write_bytes(r.read())


def hourly_prices() -> pd.Series:
    parts = []
    for p in sorted(DATA.glob(_file(0).name.replace("_0.json", "_*.json"))):
        j = json.loads(p.read_text())
        s = pd.Series(
            j["price"], index=pd.to_datetime(j["unix_seconds"], unit="s", utc=True), dtype=float
        )
        parts.append(s)
    if not parts:
        raise FileNotFoundError(f"no {ZONE} prices in {DATA}; run prices.py <years> first")
    s = pd.concat(parts).sort_index()
    s = s[~s.index.duplicated()]
    # 15-minute products from Oct 2025: average to the hour
    return s.resample("1h").mean().dropna().rename("eur_mwh")


if __name__ == "__main__":
    fetch([int(y) for y in sys.argv[1:]])
