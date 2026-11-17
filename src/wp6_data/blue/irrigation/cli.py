"""``wp6-blue-irrigation``: inspect, backtest and calibrate the blue irrigation twin.

Reads whatever database ``WP6_TSDB_URL`` points at, so the same command works
locally and against prod through a port-forward. It only reads the database.
``calibrate`` writes ``blue/irrigation/model.yaml``, which is then committed
(ADR 0009).

    wp6-blue-irrigation fuse [--start YYYY-MM-DD] [--end YYYY-MM-DD] [--csv PATH]
    wp6-blue-irrigation backtest [--export-dir DIR] [--external CSV]
    wp6-blue-irrigation calibrate
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
    backtest = sub.add_parser("backtest", help="rolling-origin bake-off of the candidate models")
    backtest.add_argument(
        "--export-dir", type=Path,
        help="write daily.csv, origins.csv and forecasts.csv here (input for model E)",
    )
    backtest.add_argument(
        "--external", type=Path,
        help="forecasts from scripts/irrigation_foundation_models.py, scored alongside",
    )
    sub.add_parser("calibrate", help="fit the production model and write model.yaml")
    return parser


async def load_daily(now: datetime) -> pd.DataFrame:
    """The daily model frame from the database and Open-Meteo, up to *now* plus forecast."""
    from wp6_data.blue.irrigation.daily import daily_frame
    from wp6_data.blue.irrigation.events import load_events
    from wp6_data.blue.irrigation.weather import load_weather
    from wp6_data.blue.provider import BlueSensorProvider
    from wp6_data.config import Settings
    from wp6_data.db.pool import close_pool, init_pool

    settings = Settings()
    config = load_config()
    provider = BlueSensorProvider()
    await init_pool(settings.tsdb_url)
    try:
        fused = await load_fused(provider, config, now=now)
        weather = await load_weather(
            provider, config, latitude=settings.blue_weather_lat,
            longitude=settings.blue_weather_lon, today=now.date(),
        )
        events = await load_events(provider, config, now=now)
    finally:
        await close_pool()
    return daily_frame(fused, weather, events, config.watering.timezone)


def _backtest(args: argparse.Namespace) -> int:
    from wp6_data.blue.irrigation import backtest as bt

    frame = asyncio.run(load_daily(datetime.now(UTC)))
    forecasts, counterfactual = bt.run(frame)
    if args.external:
        forecasts = bt.merge_external(forecasts, pd.read_csv(args.external), counterfactual)
    if args.export_dir:
        args.export_dir.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.export_dir / "daily.csv")
        pd.DataFrame({"origin": sorted(forecasts["origin"].unique())}).to_csv(
            args.export_dir / "origins.csv", index=False,
        )
        forecasts.to_csv(args.export_dir / "forecasts.csv", index=False)
        print(f"exported to {args.export_dir}")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(bt.score(forecasts, counterfactual).round(3).sort_values(f"mae_h{bt.HORIZON}"))
    return 0


def _calibrate(args: argparse.Namespace) -> int:
    from wp6_data.blue.irrigation.model import MODEL_PATH, calibrate, write_model

    frame = asyncio.run(load_daily(datetime.now(UTC)))
    model = calibrate(frame, load_config())
    write_model(model)
    print(f"wrote {MODEL_PATH}")
    print(f"  fitted on {model.origins} origins, {model.data_first_day} .. {model.data_last_day}")
    for name, value in model.params.items():
        print(f"  {name:12} {value}")
    print("  backtest:", {k: v for k, v in model.backtest.items() if not k.startswith("mae_h")})
    return 0


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
    if args.command == "backtest":
        sys.exit(_backtest(args))
    if args.command == "calibrate":
        sys.exit(_calibrate(args))
