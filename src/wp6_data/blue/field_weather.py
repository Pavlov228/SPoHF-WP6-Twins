"""Modelled hourly weather at the blue field: past and forecast, one fetch.

Blue's weather consumers (GDD, irrigation) each need some hourly Open-Meteo
variables from a start date through today, plus the forecast. The ERA5
archive lags about five days, so the forecast endpoint's ``past_days`` bridges
the recent tail. The archive wins wherever it has a value: reanalysis beats a
forecast. The bridge fills only the hours the archive lacks, which includes
hours the archive *returns* but as null.

Blue-local on purpose: the location and the stitching policy are the blue
field's, and ``shared/weather.py`` stays the twin-agnostic client underneath.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd
from cachetools import TTLCache

from wp6_data.shared.weather import OpenMeteoClient

#: Days of forecast past today. Open-Meteo serves at most 16.
FORECAST_DAYS = 14
#: Forecast days before today, covering the ERA5 lag (~5 d) plus a margin.
BRIDGE_PAST_DAYS = 10

# Per (location, start, variables). TTL ~1h: the archive is immutable; only the
# recent tail and the forecast move.
_cache: TTLCache = TTLCache(maxsize=16, ttl=3600)


@dataclass(frozen=True)
class FieldWeather:
    """Hourly weather, indexed by UTC hour, one column per requested variable."""

    #: From the start date through the end of *today*.
    actual: pd.DataFrame
    #: Hours strictly after *today*.
    forecast: pd.DataFrame


def _indexed(df: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=variables, index=pd.DatetimeIndex([], tz="UTC"))
    return df.set_index("datetime").reindex(columns=variables).sort_index()


async def field_weather(
    variables: list[str],
    *,
    latitude: float,
    longitude: float,
    start: date,
    today: date,
) -> FieldWeather:
    """Hourly *variables* at the field from *start*, split at the end of *today*."""
    key = (round(latitude, 4), round(longitude, 4), start, today, tuple(variables))
    cached = _cache.get(key)
    if cached is not None:
        return cached

    client = OpenMeteoClient(latitude=latitude, longitude=longitude)
    try:
        archive = await client.get_hourly(variables, start=start, end=today)
        recent = await client.get_hourly(
            variables, forecast_days=FORECAST_DAYS, past_days=BRIDGE_PAST_DAYS,
        )
    finally:
        await client.close()

    archive_hours = _indexed(archive, variables)
    recent_hours = _indexed(recent, variables)
    # Cell by cell, so a null archive hour is filled while a real one is kept.
    stitched = archive_hours.combine_first(recent_hours)
    stitched = stitched.loc[pd.Timestamp(start, tz="UTC"):]

    cutoff = pd.Timestamp(today + timedelta(days=1), tz="UTC")
    result = FieldWeather(
        actual=stitched.loc[stitched.index < cutoff],
        forecast=stitched.loc[stitched.index >= cutoff],
    )
    _cache[key] = result
    return result
