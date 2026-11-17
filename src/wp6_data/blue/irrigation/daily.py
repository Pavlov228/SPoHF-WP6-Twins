"""The daily frame the irrigation models work on.

Models step one day at a time (the FAO-56 resolution), on:

- ``s``: daily mean fused soil moisture (0 dry – 1 wet). It is NaN on a day
  that is too thinly covered: fewer than ``min_probes`` probes, or fewer than
  ``min_hours`` usable hours. A median over a changing handful of probes jumps
  without any water arriving (research note §4.6).
- ``rain_mm``, ``et0_mm``: daily sums of the fused rain and Open-Meteo ET0.
- ``irrigation_l``: drip dose that day, litres per plant.
- ``coverage``: mean probes used per hour, for weighting and inspection.

The day is the field's local calendar day, so an evening watering and the night
after it fall on the dates the farm would name.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from wp6_data.blue.irrigation.moisture import FusedMoisture
    from wp6_data.blue.irrigation.weather import IrrigationWeather

#: Columns of :func:`daily_frame`.
DAILY_COLUMNS = ["s", "rain_mm", "et0_mm", "irrigation_l", "coverage"]

#: A day's soil moisture needs at least this many probes per hour on average...
MIN_PROBES = 3
#: ...and at least this many usable hours.
MIN_HOURS = 18


def _local_day(index: pd.DatetimeIndex, timezone: str) -> pd.Index:
    return pd.Index(index.tz_convert(timezone).date, name="day")


def daily_frame(
    fused: FusedMoisture,
    weather: IrrigationWeather,
    events: pd.DataFrame | None,
    timezone: str,
) -> pd.DataFrame:
    """One row per local day with soil moisture, weather and watering."""
    used = fused.fused.where(fused.coverage >= MIN_PROBES)
    by_day = _local_day(used.index, timezone)
    s_mean = used.groupby(by_day).mean()
    s_hours = used.notna().groupby(by_day).sum()
    coverage = fused.coverage.groupby(by_day).mean()
    s = s_mean.where(s_hours >= MIN_HOURS)

    hours = pd.concat([weather.actual, weather.forecast])
    weather_day = _local_day(hours.index, timezone)
    rain = hours["rain_mm"].groupby(weather_day).sum(min_count=1)
    et0 = hours["et0_mm"].groupby(weather_day).sum(min_count=1)

    if events is None or events.empty:
        irrigation = pd.Series(dtype=float)
    else:
        irrigation = events["volume_l_per_plant"].groupby(
            _local_day(events.index, timezone),
        ).sum()

    days = s.index.union(rain.index)
    frame = pd.DataFrame({
        "s": s.reindex(days),
        "rain_mm": rain.reindex(days),
        "et0_mm": et0.reindex(days),
        "irrigation_l": irrigation.reindex(days).fillna(0.0),
        "coverage": coverage.reindex(days),
    })
    frame.index = pd.DatetimeIndex(frame.index, name="day")
    return frame[DAILY_COLUMNS]
