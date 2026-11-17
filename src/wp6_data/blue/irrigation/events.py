"""Watering events for the irrigation twin, read from the fertigation upload.

The farm's fertigation CSV is ingested as manual readings: one device per
treatment, stamped at the day's UTC midnight plus an ordinal second (several
runs on one day get 1 s, 2 s, ...). Here those rows become **one event per
day**:

- A treatment's runs that day are summed, giving its total dose per plant.
- The event dose is the median across treatments. In 2025 every treatment got
  the same dose, but the minimum and maximum are kept so a year with
  per-treatment differences stays visible instead of being averaged away.
- The CSV carries no clock time, and the doses don't register on the soil
  probes (research note §4.1), so the time is a configured local default.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

import pandas as pd

if TYPE_CHECKING:
    from wp6_data.blue.irrigation.config import IrrigationConfig, WateringParams
    from wp6_data.shared.twin import SensorDataProvider

#: Columns of the frame :func:`watering_events` returns, indexed by event time (UTC).
EVENT_COLUMNS = [
    "volume_l_per_plant", "volume_l_min", "volume_l_max", "duration_min", "treatments",
]

_ML_PER_L = 1000.0


def event_time(day: date, params: WateringParams) -> pd.Timestamp:
    """The assumed UTC time of a watering on *day*: the local default time."""
    local = datetime.combine(day, params.default_local_time, tzinfo=ZoneInfo(params.timezone))
    return pd.Timestamp(local).tz_convert(UTC)


def watering_events(readings: pd.DataFrame, params: WateringParams) -> pd.DataFrame:
    """Long fertigation readings (``device``/``sensor``/``time``/``value``) → one row per day."""
    empty = pd.DataFrame(columns=EVENT_COLUMNS, index=pd.DatetimeIndex([], tz=UTC))
    if readings.empty:
        return empty
    df = readings.assign(day=pd.to_datetime(readings["time"], utc=True).dt.date)

    def per_treatment(sensor: str, how: str) -> pd.DataFrame:
        rows = df[df["sensor"] == sensor]
        return rows.groupby(["day", "device"])["value"].agg(how)

    volume = per_treatment(params.volume_sensor, "sum")
    volume = volume[volume > 0]
    if volume.empty:
        return empty
    duration = per_treatment(params.duration_sensor, "sum")

    by_day = volume.groupby(level="day")
    events = pd.DataFrame({
        "volume_l_per_plant": by_day.median() / _ML_PER_L,
        "volume_l_min": by_day.min() / _ML_PER_L,
        "volume_l_max": by_day.max() / _ML_PER_L,
        "duration_min": duration.reindex(volume.index).groupby(level="day").median(),
        "treatments": by_day.size(),
    })
    events.index = pd.DatetimeIndex([event_time(d, params) for d in events.index])
    return events[EVENT_COLUMNS].sort_index()


async def load_events(
    provider: SensorDataProvider, config: IrrigationConfig, *, now: datetime,
) -> pd.DataFrame:
    """All watering events from ``history_start`` up to *now*."""
    params = config.watering
    start = datetime.combine(config.history_start, time.min, tzinfo=UTC)
    rows = await provider.fetch_data(
        sensor_tags=[params.volume_sensor, params.duration_sensor],
        start=start,
        end=now + timedelta(days=1),
    )
    return watering_events(rows, params)
