"""Weather inputs for the irrigation twin: rain, ET0, temperature, humidity.

Everything comes from Open-Meteo at the field (:mod:`wp6_data.blue.field_weather`),
except rain, which is **fused** with the on-site gauge:

- The gauge catches local showers the gridded reanalysis underestimates. In
  2025 it read 1.3–13 mm on days Open-Meteo had 0.1–6.4 mm.
- The gauge has also failed for whole months, reading nothing while it rained.

So the gauge is checked against Open-Meteo over a rolling window (research
note §5): it is used where it is plausible and replaced where it is not.
Impossible gauge values (any negative reading, or an hour above
``max_mm_per_hour``) are dropped first.

Open-Meteo labels its hourly sums (rain, ET0) at the **end** of the hour they
cover, while the gauge's hourly buckets are labelled at their start. Open-Meteo's
sums are therefore moved back an hour to line up. The cross-correlation with the
gauge peaks at exactly that offset.
Each hour records which source supplied its rain. The forecast is Open-Meteo
only, because the gauge cannot see the future.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from typing import TYPE_CHECKING

import pandas as pd

from wp6_data.blue.field_weather import field_weather

if TYPE_CHECKING:
    from wp6_data.blue.irrigation.config import IrrigationConfig, RainParams
    from wp6_data.shared.twin import SensorDataProvider

RAIN = "precipitation"
ET0 = "et0_fao_evapotranspiration"
TEMPERATURE = "temperature_2m"
HUMIDITY = "relative_humidity_2m"
VARIABLES = [RAIN, ET0, TEMPERATURE, HUMIDITY]

#: Values of the hourly ``rain_source`` column.
STATION = "station"
OPEN_METEO = "open-meteo"
FORECAST = "forecast"

#: Upper bound on raw gauge rows fetched; it reports about every 20 minutes,
#: so this covers decades.
_MAX_GAUGE_READINGS = 2_000_000


@dataclass(frozen=True)
class IrrigationWeather:
    """Hourly UTC frames, labelled at the start of each hour.

    Columns: ``rain_mm`` and ``rain_source`` (what the model uses),
    ``gauge_mm`` (the gauge's own reading, NaN where it has none) and
    ``open_meteo_mm`` (Open-Meteo's), so both stay visible side by side,
    plus ``et0_mm``, ``temperature`` and ``humidity``.
    """

    actual: pd.DataFrame
    forecast: pd.DataFrame


def gauge_faulty_days(
    station_daily: pd.Series, reference_daily: pd.Series, params: RainParams,
) -> pd.Series:
    """Per day: True where the gauge is not to be trusted.

    A day is faulty when the window around it had real rain by the reference
    (at least ``min_reference_mm``), but the gauge caught less than
    ``min_ratio`` of it. A day with no gauge data at all is faulty too.
    The window is centred, so the start of an outage is caught as well as
    its end.
    """
    days = reference_daily.index.union(station_daily.index)
    station = station_daily.reindex(days)
    reference = reference_daily.reindex(days).fillna(0.0)
    window = params.check_window_days
    station_sum = station.fillna(0.0).rolling(window, center=True, min_periods=1).sum()
    reference_sum = reference.rolling(window, center=True, min_periods=1).sum()
    under_reads = (reference_sum >= params.min_reference_mm) & (
        station_sum < params.min_ratio * reference_sum
    )
    return under_reads | station.isna()


def fuse_rain(
    station_hourly: pd.Series, reference_hourly: pd.Series, params: RainParams,
) -> tuple[pd.Series, pd.Series]:
    """Hourly rain and its source: the gauge where trusted, the reference elsewhere."""
    faulty = gauge_faulty_days(
        station_hourly.resample("D").sum(min_count=1),
        reference_hourly.resample("D").sum(),
        params,
    )
    hours = reference_hourly.index
    faulty_hour = faulty.reindex(hours.floor("D"), fill_value=True).to_numpy()
    station = station_hourly.reindex(hours)
    use_station = ~faulty_hour & station.notna().to_numpy()
    rain = reference_hourly.where(~use_station, station)
    source = pd.Series(OPEN_METEO, index=hours).where(~use_station, STATION)
    return rain, source


#: Open-Meteo variables that are sums over the preceding hour, not instants.
_HOURLY_SUMS = (RAIN, ET0)


def align_hourly_sums(hours: pd.DataFrame) -> pd.DataFrame:
    """Move Open-Meteo's end-of-hour sums back to the start of their hour."""
    aligned = hours.copy()
    for column in _HOURLY_SUMS:
        if column in aligned:
            aligned[column] = aligned[column].shift(-1, freq="h").reindex(aligned.index)
    return aligned


def plausible_gauge_hours(readings: pd.DataFrame, params: RainParams) -> pd.Series:
    """Hourly gauge rain from raw readings, without impossible values.

    An hour containing any negative reading, or summing above
    ``max_mm_per_hour``, is dropped (NaN) rather than clipped. Such an hour
    is a broken gauge, not a big shower.
    """
    if readings.empty:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC"))
    values = pd.Series(
        readings["value"].to_numpy(dtype=float),
        index=pd.to_datetime(readings["time"], utc=True).dt.floor("h"),
    )
    hourly = values.groupby(level=0).agg(["sum", "min"])
    ok = (hourly["min"] >= 0) & (hourly["sum"] <= params.max_mm_per_hour)
    return hourly["sum"].where(ok)


def _shape(
    hours: pd.DataFrame, rain: pd.Series, source: pd.Series, gauge: pd.Series,
) -> pd.DataFrame:
    return pd.DataFrame({
        "rain_mm": rain,
        "rain_source": source,
        "gauge_mm": gauge.reindex(hours.index),
        "open_meteo_mm": hours[RAIN],
        "et0_mm": hours[ET0],
        "temperature": hours[TEMPERATURE],
        "humidity": hours[HUMIDITY],
    }, index=hours.index)


async def load_weather(
    provider: SensorDataProvider,
    config: IrrigationConfig,
    *,
    latitude: float,
    longitude: float,
    today: date,
) -> IrrigationWeather:
    """Hourly irrigation weather from ``history_start`` through the forecast."""
    weather = await field_weather(
        VARIABLES, latitude=latitude, longitude=longitude,
        start=config.history_start, today=today,
    )
    start = datetime.combine(config.history_start, time.min, tzinfo=UTC)
    end = datetime.combine(today + timedelta(days=1), time.min, tzinfo=UTC)
    # Raw readings, not hourly sums: a negative reading must be seen to be
    # rejected, and a database-side sum would net it off against real rain.
    rows = await provider.fetch_data(
        sensor_tags=[config.rain.station_sensor],
        device_names=[config.rain.station_device],
        start=start, end=end,
        limit=_MAX_GAUGE_READINGS,
    )
    station = plausible_gauge_hours(rows, config.rain)

    # Align across the actual/forecast seam: the first forecast hour's sum
    # belongs to the last actual hour.
    both = align_hourly_sums(pd.concat([weather.actual, weather.forecast]))
    cutoff = len(weather.actual)
    weather = type(weather)(actual=both.iloc[:cutoff], forecast=both.iloc[cutoff:])

    actual = weather.actual
    rain, source = fuse_rain(station, actual[RAIN].fillna(0.0), config.rain)
    forecast = weather.forecast
    no_gauge = pd.Series(dtype=float)
    return IrrigationWeather(
        actual=_shape(actual, rain, source, station),
        forecast=_shape(
            forecast, forecast[RAIN].fillna(0.0),
            pd.Series(FORECAST, index=forecast.index), no_gauge,
        ),
    )
