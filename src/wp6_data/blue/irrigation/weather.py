"""Weather inputs for the irrigation twin: rain, ET0, temperature, humidity.

Everything comes from Open-Meteo at the field (:mod:`wp6_data.blue.field_weather`),
except rain, which is **fused** with the on-site gauge:

- The gauge catches local showers the gridded reanalysis underestimates. In
  2025 it read 1.3–13 mm on days Open-Meteo had 0.1–6.4 mm.
- The gauge has also failed for whole months, reading nothing while it rained.

So the gauge is checked against Open-Meteo over a rolling window (research
note §5): it is used where it is plausible and replaced where it is not.
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


@dataclass(frozen=True)
class IrrigationWeather:
    """Hourly UTC frames with columns ``rain_mm``, ``rain_source``, ``et0_mm``,
    ``temperature``, ``humidity``."""

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


def _shape(hours: pd.DataFrame, rain: pd.Series, source: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({
        "rain_mm": rain,
        "rain_source": source,
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
    rows = await provider.fetch_data(
        sensor_tags=[config.rain.station_sensor],
        device_names=[config.rain.station_device],
        start=start, end=end,
        bucket=timedelta(hours=1), agg="sum",
        limit=int((end - start) / timedelta(hours=1)) + 1,
    )
    station = (
        pd.Series(rows["value"].to_numpy(), index=pd.to_datetime(rows["time"], utc=True))
        .groupby(level=0).sum()
        if not rows.empty else pd.Series(dtype=float)
    )
    station.index = station.index.floor("h") if len(station) else station.index

    actual = weather.actual
    rain, source = fuse_rain(station, actual[RAIN].fillna(0.0), config.rain)
    forecast = weather.forecast
    return IrrigationWeather(
        actual=_shape(actual, rain, source),
        forecast=_shape(
            forecast, forecast[RAIN].fillna(0.0),
            pd.Series(FORECAST, index=forecast.index),
        ),
    )
