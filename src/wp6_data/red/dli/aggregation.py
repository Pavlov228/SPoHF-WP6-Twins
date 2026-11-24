"""Aggregation utilities for DLI data processing.

Generic functions (aggregate_to_daily, align_daily_dataframes, etc.) live in
shared.aggregation and are re-exported here for backwards compatibility.
"""

import pandas as pd

from wp6_data.red.dli.calculator import integrate_over_time
from wp6_data.red.dli.constants import (
    MIN_INDOOR_PAR_INTEGRAL,
    SECONDS_PER_HOUR,
)
from wp6_data.shared.aggregation import (
    add_day_of_year_features,
    aggregate_to_daily,
    align_daily_dataframes,
    encode_day_of_year,
)

__all__ = [
    "add_day_of_year_features",
    "aggregate_to_daily",
    "align_daily_dataframes",
    "align_weather_to_indoor_daily",
    "encode_day_of_year",
]


def _daily_integral(
    frame: pd.DataFrame, value_col: str, *, time_col: str, per_hour: bool = False
) -> pd.DataFrame:
    """Per-day time integral of ``value_col``, as a ``date`` + value frame.

    A daily total has to be an integral over the timestamps, never a sum of
    rows. A sum measures how often the sensor reported: red's s1000 fell from
    ~270 readings a day to 97 in June 2026 with the year's highest mean lux, so
    its summed daily total collapsed to a third on the brightest days — which is
    exactly the signal the old outdoor-lux stage was being asked to learn.
    """
    rows = [
        {
            "date": day,
            value_col: integrate_over_time(
                group[time_col].to_numpy(), group[value_col].to_numpy()
            ) / (SECONDS_PER_HOUR if per_hour else 1.0),
        }
        for day, group in frame.sort_values(time_col).groupby("date")
    ]
    return pd.DataFrame(rows, columns=["date", value_col])


def _weather_daily(weather_df: pd.DataFrame) -> pd.DataFrame:
    """OpenMeteo hourly rows as one row per day, under the model's feature names."""
    weather = weather_df.copy()
    weather["datetime"] = pd.to_datetime(weather["datetime"], utc=True)
    weather["date"] = weather["datetime"].dt.date

    agg_dict: dict[str, str] = {}
    # Global horizontal irradiance, under its own name. The legacy
    # solar_radiation -> direct_radiation_sum rename below is kept for callers
    # that still pass a beam column through the generic slot, but the model asks
    # for shortwave explicitly so it can never be handed a different quantity.
    if "shortwave_radiation" in weather.columns:
        agg_dict["shortwave_radiation"] = "sum"
    if "direct_radiation" in weather.columns:
        agg_dict["direct_radiation"] = "sum"
    elif "solar_radiation" in weather.columns:
        agg_dict["solar_radiation"] = "sum"
    if "diffuse_radiation" in weather.columns:
        agg_dict["diffuse_radiation"] = "sum"
    if "cloud_cover" in weather.columns:
        agg_dict["cloud_cover"] = "mean"
    if not agg_dict:
        raise ValueError("No radiation columns found in weather data")

    return (
        weather.groupby("date").agg(agg_dict).reset_index().rename(columns={
            "shortwave_radiation": "shortwave_sum",
            "solar_radiation": "direct_radiation_sum",
            "direct_radiation": "direct_radiation_sum",
            "diffuse_radiation": "diffuse_radiation_sum",
            "cloud_cover": "cloud_cover_avg",
        })
    )


def align_weather_to_indoor_daily(
    weather_df: pd.DataFrame,
    indoor_df: pd.DataFrame,
    min_par: float = MIN_INDOOR_PAR_INTEGRAL,
) -> pd.DataFrame:
    """Daily OpenMeteo weather next to the daily PAR integral it should predict.

    One join, weather straight to the above-lamp sensor. There used to be an
    outdoor-lux step in between (s1000), but at forecast time that step only
    passed a prediction through, and two chained fits compound their errors:
    the direct fit scores better out of sample than the chain did in sample
    (MODEL_PAPER §5.7).

    Args:
        weather_df: OpenMeteo hourly rows with ``datetime`` and radiation columns
        indoor_df: PAR readings with ``time``/``datetime`` and ``value``/``par``
        min_par: Minimum daily PAR integral (μmol/m²) for a day the sensor was
            really measuring

    Returns:
        One row per day both cover: the weather sums and ``par_integral``.
    """
    indoor = indoor_df.copy()
    time_col = "time" if "time" in indoor.columns else "datetime"
    indoor["datetime"] = pd.to_datetime(indoor[time_col], utc=True)
    if "value" in indoor.columns:
        indoor["par"] = indoor["value"]
    indoor["date"] = indoor["datetime"].dt.date

    indoor_daily = _daily_integral(indoor, "par", time_col="datetime")
    indoor_daily.columns = ["date", "par_integral"]

    merged = _weather_daily(weather_df).merge(indoor_daily, on="date", how="inner")
    if merged.empty:
        return merged
    return merged[merged["par_integral"] > min_par]
