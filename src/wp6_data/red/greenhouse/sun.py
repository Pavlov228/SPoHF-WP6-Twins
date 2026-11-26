"""Where the sun stands, for drawing it over the greenhouse.

NOAA's low-precision solar position (declination and equation of time from
the fractional year): within ~0.5° — plenty for a sky and a light direction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def position(times: pd.DatetimeIndex, latitude: float, longitude: float) -> pd.DataFrame:
    """Solar ``elevation`` and ``azimuth`` (degrees, azimuth clockwise from north)."""
    utc = times.tz_convert("UTC")
    day_of_year = utc.dayofyear.to_numpy()
    hour = (utc.hour + utc.minute / 60 + utc.second / 3600).to_numpy()
    g = 2 * np.pi / 365 * (day_of_year - 1 + (hour - 12) / 24)
    eq_time = 229.18 * (
        0.000075
        + 0.001868 * np.cos(g)
        - 0.032077 * np.sin(g)
        - 0.014615 * np.cos(2 * g)
        - 0.040849 * np.sin(2 * g)
    )
    decl = (
        0.006918
        - 0.399912 * np.cos(g)
        + 0.070257 * np.sin(g)
        - 0.006758 * np.cos(2 * g)
        + 0.000907 * np.sin(2 * g)
        - 0.002697 * np.cos(3 * g)
        + 0.00148 * np.sin(3 * g)
    )
    solar_minutes = hour * 60 + eq_time + 4 * longitude
    hour_angle = np.radians(solar_minutes / 4 - 180)
    lat = np.radians(latitude)
    cos_zen = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(hour_angle)
    zen = np.arccos(np.clip(cos_zen, -1, 1))
    az = (
        np.degrees(
            np.arctan2(
                np.sin(hour_angle),
                np.cos(hour_angle) * np.sin(lat) - np.tan(decl) * np.cos(lat),
            )
        )
        + 180
    )
    return pd.DataFrame({"elevation": 90 - np.degrees(zen), "azimuth": az % 360}, index=times)


def rise_and_set(day_start: pd.Timestamp, latitude: float, longitude: float) -> dict:
    """Sunrise and sunset (local HH:MM) and the hours between, at 5-minute resolution."""
    times = pd.date_range(
        day_start, day_start + pd.Timedelta(days=1), freq="5min", inclusive="left"
    )
    up = position(times, latitude, longitude)["elevation"] > 0
    if not up.any():
        return {"sunrise": None, "sunset": None, "daylight_h": 0.0}
    rise, last = up[up].index[0], up[up].index[-1] + pd.Timedelta(minutes=5)
    return {
        "sunrise": rise.strftime("%H:%M"),
        "sunset": last.strftime("%H:%M"),
        "daylight_h": round(float(up.sum()) * 5 / 60, 2),
    }
