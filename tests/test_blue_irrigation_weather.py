"""Irrigation weather: rain fused from the on-site gauge and Open-Meteo.

The two gauge behaviours seen in the data (research note §4.1, §5) drive these:
it catches a local shower that Open-Meteo underestimates, and it fails for
weeks at a time.
"""

from datetime import UTC, date

import numpy as np
import pandas as pd
import pytest
import yaml

from wp6_data.blue import field_weather as field_weather_mod
from wp6_data.blue.irrigation import weather
from wp6_data.blue.irrigation.config import CONFIG_PATH, IrrigationConfig
from wp6_data.blue.irrigation.weather import (
    ET0,
    FORECAST,
    OPEN_METEO,
    RAIN,
    STATION,
    TEMPERATURE,
    align_hourly_sums,
    fuse_rain,
    gauge_faulty_days,
    plausible_gauge_hours,
)

DAYS = 90


def _config() -> IrrigationConfig:
    with CONFIG_PATH.open(encoding="utf-8") as f:
        return IrrigationConfig(**yaml.safe_load(f))


def _hours() -> pd.DatetimeIndex:
    return pd.date_range("2025-06-01", periods=DAYS * 24, freq="h", tz=UTC)


def _reference() -> pd.Series:
    """Open-Meteo: a shower at 15:00 every third day, sized so every check
    window holds twice the rain needed to judge the gauge."""
    params = _config().rain
    showers_per_window = params.check_window_days // 3
    per_shower = 2 * params.min_reference_mm / showers_per_window
    hours = _hours()
    rain = np.where((hours.hour == 15) & (hours.day % 3 == 0), per_shower, 0.0)
    return pd.Series(rain, index=hours)


def test_a_healthy_gauge_is_used_including_a_local_shower():
    params = _config().rain
    reference = _reference()
    station = reference * 1.1
    shower = pd.Timestamp("2025-07-10 18:00", tz=UTC)
    station.loc[shower] = 9.0  # local shower Open-Meteo missed

    rain, source = fuse_rain(station, reference, params)

    assert (source == STATION).all()
    assert rain.loc[shower] == 9.0


def test_a_gauge_that_reads_nothing_for_weeks_is_replaced():
    params = _config().rain
    reference = _reference()
    station = reference.copy()
    outage = slice("2025-07-01", "2025-07-31")
    station.loc[outage] = 0.0

    rain, source = fuse_rain(station, reference, params)

    # Deep inside the outage the gauge is replaced, and the rain is Open-Meteo's.
    inner = slice("2025-07-08", "2025-07-24")
    assert (source.loc[inner] == OPEN_METEO).all()
    assert rain.loc[inner].sum() == pytest.approx(reference.loc[inner].sum())
    # Well away from it the gauge is trusted again.
    assert (source.loc["2025-06-01":"2025-06-15"] == STATION).all()
    assert (source.loc["2025-08-15":] == STATION).all()


def test_a_dry_spell_is_not_mistaken_for_a_broken_gauge():
    params = _config().rain
    hours = _hours()
    reference = pd.Series(0.0, index=hours)
    station = pd.Series(0.0, index=hours)

    _rain, source = fuse_rain(station, reference, params)

    assert (source == STATION).all()


def test_days_without_any_gauge_data_fall_back():
    params = _config().rain
    reference = _reference()
    station = reference.copy().drop(reference.loc["2025-06-20":"2025-06-21"].index)

    faulty = gauge_faulty_days(
        station.resample("D").sum(min_count=1), reference.resample("D").sum(), params,
    )

    assert faulty.loc["2025-06-20":"2025-06-21"].all()
    assert not faulty.loc["2025-06-01":"2025-06-10"].any()


@pytest.mark.asyncio
async def test_load_weather_fuses_actuals_and_marks_the_forecast(monkeypatch):
    config = _config().model_copy(update={"history_start": date(2025, 6, 1)})
    today = date(2025, 6, 10)

    class FakeOpenMeteo:
        def __init__(self, **_):
            pass

        async def get_hourly(self, variables, *, start=None, end=None,
                             forecast_days=0, past_days=0):
            if start is not None:
                first, last = pd.Timestamp(start, tz=UTC), pd.Timestamp(end, tz=UTC)
            else:
                first = pd.Timestamp(today, tz=UTC) - pd.Timedelta(days=past_days)
                last = pd.Timestamp(today, tz=UTC) + pd.Timedelta(days=forecast_days)
            times = pd.date_range(first, last + pd.Timedelta(hours=23), freq="h")
            return pd.DataFrame({"datetime": times, **{v: 0.5 for v in variables}})

        async def close(self):
            pass

    class FakeProvider:
        async def fetch_data(self, **kwargs):
            # Raw readings: negatives must be visible to be rejected.
            assert kwargs.get("bucket") is None
            readings = pd.date_range("2025-06-01", "2025-06-10 23:40", freq="20min", tz=UTC)
            return pd.DataFrame({"time": readings, "value": 0.2})

    field_weather_mod._cache.clear()
    monkeypatch.setattr(field_weather_mod, "OpenMeteoClient", FakeOpenMeteo)

    result = await weather.load_weather(
        FakeProvider(), config, latitude=51.4, longitude=6.1, today=today,
    )
    field_weather_mod._cache.clear()

    assert list(result.actual.columns) == [
        "rain_mm", "rain_source", "gauge_mm", "open_meteo_mm",
        "et0_mm", "temperature", "humidity",
    ]
    assert result.actual.index.max() < pd.Timestamp("2025-06-11", tz=UTC)
    assert (result.actual["rain_source"] == STATION).all()
    # Three 0.2 mm readings per hour.
    assert result.actual["rain_mm"].to_numpy() == pytest.approx(0.6)
    assert result.actual["gauge_mm"].to_numpy() == pytest.approx(0.6)
    assert (result.actual["open_meteo_mm"] == 0.5).all()
    assert (result.forecast["rain_source"] == FORECAST).all()
    assert result.forecast.index.min() == pd.Timestamp("2025-06-11", tz=UTC)


@pytest.mark.asyncio
async def test_irrigation_page_draws_rain_by_source(monkeypatch):
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient

    from wp6_data.blue.irrigation.moisture import fuse
    from wp6_data.blue.irrigation.weather import IrrigationWeather
    from wp6_data.blue.routes import irrigation
    from wp6_data.shared.auth import verify_session_user
    from wp6_data.shared.templates.config import configure_dashboard

    configure_dashboard("blue")
    hours = _hours()
    probes = pd.DataFrame({p: 20.0 + np.sin(np.arange(len(hours)) / 50) * (i + 1)
                           for i, p in enumerate(_config().probes)}, index=hours)
    config = _config()
    fused = fuse(probes, config)

    actual = pd.DataFrame({
        "rain_mm": _reference(), "rain_source": STATION,
        "gauge_mm": _reference(), "open_meteo_mm": _reference(), "et0_mm": 0.2,
        "temperature": 15.0, "humidity": 80.0,
    }, index=hours)
    actual.loc["2025-07-01":"2025-07-20", "rain_source"] = OPEN_METEO

    async def fake_fused():
        return fused

    async def fake_weather():
        return IrrigationWeather(actual=actual, forecast=actual.iloc[:0])

    monkeypatch.setattr(irrigation, "fused_moisture", fake_fused)
    monkeypatch.setattr(irrigation, "irrigation_weather", fake_weather)

    app = FastAPI()
    app.include_router(irrigation.router)
    app.dependency_overrides[verify_session_user] = lambda: "test-user"
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/irrigation?year=2025")

    assert response.status_code == 200
    # Both sources are always drawn, and the gauge is split by whether it was used.
    for key in ("gauge_used", "gauge_unused", "open_meteo"):
        assert irrigation.RAIN_TRACES[key][0] in response.text
    assert "Reference ET0" in response.text


def test_impossible_gauge_hours_are_dropped_not_clipped():
    params = _config().rain
    readings = pd.DataFrame({
        "time": pd.to_datetime([
            "2024-03-26 08:00", "2024-03-26 08:20",          # a negative reading
            "2024-03-26 09:00", "2024-03-26 09:20",          # above the hourly maximum
            "2024-03-26 10:00", "2024-03-26 10:20",          # an ordinary shower
        ], utc=True),
        "value": [-0.02, 1.0, params.max_mm_per_hour, 1.0, 1.5, 0.5],
    })

    hourly = plausible_gauge_hours(readings, params)

    assert np.isnan(hourly.loc[pd.Timestamp("2024-03-26 08:00", tz=UTC)])
    assert np.isnan(hourly.loc[pd.Timestamp("2024-03-26 09:00", tz=UTC)])
    assert hourly.loc[pd.Timestamp("2024-03-26 10:00", tz=UTC)] == 2.0


def test_open_meteo_sums_move_to_the_start_of_their_hour():
    """Open-Meteo stamps a sum at the end of its hour, and the gauge stamps the
    start. Instants such as temperature stay where they are."""
    hours = pd.date_range("2025-06-01", periods=4, freq="h", tz=UTC)
    frame = pd.DataFrame({
        RAIN: [0.0, 3.0, 0.0, 0.0], ET0: [0.1, 0.2, 0.3, 0.4],
        TEMPERATURE: [10.0, 11.0, 12.0, 13.0],
    }, index=hours)

    aligned = align_hourly_sums(frame)

    assert aligned[RAIN].tolist()[:3] == [3.0, 0.0, 0.0]
    assert aligned[ET0].tolist()[:3] == [0.2, 0.3, 0.4]
    assert np.isnan(aligned[RAIN].iloc[-1])
    assert aligned[TEMPERATURE].tolist() == frame[TEMPERATURE].tolist()
