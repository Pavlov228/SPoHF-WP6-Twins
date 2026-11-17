"""Watering events: fertigation readings → one event per day at the local default time."""

from datetime import UTC, date, datetime, timedelta

import pandas as pd
import pytest
import yaml

from wp6_data.blue.irrigation.config import CONFIG_PATH, IrrigationConfig
from wp6_data.blue.irrigation.events import (
    EVENT_COLUMNS,
    event_time,
    load_events,
    watering_events,
)


def _config() -> IrrigationConfig:
    with CONFIG_PATH.open(encoding="utf-8") as f:
        return IrrigationConfig(**yaml.safe_load(f))


def _rows(day: date, treatments: dict[str, list[float]], duration: float) -> list[dict]:
    """Readings as the fertigation ingest stores them: per treatment, midnight
    UTC plus an ordinal second per run that day."""
    params = _config().watering
    rows = []
    for device, volumes in treatments.items():
        for i, volume in enumerate(volumes, start=1):
            t = datetime(day.year, day.month, day.day, tzinfo=UTC) + timedelta(seconds=i)
            rows.append({"device": device, "sensor": params.volume_sensor, "time": t,
                         "value": volume})
            rows.append({"device": device, "sensor": params.duration_sensor, "time": t,
                         "value": duration})
    return rows


def test_one_event_per_day_in_litres_at_the_local_default_time():
    params = _config().watering
    day = date(2025, 7, 2)
    rows = pd.DataFrame(_rows(day, {"Ca": [2400.0], "K": [2400.0], "Std": [2400.0]}, 18.0))

    events = watering_events(rows, params)

    assert list(events.columns) == EVENT_COLUMNS
    assert len(events) == 1
    event = events.iloc[0]
    assert event["volume_l_per_plant"] == pytest.approx(2.4)
    assert event["duration_min"] == 18.0
    assert event["treatments"] == 3
    assert events.index[0] == event_time(day, params)


def test_default_time_is_local_so_summer_and_winter_differ_in_utc():
    params = _config().watering
    summer = event_time(date(2025, 7, 2), params)
    winter = event_time(date(2025, 1, 15), params)

    local = params.default_local_time
    assert summer.tz_convert(params.timezone).time() == local
    assert winter.tz_convert(params.timezone).time() == local
    # Daylight saving: the same local time is an hour earlier in UTC in summer.
    assert winter.hour - summer.hour == 1


def test_runs_on_one_day_add_up_and_differing_treatments_stay_visible():
    params = _config().watering
    rows = pd.DataFrame(_rows(
        date(2026, 6, 1), {"Ca": [1000.0, 500.0], "K": [1200.0], "Std": [2000.0]}, 10.0,
    ))

    event = watering_events(rows, params).iloc[0]

    # Per treatment: Ca 1.5 L (two runs), K 1.2 L, Std 2.0 L → median 1.5 L.
    assert event["volume_l_per_plant"] == pytest.approx(1.5)
    assert event["volume_l_min"] == pytest.approx(1.2)
    assert event["volume_l_max"] == pytest.approx(2.0)


def test_zero_volume_rows_are_not_events():
    params = _config().watering
    rows = pd.DataFrame(_rows(date(2025, 5, 1), {"Ca": [0.0]}, 0.0))

    assert watering_events(rows, params).empty


@pytest.mark.asyncio
async def test_load_events_asks_for_both_sensors():
    config = _config()
    calls = {}

    class FakeProvider:
        async def fetch_data(self, **kwargs):
            calls.update(kwargs)
            return pd.DataFrame(_rows(date(2025, 4, 29), {"Ca": [1333.2]}, 10.0))

    events = await load_events(FakeProvider(), config, now=datetime(2025, 5, 1, tzinfo=UTC))

    assert set(calls["sensor_tags"]) == {
        config.watering.volume_sensor, config.watering.duration_sensor,
    }
    assert events.iloc[0]["volume_l_per_plant"] == pytest.approx(1.3332)


@pytest.mark.asyncio
async def test_irrigation_page_marks_waterings_with_their_dose(monkeypatch):
    import numpy as np
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient

    from wp6_data.blue.irrigation.moisture import fuse
    from wp6_data.blue.routes import irrigation
    from wp6_data.shared.auth import verify_session_user
    from wp6_data.shared.templates.config import configure_dashboard

    configure_dashboard("blue")
    config = _config()
    hours = pd.date_range("2025-06-01", periods=60 * 24, freq="h", tz=UTC)
    probes = pd.DataFrame({p: 20.0 + np.sin(np.arange(len(hours)) / 40) * (i + 1)
                           for i, p in enumerate(config.probes)}, index=hours)
    fused = fuse(probes, config)
    events = watering_events(
        pd.DataFrame(_rows(date(2025, 7, 2), {"Ca": [2400.0], "K": [2400.0]}, 18.0)),
        config.watering,
    )

    async def fake_fused():
        return fused

    async def fake_events():
        return events

    async def no_weather():
        raise ConnectionError("offline")

    monkeypatch.setattr(irrigation, "fused_moisture", fake_fused)
    monkeypatch.setattr(irrigation, "irrigation_events", fake_events)
    monkeypatch.setattr(irrigation, "irrigation_weather", no_weather)

    app = FastAPI()
    app.include_router(irrigation.router)
    app.dependency_overrides[verify_session_user] = lambda: "test-user"
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/irrigation?year=2025")

    assert response.status_code == 200
    assert "Watering (drip)" in response.text
    assert "time assumed" in response.text
