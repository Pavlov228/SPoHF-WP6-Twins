"""Watering advice and what-ifs on the 7-day outlook (issues 071, 072).

Constructed weather drives each case, and levels come from the shipped model
and config, never hard-coded, so the tests follow a recalibration.
"""

from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest
import yaml

from wp6_data.blue.irrigation.advice import (
    EVENING,
    MORNING,
    advise,
    levels,
    outlook,
    scenario,
    trajectory,
)
from wp6_data.blue.irrigation.config import CONFIG_PATH, IrrigationConfig
from wp6_data.blue.irrigation.model import MODEL_PATH, read_model

TODAY = date(2026, 7, 10)


def _config() -> IrrigationConfig:
    with CONFIG_PATH.open(encoding="utf-8") as f:
        return IrrigationConfig(**yaml.safe_load(f))


def _frame(s_now: float, rain: list[float] | None = None, et0: float = 5.0,
           last_reading: date = TODAY) -> pd.DataFrame:
    model = read_model(MODEL_PATH)
    days = pd.date_range(pd.Timestamp(last_reading) - pd.Timedelta(days=20),
                         pd.Timestamp(last_reading) + pd.Timedelta(days=model.horizon), freq="D")
    future = days > pd.Timestamp(last_reading)
    rain_mm = np.zeros(len(days))
    if rain is not None:
        rain_mm[future] = rain
    return pd.DataFrame({
        "s": np.where(future, np.nan, s_now), "rain_mm": rain_mm, "et0_mm": et0,
        "irrigation_l": 0.0, "coverage": 10.0,
    }, index=days)


def test_outlook_runs_from_the_last_reading_and_dries_without_water():
    model, config = read_model(MODEL_PATH), _config()
    o = outlook(_frame(0.6), model, TODAY)

    path = trajectory(o, model, config)

    assert len(path) == model.horizon
    assert path.index[0] == pd.Timestamp(TODAY) + pd.Timedelta(days=1)
    assert (np.diff(np.concatenate([[0.6], path["mean"]])) < 0).all()
    assert ((path["low"] <= path["mean"]) & (path["mean"] <= path["high"])).all()


def test_no_outlook_from_a_stale_reading_or_without_weather():
    model = read_model(MODEL_PATH)

    assert outlook(_frame(0.6, last_reading=date(2026, 7, 1)), model, TODAY) is None
    short = _frame(0.6).iloc[:-3]
    assert outlook(short, model, TODAY) is None


def test_a_dry_spell_gets_a_day_and_a_dose_that_refills_to_the_target():
    model, config = read_model(MODEL_PATH), _config()
    lv = levels(model, config)
    o = outlook(_frame(lv.trigger + 0.03), model, TODAY)

    advice = advise(o, model, config)

    path = trajectory(o, model, config)
    first_below = path.index[path["mean"] < lv.trigger][0]
    assert advice.needed
    assert advice.water_on == first_below.date()
    # The advised dose, given that morning, brings that day back to the target.
    dosed = trajectory(scenario(o, dose_l=advice.dose_l_per_plant, dose_day=advice.water_on),
                       model, config)
    assert dosed["mean"].loc[first_below] == pytest.approx(lv.target, abs=1e-6)


def test_rain_on_the_way_means_no_watering():
    model, config = read_model(MODEL_PATH), _config()
    lv = levels(model, config)
    o = outlook(_frame(lv.trigger + 0.03, rain=[15.0] * model.horizon, et0=2.0), model, TODAY)

    advice = advise(o, model, config)

    assert not advice.needed
    assert advice.dose_l_per_plant == 0.0


def test_already_below_the_trigger_means_water_at_the_first_chance():
    model, config = read_model(MODEL_PATH), _config()
    lv = levels(model, config)
    o = outlook(_frame(lv.trigger - 0.1), model, TODAY)

    advice = advise(o, model, config)

    assert advice.needed
    assert advice.water_on == TODAY + timedelta(days=1)


def test_delayed_rain_moves_and_less_rain_shrinks():
    model = read_model(MODEL_PATH)
    rain = [10.0] + [0.0] * (model.horizon - 1)
    o = outlook(_frame(0.5, rain=rain), model, TODAY)

    later = scenario(o, rain_delay_days=2)
    less = scenario(o, rain_factor=0.5)

    assert later.inputs.rain[1:, 0].tolist()[:3] == [0.0, 0.0, 10.0]
    assert less.inputs.rain[1, 0] == 5.0
    assert o.inputs.rain[1, 0] == 10.0  # the base outlook is untouched


def test_an_evening_dose_counts_on_the_next_day():
    model, config = read_model(MODEL_PATH), _config()
    o = outlook(_frame(0.5), model, TODAY)
    day = TODAY + timedelta(days=2)

    morning = trajectory(scenario(o, dose_l=10.0, dose_day=day, dose_time=MORNING), model, config)
    evening = trajectory(scenario(o, dose_l=10.0, dose_day=day, dose_time=EVENING), model, config)
    base = trajectory(o, model, config)

    assert morning["mean"].loc[pd.Timestamp(day)] > base["mean"].loc[pd.Timestamp(day)]
    assert evening["mean"].loc[pd.Timestamp(day)] == pytest.approx(
        base["mean"].loc[pd.Timestamp(day)])
    assert evening["mean"].iloc[-1] > base["mean"].iloc[-1]


@pytest.mark.asyncio
@pytest.mark.parametrize("as_of", ["", str(TODAY)])
async def test_simulator_runs_a_what_if_now_or_from_a_past_day(monkeypatch, as_of):
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient

    from wp6_data.blue.routes import irrigation_simulator as sim
    from wp6_data.shared.auth import verify_session_user
    from wp6_data.shared.templates.config import configure_dashboard

    configure_dashboard("blue")
    frame = _frame(0.55)
    hours = pd.date_range(pd.Timestamp(TODAY, tz="UTC") - pd.Timedelta(days=3),
                          pd.Timestamp(TODAY, tz="UTC") + pd.Timedelta(days=8), freq="h")
    fused = type("F", (), {"fused": pd.Series(0.55, index=hours)})()

    async def loaded():
        return fused

    monkeypatch.setattr(sim, "fused_moisture", loaded)
    monkeypatch.setattr(sim, "irrigation_weather", loaded)
    monkeypatch.setattr(sim, "irrigation_events", lambda: _no_events())
    monkeypatch.setattr(sim, "outlook_frame", lambda *_: frame)
    # "Now" is the frame's own last reading day, so both cases have an outlook.
    monkeypatch.setattr(sim, "datetime", _Frozen)

    app = FastAPI()
    app.include_router(sim.router)
    app.dependency_overrides[verify_session_user] = lambda: "test-user"
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get(
            f"/irrigation/simulator?as_of={as_of}&dose=2.4&dose_day={TODAY + timedelta(days=1)}"
            f"&rain_pct=50&rain_delay=1&dose_time={EVENING}",
        )

    assert response.status_code == 200
    assert "What-if" in response.text
    assert "No watering (forecast)" in response.text
    assert 'value="50"' in response.text  # the form keeps the chosen rain %


async def _no_events():
    return pd.DataFrame(columns=["volume_l_per_plant"], index=pd.DatetimeIndex([], tz="UTC"))


class _Frozen(datetime):
    @classmethod
    def now(cls, tz=None):
        return datetime.combine(TODAY + timedelta(days=1), datetime.min.time(), tzinfo=tz)


def test_simulator_table_dates_are_local_days():
    """The table summarises local calendar days. Using the UTC-shifted index put
    every date a day early (local midnight is 22:00 UTC the day before)."""
    from wp6_data.blue.routes.irrigation_simulator import _summary

    model, config = read_model(MODEL_PATH), _config()
    lv = levels(model, config)
    o = outlook(_frame(lv.trigger + 0.01), model, TODAY)
    path = trajectory(o, model, config)

    first_below = _summary(path["mean"], lv)[2]

    expected = path.index[path["mean"] < lv.trigger][0]
    assert first_below == f"{expected:%a %d %b}"
    assert expected > pd.Timestamp(TODAY)
