"""The shipped forecast model (ADR 0009): the committed file must match the code,
and forecasts must start from a fresh reading."""

from datetime import date

import numpy as np
import pandas as pd
import pytest
import yaml

from wp6_data.blue.irrigation.config import CONFIG_PATH, IrrigationConfig
from wp6_data.blue.irrigation.model import (
    MODEL_PATH,
    ShippedModel,
    check_against,
    forecast,
    production_spec,
    read_model,
    write_model,
)


def _config() -> IrrigationConfig:
    with CONFIG_PATH.open(encoding="utf-8") as f:
        return IrrigationConfig(**yaml.safe_load(f))


def test_the_committed_model_matches_the_code_and_config():
    """Fails when the model's parameters or config change without re-running
    `wp6-blue-irrigation calibrate` and committing model.yaml."""
    model = read_model(MODEL_PATH)

    check_against(model, _config())
    assert len(model.band_low) == len(model.band_high)
    assert all(lo <= hi for lo, hi in zip(model.band_low, model.band_high, strict=True))


def test_a_model_with_other_parameters_is_refused():
    config = _config()
    model = read_model(MODEL_PATH)
    stale = ShippedModel(**(model.__dict__ | {"params": model.params | {"irr_eff": 0.01}}))

    with pytest.raises(ValueError, match="calibrate"):
        check_against(stale, config)


def test_write_then_read_round_trips(tmp_path):
    model = read_model(MODEL_PATH)
    path = tmp_path / "model.yaml"

    write_model(model, path)

    assert read_model(path) == model


def _frame(last_reading: date, days_ahead: int) -> pd.DataFrame:
    days = pd.date_range(pd.Timestamp(last_reading) - pd.Timedelta(days=20),
                         pd.Timestamp(last_reading) + pd.Timedelta(days=days_ahead), freq="D")
    frame = pd.DataFrame({
        "s": np.where(days <= pd.Timestamp(last_reading), 0.6, np.nan),
        "rain_mm": 0.0, "et0_mm": 4.0, "irrigation_l": 0.0, "coverage": 10.0,
    }, index=days)
    return frame


def test_forecast_runs_from_the_last_reading_and_dries_without_water():
    config = _config()
    model = read_model(MODEL_PATH)
    today = date(2026, 7, 10)

    out = forecast(_frame(today, model.horizon), model, config, today)

    assert len(out) == model.horizon
    assert out.index[0] == pd.Timestamp(today) + pd.Timedelta(days=1)
    assert (np.diff(np.concatenate([[0.6], out["mean"]])) < 0).all()
    assert ((out["low"] <= out["mean"]) & (out["mean"] <= out["high"])).all()


def test_no_forecast_from_a_stale_reading_or_without_weather():
    config = _config()
    model = read_model(MODEL_PATH)
    stale_reading = date(2026, 7, 1)

    assert forecast(_frame(stale_reading, 30), model, config, date(2026, 7, 20)).empty
    assert forecast(_frame(date(2026, 7, 10), 2), model, config, date(2026, 7, 10)).empty


def test_the_production_spec_ties_the_dose_to_rain():
    spec = production_spec(_config())

    assert "irr_eff" not in spec.params
    assert spec.area_per_plant_m2 == _config().model.area_per_plant_m2
