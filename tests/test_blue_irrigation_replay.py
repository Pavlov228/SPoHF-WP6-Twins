"""Season replay (issue 072): anchored to the measurements, gated on crop activity."""

import numpy as np
import pandas as pd
import pytest
import yaml

from wp6_data.blue.irrigation import replay as rp
from wp6_data.blue.irrigation.advice import levels
from wp6_data.blue.irrigation.config import CONFIG_PATH, IrrigationConfig
from wp6_data.blue.irrigation.model import MODEL_PATH, read_model

YEAR = 2025


def _config() -> IrrigationConfig:
    with CONFIG_PATH.open(encoding="utf-8") as f:
        return IrrigationConfig(**yaml.safe_load(f))


def _season(et0_summer: float = 5.0, warm: bool = True) -> pd.DataFrame:
    """A season with a summer drought that the measurements follow, and a cool autumn."""
    days = pd.date_range(f"{YEAR}-04-01", f"{YEAR}-10-31", freq="D", name="day")
    summer = (days.month >= 5) & (days.month <= 8)
    autumn = days.month >= 10
    rng = np.random.default_rng(0)
    s = np.clip(0.75 - 0.004 * np.arange(len(days)) + rng.normal(0, 0.005, len(days)), 0.05, 1)
    return pd.DataFrame({
        "s": s,
        "rain_mm": np.where(summer, 0.0, 2.0),
        "et0_mm": np.where(summer, et0_summer, np.where(autumn, 0.5, 2.5)),
        "irrigation_l": np.where(days.day == 15, 2.4, 0.0),
        "coverage": 10.0,
        "soil_temp_c": np.where(autumn, 8.0, 18.0) if warm else 8.0,
    }, index=days)


def _run(frame: pd.DataFrame, **kwargs):
    model, config = read_model(MODEL_PATH), _config()
    return rp.run_season(frame, YEAR, model, config, every_days=10, fixed_dose_l=2.4, **kwargs)


def test_as_logged_reproduces_the_measurements_even_across_a_gap():
    frame = _season()
    frame.loc[f"{YEAR}-06-10":f"{YEAR}-06-14", "s"] = np.nan

    runs, measured, _gate = _run(frame)

    path = runs[rp.AS_LOGGED].path
    both = measured.notna()
    assert np.allclose(path[both], measured[both])


def test_watering_never_leaves_the_soil_drier_than_not_watering():
    runs, _measured, _gate = _run(_season())

    assert (runs[rp.FIXED].path >= runs[rp.NO_WATERING].path - 1e-9).all()
    assert (runs[rp.ADVICE_RULE].path >= runs[rp.NO_WATERING].path - 1e-9).all()


def test_the_advice_rule_waters_in_a_summer_drought_but_not_in_a_cold_autumn():
    runs, _measured, gate = _run(_season())
    doses = runs[rp.ADVICE_RULE].doses

    watered = doses[doses > 0].index
    assert len(watered) > 0
    assert all(d.month < 10 for d in watered)
    assert not any(gate.active(i) for i, d in enumerate(doses.index) if d.month == 10)


def test_stress_counts_only_while_the_crop_is_active():
    model, config = read_model(MODEL_PATH), _config()
    lv = levels(model, config)
    cold = _season(warm=False)  # never warm enough: the crop is never active

    runs, _measured, gate = _run(cold)

    assert rp.summarise(runs[rp.NO_WATERING], lv, gate)["stress_days"] == 0
    assert rp.summarise(runs[rp.ADVICE_RULE], lv, gate)["water_l_per_plant"] == 0


def test_a_max_dose_caps_each_advised_watering():
    runs, _measured, _gate = _run(_season(), max_dose_l=10.0)

    doses = runs[rp.ADVICE_RULE].doses
    assert doses.max() == pytest.approx(10.0)


def test_no_replay_without_measurements():
    frame = _season().assign(s=np.nan)

    assert _run(frame) is None
