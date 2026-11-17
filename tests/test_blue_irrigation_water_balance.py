"""The daily water balance and the backtest that rates it.

The physics must behave (drying without inputs, wetting with them, never a
negative response to water), and the backtest must catch a model that
doesn't.
"""

import numpy as np
import pandas as pd
import pytest

from wp6_data.blue.irrigation import backtest as bt
from wp6_data.blue.irrigation.water_balance import (
    SINGLE_BUCKET,
    TWO_BUCKET,
    Inputs,
    fit,
    kalman_noise,
    simulate,
)

DAYS = 7


def _x(spec, **overrides) -> np.ndarray:
    values = dict(zip(spec.params, spec.x0, strict=True)) | overrides
    return np.array([values[p] for p in spec.params])


def _inputs(n: int = 1, rain=0.0, et0=4.0, irrigation=0.0, doy=196.0) -> Inputs:
    shape = (DAYS + 1, n)
    return Inputs(
        rain=np.full(shape, rain), et0=np.full(shape, et0),
        irrigation=np.full(shape, irrigation), doy=np.full(shape, doy),
    )


def test_without_water_the_root_zone_only_dries_and_drying_slows_when_dry():
    x = _x(SINGLE_BUCKET, et_scale=0.05, theta=0.5)

    path = simulate(SINGLE_BUCKET, x, np.array([0.6]), _inputs())[:, 0]

    steps = np.diff(np.concatenate([[0.6], path]))
    assert (steps < 0).all()
    # Below theta the stress factor shrinks water use, so the steps get smaller.
    assert abs(steps[-1]) < abs(steps[0])


def test_a_dose_wets_by_its_effectiveness_and_never_dries():
    x = _x(SINGLE_BUCKET, irr_eff=0.05, mix=1.0, et_scale=0.0, drain_rate=0.0)
    dosed = _inputs()
    dosed.irrigation[1] = 2.0

    base = simulate(SINGLE_BUCKET, x, np.array([0.5]), _inputs())[:, 0]
    wet = simulate(SINGLE_BUCKET, x, np.array([0.5]), dosed)[:, 0]

    assert wet[0] - base[0] == pytest.approx(0.05 * 2.0)
    assert (wet - base >= 0).all()


def test_drainage_pulls_an_overfull_root_zone_back_towards_field_capacity():
    x = _x(SINGLE_BUCKET, et_scale=0.0, drain_rate=0.5, s_fc=0.8)

    path = simulate(SINGLE_BUCKET, x, np.array([1.2]), _inputs())[:, 0]

    assert path[-1] < 1.2 and path[-1] > 0.8


def test_the_bed_tops_up_a_drying_root_zone():
    x = _x(TWO_BUCKET, et_scale=0.03, exchange=0.2, bed_relax=0.0)
    single = _x(SINGLE_BUCKET, et_scale=0.03)
    s0, wet_bed = np.array([0.3]), np.array([0.9])

    with_bed = simulate(TWO_BUCKET, x, s0, _inputs(), b0=wet_bed)[:, 0]
    alone = simulate(SINGLE_BUCKET, single, s0, _inputs())[:, 0]

    assert (with_bed > alone).all()


def test_fit_recovers_the_parameters_that_made_the_data():
    rng = np.random.default_rng(1)
    n = 300
    truth = _x(SINGLE_BUCKET, rain_eff=0.03, et_scale=0.04, theta=0.6, drain_rate=0.3)
    inputs = Inputs(
        rain=rng.exponential(2.0, (DAYS + 1, n)) * (rng.random((DAYS + 1, n)) < 0.3),
        et0=rng.uniform(1.0, 5.0, (DAYS + 1, n)),
        irrigation=np.zeros((DAYS + 1, n)),
        doy=np.full((DAYS + 1, n), 196.0),
    )
    s0 = rng.uniform(0.2, 1.0, n)
    observed = simulate(SINGLE_BUCKET, truth, s0, inputs)

    x = fit(SINGLE_BUCKET, s0, inputs, observed)

    fitted = dict(zip(SINGLE_BUCKET.params, x, strict=True))
    assert fitted["rain_eff"] == pytest.approx(0.03, rel=0.1)
    assert fitted["et_scale"] == pytest.approx(0.04, rel=0.15)


def test_kalman_noise_reads_model_and_observation_error_from_the_horizon_slope():
    rng = np.random.default_rng(2)
    q, r = 0.001, 0.004
    horizons = np.arange(1, DAYS + 1)
    residuals = np.vstack([rng.normal(0, np.sqrt(h * q + 2 * r), 20_000) for h in horizons])

    q_hat, r_hat = kalman_noise(residuals)

    assert q_hat == pytest.approx(q, rel=0.2)
    assert r_hat == pytest.approx(r, rel=0.2)


def _frame(days: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    index = pd.date_range("2024-04-01", periods=days, freq="D", name="day")
    rain = rng.exponential(3.0, days) * (rng.random(days) < 0.3)
    et0 = np.full(days, 3.0)
    x = _x(SINGLE_BUCKET, rain_eff=0.03, et_scale=0.03)
    s = np.empty(days)
    s[0] = 0.6
    for t in range(1, days):
        step = Inputs(rain=rain[t - 1:t + 1, None], et0=et0[t - 1:t + 1, None],
                      irrigation=np.zeros((2, 1)), doy=np.full((2, 1), 196.0))
        s[t] = simulate(SINGLE_BUCKET, x, np.array([s[t - 1]]), step)[0, 0]
    return pd.DataFrame({"s": s, "rain_mm": rain, "et0_mm": et0, "irrigation_l": 0.0,
                         "coverage": 10.0}, index=index)


def test_origins_are_in_season_with_complete_drivers():
    o = bt.origins(_frame())

    assert all(d.month in bt.SEASON_MONTHS for d in o.days)
    assert o.observed.shape == (bt.HORIZON, len(o.days))
    assert o.inputs.rain.shape == (bt.HORIZON + 1, len(o.days))


def test_the_counterfactual_checks_catch_a_model_that_dries_on_water():
    def backwards(train, history):
        def predict(o):
            # Wrong on purpose: more water → drier, a dry week → wetter.
            mean = (o.s0[None, :] - 0.1 * np.cumsum(o.inputs.irrigation[1:], axis=0)
                    + 0.01 * np.cumsum(o.inputs.et0[1:], axis=0))
            return bt.Forecast(mean, mean - 0.1, mean + 0.1)
        return predict

    frame = _frame()
    models = {"backwards": backwards, "A": bt.bucket(SINGLE_BUCKET)}
    forecasts, counterfactual = bt.run(frame, models, test_from="2025-01-01")
    table = bt.score(forecasts, counterfactual)

    assert table.loc["backwards", "cf_irrigation_ok"] == 0.0
    assert table.loc["backwards", "cf_dry_week_ok"] == 0.0
    assert table.loc["A", "cf_irrigation_ok"] == 1.0
    assert table.loc["A", "cf_dry_week_ok"] == 1.0
