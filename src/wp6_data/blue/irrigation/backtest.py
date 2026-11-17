"""Rolling-origin backtest of soil-moisture forecast models (issue 068).

Offline only; nothing here is served. For every in-season day with a soil
reading (an *origin*), each model forecasts the daily soil moisture 1..H days
ahead. Each model is refitted at the start of every month, on origins whose
targets all lie before that month. Nothing a model sees reaches past its
forecast origin.

Weather is **as observed** (a perfect forecast), so the scores rate the
models, not Open-Meteo's forecast skill. In operation, forecast error adds to
them.

Scored per model and lead time:

- **MAE**, and **skill** against persistence (1 − MAE / MAE_persistence).
- **Band coverage**: the share of outcomes inside the model's 80 % band.
- **Counterfactual checks**: an extra watering must never lower the forecast;
  a week without rain or watering must never raise it. A model that fails
  these cannot advise, whatever its MAE.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from wp6_data.blue.irrigation.water_balance import (
    SINGLE_BUCKET,
    TWO_BUCKET,
    BucketSpec,
    Inputs,
    fit,
    kalman_filter,
    kalman_noise,
    simulate,
    spin_up_bed,
)

HORIZON = 7
#: Months whose days can be forecast origins and training examples.
SEASON_MONTHS = range(4, 11)
#: The 80 % band.
BAND = (0.1, 0.9)
#: Watering added in the counterfactual check, litres per plant (the farm's dose).
CF_DOSE_L = 2.4
#: ET0 for the "dry week" counterfactual, mm/day: a typical summer day.
CF_DRY_ET0 = 4.0
#: Tolerance on the counterfactual sign checks (S units).
CF_TOL = 1e-6


_DRIVERS = ("rain", "et0", "irrigation", "doy")


@dataclass(frozen=True)
class Origins:
    """Every forecast origin in a daily frame, with the arrays the models need."""

    days: pd.DatetimeIndex
    s0: np.ndarray            # (N,)
    inputs: Inputs            # (H+1, N)
    observed: np.ndarray      # (H, N)


def origins(frame: pd.DataFrame, horizon: int = HORIZON) -> Origins:
    """Origins: in-season days with a reading and complete drivers for the next *horizon* days."""
    values = frame.reindex(pd.date_range(frame.index.min(), frame.index.max(), freq="D"))
    n = len(values)
    s = values["s"].to_numpy()
    drivers = values[["rain_mm", "et0_mm", "irrigation_l"]].to_numpy()
    doy = values.index.dayofyear.to_numpy(dtype=float)
    keep = []
    for t in range(n - horizon):
        if not np.isfinite(s[t]) or values.index[t].month not in SEASON_MONTHS:
            continue
        if not np.isfinite(drivers[t:t + horizon + 1, :2]).all():
            continue
        keep.append(t)
    idx = np.array(keep, dtype=int)
    window = idx[None, :] + np.arange(horizon + 1)[:, None]
    return Origins(
        days=values.index[idx],
        s0=s[idx],
        inputs=Inputs(
            rain=drivers[window, 0],
            et0=drivers[window, 1],
            irrigation=np.nan_to_num(drivers[window, 2]),
            doy=doy[window],
        ),
        observed=s[window[1:]],
    )


def subset(o: Origins, mask: np.ndarray) -> Origins:
    return Origins(
        days=o.days[mask],
        s0=o.s0[mask],
        inputs=Inputs(*(getattr(o.inputs, f)[:, mask] for f in _DRIVERS)),
        observed=o.observed[:, mask],
    )


def with_inputs(o: Origins, **arrays: np.ndarray) -> Origins:
    """A copy of *o* with some drivers replaced (counterfactuals)."""
    fields = {f: getattr(o.inputs, f) for f in _DRIVERS} | arrays
    return Origins(days=o.days, s0=o.s0, inputs=Inputs(**fields), observed=o.observed)


@dataclass(frozen=True)
class Forecast:
    mean: np.ndarray  # (H, N)
    low: np.ndarray
    high: np.ndarray


#: A fitted model: forecasts any origins (possibly counterfactual ones).
Predictor = Callable[[Origins], Forecast]
#: Fits on training origins and the full daily history before the refit date.
Trainer = Callable[[Origins, pd.DataFrame], Predictor]


def _band(residuals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-horizon residual quantiles, shaped (H, 1) for broadcasting."""
    lo = np.nanquantile(residuals, BAND[0], axis=1)[:, None]
    hi = np.nanquantile(residuals, BAND[1], axis=1)[:, None]
    return lo, hi


def persistence(train: Origins, _history: pd.DataFrame) -> Predictor:
    lo, hi = _band(train.observed - train.s0[None, :])

    def predict(o: Origins) -> Forecast:
        mean = np.repeat(o.s0[None, :], o.inputs.horizon, axis=0)
        return Forecast(mean, mean + lo, mean + hi)

    return predict


def bucket(spec: BucketSpec) -> Trainer:
    """Model A (single bucket) or A2 (with the soil bed)."""

    def train(train_o: Origins, history: pd.DataFrame) -> Predictor:
        b0_train = _bed_states(spec, None, train_o, history) if spec.two_bucket else None
        x = fit(spec, train_o.s0, train_o.inputs, train_o.observed, b0_train)
        if spec.two_bucket:
            # Refit once more with the bed spun up under the fitted parameters.
            b0_train = _bed_states(spec, x, train_o, history)
            x = fit(spec, train_o.s0, train_o.inputs, train_o.observed, b0_train)
        lo, hi = _band(train_o.observed - simulate(spec, x, train_o.s0, train_o.inputs, b0_train))

        def predict(o: Origins) -> Forecast:
            b0 = _bed_states(spec, x, o, history) if spec.two_bucket else None
            mean = simulate(spec, x, o.s0, o.inputs, b0)
            return Forecast(mean, mean + lo, mean + hi)

        predict.params = dict(zip(spec.params, x, strict=True))  # type: ignore[attr-defined]
        return predict

    return train


def _bed_states(
    spec: BucketSpec, x: np.ndarray | None, o: Origins, history: pd.DataFrame,
) -> np.ndarray:
    """The bed state at each origin, spun up over the 60 days before it."""
    x = np.asarray(spec.x0) if x is None else x
    out = np.empty(len(o.days))
    for i, day in enumerate(o.days):
        past = history.loc[day - pd.Timedelta(days=60):day]
        out[i] = spin_up_bed(spec, x, past["s"].to_numpy(), _history_inputs(past))
    return out


def _history_inputs(days: pd.DataFrame) -> Inputs:
    return Inputs(
        rain=days["rain_mm"].fillna(0.0).to_numpy(),
        et0=days["et0_mm"].fillna(0.0).to_numpy(),
        irrigation=days["irrigation_l"].fillna(0.0).to_numpy(),
        doy=days.index.dayofyear.to_numpy(dtype=float),
    )


def kalman(train_o: Origins, history: pd.DataFrame) -> Predictor:
    """Model B: A's dynamics, with the starting state filtered and the band grown per day."""
    spec = SINGLE_BUCKET
    x = fit(spec, train_o.s0, train_o.inputs, train_o.observed)
    q, r = kalman_noise(train_o.observed - simulate(spec, x, train_o.s0, train_o.inputs))
    z = 1.2816  # 80 % two-sided normal band

    def predict(o: Origins) -> Forecast:
        s0 = np.empty(len(o.days))
        var0 = np.empty(len(o.days))
        for i, day in enumerate(o.days):
            past = history.loc[day - pd.Timedelta(days=60):day]
            obs = past["s"].to_numpy().copy()
            obs[-1] = o.s0[i]  # the origin's own reading (counterfactuals keep it)
            s0[i], var0[i] = kalman_filter(spec, x, obs, _history_inputs(past), q, r)
        mean = simulate(spec, x, s0, o.inputs)
        h = np.arange(1, o.inputs.horizon + 1)[:, None]
        half = z * np.sqrt(var0[None, :] + h * q + r)
        return Forecast(mean, mean - half, mean + half)

    return predict


def _cumulative(o: Origins, h: int) -> dict[str, np.ndarray]:
    return {
        "rain": o.inputs.rain[1:h + 1].sum(axis=0),
        "et0": o.inputs.et0[1:h + 1].sum(axis=0),
        "irr": o.inputs.irrigation[0:h + 1].sum(axis=0),
    }


def _lags(o: Origins, history: pd.DataFrame) -> np.ndarray:
    s = history["s"]
    out = []
    for lag in (1, 3, 7):
        out.append(s.reindex(o.days - pd.Timedelta(days=lag)).to_numpy())
    return np.column_stack(out)


def _gbm() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=10,
        l2_regularization=1.0, random_state=0,
    )


def _holdout_split(n: int) -> tuple[np.ndarray, np.ndarray]:
    """The last 20 % of training origins (by time) as a holdout for honest bands."""
    cut = int(n * 0.8)
    return np.arange(cut), np.arange(cut, n)


def gbm_direct(train_o: Origins, history: pd.DataFrame) -> Predictor:
    """Model D: one gradient-boosting model per horizon. Watering is an input like any other."""

    def features(o: Origins, h: int) -> np.ndarray:
        c = _cumulative(o, h)
        past_irr = np.array([
            history["irrigation_l"].loc[d - pd.Timedelta(days=7):d].sum() for d in o.days
        ])
        return np.column_stack([
            o.s0, _lags(o, history), c["rain"], c["et0"], c["irr"], past_irr,
            o.inputs.doy[0],
        ])

    models, lo, hi = [], [], []
    for h in range(1, train_o.inputs.horizon + 1):
        y = train_o.observed[h - 1]
        x = features(train_o, h)
        ok = np.isfinite(y)
        first, last = _holdout_split(int(ok.sum()))
        xs, ys = x[ok], y[ok]
        holdout = _gbm().fit(xs[first], ys[first])
        err = ys[last] - holdout.predict(xs[last])
        lo.append(np.quantile(err, BAND[0]))
        hi.append(np.quantile(err, BAND[1]))
        models.append(_gbm().fit(xs, ys))

    def predict(o: Origins) -> Forecast:
        mean = np.vstack([m.predict(features(o, h)) for h, m in enumerate(models, start=1)])
        return Forecast(mean, mean + np.array(lo)[:, None], mean + np.array(hi)[:, None])

    return predict


def hybrid(train_o: Origins, history: pd.DataFrame) -> Predictor:
    """Model C: A, plus a gradient-boosting correction of its residual.

    The correction never sees watering, so watering's effect stays A's: the
    physics alone decides the response to a dose.
    """
    base = bucket(SINGLE_BUCKET)(train_o, history)

    def features(o: Origins, h: int) -> np.ndarray:
        c = _cumulative(o, h)
        return np.column_stack([o.s0, c["rain"], c["et0"], o.inputs.doy[0]])

    base_train = base(train_o).mean
    models, lo, hi = [], [], []
    for h in range(1, train_o.inputs.horizon + 1):
        y = train_o.observed[h - 1] - base_train[h - 1]
        x = features(train_o, h)
        ok = np.isfinite(y)
        first, last = _holdout_split(int(ok.sum()))
        xs, ys = x[ok], y[ok]
        holdout = _gbm().fit(xs[first], ys[first])
        err = ys[last] - holdout.predict(xs[last])
        lo.append(np.quantile(err, BAND[0]))
        hi.append(np.quantile(err, BAND[1]))
        models.append(_gbm().fit(xs, ys))

    def predict(o: Origins) -> Forecast:
        a = base(o).mean
        mean = a + np.vstack([m.predict(features(o, h)) for h, m in enumerate(models, start=1)])
        return Forecast(mean, mean + np.array(lo)[:, None], mean + np.array(hi)[:, None])

    return predict


#: The in-process candidates. E (foundation models) runs outside; see
#: ``scripts/irrigation_foundation_models.py``.
MODELS: dict[str, Trainer] = {
    "persistence": persistence,
    "A single bucket": bucket(SINGLE_BUCKET),
    "A2 two bucket": bucket(TWO_BUCKET),
    "B kalman": kalman,
    "C hybrid": hybrid,
    "D gbm direct": gbm_direct,
}


def run(
    frame: pd.DataFrame,
    models: dict[str, Trainer] = MODELS,
    test_from: str = "2025-04-01",
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Rolling-origin forecasts for every model.

    Returns the forecasts (one row per model, origin and horizon) and, per
    model, its counterfactual results.
    """
    o = origins(frame)
    test_months = sorted({d.to_period("M") for d in o.days if d >= pd.Timestamp(test_from)})
    rows = []
    counterfactual: dict[str, dict] = {name: {"irr_ok": [], "irr_gain": [], "dry_ok": []}
                                       for name in models}
    for month in test_months:
        refit = month.start_time
        target_end = o.days + pd.Timedelta(days=HORIZON)
        train = subset(o, np.asarray(target_end < refit))
        test = subset(o, np.asarray((o.days >= refit) & (o.days <= month.end_time)))
        if len(test.days) == 0 or len(train.days) < 60:
            continue
        # Every use of history inside a model is a window ending at its own
        # origin (lags, bed spin-up, filter), so history up to the month's end
        # cannot leak a target into a forecast.
        history = frame.loc[:month.end_time]
        for name, trainer in models.items():
            predictor = trainer(train, history)
            fc = predictor(test)
            for h in range(HORIZON):
                rows.append(pd.DataFrame({
                    "model": name, "origin": test.days, "h": h + 1,
                    "observed": test.observed[h], "mean": fc.mean[h],
                    "low": fc.low[h], "high": fc.high[h],
                }))
            _counterfactuals(predictor, test, counterfactual[name])
    return pd.concat(rows, ignore_index=True), counterfactual


def _counterfactuals(predict: Predictor, test: Origins, out: dict) -> None:
    base = predict(test).mean
    dosed = test.inputs.irrigation.copy()
    dosed[1] += CF_DOSE_L
    more = predict(with_inputs(test, irrigation=dosed)).mean
    out["irr_ok"].append(np.all(more - base >= -CF_TOL, axis=0))
    out["irr_gain"].append((more - base)[2])
    dry = predict(with_inputs(
        test,
        rain=np.zeros_like(test.inputs.rain),
        irrigation=np.zeros_like(test.inputs.irrigation),
        et0=np.full_like(test.inputs.et0, CF_DRY_ET0),
    )).mean
    # Along the model's own path: a filtered start (B) may legitimately differ
    # from the raw reading, and that first jump isn't water arriving.
    out["dry_ok"].append(np.all(np.diff(dry, axis=0) <= CF_TOL, axis=0))


def score(forecasts: pd.DataFrame, counterfactual: dict[str, dict]) -> pd.DataFrame:
    """One row per model: MAE by horizon, skill vs persistence, band coverage, checks."""
    f = forecasts.dropna(subset=["observed", "mean"])
    f = f.assign(err=(f["mean"] - f["observed"]).abs(),
                 inside=(f["observed"] >= f["low"]) & (f["observed"] <= f["high"]))
    mae = f.pivot_table(index="model", columns="h", values="err", aggfunc="mean")
    table = mae.rename(columns=lambda h: f"mae_h{h}")
    persist = mae.loc["persistence"] if "persistence" in mae.index else None
    if persist is not None:
        table["skill_h1"] = 1 - mae[1] / persist[1]
        table["skill_h7"] = 1 - mae[HORIZON] / persist[HORIZON]
    table["band80_cover"] = f.groupby("model")["inside"].mean()
    for name, cf in counterfactual.items():
        if name in table.index and cf["irr_ok"]:
            table.loc[name, "cf_irrigation_ok"] = np.concatenate(cf["irr_ok"]).mean()
            table.loc[name, "cf_dose_gain_h2"] = np.median(np.concatenate(cf["irr_gain"]))
            table.loc[name, "cf_dry_week_ok"] = np.concatenate(cf["dry_ok"]).mean()
    return table


def merge_external(
    forecasts: pd.DataFrame, external: pd.DataFrame, counterfactual: dict[str, dict],
) -> pd.DataFrame:
    """Add forecasts made outside the project environment (model E) for scoring.

    *external* has model, origin, h, mean, low, high and, where the model takes
    covariates, mean_dosed / mean_dry for the counterfactual checks.
    """
    observed = forecasts.drop_duplicates(["origin", "h"])[["origin", "h", "observed"]]
    ext = external.assign(origin=pd.to_datetime(external["origin"]))
    ext = ext.merge(observed, on=["origin", "h"], how="inner")
    for name, rows in ext.groupby("model"):
        if {"mean_dosed", "mean_dry"} <= set(rows.columns) and rows["mean_dosed"].notna().any():
            wide = rows.pivot_table(index="h", columns="origin", values=["mean", "mean_dosed",
                                                                         "mean_dry"])
            base, dosed, dry = (wide[c].to_numpy() for c in ("mean", "mean_dosed", "mean_dry"))
            counterfactual[name] = {
                "irr_ok": [np.all(dosed - base >= -CF_TOL, axis=0)],
                "irr_gain": [(dosed - base)[1]],
                "dry_ok": [np.all(np.diff(dry, axis=0) <= CF_TOL, axis=0)],
            }
    columns = ["model", "origin", "h", "observed", "mean", "low", "high"]
    return pd.concat([forecasts, ext[columns]], ignore_index=True)
