"""Model E of the irrigation bake-off: pretrained time-series foundation models.

Runs OUTSIDE the project environment, so torch never enters pyproject.toml or
the image:

    uv run --python 3.12 --with chronos-forecasting --with pandas \
        python -I scripts/irrigation_foundation_models.py DAILY_CSV ORIGINS_CSV OUT_CSV

- DAILY_CSV: the daily frame (day, s, rain_mm, et0_mm, irrigation_l, ...)
  written by ``wp6-blue-irrigation backtest --export-dir``.
- ORIGINS_CSV: one ``origin`` date per row to forecast from.
- OUT_CSV: rows of model, origin, h, mean, low, high, plus the counterfactual
  means (mean_dosed, mean_dry) where the model takes covariates.

Models:

- **Chronos-Bolt** (zero-shot, univariate). It sees only soil-moisture history,
  so it cannot respond to rain or watering, and the counterfactuals are n/a.
- **Chronos-2** (zero-shot, with covariates). Rain, ET0 and watering are passed
  as past and known-future covariates, so the counterfactuals can be run.

Weather over the horizon is as observed, the same as the in-process models.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

HORIZON = 7
CONTEXT_DAYS = 365
QUANTILES = [0.1, 0.5, 0.9]
CF_DOSE_L = 2.4
CF_DRY_ET0 = 4.0
COVARIATES = ["rain_mm", "et0_mm", "irrigation_l"]


def _context(daily: pd.DataFrame, origin: pd.Timestamp) -> pd.DataFrame:
    past = daily.loc[origin - pd.Timedelta(days=CONTEXT_DAYS - 1):origin].copy()
    # Short gaps interpolated; models get a regular daily series.
    past["s"] = past["s"].interpolate(limit=7, limit_area="inside")
    return past


def chronos_bolt(daily: pd.DataFrame, origins: list[pd.Timestamp]) -> pd.DataFrame:
    import torch
    from chronos import BaseChronosPipeline

    pipe = BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-small", device_map="cpu")
    contexts = [torch.tensor(_context(daily, o)["s"].to_numpy(dtype="float32")) for o in origins]
    quantiles, _mean = pipe.predict_quantiles(
        contexts, prediction_length=HORIZON, quantile_levels=QUANTILES,
    )
    rows = []
    for i, origin in enumerate(origins):
        for h in range(HORIZON):
            q = quantiles[i, h].numpy()
            rows.append({"model": "E chronos-bolt", "origin": origin, "h": h + 1,
                         "mean": q[1], "low": q[0], "high": q[2]})
    return pd.DataFrame(rows)


def chronos_2(daily: pd.DataFrame, origins: list[pd.Timestamp]) -> pd.DataFrame:
    from chronos import Chronos2Pipeline

    pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cpu")

    def frames(scenario: str):
        contexts, futures = [], []
        for origin in origins:
            past = _context(daily, origin).reset_index()
            past["id"] = str(origin.date())
            future = daily.loc[origin + pd.Timedelta(days=1):
                               origin + pd.Timedelta(days=HORIZON), COVARIATES].copy()
            if scenario == "dosed":
                future.iloc[0, future.columns.get_loc("irrigation_l")] += CF_DOSE_L
            elif scenario == "dry":
                future["rain_mm"] = 0.0
                future["irrigation_l"] = 0.0
                future["et0_mm"] = CF_DRY_ET0
            future = future.reset_index()
            future["id"] = str(origin.date())
            contexts.append(past[["id", "day", "s", *COVARIATES]])
            futures.append(future[["id", "day", *COVARIATES]])
        return pd.concat(contexts), pd.concat(futures)

    out = {}
    for scenario in ("base", "dosed", "dry"):
        context, future = frames(scenario)
        pred = pipe.predict_df(
            context, future_df=future, prediction_length=HORIZON,
            quantile_levels=QUANTILES, id_column="id", timestamp_column="day", target="s",
        )
        out[scenario] = pred
    base = out["base"].copy()
    base["origin"] = pd.to_datetime(base["id"])
    base["h"] = base.groupby("id").cumcount() + 1
    base = base.rename(columns={"0.1": "low", "0.5": "mean", "0.9": "high"})
    base["mean_dosed"] = out["dosed"]["0.5"].to_numpy()
    base["mean_dry"] = out["dry"]["0.5"].to_numpy()
    base["model"] = "E chronos-2"
    return base[["model", "origin", "h", "mean", "low", "high", "mean_dosed", "mean_dry"]]


def main() -> None:
    daily_csv, origins_csv, out_csv = sys.argv[1:4]
    daily = pd.read_csv(daily_csv, parse_dates=["day"]).set_index("day").asfreq("D")
    origins = list(pd.to_datetime(pd.read_csv(origins_csv)["origin"]))
    results = []
    for runner in (chronos_bolt, chronos_2):
        try:
            results.append(runner(daily, origins))
            print(f"{runner.__name__}: ok")
        except Exception as exc:  # report and carry on with the other model
            print(f"{runner.__name__}: failed: {exc!r}")
    if not results:
        raise SystemExit("no foundation model ran")
    out = pd.concat(results, ignore_index=True)
    out["origin"] = pd.to_datetime(out["origin"]).dt.date
    out.to_csv(out_csv, index=False)
    print(f"wrote {len(out)} rows to {out_csv}")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
