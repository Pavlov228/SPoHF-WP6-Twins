"""The production forecast model: fitted locally, shipped as ``model.yaml`` (ADR 0009).

Production never trains. ``wp6-blue-irrigation calibrate`` fits model A on
everything locally, backtests the same recipe, and writes ``model.yaml`` next
to this module, which is then committed. The dashboard only reads it.

The model forecasts the daily fused soil moisture from the last day with a
reading, through Open-Meteo's weather forecast, **assuming no watering**. The
advice (issue 071) then asks what a dose would change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import yaml

from wp6_data.blue.irrigation import backtest as bt
from wp6_data.blue.irrigation.water_balance import (
    SINGLE_BUCKET,
    BucketSpec,
    Inputs,
    dose_as_rain,
    simulate,
)

if TYPE_CHECKING:
    from wp6_data.blue.irrigation.config import IrrigationConfig

MODEL_PATH = Path(__file__).with_name("model.yaml")
#: Bump when the model's structure changes; a shipped file of another version
#: is refused rather than misread.
MODEL_VERSION = 1


def production_spec(config: IrrigationConfig) -> BucketSpec:
    return dose_as_rain(SINGLE_BUCKET, config.model.area_per_plant_m2)


@dataclass(frozen=True)
class ShippedModel:
    """Fitted parameters and band, plus what they were fitted on and how they scored."""

    params: dict[str, float]
    band_low: list[float]       # per lead time 1..H, added to the mean
    band_high: list[float]
    area_per_plant_m2: float
    fitted_at: str
    data_first_day: str
    data_last_day: str
    origins: int
    backtest: dict[str, float] = field(default_factory=dict)
    version: int = MODEL_VERSION

    @property
    def horizon(self) -> int:
        return len(self.band_low)

    def x(self, spec: BucketSpec) -> np.ndarray:
        return np.array([self.params[name] for name in spec.params])


def check_against(model: ShippedModel, config: IrrigationConfig) -> None:
    """Refuse a shipped model that doesn't match the code or the config."""
    if model.version != MODEL_VERSION:
        raise ValueError(f"model.yaml is version {model.version}, code expects {MODEL_VERSION}")
    spec = production_spec(config)
    if set(model.params) != set(spec.params):
        raise ValueError(
            f"model.yaml parameters {sorted(model.params)} do not match the model's "
            f"{sorted(spec.params)}; re-run `wp6-blue-irrigation calibrate`",
        )
    if model.area_per_plant_m2 != config.model.area_per_plant_m2:
        raise ValueError(
            "model.yaml was fitted with a different area_per_plant_m2 than config.yaml; "
            "re-run `wp6-blue-irrigation calibrate`",
        )


def write_model(model: ShippedModel, path: Path = MODEL_PATH) -> None:
    header = (
        "# Fitted by `wp6-blue-irrigation calibrate` (ADR 0009). Do not edit by hand;\n"
        "# re-run the CLI locally and commit the result.\n"
    )
    body = yaml.safe_dump(model.__dict__, sort_keys=False, allow_unicode=True)
    path.write_text(header + body, encoding="utf-8")


def read_model(path: Path = MODEL_PATH) -> ShippedModel:
    with path.open(encoding="utf-8") as f:
        return ShippedModel(**yaml.safe_load(f))


@lru_cache(maxsize=1)
def load_model() -> ShippedModel:
    from wp6_data.blue.irrigation.config import load_config

    model = read_model()
    check_against(model, load_config())
    return model


def calibrate(frame: pd.DataFrame, config: IrrigationConfig) -> ShippedModel:
    """Backtest the production recipe, then fit it on everything."""
    spec = production_spec(config)
    trainer = bt.bucket(spec, held_out_band=True)
    forecasts, counterfactual = bt.run(frame, {"persistence": bt.persistence, "model": trainer})
    scores = bt.score(forecasts, counterfactual).loc["model"]

    all_origins = bt.origins(frame)
    final = trainer(all_origins, frame)
    low, high = final.band  # type: ignore[attr-defined]
    return ShippedModel(
        params={k: round(float(v), 6) for k, v in final.params.items()},  # type: ignore[attr-defined]
        band_low=[round(float(v), 4) for v in low],
        band_high=[round(float(v), 4) for v in high],
        area_per_plant_m2=config.model.area_per_plant_m2,
        fitted_at=datetime.now(UTC).isoformat(timespec="seconds"),
        data_first_day=str(all_origins.days.min().date()),
        data_last_day=str(all_origins.days.max().date()),
        origins=len(all_origins.days),
        backtest={k: round(float(v), 4) for k, v in scores.items() if pd.notna(v)},
    )


def forecast(
    frame: pd.DataFrame, model: ShippedModel, config: IrrigationConfig, today: date,
) -> pd.DataFrame:
    """Daily mean, low and high for the days after the last reading, no watering.

    Returns an empty frame if there is no reading in the last week, or no weather
    for the days ahead. A forecast from a stale start would look precise and
    mean nothing.
    """
    empty = pd.DataFrame(columns=["mean", "low", "high"])
    readings = frame["s"].loc[:pd.Timestamp(today)].dropna()
    if readings.empty or (pd.Timestamp(today) - readings.index[-1]).days > 7:
        return empty
    origin = readings.index[-1]
    days = pd.date_range(origin, periods=model.horizon + 1, freq="D")
    drivers = frame.reindex(days)
    if drivers[["rain_mm", "et0_mm"]].iloc[1:].isna().any().any():
        return empty

    def column(name: str) -> np.ndarray:
        return drivers[name].fillna(0.0).to_numpy()[:, None]

    inputs = Inputs(
        rain=column("rain_mm"), et0=column("et0_mm"),
        irrigation=np.zeros((len(days), 1)),  # the "no watering" path
        doy=days.dayofyear.to_numpy(dtype=float)[:, None],
    )
    spec = production_spec(config)
    mean = simulate(spec, model.x(spec), np.array([readings.iloc[-1]]), inputs)[:, 0]
    return pd.DataFrame({
        "mean": mean,
        "low": mean + np.array(model.band_low),
        "high": mean + np.array(model.band_high),
    }, index=days[1:])
