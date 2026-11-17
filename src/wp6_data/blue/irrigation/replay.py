"""Replay a past season under different watering policies (issue 072).

The shipped model (ADR 0009) runs day by day through a season on the weather
that actually happened, starting from the measured soil moisture on the first
day. Each **policy** decides every morning whether to water and how much:

- **as logged**: the farm's recorded doses. Its path set against the measured
  soil moisture shows how far an open-loop season replay can be trusted.
- **no watering**
- **advice rule**: below the trigger in the morning → refill to the target,
  but only while the crop is active. That uses the same levels and the same
  two gates as the live advice: the soil-temperature mean of the past days,
  and the crop use over the coming week.
- **fixed schedule**: every N days, a fixed dose.

**Anchored to the measurements.** Run open loop, the model drifts over a season
(the "as logged" replay was off by 0.10 on average in 2025 and 0.26 in 2026). So
the model is only ever trusted for **one day, from a real level**:

    policy(t+1) = measured(t+1) + [ step(policy(t), policy's dose)
                                    − step(measured(t), logged dose) ]

The model's one-day error at the measured level cancels out, and only the
policy's difference is carried forward. That difference fades through extra
evaporation and drainage, computed at the level the soil is really at. "As
logged" reproduces the measurements exactly, by construction. Across a gap in the
measurements, the reference is stepped by the model under the logged doses.
A season without a watering log (2026, so far) treats "as logged" as no
watering, so policy effects are relative to whatever was actually done.

Per policy: water used, days below the trigger, the longest run below it, the
lowest level, and water lost to drainage (filled above field capacity).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from wp6_data.blue.irrigation.advice import Levels, levels
from wp6_data.blue.irrigation.model import ShippedModel, production_spec
from wp6_data.blue.irrigation.water_balance import KC_PIVOT_DOY, Inputs, simulate

if TYPE_CHECKING:
    from wp6_data.blue.irrigation.config import IrrigationConfig

#: The replayed season: April through October.
SEASON = ((4, 1), (10, 31))

AS_LOGGED = "as logged"
NO_WATERING = "no watering"
ADVICE_RULE = "advice rule"
FIXED = "fixed schedule"

#: A policy: (day index, today's anchored level, logged dose today) → litres per plant.
Policy = Callable[[int, float, float], float]


@dataclass(frozen=True)
class Replay:
    path: pd.Series          # simulated daily level
    doses: pd.Series         # litres per plant per day
    drained_mm: pd.Series    # water lost above field capacity, mm rain-equivalent


def season_days(frame: pd.DataFrame, year: int) -> pd.DataFrame:
    (m0, d0), (m1, d1) = SEASON
    return frame.loc[pd.Timestamp(date(year, m0, d0)):pd.Timestamp(date(year, m1, d1))]


def replay(
    season: pd.DataFrame, s0: float, model: ShippedModel, config: IrrigationConfig,
    policy: Policy, measured: pd.Series | None = None,
) -> Replay:
    """Run the season one day at a time, letting *policy* water each morning.

    With *measured*, each day is anchored as in the module docstring. Without
    it, the replay runs open loop (for comparison).
    """
    spec = production_spec(config)
    x = model.x(spec)
    p = model.params
    rain = season["rain_mm"].fillna(0.0).to_numpy()
    et0 = season["et0_mm"].fillna(0.0).to_numpy()
    logged = season["irrigation_l"].fillna(0.0).to_numpy()
    doy = season.index.dayofyear.to_numpy(dtype=float)
    real = (measured.reindex(season.index).to_numpy() if measured is not None
            else np.full(len(season), np.nan))

    def step(day: int, start: float, irrigation: np.ndarray) -> float:
        window = slice(day - 1, day + 1)
        inputs = Inputs(
            rain=rain[window, None], et0=et0[window, None],
            irrigation=irrigation[window, None], doy=doy[window, None],
        )
        return float(simulate(spec, x, np.array([start]), inputs)[0, 0])

    n = len(season)
    anchored = np.isfinite(real).any()
    # The reference: measured where it exists; across a gap, stepped by the
    # model under the logged doses, so the anchor never breaks.
    reference = np.empty(n)
    reference[0] = s0
    for day in range(1, n):
        reference[day] = real[day] if np.isfinite(real[day]) else step(day, reference[day - 1],
                                                                        logged)
    level = np.empty(n)
    doses = np.zeros(n)
    drained = np.zeros(n)
    level[0] = s0
    for day in range(1, n):
        # A morning dose counts on its own day, so decide before stepping into it.
        doses[day] = policy(day, level[day - 1], logged[day])
        drained[day] = p["drain_rate"] * max(level[day - 1] - p["s_fc"], 0.0) / p["rain_eff"]
        own = step(day, level[day - 1], doses)
        if anchored:
            level[day] = reference[day] + own - step(day, reference[day - 1], logged)
        else:
            level[day] = own
    return Replay(
        path=pd.Series(level, index=season.index),
        doses=pd.Series(doses, index=season.index),
        drained_mm=pd.Series(drained, index=season.index),
    )


@dataclass(frozen=True)
class SeasonGate:
    """The live advice's crop-activity gates, evaluated per replay day.

    Soil temperature is the mean over the days up to yesterday. Demand is the
    crop's *potential* water use over the coming week (no drought stress
    factor), so a drought never makes the crop look dormant.
    """

    recent_temp: np.ndarray
    potential_mm: np.ndarray
    week: int
    min_temp: float
    min_use: float

    @classmethod
    def for_season(
        cls, season: pd.DataFrame, model: ShippedModel, config: IrrigationConfig,
    ) -> SeasonGate:
        p, a = model.params, config.advice
        soil_temp = (season["soil_temp_c"] if "soil_temp_c" in season
                     else pd.Series(np.nan, index=season.index))
        kc = np.clip(1.0 + p["kc_slope"] * (season.index.dayofyear - KC_PIVOT_DOY) / 100.0,
                     0.05, None)
        return cls(
            recent_temp=soil_temp.rolling(a.soil_temp_window_days, min_periods=1)
            .mean().to_numpy(),
            potential_mm=p["et_scale"] * kc * season["et0_mm"].fillna(0.0).to_numpy()
            / p["rain_eff"],
            # Look as far ahead as the live advice does: the model's horizon.
            week=model.horizon,
            min_temp=a.min_soil_temp_c,
            min_use=a.min_crop_use_mm_per_day,
        )

    def active(self, day: int) -> bool:
        temp = self.recent_temp[max(day - 1, 0)]
        warm = not np.isfinite(temp) or temp >= self.min_temp
        demand = float(np.mean(self.potential_mm[day:day + self.week]))
        return warm and demand >= self.min_use


def policies(
    gate: SeasonGate, model: ShippedModel, config: IrrigationConfig, every_days: int,
    fixed_dose_l: float, max_dose_l: float | None = None,
) -> dict[str, Policy]:
    lv = levels(model, config)
    refill_per_l = model.params["rain_eff"] / model.area_per_plant_m2

    def advice_rule(day: int, s: float, _logged: float) -> float:
        if s >= lv.trigger or not gate.active(day):
            return 0.0
        dose = (lv.target - s) / refill_per_l
        return min(dose, max_dose_l) if max_dose_l else dose

    def fixed(day: int, _s: float, _logged: float) -> float:
        return fixed_dose_l if every_days > 0 and day % every_days == 0 else 0.0

    return {
        AS_LOGGED: lambda _day, _s, logged: logged,
        NO_WATERING: lambda *_: 0.0,
        ADVICE_RULE: advice_rule,
        FIXED: fixed,
    }


def longest_run(below: np.ndarray) -> int:
    best = run = 0
    for flag in below:
        run = run + 1 if flag else 0
        best = max(best, run)
    return best


def summarise(r: Replay, lv: Levels, gate: SeasonGate) -> dict[str, float]:
    """Totals for one policy. Stress counts only while the crop is active: a low
    reading in a dormant autumn isn't a thirsty plant."""
    level = r.path.to_numpy()
    active = np.array([gate.active(day) for day in range(len(level))])
    stressed = (level < lv.trigger) & active
    return {
        "water_l_per_plant": float(r.doses.sum()),
        "waterings": int((r.doses > 0).sum()),
        "stress_days": int(stressed.sum()),
        "longest_stress": longest_run(stressed),
        "lowest_active": float(level[active].min()) if active.any() else float("nan"),
        "drained_mm": float(r.drained_mm.sum()),
    }


def run_season(
    frame: pd.DataFrame, year: int, model: ShippedModel, config: IrrigationConfig,
    every_days: int, fixed_dose_l: float, max_dose_l: float | None = None,
) -> tuple[dict[str, Replay], pd.Series, SeasonGate] | None:
    """Every policy over *year*'s season, anchored to the measured path.

    Returns None when the season has no measurement or no weather to start from.
    """
    season = season_days(frame, year)
    measured = season["s"]
    if measured.dropna().empty or season[["rain_mm", "et0_mm"]].isna().all().any():
        return None
    start = measured.first_valid_index()
    season = season.loc[start:]
    measured = measured.loc[start:]
    s0 = float(measured.loc[start])
    gate = SeasonGate.for_season(season, model, config)
    runs = {
        name: replay(season, s0, model, config, policy, measured=measured)
        for name, policy in policies(
            gate, model, config, every_days, fixed_dose_l, max_dose_l,
        ).items()
    }
    return runs, measured, gate
