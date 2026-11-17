"""Watering advice and what-if scenarios on the 7-day soil-moisture outlook.

Everything runs from one :class:`Outlook`: the last day with a reading, its
soil moisture, and the drivers (weather forecast, watering) for the days
after it. On that outlook:

- :func:`scenario` simulates any what-if (rain scaled or delayed, a dose on
  some day, morning or evening) with the shipped model and its band.
- :func:`advise` answers "when, and how much?". *When* is the first morning the
  no-watering path falls below the trigger. *How much* is the dose that brings
  that day back to the target. Advice is only given while the crop is
  **active** (:func:`activity`): soil warm enough for roots to work, and real
  water demand ahead. A low reading in autumn is not a thirsty plant.

Levels live on the fused 0–1 scale, anchored on the model's fitted field
capacity ``s_fc``:

    trigger = s_fc × (1 − allowable_depletion)        (0 taken as the dry limit)
    target  = s_fc − target_margin

Drip barely registers on the probes, so a dose's effect is tied to rain over
the plant's area (ADR 0009). The litres advised rest on that assumption, and
the advice says so.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, datetime, time
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from wp6_data.blue.irrigation.model import ShippedModel, production_spec
from wp6_data.blue.irrigation.water_balance import KC_PIVOT_DOY, Inputs, simulate

if TYPE_CHECKING:
    from wp6_data.blue.irrigation.config import IrrigationConfig

#: Readings older than this can't start an outlook: the start would be stale.
MAX_READING_AGE_DAYS = 7
MORNING = "morning"
EVENING = "evening"


@dataclass(frozen=True)
class Outlook:
    """The start of a forecast and its drivers. Row 0 of *inputs* is the origin day."""

    origin: pd.Timestamp
    s0: float
    days: pd.DatetimeIndex     # origin + 1 .. origin + H
    inputs: Inputs             # (H + 1, 1)
    #: Mean soil temperature over the days up to the origin (°C), if measured.
    soil_temp_c: float | None = None


@dataclass(frozen=True)
class Levels:
    trigger: float
    target: float
    field_capacity: float


def levels(model: ShippedModel, config: IrrigationConfig) -> Levels:
    fc = model.params["s_fc"]
    return Levels(
        trigger=fc * (1 - config.advice.allowable_depletion),
        target=fc - config.advice.target_margin,
        field_capacity=fc,
    )


def outlook(
    frame: pd.DataFrame, model: ShippedModel, today: date, soil_temp_window_days: int = 7,
) -> Outlook | None:
    """The outlook from the last reading, or None if there isn't a fresh one or no weather."""
    readings = frame["s"].loc[:pd.Timestamp(today)].dropna()
    if readings.empty or (pd.Timestamp(today) - readings.index[-1]).days > MAX_READING_AGE_DAYS:
        return None
    origin = readings.index[-1]
    window = pd.date_range(origin, periods=model.horizon + 1, freq="D")
    drivers = frame.reindex(window)
    if drivers[["rain_mm", "et0_mm"]].iloc[1:].isna().any().any():
        return None

    def column(name: str) -> np.ndarray:
        return drivers[name].fillna(0.0).to_numpy(dtype=float)[:, None]

    recent_temp = (
        frame["soil_temp_c"].loc[origin - pd.Timedelta(days=soil_temp_window_days - 1):origin]
        if "soil_temp_c" in frame else pd.Series(dtype=float)
    ).dropna()
    return Outlook(
        origin=origin,
        s0=float(readings.iloc[-1]),
        soil_temp_c=float(recent_temp.mean()) if not recent_temp.empty else None,
        days=window[1:],
        inputs=Inputs(
            rain=column("rain_mm"), et0=column("et0_mm"),
            # Forward-looking: no watering unless a scenario adds it.
            irrigation=np.zeros((len(window), 1)),
            doy=window.dayofyear.to_numpy(dtype=float)[:, None],
        ),
    )


def scenario(
    base: Outlook,
    *,
    rain_factor: float = 1.0,
    rain_delay_days: int = 0,
    dose_l: float = 0.0,
    dose_day: date | None = None,
    dose_time: str = MORNING,
) -> Outlook:
    """A what-if on *base*: forecast rain scaled and/or delayed, and an optional dose.

    The model works on daily means, and its fitted ``mix`` puts a day's inputs
    in that day's mean. A morning dose therefore counts on its own day; an
    evening dose arrives after most of the day has passed, so it counts on the
    next. That is the only time-of-day effect a daily model can express (with
    drip it is small anyway, research note §3).
    """
    rain = base.inputs.rain.copy()
    future = rain[1:, 0] * rain_factor
    if rain_delay_days > 0:
        future = np.concatenate([np.zeros(rain_delay_days), future])[: len(future)]
    rain[1:, 0] = future

    irrigation = base.inputs.irrigation.copy()
    if dose_l > 0 and dose_day is not None:
        all_days = base.days.insert(0, base.origin)
        target_day = pd.Timestamp(dose_day) + pd.Timedelta(days=1 if dose_time == EVENING else 0)
        if target_day in all_days:
            irrigation[all_days.get_loc(target_day), 0] += dose_l
    return replace(base, inputs=replace(base.inputs, rain=rain, irrigation=irrigation))


def trajectory(o: Outlook, model: ShippedModel, config: IrrigationConfig) -> pd.DataFrame:
    """Daily mean and 80 % band along an outlook."""
    spec = production_spec(config)
    mean = simulate(spec, model.x(spec), np.array([o.s0]), o.inputs)[:, 0]
    return pd.DataFrame({
        "mean": mean,
        "low": mean + np.array(model.band_low),
        "high": mean + np.array(model.band_high),
    }, index=o.days)


def crop_use_mm(o: Outlook, model: ShippedModel, config: IrrigationConfig) -> float:
    """Forecast crop water use over the outlook (no watering), in mm.

    The model's ET term is in soil-moisture units; dividing by ``rain_eff``
    turns it back into millimetres, the unit the farm can compare with rain.
    """
    p = model.params
    path = np.concatenate([[o.s0], trajectory(o, model, config)["mean"].to_numpy()])
    kc = np.clip(1.0 + p["kc_slope"] * (o.inputs.doy[1:, 0] - KC_PIVOT_DOY) / 100.0, 0.05, None)
    ks = np.clip(path[:-1] / p["theta"], 0.0, 1.0)
    et = p["et_scale"] * kc * ks * o.inputs.et0[1:, 0]
    return float(et.sum() / p["rain_eff"])


@dataclass(frozen=True)
class Activity:
    """Whether the crop is active enough for watering to matter, and why not."""

    active: bool
    reason: str | None           # why advice is withheld, for the card
    soil_temp_c: float | None
    crop_use_mm_per_day: float


def potential_demand_mm(o: Outlook, model: ShippedModel) -> float:
    """The crop's water demand over the outlook *before* drought stress, mm/day.

    ET0 × the seasonal crop factor, in rain-equivalent mm. Unlike
    :func:`crop_use_mm`, it doesn't shrink as the soil dries, so a drought can't
    make a thirsty crop look dormant.
    """
    p = model.params
    kc = np.clip(1.0 + p["kc_slope"] * (o.inputs.doy[1:, 0] - KC_PIVOT_DOY) / 100.0, 0.05, None)
    return float(np.mean(p["et_scale"] * kc * o.inputs.et0[1:, 0]) / p["rain_eff"])


def activity(o: Outlook, model: ShippedModel, config: IrrigationConfig) -> Activity:
    """Both gates from the config: soil warm enough, and real demand ahead.

    A missing soil temperature doesn't block advice. The demand gate still
    holds, and an unknown isn't evidence of dormancy.
    """
    a = config.advice
    use = potential_demand_mm(o, model)
    reasons = []
    if o.soil_temp_c is not None and o.soil_temp_c < a.min_soil_temp_c:
        reasons.append(
            f"the soil is {o.soil_temp_c:.0f} °C, below the {a.min_soil_temp_c:g} °C at "
            f"which blueberry roots are active",
        )
    if use < a.min_crop_use_mm_per_day:
        reasons.append(
            f"the crop's water demand is only {use:.1f} mm/day, below "
            f"{a.min_crop_use_mm_per_day:g} mm/day",
        )
    return Activity(
        active=not reasons, reason=" and ".join(reasons) or None,
        soil_temp_c=o.soil_temp_c, crop_use_mm_per_day=use,
    )


@dataclass(frozen=True)
class Advice:
    """What to do, and the numbers behind it."""

    needed: bool
    water_on: date | None        # the local day to water
    water_at: time               # preferred local time
    dose_mm: float               # rain-equivalent depth
    dose_l_per_plant: float      # under the area assumption
    crossing_day: date | None    # first day the no-watering path falls below the trigger
    possibly_on: date | None     # first day the band's dry edge falls below it
    levels: Levels
    rain_forecast_mm: float
    crop_use_mm: float
    area_per_plant_m2: float
    activity: Activity


def advise(o: Outlook, model: ShippedModel, config: IrrigationConfig) -> Advice:
    """When the no-watering path crosses the trigger, and the dose that refills that day."""
    lv = levels(model, config)
    path = trajectory(o, model, config)
    below = path.index[path["mean"] < lv.trigger]
    possibly = path.index[path["low"] < lv.trigger]
    crossing = below[0] if len(below) else None
    if o.s0 < lv.trigger:
        crossing = o.days[0]  # already below: water at the first chance
    rain = float(o.inputs.rain[1:, 0].sum())
    use = crop_use_mm(o, model, config)
    area = model.area_per_plant_m2
    act = activity(o, model, config)
    if crossing is None or not act.active:
        return Advice(
            needed=False, water_on=None, water_at=config.advice.preferred_local_time,
            dose_mm=0.0, dose_l_per_plant=0.0, crossing_day=None,
            possibly_on=possibly[0].date() if len(possibly) else None,
            levels=lv, rain_forecast_mm=rain, crop_use_mm=use, area_per_plant_m2=area,
            activity=act,
        )
    # Refill the level the no-watering path would have on the day of watering.
    deficit = max(lv.target - float(path["mean"].loc[crossing]), 0.0)
    dose_mm = deficit / model.params["rain_eff"]
    return Advice(
        needed=True, water_on=crossing.date(), water_at=config.advice.preferred_local_time,
        dose_mm=dose_mm, dose_l_per_plant=dose_mm * area, crossing_day=crossing.date(),
        possibly_on=possibly[0].date() if len(possibly) else None,
        levels=lv, rain_forecast_mm=rain, crop_use_mm=use, area_per_plant_m2=area,
        activity=act,
    )


def local_moment(day: date, at: time, timezone: str) -> datetime:
    return datetime.combine(day, at, tzinfo=ZoneInfo(timezone))
