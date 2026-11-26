"""One day of LetsGrow, shaped for the balance (CONTEXT "LetsGrow").

LetsGrow logs every 5 minutes. A day is read with a margin either side — the
running counters reset around midnight, and the transpiration slope needs the
readings around each edge — kept at 5 minutes for the scale, and averaged to
15-minute steps for the balance and the page.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from wp6_data.red.greenhouse import transpiration as T
from wp6_data.red.greenhouse.config import GreenhouseConfig
from wp6_data.red.greenhouse.physics import pipe_heat, sun_inside

STEP = "15min"
MARGIN = timedelta(hours=3)

#: (device role, LetsGrow tag) → column name on the page's frames.
COLUMNS: dict[tuple[str, str], str] = {
    ("climate", "lg_temp_greenhouse"): "t_in",
    ("climate", "lg_rh_greenhouse"): "rh_in",
    ("climate", "lg_abs_hum_greenhouse"): "ah_in",
    ("climate", "lg_hum_deficit_greenhouse"): "hd_in",
    ("climate", "lg_co2"): "co2",
    ("climate", "lg_heating_setpoint"): "heat_sp",
    ("climate", "lg_vent_setpoint"): "vent_sp",
    ("climate", "lg_vent_lee"): "vent_lee",
    ("climate", "lg_vent_wind"): "vent_wind",
    ("climate", "lg_energy_curtain"): "screen_pct",
    ("climate", "lg_blackout_curtain"): "blackout_pct",
    ("climate", "lg_pipe_lower"): "lg_pipe_lower",
    ("climate", "lg_pipe_growth"): "lg_pipe_growth",
    ("climate", "lg_pipe_lower_calc"): "lg_pipe_lower_calc",
    ("climate", "lg_pipe_growth_calc"): "lg_pipe_growth_calc",
    ("lamps", "lg_assim_light_on"): "lamp_on",
    ("lamps", "lg_assim_light_pct"): "lamp_pct",
    ("water", "lg_substrate_weight"): "weight",
    ("water", "lg_irrigation"): "irrigation",
    ("water", "lg_drain"): "drain",
    ("water", "lg_irrigation_starts"): "starts",
    ("water", "lg_irrigation_ec"): "ec_irrigation",
    ("water", "lg_drain_ec"): "ec_drain",
    ("weather", "lg_temp_outside"): "t_out",
    ("weather", "lg_rh_outside"): "rh_out",
    ("weather", "lg_abs_hum_outside"): "ah_out",
    ("weather", "lg_rad"): "radiation",
    ("weather", "lg_wind_speed"): "wind",
    ("weather", "lg_wind_dir"): "wind_dir",
    ("weather", "lg_rain"): "rain",
}


def window(day: date, tz: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """The local day with its margins, as tz-aware timestamps."""
    start = pd.Timestamp(day, tz=tz)
    return start - MARGIN, start + timedelta(days=1) + MARGIN


def five_minute(long: pd.DataFrame, cfg: GreenhouseConfig, tz: str) -> pd.DataFrame:
    """``fetch_data_tsdb``'s long rows → one 5-minute column per quantity, local time."""
    role = {getattr(cfg.devices, r): r for r in ("climate", "lamps", "water", "weather")}
    if long.empty:
        return pd.DataFrame(columns=list(COLUMNS.values()))
    df = long.assign(role=long["device"].map(role)).dropna(subset=["role"])
    df["column"] = [COLUMNS.get((r, s)) for r, s in zip(df["role"], df["sensor"], strict=True)]
    df = df.dropna(subset=["column"])
    df["time"] = df["time"].dt.floor("5min")
    wide = df.pivot_table(index="time", columns="column", values="value", aggfunc="mean")
    wide = wide.reindex(columns=list(COLUMNS.values()))
    wide.index = wide.index.tz_convert(tz)
    full = pd.date_range(wide.index.min(), wide.index.max(), freq="5min")
    return wide.reindex(full)


def day_frame(f5: pd.DataFrame, day: date, cfg: GreenhouseConfig, tz: str) -> pd.DataFrame:
    """The 15-minute state of ``day`` for the balance, with transpiration filled in.

    Adds ``lamps``, ``vent`` and ``screen`` as 0-1 shares, ``transpiration`` with
    its ``transpiration_source``, and ``screen_logged`` (False before LetsGrow
    delivered the curtain, when the screen is drawn open and said to be unknown).
    """
    f = f5.copy()
    f["lamps"] = (f["lamp_on"].clip(0, 1) * f["lamp_pct"].clip(0, 100) / 100).astype(float)
    f["vent"] = f[["vent_lee", "vent_wind"]].mean(axis=1) / 100
    f["screen_logged"] = f["screen_pct"].notna()
    f["screen"] = (f["screen_pct"] / 100).clip(0, 1)
    f["radiation"] = f["radiation"].clip(lower=0)
    # transpiration: clean scale slots, and the shape everywhere
    measured = T.quiet_rate(f["weight"], f["irrigation"], cfg)
    pipes = sum(
        pd.Series(pipe_heat(f[p.tag], f["t_in"], p.diameter, p.length), index=f.index).fillna(0)
        for p in cfg.pipes
    )
    shape = T.shape(
        pd.Series(sun_inside(f["radiation"], f["screen"].fillna(0), cfg), index=f.index),
        f["lamps"] * cfg.lamps.power_w,
        f["hd_in"],
        pipes,
        cfg,
    )
    start = pd.Timestamp(day, tz=tz)
    end = start + timedelta(days=1)
    total = water_balance(f, day, cfg)["transpiration"]
    inside = (f.index >= start) & (f.index < end)
    e = T.day_series(measured[inside], shape[inside], total)
    agg = {c: "mean" for c in f.columns if c not in ("screen_logged",)}
    agg["screen_logged"] = "max"
    s = f[inside].resample(STEP).agg(agg)
    s["transpiration"] = e["transpiration"].resample(STEP).mean()
    s["transpiration_source"] = (
        (e["source"] == "scale")
        .resample(STEP)
        .mean()
        .map(lambda x: "scale" if x >= 0.5 else "model")
    )
    s["screen"] = s["screen"].where(s["screen_logged"].astype(bool), 0.0)
    return s


def water_balance(f5: pd.DataFrame, day: date, cfg: GreenhouseConfig) -> dict[str, float | None]:
    """The day's irrigation, drain, starts and transpiration (l/m²; None when unknown).

    Transpiration = irrigation − drain − the scale's change in water over the
    day; implausible results (a changeover, a counter fault) are dropped.
    """
    key = pd.Timestamp(day)
    out: dict[str, float | None] = {}
    for col in ("irrigation", "drain", "starts"):
        totals = T.counter_totals(f5[col]) if col in f5 else pd.Series(dtype=float)
        out[col] = float(totals[key]) if key in totals.index and np.isfinite(totals[key]) else None
    tz = f5.index.tz
    start = pd.Timestamp(day, tz=tz)
    w = f5["weight"].dropna()
    w0 = w[(w.index >= start - timedelta(minutes=30)) & (w.index <= start + timedelta(minutes=30))]
    w1 = w[
        (w.index >= start + timedelta(hours=23, minutes=30))
        & (w.index <= start + timedelta(days=1, minutes=30))
    ]
    total = None
    if out["irrigation"] is not None and out["drain"] is not None and len(w0) and len(w1):
        total = out["irrigation"] - out["drain"] - (w1.median() - w0.median()) / cfg.scale_area_m2
        if not 0.2 < total < 12:
            total = None
    out["transpiration"] = total
    return out
