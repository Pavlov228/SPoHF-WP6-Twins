"""Derived 5-minute and daily frames from the LetsGrow wide frame.

Everything physical that is *assumed* rather than measured is a named constant
at the top, so the report can state it and a sensitivity run can move it.
Research code: if any of this moves into `red/`, these constants belong in
`metadata.yaml`, next to `lamps` and `heating`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from wp6_data.red.heating import load_heat_output

_HEATING = load_heat_output(Path(__file__).resolve().parents[2] / "src/wp6_data/red/metadata.yaml")

TZ = "Europe/Amsterdam"
DT_H = 5 / 60  # one 5-minute slot, in hours

# --- lamps: the grower's Block B conversion sheets (as transcribed in the
# plant-activity research branch; LetsGrow reports one switch, always at 100%) ---
LAMP_PAR_NOMINAL = 257.3  # µmol/m²/s at 100%, both installations
LAMP_JCM2_PER_HOUR = 64.46  # outdoor-equivalent J/cm² per full lamp-hour
# ASSUMED: LED efficacy. Not on the sheets. 2.7-3.3 µmol/J spans the
# 2018-2023 top-light generations; 3.0 is the central guess.
LED_EFFICACY = 3.0  # µmol/J
LAMP_KW_PER_M2 = LAMP_PAR_NOMINAL / LED_EFFICACY / 1000  # kW/m² at 100%

# --- heating pipes: ASSUMED geometry (GreenLight/Vanthoor reference greenhouse) ---
PIPES = {
    "lower": {"col": "climate:lg_pipe_lower", "d": 0.051, "l": 1.25},
    "growth": {"col": "climate:lg_pipe_growth", "d": 0.035, "l": 1.655},
}
MAX_PIPE = _HEATING.max_pipe  # °C, above this is a sensor fault
EPS_PIPE = 0.88
SIGMA = 5.670e-8

# --- gas: boiler heat per m³ natural gas (Groningen-quality LHV × efficiency) ---
GAS_MJ_PER_M3 = 31.65
BOILER_EFF = 0.92

SHORT = {
    "climate:lg_temp_greenhouse": "t_in",
    "climate:lg_rh_greenhouse": "rh_in",
    "climate:lg_abs_hum_greenhouse": "ah_in",
    "climate:lg_hum_deficit_greenhouse": "hd_in",
    "climate:lg_vpd": "vpd",
    "climate:lg_co2": "co2",
    "climate:lg_co2_kg": "co2_kg",
    "climate:lg_heating_setpoint": "heat_sp",
    "climate:lg_pipe_lower": "pipe_lower",
    "climate:lg_pipe_lower_calc": "pipe_lower_calc",
    "climate:lg_pipe_growth": "pipe_growth",
    "climate:lg_pipe_growth_calc": "pipe_growth_calc",
    "climate:lg_vent_lee": "vent_lee",
    "climate:lg_vent_wind": "vent_wind",
    "climate:lg_vent_setpoint": "vent_sp",
    "climate:lg_energy_curtain": "screen",
    "climate:lg_blackout_curtain": "blackout",
    "climate:lg_fans": "fans",
    "climate:lg_par": "par_in",
    "grow-light:lg_assim_light_on": "lamp_on",
    "grow-light:lg_assim_light_pct": "lamp_pct",
    "grow-light:lg_led1_on": "led1",
    "water:lg_substrate_weight": "weight",
    "water:lg_irrigation": "irr_cum",
    "water:lg_drain": "drain_cum",
    "water:lg_water_uptake": "uptake_cum",
    "water:lg_irrigation_starts": "starts_cum",
    "water:lg_drain_pct": "drain_pct",
    "water:lg_irrigation_ec": "ec_irr",
    "water:lg_drain_ec": "ec_drain",
    "water:lg_irrigation_ph": "ph_irr",
    "water:lg_drain_ph": "ph_drain",
    "weather-station:lg_temp_outside": "t_out",
    "weather-station:lg_rh_outside": "rh_out",
    "weather-station:lg_abs_hum_outside": "ah_out",
    "weather-station:lg_rad": "rad",
    "weather-station:lg_rad_sum": "rad_sum_cum",
    "weather-station:lg_wind_speed": "wind",
    "weather-station:lg_rain": "rain",
}


def pipe_heat_wm2(pipe: pd.Series, air: pd.Series, d: float, length: float) -> pd.Series:
    """Heat a pipe circuit gives the greenhouse, W per m² of floor.

    Convection after Bot (1983) as in Vanthoor (2011)/GreenLight:
    1.99·π·d·L·ΔT^1.32, plus long-wave radiation to surroundings at air
    temperature: ε·σ·π·d·L·(Tp⁴ − Ta⁴). Geometry is ASSUMED (see constants).
    """
    pipe = pipe.where(pipe <= MAX_PIPE)
    dT = (pipe - air).clip(lower=0)
    conv = 1.99 * np.pi * d * length * dT**1.32
    tp, ta = pipe + 273.15, air + 273.15
    rad = (EPS_PIPE * SIGMA * np.pi * d * length * (tp**4 - ta**4)).clip(lower=0)
    return conv + rad


def five_min(wide: pd.DataFrame) -> pd.DataFrame:
    f = wide.rename(columns=SHORT)[[c for c in SHORT.values() if c in wide.rename(columns=SHORT)]]
    f = f.copy()
    f.index = f.index.tz_convert(TZ)
    f["lamp_frac"] = (f["lamp_on"].clip(0, 1) * f["lamp_pct"].clip(0, 100) / 100).astype(float)
    q = 0.0
    idx = 0.0
    for name, p in PIPES.items():
        col = SHORT[p["col"]]
        f[f"q_{name}"] = pipe_heat_wm2(f[col], f["t_in"], p["d"], p["l"])
        q = q + f[f"q_{name}"]
        idx = idx + (f[col].where(f[col] <= MAX_PIPE) - f["t_in"]).clip(lower=0) ** 1.25
    f["q_pipe"] = q  # W/m² estimate
    f["heat_index"] = idx  # red's /energy index, per reading
    f["vent"] = f[["vent_lee", "vent_wind"]].mean(axis=1)
    f["sun"] = f["rad"].clip(lower=0)
    f["lamp_kw"] = f["lamp_frac"] * LAMP_KW_PER_M2
    return f


def counter_totals(c: pd.Series) -> pd.Series:
    """Per-day totals of a LetsGrow running counter, read at its resets.

    The reset clock is not fixed: the water counters reset at 23:00 in 2025 and
    at 00:00 in late 2026, and the first slot after midnight still carries the
    previous day's total. So a reset is found where the value falls below half
    of the one before it, and that earlier value is the period's total,
    labelled with the date 30 minutes before it (23:00 → that day, 00:00 → the
    day before). Days with data but no reset (a counter that stayed at zero)
    take the day's maximum.
    """
    s = c.dropna()
    prev = s.shift(1)
    reset = (s < 0.5 * prev) & (prev > 0.02)
    at = prev.index[np.flatnonzero(reset.to_numpy()) - 1]
    totals = pd.Series(
        prev[reset].to_numpy(), index=(at - pd.Timedelta(minutes=30)).normalize().tz_localize(None)
    )
    totals = totals.groupby(level=0).max()
    dmax = s.groupby(s.index.normalize().tz_localize(None)).max()
    return totals.reindex(dmax.index).fillna(dmax.where(dmax <= 0.02))


def daily(f: pd.DataFrame) -> pd.DataFrame:
    day = f.index.normalize().tz_localize(None)
    g = f.groupby(day)
    n = g["t_in"].count()
    out = pd.DataFrame(index=g.size().index)
    out["coverage"] = n / 288
    out["sun_jcm2"] = g["sun"].sum() * 300 / 1e4  # W/m² × s → J/cm²
    out["rad_sum_lg"] = counter_totals(f["rad_sum_cum"])
    out["lamp_h"] = g["lamp_frac"].sum() * DT_H
    out["lamp_jcm2"] = out["lamp_h"] * LAMP_JCM2_PER_HOUR
    out["light_jcm2"] = out["sun_jcm2"] + out["lamp_jcm2"]
    out["lamp_kwh"] = g["lamp_kw"].sum() * DT_H
    out["heat_mj"] = g["q_pipe"].sum() * 300 / 1e6  # MJ/m²/day
    out["heat_index"] = g["heat_index"].sum() * DT_H
    out["gas_m3"] = out["heat_mj"] / (GAS_MJ_PER_M3 * BOILER_EFF)
    out["t24"] = g["t_in"].mean()
    out["t_min"] = g["t_in"].min()
    out["t_max"] = g["t_in"].max()
    out["t_out"] = g["t_out"].mean()
    out["t_out_min"] = g["t_out"].min()
    out["wind"] = g["wind"].mean()
    out["heat_sp_min"] = g["heat_sp"].min()
    out["heat_sp_mean"] = g["heat_sp"].mean()
    out["vent"] = g["vent"].mean()
    out["rh_in"] = g["rh_in"].mean()
    out["hd_in"] = g["hd_in"].mean()
    out["h_rh90"] = (f["rh_in"] >= 90).groupby(day).sum() * DT_H
    out["h_rh85"] = (f["rh_in"] >= 85).groupby(day).sum() * DT_H
    lit = (f["sun"] > 0) | (f["lamp_frac"] > 0)
    out["light_h"] = lit.groupby(day).sum() * DT_H
    out["co2_lit"] = f["co2"].where(lit).groupby(day).mean()
    out["co2_kg"] = g["co2_kg"].mean()
    hv = (f["pipe_lower"] > 35) & (f["vent_lee"] > 20)
    out["h_heat_vent"] = hv.groupby(day).sum() * DT_H
    out["heat_mj_while_vent"] = (
        (f["q_pipe"].where(f["vent"] > 10, 0)).groupby(day).sum() * 300 / 1e6
    )
    out["screen"] = g["screen"].mean() if "screen" in f else np.nan
    out["dT_in_out"] = out["t24"] - out["t_out"]
    # water: totals read at the counters' own resets
    out["irr"] = counter_totals(f["irr_cum"])
    out["drain"] = counter_totals(f["drain_cum"])
    out["starts"] = counter_totals(f["starts_cum"])
    out["uptake"] = out["irr"] - out["drain"]
    out["ec_irr"] = g["ec_irr"].mean()
    out["ec_drain"] = g["ec_drain"].mean()
    out["weight"] = g["weight"].median()
    out["drain_share"] = out["drain"] / out["irr"]
    return out


def longest_dark_h(f: pd.DataFrame) -> pd.Series:
    """Longest unbroken spell of no sun and no lamp, credited to the day it ends."""
    dark = ~((f["sun"] > 0) | (f["lamp_frac"] > 0))
    run_id = (dark != dark.shift()).cumsum()
    runs = dark.groupby(run_id).agg(["first", "size"])
    ends = f.index.to_series().groupby(run_id).last()
    runs = runs[runs["first"]]
    s = pd.Series(
        runs["size"].to_numpy() * DT_H,
        index=ends.loc[runs.index].dt.normalize().dt.tz_localize(None),
    )
    return s.groupby(level=0).max()
