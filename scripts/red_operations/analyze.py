"""Season KPIs, weather-normalised heat and lamp use, waste indicators.

usage: python analyze.py <lg5.parquet> <out_dir>
Writes daily.csv, kpis.csv, monthly.csv, normalised.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F  # noqa: E402
from prices import hourly_prices  # noqa: E402


def season_of(idx: pd.DatetimeIndex) -> pd.Series:
    """Jul→Jun seasons, labelled '2023/24'."""
    y = np.where(idx.month >= 7, idx.year, idx.year - 1)
    return pd.Series([f"{a}/{str(a + 1)[2:]}" for a in y], index=idx)


def crop_present(d: pd.DataFrame) -> pd.Series:
    """A crop is in the house when it is being irrigated and the slab is wet."""
    return (d["irr"].fillna(0) > 0.3) & (d["weight"].fillna(0) > 10)


def build(wide: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    f = F.five_min(wide)
    d = F.daily(f)
    d["dark_h"] = F.longest_dark_h(f)
    # lamps running under bright sun (> 300 W/m² outside)
    d["lamp_h_bright"] = (f["lamp_frac"] * (f["sun"] > 300)).groupby(
        f.index.normalize().tz_localize(None)
    ).sum() * F.DT_H
    # electricity cost of the lamps at DE-LU day-ahead
    p = hourly_prices()
    p.index = p.index.tz_convert(F.TZ)
    price5 = p.reindex(f.index, method="ffill", limit=11)
    day = f.index.normalize().tz_localize(None)
    d["lamp_eur"] = (f["lamp_kw"] * F.DT_H * price5 / 1000).groupby(day).sum(min_count=1)
    d["price_day"] = price5.groupby(day).mean()
    d["price_lamp"] = (price5 * f["lamp_frac"]).groupby(day).sum() / f["lamp_frac"].groupby(
        day
    ).sum()
    # heat delivered while the vents are open (> 10%), and the humid share of it
    vent_open = f["vent"] > 10
    humid = f["rh_in"] >= 85
    d["heat_mj_vent_dry"] = (
        (f["q_pipe"].where(vent_open & ~humid, 0)).groupby(day).sum() * 300 / 1e6
    )
    d["heat_mj_vent_humid"] = (
        (f["q_pipe"].where(vent_open & humid, 0)).groupby(day).sum() * 300 / 1e6
    )
    d["heat_mj_above_sp"] = (
        (f["q_pipe"].where(vent_open & (f["t_in"] > f["heat_sp"] + 1), 0)).groupby(day).sum()
        * 300
        / 1e6
    )
    # heat going in while the air is already > 1 °C above its heating setpoint:
    # not asked for by temperature — minimum pipe or humidity control
    d["heat_mj_unasked"] = (
        (f["q_pipe"].where(f["t_in"] > f["heat_sp"] + 1, 0)).groupby(day).sum() * 300 / 1e6
    )
    d["pipe_lower_min"] = f["pipe_lower"].groupby(day).quantile(0.05)
    d["season"] = season_of(d.index)
    d["month"] = d.index.month
    d["crop"] = crop_present(d)
    return f, d


KPI = {
    "sun_jcm2": "sum",
    "lamp_h": "sum",
    "lamp_kwh": "sum",
    "lamp_eur": "sum",
    "heat_mj": "sum",
    "gas_m3": "sum",
    "heat_index": "sum",
    "irr": "sum",
    "drain": "sum",
    "uptake": "sum",
    "h_rh90": "mean",
    "t24": "mean",
    "light_jcm2": "mean",
    "heat_mj_vent_dry": "sum",
    "heat_mj_vent_humid": "sum",
    "heat_mj_above_sp": "sum",
    "lamp_h_bright": "sum",
    "co2_lit": "mean",
    "heat_mj_unasked": "sum",
    "pipe_lower_min": "median",
}


def kpis(d: pd.DataFrame, months: list[int] | None = None) -> pd.DataFrame:
    x = d[d["coverage"] > 0.9]
    if months:
        x = x[x["month"].isin(months)]
    g = x.groupby("season")
    k = g.agg(KPI)
    k["days"] = g.size()
    k["drain_share"] = k["drain"] / k["irr"]
    k["heat_gj"] = k["heat_mj"] / 1000
    k["vent_heat_share"] = (k["heat_mj_vent_dry"] + k["heat_mj_vent_humid"]) / k["heat_mj"]
    k["unasked_share"] = k["heat_mj_unasked"] / k["heat_mj"]
    k["price_capture"] = (
        x.groupby("season")["lamp_eur"].sum()
        / (x["lamp_kwh"] * x["price_day"] / 1000).groupby(x["season"]).sum()
    )
    return k


BASE = "2023/24"  # first full season with LetsGrow


def normalised_heat(d: pd.DataFrame) -> dict:
    """Season effect on daily heat at equal weather (OLS, Oct-Apr, crop in)."""
    x = (
        d[
            (d["coverage"] > 0.9)
            & d["crop"]
            & (d["season"] >= BASE)
            & d["month"].isin([10, 11, 12, 1, 2, 3, 4])
        ]
        .dropna(subset=["heat_mj", "t_out", "t24", "sun_jcm2", "wind", "lamp_kwh"])
        .copy()
    )
    x["hdd"] = (18 - x["t_out"]).clip(lower=0)
    x["dT"] = x["t24"] - x["t_out"]
    res = {}
    # M1: weather only — what the season cost at the same weather (strategy included)
    m1 = smf.ols(
        "heat_mj ~ C(season, Treatment(reference=BASE)) + hdd + I(hdd*wind) + sun_jcm2 + C(month)",
        data=x,
    ).fit()
    # M2: at equal inside-outside difference — the envelope and operation
    m2 = smf.ols(
        "heat_mj ~ C(season, Treatment(reference=BASE)) + dT + I(dT*wind)"
        " + sun_jcm2 + lamp_kwh + C(month)",
        data=x,
    ).fit()
    for name, m in [("weather_only", m1), ("equal_dT", m2)]:
        eff = {
            k.split("[T.")[1].rstrip("]"): (m.params[k], m.bse[k])
            for k in m.params.index
            if k.startswith("C(season")
        }
        res[name] = {
            "base": BASE,
            "effects": eff,
            "r2": m.rsquared,
            "n": int(m.nobs),
            "mean_heat": float(x["heat_mj"].mean()),
            "coef": {k: float(v) for k, v in m.params.items() if not k.startswith("C(")},
        }
    return res


def normalised_lamps(d: pd.DataFrame) -> dict:
    x = d[
        (d["coverage"] > 0.9)
        & d["crop"]
        & (d["season"] >= BASE)
        & d["month"].isin([10, 11, 12, 1, 2, 3])
    ].dropna(subset=["lamp_h", "sun_jcm2"])
    m = smf.ols(
        "lamp_h ~ C(season, Treatment(reference=BASE)) + sun_jcm2 + I(sun_jcm2**2) + C(month)",
        data=x,
    ).fit()
    eff = {
        k.split("[T.")[1].rstrip("]"): (m.params[k], m.bse[k])
        for k in m.params.index
        if k.startswith("C(season")
    }
    return {"base": BASE, "effects": eff, "r2": m.rsquared, "n": int(m.nobs)}


def main() -> None:
    wide = pd.read_parquet(sys.argv[1])
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    f, d = build(wide)
    d.to_csv(out / "daily.csv")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    full = kpis(d)
    winter = kpis(d, [11, 12, 1, 2])
    full.to_csv(out / "kpis_season.csv")
    winter.to_csv(out / "kpis_winter.csv")
    print("== season (Jul-Jun) ==")
    print(full.round(2).T.to_string())
    print("== core winter (Nov-Feb) ==")
    print(winter.round(2).T.to_string())
    nh = normalised_heat(d)
    nl = normalised_lamps(d)
    (out / "normalised.json").write_text(
        json.dumps({"heat": nh, "lamps": nl}, indent=1, default=float)
    )
    print(json.dumps({"heat": nh, "lamps": nl}, indent=1, default=lambda v: round(float(v), 3)))
    x = d[d["coverage"] > 0.9]
    m = x.groupby([x.index.year, x.index.month]).agg(
        KPI
        | {
            "dark_h": "median",
            "heat_sp_min": "median",
            "t_min": "mean",
            "t_out": "mean",
            "vent": "mean",
        }
    )
    m.to_csv(out / "monthly.csv")


if __name__ == "__main__":
    main()
