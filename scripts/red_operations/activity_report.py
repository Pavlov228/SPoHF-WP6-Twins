"""Crop-activity numbers for §3: scale calibration, event studies, drivers.

usage: python activity_report.py <lg5.parquet> <daily.csv> <out.json>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import activity as A  # noqa: E402
import features as F  # noqa: E402


def scale_area(f: pd.DataFrame) -> pd.DataFrame:
    """kg of weight gain per l/m² irrigated, per quarter, corrected for transpiration."""
    dI = f["irr_cum"].diff().clip(lower=0)
    dD = f["drain_cum"].diff().clip(lower=0)
    dW = f["weight"].diff()
    quiet = (dI.rolling(7, center=True).sum() == 0) & (dD.rolling(7, center=True).sum() == 0)
    loss = (-dW.where(quiet)).rolling(19, center=True, min_periods=4).median()
    r = (
        pd.DataFrame({"dI": dI, "dD": dD, "dW": dW, "loss": loss, "sun": f["sun"]})
        .resample("15min")
        .agg({"dI": "sum", "dD": "sum", "dW": "sum", "loss": "mean", "sun": "mean"})
    )
    x = r[(r.dI > 0.05) & (r.dI < 0.8) & (r.dD < 0.005) & r.loss.notna() & (r.sun < 30)].copy()
    x["dWc"] = x.dW + 3 * x.loss
    q = x.index.tz_localize(None).to_period("Q")
    return x.groupby(q).apply(
        lambda g: pd.Series({"area": (g.dWc * g.dI).sum() / (g.dI**2).sum(), "n": len(g)})
    )


def event_profile(series: pd.Series, events: pd.DatetimeIndex, before: float, after: float) -> dict:
    al = A.aligned(series, events, before, after) * 1000
    med = al.median(axis=1)
    return {
        int(k): (None if pd.isna(v) else round(float(v), 1)) for k, v in med.iloc[::6].items()
    } | {"n": len(events)}


def main() -> None:
    f = F.five_min(pd.read_parquet(sys.argv[1]))
    f = f[f["weight"].notna()]
    d = pd.read_csv(sys.argv[2], index_col=0, parse_dates=True)
    out: dict = {}
    sa = scale_area(f)
    out["scale_area_by_quarter"] = {str(k): round(float(v), 2) for k, v in sa["area"].items()}
    q = A.quiet_rate(f)
    ev_on = A.lamp_on_events(f)
    on = f["lamp_frac"] > 0.5
    ev_off = f.index[((~on) & on.shift(1, fill_value=False)) & (f["sun"] <= 0)]
    ev_sr = A.sunrise_events(f)
    ev_sr = ev_sr[np.isin(ev_sr.month, [4, 5, 6, 7, 8])]
    out["lamp_on"] = event_profile(q.E, ev_on, 3, 4)
    out["lamp_off"] = event_profile(q.E, ev_off, 2, 4)
    out["sunrise"] = event_profile(q.E, ev_sr, 2, 2)
    for col in ["t_in", "rh_in", "hd_in", "q_pipe", "vent", "co2"]:
        out[f"lamp_on_{col}"] = {
            int(k): round(float(v), 2)
            for k, v in A.aligned(f[col], ev_on, 3, 4).median(axis=1).iloc[::6].items()
        }
    # hourly drivers
    grp = (on != on.shift()).cumsum()
    since = on.groupby(grp).cumcount() * 5 * on
    x = pd.DataFrame(
        {
            "E": q.E * 1000,
            "sun": f.sun,
            "lamp": f.lamp_frac,
            "hd": f.hd_in,
            "t": f.t_in,
            "qp": f.q_pipe,
            "since": since,
            "weight": f.weight,
        }
    )
    h = x.resample("1h").mean()
    h["n"] = q.E.resample("1h").count()
    h = h[(h.n >= 6) & (h.weight > 10)].dropna(subset=["E", "sun", "lamp", "hd", "qp"])
    h["warm"] = h.since.clip(upper=180) / 180
    dark = h[(h.sun < 1) & (h.lamp < 0.05)]
    lit = h[(h.sun < 1) & (h.lamp > 0.95)]
    out["dark_mean"] = round(float(dark.E.mean()), 1)
    out["lampnight_mean"] = round(float(lit.E.mean()), 1)
    out["dark_by_pipe_heat"] = {
        str(k): round(float(v), 1)
        for k, v in dark.groupby(pd.cut(dark.qp, [0, 20, 40, 60, 80, 100, 150]), observed=True)
        .E.median()
        .items()
    }
    out["lamp_by_hd"] = {
        str(k): round(float(v), 1)
        for k, v in lit.groupby(pd.cut(lit.hd, [0, 1, 2, 3, 4, 6, 10]), observed=True)
        .E.median()
        .items()
    }
    m = smf.ols("E ~ hd + qp + t + warm", data=lit).fit()
    out["lampnight_ols"] = {k: round(float(v), 2) for k, v in m.params.items()} | {
        "r2": round(m.rsquared, 2)
    }
    m = smf.ols("E ~ hd + qp + t", data=dark).fit()
    out["dark_ols"] = {k: round(float(v), 3) for k, v in m.params.items()} | {
        "r2": round(m.rsquared, 2)
    }
    # daily uptake per joule
    dd = d[(d.coverage > 0.9) & d.crop].dropna(
        subset=["uptake", "sun_jcm2", "lamp_jcm2", "hd_in", "t24"]
    )
    dd = dd[(dd.uptake > 0.3) & (dd.uptake < 10)]
    m = smf.ols("uptake ~ sun_jcm2 + lamp_jcm2 + hd_in + t24 + C(season)", data=dd).fit()
    out["daily_ols"] = {k: round(float(v), 5) for k, v in m.params.items()} | {
        "r2": round(m.rsquared, 2),
        "n": int(m.nobs),
    }
    dd = dd.assign(ml_per_j=dd.uptake / dd.light_jcm2 * 1000)
    out["ml_per_j_by_month"] = {
        f"{y}-{mo:02d}": round(float(v), 2)
        for (y, mo), v in dd.groupby([dd.index.year, dd.index.month]).ml_per_j.median().items()
    }
    Path(sys.argv[3]).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
