"""Calibrate the greenhouse balance's two free parameters against LetsGrow.

1. Cover heat loss U (W/m² floor per K): dark, lamp-free, vents-closed nights,
   where pipe heat ≈ U·(T_in − T_out) + latent heat of transpiration.
2. Ventilation: hours with measured transpiration E (substrate scale) and the
   cover too warm to condense, where E ≈ g_v·(AH_in − AH_out); g_v regressed
   on vent opening × the wind/stack driving term.

usage: python calibrate_physics.py <lg5.parquet>
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import activity as A  # noqa: E402
import features as F  # noqa: E402

LAMBDA = 2.45e6  # J/kg


def hourly(f: pd.DataFrame) -> pd.DataFrame:
    q = A.quiet_rate(f)
    cols = [
        "t_in",
        "t_out",
        "ah_in",
        "ah_out",
        "wind",
        "sun",
        "lamp_frac",
        "q_pipe",
        "vent_lee",
        "vent_wind",
        "screen",
        "weight",
        "rh_in",
    ]
    h = f[cols].resample("1h").mean()
    h["E"] = q.E.resample("1h").mean()  # l/m²/h
    h["nE"] = q.E.resample("1h").count()
    h["dT"] = h.t_in - h.t_out
    h["dTdt"] = h.t_in.diff()
    h["open"] = h[["vent_lee", "vent_wind"]].mean(axis=1) / 100
    return h


def main() -> None:
    f = F.five_min(pd.read_parquet(sys.argv[1]))
    h = hourly(f)
    crop = h.weight > 10

    # 1. cover U on closed, dark nights with steady temperature
    n = h[
        crop
        & (h.sun <= 0)
        & (h.lamp_frac < 0.01)
        & (h.open < 0.02)
        & (h.dT > 3)
        & (h.dTdt.abs() < 0.3)
        & ((h.screen.isna()) | (h.screen < 5))
    ].dropna(subset=["q_pipe", "dT", "wind"])
    n = n.assign(latent=(n.E.fillna(0.04)) * LAMBDA / 3600)
    n = n.assign(y=(n.q_pipe - n.latent) / n.dT)
    m = smf.ols("y ~ wind", data=n[(n.y > 0) & (n.y < 30)]).fit()
    print(
        f"cover U = {m.params['Intercept']:.2f} + {m.params['wind']:.2f}·wind  W/m²K "
        f"(n={int(m.nobs)}, R²={m.rsquared:.2f}); median U {n.y.median():.2f}"
    )

    # 2. ventilation from the vapour balance (no condensation: cover warm)
    v = h[crop & h.E.notna() & (h.nE >= 6) & (h.ah_in - h.ah_out > 1.0) & (h.t_out > 12)].copy()
    v["gv_obs"] = v.E / 3600 / ((v.ah_in - v.ah_out) / 1000)  # m³/m²/s
    v["drive"] = np.sqrt(0.09 * v.wind**2 + 0.001 * v.dT.clip(lower=0) * 9.81 * 4)
    v["x"] = v.open * v.drive
    m2 = smf.ols("gv_obs ~ x", data=v[(v.gv_obs > 0) & (v.gv_obs < 0.05)]).fit()
    print(
        f"g_v = {m2.params['Intercept'] * 1e4:.2f}e-4 + {m2.params['x']:.4f}·open·drive  m³/m²/s "
        f"(n={int(m2.nobs)}, R²={m2.rsquared:.2f})"
    )
    for lo, hi in [(0, 0.02), (0.02, 0.1), (0.1, 0.3), (0.3, 0.6), (0.6, 1.01)]:
        s = v[(v.open >= lo) & (v.open < hi)]
        print(
            f"  open {lo:.2f}-{hi:.2f}: n={len(s):5d} median g_v {s.gv_obs.median() * 1e4:6.1f}e-4 "
            f"median drive {s.drive.median():.2f}"
        )


if __name__ == "__main__":
    main()
