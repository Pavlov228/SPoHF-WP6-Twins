"""Evaluate the lamp planner against what the grower did, day by day.

Scenarios (all priced at DE-LU day-ahead, lamp power from features.LAMP_KW_PER_M2):
  actual     what LetsGrow switched
  same       the same effective light as the grower gave, cheapest whole hours, dark ≥ 6 h
  t1000      1000 J/cm² effective, perfect sun knowledge
  t1000_fc   1000 J/cm², re-planned hourly on yesterday's sun as forecast, scored on the real sun

usage: python run_planner.py <lg5.parquet> <out.csv>
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F  # noqa: E402
import planner as P  # noqa: E402
from prices import hourly_prices  # noqa: E402

K = F.LAMP_JCM2_PER_HOUR
KW = F.LAMP_KW_PER_M2


def eff_light(sun: np.ndarray, u: np.ndarray, prev_on: bool) -> float:
    return float(sun.sum() + K * (u.sum() - P.WAKE * P.starts(u, prev_on)))


def main() -> None:
    wide = pd.read_parquet(sys.argv[1])
    target = float(sys.argv[3]) if len(sys.argv) > 3 else 1000.0
    f5 = F.five_min(wide)
    h = P.hourly_table(f5, hourly_prices())
    rows = []
    prev = {"actual": False, "same": False, "t1000": False, "t1000_fc": False}
    last_sun = None
    for day, g in h.groupby(h.index.normalize()):
        ok = (
            len(g) == 24
            and g["n"].min() >= 10
            and not g["price"].isna().any()
            and not g["lamp"].isna().any()
        )
        if not ok:
            last_sun = None
            prev = dict.fromkeys(prev, False)
            continue
        sun = g["sun_j"].fillna(0).to_numpy()
        price = g["price"].to_numpy()
        lamp = (g["lamp"].to_numpy() > 0.5).astype(float)
        if g["lamp"].sum() < 0.5:
            prev = dict.fromkeys(prev, False)
            last_sun = sun
            continue  # lamp season only
        r = {
            "date": day.tz_localize(None),
            "sun": sun.sum(),
            "price_mean": price.mean(),
            "lamp_h_actual": g["lamp"].sum(),
            "cost_actual": KW * float(np.dot(g["lamp"].to_numpy(), price)) / 1000,
            "light_actual": eff_light(sun, lamp, prev["actual"]),
        }
        dark = (sun <= 0.5) & (lamp < 0.5)
        run = best = 0
        for x in dark:
            run = run + 1 if x else 0
            best = max(best, run)
        r["dark_actual"] = best
        plans = {
            "same": (r["light_actual"], sun),
            "t1000": (max(target, sun.sum()), sun),
        }
        if last_sun is not None:
            plans["t1000_fc"] = (target, last_sun)
        for name, (tgt, sun_plan) in plans.items():
            if name == "t1000_fc":
                u = P.rolling_plan(sun, sun_plan, price, target, K, prev_on=prev[name])
            else:
                u = P.plan_day(sun_plan, price, tgt, K, dark_min=6, prev_on=prev[name])
            if u is None:
                r[f"cost_{name}"] = np.nan
                r[f"lamp_h_{name}"] = np.nan
                r[f"light_{name}"] = np.nan
                prev[name] = False
                continue
            r[f"cost_{name}"] = KW * float(np.dot(u, price)) / 1000
            r[f"lamp_h_{name}"] = u.sum()
            r[f"light_{name}"] = eff_light(sun, u, prev[name])  # scored on the real sun
            r[f"starts_{name}"] = P.starts(u, prev[name])
            prev[name] = bool(u[-1] > 0.5)
        r["starts_actual"] = P.starts(lamp, prev["actual"])
        prev["actual"] = bool(lamp[-1] > 0.5)
        last_sun = sun
        rows.append(r)
    out = pd.DataFrame(rows).set_index("date")
    out.to_csv(sys.argv[2])
    print(out.describe().T.round(3).to_string())


if __name__ == "__main__":
    main()
