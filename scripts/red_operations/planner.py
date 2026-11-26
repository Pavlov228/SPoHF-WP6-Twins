"""The lamp planner now lives in ``wp6_data.red.greenhouse.planner``; this keeps
the research scripts' imports working and adds their hourly table.
"""

from __future__ import annotations

import pandas as pd

from wp6_data.red.greenhouse.planner import WAKE, plan_day, rolling_plan, starts

__all__ = ["WAKE", "hourly_table", "plan_day", "rolling_plan", "starts"]


def hourly_table(f5: pd.DataFrame, prices: pd.Series) -> pd.DataFrame:
    """Hourly sun (J/cm²), lamp share and price, local time."""
    h = pd.DataFrame(
        {
            "sun_j": f5["sun"].resample("1h").sum(min_count=6) * 300 / 1e4,
            "lamp": f5["lamp_frac"].resample("1h").mean(),
            "n": f5["sun"].resample("1h").count(),
        }
    )
    p = prices.copy()
    p.index = p.index.tz_convert(h.index.tz)
    h["price"] = p.reindex(h.index)
    return h
