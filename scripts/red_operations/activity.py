"""Crop activity from the substrate scale: transpiration in the clean windows.

The scale weighs ~4 m² of crop (calibrated from irrigation pulses: 4.0 kg per
l/m², stable 3.8-4.3 across nine quarters). Between irrigations its weight
loss is transpiration plus slab drainage; drainage stops ~30 min after a
pulse, so slots more than QUIET_AFTER_IRR after the last irrigation give
transpiration directly: E = -dW/dt / A.

The drain gauge lags the scale by hours, so the full hourly balance
(irrigation - drain - dW/A) is only usable summed over a whole day.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from wp6_data.red.greenhouse import transpiration as _T
from wp6_data.red.greenhouse.config import load_greenhouse

_CFG = load_greenhouse(Path(__file__).resolve().parents[2] / "src/wp6_data/red/metadata.yaml")

SCALE_AREA_M2 = _CFG.scale_area_m2


def quiet_rate(f: pd.DataFrame) -> pd.DataFrame:
    """Per 5-min slot: transpiration rate (l/m²/h) where the slab is undisturbed.

    Delegates to the app's ``red.greenhouse.transpiration.quiet_rate``, which
    the /greenhouse page uses too, so research and page cannot drift apart.
    """
    e = _T.quiet_rate(f["weight"], f["irr_cum"], _CFG)
    return pd.DataFrame({"E": e, "quiet": e.notna()})


def aligned(
    series: pd.Series, events: pd.DatetimeIndex, before_h: float, after_h: float
) -> pd.DataFrame:
    """Series values on a 5-min grid aligned at each event (rows = minutes from event)."""
    offs = pd.timedelta_range(
        -pd.Timedelta(hours=before_h), pd.Timedelta(hours=after_h), freq="5min"
    )
    cols = {}
    for ev in events:
        idx = ev + offs
        cols[ev] = series.reindex(idx).to_numpy()
    out = pd.DataFrame(cols, index=(offs.total_seconds() / 60).astype(int))
    return out


def lamp_on_events(f: pd.DataFrame, min_off_h: float = 4) -> pd.DatetimeIndex:
    """Switch-on moments in the dark (no sun) after at least ``min_off_h`` off."""
    on = f["lamp_frac"] > 0.5
    start = on & ~on.shift(1, fill_value=False)
    off_before = (~on).rolling(int(min_off_h * 12)).sum().shift(1) >= min_off_h * 12
    dark = f["sun"] <= 0
    return f.index[start & off_before & dark]


def sunrise_events(f: pd.DataFrame) -> pd.DatetimeIndex:
    """First slot of each day with outdoor radiation > 0, on lamp-free mornings."""
    lit = f["sun"] > 0
    first = lit & ~lit.shift(1, fill_value=False)
    ev = f.index[first & (f.index.hour >= 3) & (f.index.hour <= 9)]
    ev = pd.DatetimeIndex(pd.Series(ev).groupby(pd.Series(ev).dt.date).first().to_numpy())
    lamps = f["lamp_frac"].rolling(36).max()  # any lamp in the 3 h before
    return pd.DatetimeIndex([e for e in ev if lamps.get(e, 1) < 0.01])
