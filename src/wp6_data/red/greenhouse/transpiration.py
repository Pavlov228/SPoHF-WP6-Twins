"""Crop transpiration from the substrate scale (CONTEXT "Crop activity").

The scale weighs about ``scale_area_m2`` of crop. Between irrigation pulses —
once the slab has stopped draining, ~40 minutes after the last one — its loss
of weight is the crop's transpiration: E = −ΔW / area. LetsGrow's own "water
uptake" is irrigation minus drain, which says how much but not when: its drain
gauge registers hours after the pulse that caused it.

Per day, the total comes from the water balance (irrigation − drain − ΔW/area),
which matches LetsGrow's uptake; the hourly shape comes from the scale where it
is clean and, in between, from the configured transpiration shape, scaled so
the day adds up.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from wp6_data.red.greenhouse.config import GreenhouseConfig

QUIET_AFTER_IRRIGATION = pd.Timedelta(minutes=40)
#: A reset is a running counter falling below this share of its previous value.
RESET_DROP = 0.5


def counter_totals(counter: pd.Series) -> pd.Series:
    """Per-day totals of a LetsGrow running counter, read at its resets.

    The reset clock is not fixed — the water counters reset at 23:00 through
    2025 and at 00:00 later — and the first slot after midnight still carries
    the previous day's total. So a reset is where the value falls below half
    the one before it, and that earlier value is the total, labelled with the
    date 30 minutes before it. Days with data but no reset take their maximum.
    Index: tz-aware; returned index: naive local dates.
    """
    s = counter.dropna()
    if s.empty:
        return pd.Series(dtype=float)
    prev = s.shift(1)
    reset = (s < RESET_DROP * prev) & (prev > 0.02)
    at = prev.index[np.flatnonzero(reset.to_numpy()) - 1]
    totals = pd.Series(
        prev[reset].to_numpy(), index=(at - pd.Timedelta(minutes=30)).normalize().tz_localize(None)
    )
    totals = totals.groupby(level=0).max()
    dmax = s.groupby(s.index.normalize().tz_localize(None)).max()
    return totals.reindex(dmax.index).fillna(dmax.where(dmax <= 0.02)).combine_first(totals)


def quiet_rate(weight: pd.Series, irrigation: pd.Series, cfg: GreenhouseConfig) -> pd.Series:
    """Transpiration (l/m²/h) per 5-minute slot where the slab is undisturbed.

    ``weight`` (kg) and ``irrigation`` (the running counter) on a regular
    5-minute index. NaN where a pulse, its drainage or a counter reset makes
    the weight change something other than transpiration.
    """
    d_irr = irrigation.diff()
    pulse = d_irr > 0.005
    # DatetimeIndex.where keeps the timezone even when no slot is a pulse
    last = pd.Series(irrigation.index.where(pulse.to_numpy()), index=irrigation.index).ffill()
    since = irrigation.index.to_series() - last
    quiet = ((since > QUIET_AFTER_IRRIGATION) | last.isna()) & ~(d_irr < -0.005)
    w = weight.where(quiet)
    slope = (w.shift(-3) - w.shift(3)) / 0.5  # kg/h over a centred 30 minutes
    ok = quiet & quiet.shift(-3, fill_value=False) & quiet.shift(3, fill_value=False)
    return (-slope / cfg.scale_area_m2).where(ok)


def shape(
    sun_w: pd.Series, lamp_w: pd.Series, hd: pd.Series, pipe_w: pd.Series, cfg: GreenhouseConfig
) -> pd.Series:
    """The configured transpiration shape, l/m²/h (never negative)."""
    t = cfg.transpiration
    ml = (
        t.intercept
        + t.per_sun_w * sun_w.fillna(0)
        + t.per_lamp_w * lamp_w.fillna(0)
        + t.per_hd * hd.fillna(0)
        + t.per_pipe_w * pipe_w.fillna(0)
    )
    return (ml / 1000).clip(lower=0)


def day_series(measured: pd.Series, modelled: pd.Series, total: float | None) -> pd.DataFrame:
    """One day's transpiration: measured where clean, the shape elsewhere.

    ``measured`` and ``modelled`` share one regular index (l/m²/h). When the
    water balance gives the day's ``total`` (l/m²), the modelled slots are
    scaled so the day sums to it — within a factor of three either way; past
    that the balance is more likely broken (a changeover, a counter fault)
    than the shape, and the shape is kept as is.
    Returns ``transpiration`` and ``source`` ("scale" or "model").
    """
    step_h = (
        (measured.index[1] - measured.index[0]).total_seconds() / 3600 if len(measured) > 1 else 1
    )
    have = measured.notna() & (measured >= 0)
    e = measured.where(have)
    gap = modelled.where(~have)
    scale = 1.0
    if total is not None and np.isfinite(total) and total > 0:
        rest = total - e.sum() * step_h
        modelled_rest = gap.sum() * step_h
        if modelled_rest > 0 and rest > 0:
            scale = float(np.clip(rest / modelled_rest, 1 / 3, 3))
    out = pd.DataFrame(index=measured.index)
    out["transpiration"] = e.fillna(gap * scale).clip(lower=0)
    out["source"] = np.where(have, "scale", "model")
    return out


#: A wake-up needs the plateau this far above the level before the trigger (l/m²/h).
MIN_WAKE_RISE = 0.010


def wake_time(e: pd.Series, trigger: pd.Timestamp) -> dict | None:
    """When the crop woke after ``trigger`` (lamps on, or sunrise): its half-response time.

    The level before (median of the 2 h before the trigger) and the plateau
    after (2-4 h after) bound the response; the crop is "awake" once a 15-minute
    mean of transpiration has covered half the way. The same measure as the
    research's switch-on curve (~45-60 minutes). None when there is no clear
    rise, or not enough clean readings to tell.
    """
    x = e.dropna()
    before = x[(x.index >= trigger - pd.Timedelta(hours=2)) & (x.index < trigger)]
    plateau = x[
        (x.index >= trigger + pd.Timedelta(hours=2)) & (x.index < trigger + pd.Timedelta(hours=4))
    ]
    if len(before) < 6 or len(plateau) < 6:
        return None
    lo, hi = float(before.median()), float(plateau.median())
    if hi - lo < MIN_WAKE_RISE:
        return None
    after = x[(x.index >= trigger) & (x.index < trigger + pd.Timedelta(hours=4))]
    smooth = after.rolling(3, min_periods=2).mean()
    hit = smooth[smooth >= lo + 0.5 * (hi - lo)]
    if hit.empty:
        return None
    return {"time": hit.index[0] - pd.Timedelta(minutes=5), "before": lo, "plateau": hi}
