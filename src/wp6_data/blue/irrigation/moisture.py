"""One fused soil-moisture stream from blue's disagreeing soil probes.

The probes agree on *how* the soil wets and dries but not on the level: offsets of
about 10 %VWC, and readings far below what peat holds, which points at a
mineral-soil calibration (docs/research/blueberry-irrigation.md §2, §4). So no
absolute value is trusted. Each probe is masked mechanically, put on its own
0..1 scale, and the scaled probes are fused by median.

The pipeline is pure over an hourly wide frame (one column per probe), so it is
tested on synthetic probes; :func:`load_fused` is the only part that touches
the database.

Mask reasons, applied in this order (the first one wins):

- ``missing``: no reading that hour.
- ``excluded``: a manual override from the config.
- ``frozen``: the probe's own soil temperature is near freezing. Ice reads
  like dry soil to a permittivity probe, so a frost looks like drying and a
  thaw like rain. Neither is water arriving or leaving.
- ``dropout``: below the floor, or within the padding of such a run. Drying
  peat shrinks away from the probe, so these are lost contact, not dry soil.
- ``spike``: a one-off jump that a centred Hampel filter rejects. A rain step
  is not a spike, because the level stays where it jumped to.
- ``disagrees``: the probe's day-to-day changes stopped tracking the median
  of the other probes over the consensus window.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from wp6_data.blue.irrigation.config import Exclusion, FusionParams, IrrigationConfig
    from wp6_data.shared.twin import SensorDataProvider

OK = ""
MISSING = "missing"
EXCLUDED = "excluded"
FROZEN = "frozen"
DROPOUT = "dropout"
SPIKE = "spike"
DISAGREES = "disagrees"

#: Scale factor turning a median absolute deviation into a normal-equivalent SD.
_MAD_TO_SD = 1.4826


@dataclass(frozen=True)
class FusedMoisture:
    """The fused stream plus everything needed to inspect how it was made."""

    #: Median of the scaled, unmasked probes; 0 = a probe's dry reference,
    #: 1 = its wet reference.
    fused: pd.Series
    #: How many probes contributed each hour.
    coverage: pd.Series
    #: Spread across the contributing probes (25th/75th percentile).
    low: pd.Series
    high: pd.Series
    #: Per-probe scaled values, NaN where masked.
    scaled: pd.DataFrame
    #: Per-probe mask reason, ``OK`` where the reading was used.
    reasons: pd.DataFrame
    #: Per-probe (dry, wet) reference in the probe's own unit.
    references: dict[str, tuple[float, float]]
    #: Per-probe hourly readings as received, in the probe's own unit, so every
    #: reading (used or dropped) stays inspectable.
    raw: pd.DataFrame
    #: Per-probe hourly soil temperature (°C), when it was loaded.
    soil_temperature: pd.DataFrame | None = None


def mask_readings(
    raw: pd.DataFrame,
    params: FusionParams,
    exclusions: tuple[Exclusion, ...] = (),
    soil_temperature: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Per-hour mask reason for every probe; ``OK`` where the reading is kept.

    *soil_temperature* is the same shape as *raw* (each probe's own
    temperature). A probe without a temperature reading is not masked as frozen.
    """
    reasons = pd.DataFrame(OK, index=raw.index, columns=raw.columns)

    def mark(condition: pd.DataFrame, reason: str) -> None:
        # Earlier reasons win: only label cells nothing has claimed yet.
        reasons[(reasons == OK) & condition] = reason

    mark(raw.isna(), MISSING)

    if exclusions:
        excluded = pd.DataFrame(False, index=raw.index, columns=raw.columns)
        for e in exclusions:
            if e.probe in excluded.columns:
                excluded.loc[e.start:e.end, e.probe] = True
        mark(excluded, EXCLUDED)

    if soil_temperature is not None:
        temperature = soil_temperature.reindex(index=raw.index, columns=raw.columns)
        mark(temperature < params.frozen_below_c, FROZEN)

    below = (raw < params.floor).astype(float)
    pad = 2 * params.dropout_pad_hours + 1
    mark(below.rolling(pad, center=True, min_periods=1).max() > 0, DROPOUT)

    kept = raw.where(reasons == OK)
    window = params.spike_window_hours
    centre = kept.rolling(window, center=True, min_periods=window // 2 + 1).median()
    deviation = (kept - centre).abs()
    spread = deviation.rolling(window, center=True, min_periods=window // 2 + 1).median()
    spread = (spread * _MAD_TO_SD).clip(lower=params.spike_min_spread)
    mark(deviation > params.spike_k * spread, SPIKE)

    return reasons


def references(
    raw: pd.DataFrame, reasons: pd.DataFrame, params: FusionParams,
) -> dict[str, tuple[float, float]]:
    """Each probe's dry and wet reference: low and high quantiles of its clean history."""
    clean = raw.where(reasons == OK)
    refs: dict[str, tuple[float, float]] = {}
    for probe in clean.columns:
        values = clean[probe].dropna()
        if values.empty:
            continue
        lo = float(values.quantile(params.reference_low_quantile))
        hi = float(values.quantile(params.reference_high_quantile))
        if hi > lo:
            refs[probe] = (lo, hi)
    return refs


def scale(raw: pd.DataFrame, refs: dict[str, tuple[float, float]]) -> pd.DataFrame:
    """Map each probe onto 0 (its dry reference) .. 1 (its wet reference)."""
    scaled = pd.DataFrame(np.nan, index=raw.index, columns=raw.columns)
    for probe, (lo, hi) in refs.items():
        scaled[probe] = (raw[probe] - lo) / (hi - lo)
    return scaled


def mask_disagreement(
    scaled: pd.DataFrame, reasons: pd.DataFrame, params: FusionParams,
) -> pd.DataFrame:
    """Mark the stretches where a probe's daily changes stop tracking the rest.

    A level offset is fine, since scaling absorbs it. A probe that rises when
    the others fall is not measuring the same soil water. Each probe is compared
    against the median of the *other* probes, so one bad probe cannot pull the
    reference towards itself.
    """
    reasons = reasons.copy()
    clean = scaled.where(reasons == OK)
    daily = clean.resample("D").median()
    changes = daily.diff()
    window = params.consensus_window_days
    for probe in changes.columns:
        others = changes.drop(columns=probe)
        enough = others.notna().sum(axis=1) >= params.consensus_min_probes - 1
        consensus = others.median(axis=1).where(enough)
        corr = changes[probe].rolling(window, min_periods=window // 2).corr(consensus)
        bad_days = corr.index[corr < params.consensus_min_corr]
        if len(bad_days) == 0:
            continue
        hours = reasons.index.floor("D").isin(bad_days)
        cell = reasons[probe]
        reasons.loc[hours & (cell == OK).to_numpy(), probe] = DISAGREES
    return reasons


def fuse(
    raw: pd.DataFrame,
    config: IrrigationConfig,
    soil_temperature: pd.DataFrame | None = None,
) -> FusedMoisture:
    """Mask, scale and fuse an hourly wide frame (one column per probe)."""
    params = config.fusion
    reasons = mask_readings(raw, params, config.exclude, soil_temperature)
    refs = references(raw, reasons, params)
    scaled = scale(raw, refs)
    reasons = mask_disagreement(scaled, reasons, params)
    # A probe with no usable reference cannot be scaled at all.
    for probe in raw.columns:
        if probe not in refs:
            reasons.loc[reasons[probe] == OK, probe] = MISSING
    used = scaled.where(reasons == OK)
    return FusedMoisture(
        fused=used.median(axis=1),
        coverage=used.notna().sum(axis=1),
        low=used.quantile(0.25, axis=1),
        high=used.quantile(0.75, axis=1),
        scaled=used,
        reasons=reasons,
        references=refs,
        raw=raw,
        soil_temperature=soil_temperature,
    )


def quality_table(result: FusedMoisture, start: datetime, end: datetime) -> pd.DataFrame:
    """Per probe over [start, end]: hours by mask reason, its references, its share.

    ``share`` is the probe's part of all used probe-hours, so the shares add up to 1.
    """
    reasons = result.reasons.loc[start:end]
    used = (reasons == OK).sum()
    total_used = int(used.sum())
    rows = []
    for probe in reasons.columns:
        counts = reasons[probe].value_counts()
        lo, hi = result.references.get(probe, (float("nan"), float("nan")))
        rows.append({
            "probe": probe,
            "used_h": int(counts.get(OK, 0)),
            **{r: int(n) for r, n in counts.items() if r != OK},
            "dry_ref": round(lo, 1),
            "wet_ref": round(hi, 1),
            "share": round(used[probe] / total_used, 2) if total_used else 0.0,
        })
    table = pd.DataFrame(rows).set_index("probe").fillna(0)
    reason_cols = [r for r in (MISSING, EXCLUDED, FROZEN, DROPOUT, SPIKE, DISAGREES) if r in table]
    table[["used_h", *reason_cols]] = table[["used_h", *reason_cols]].astype(int)
    return table[["used_h", *reason_cols, "dry_ref", "wet_ref", "share"]]


def to_hourly_frame(
    readings: pd.DataFrame, probes: tuple[str, ...], start: datetime, end: datetime,
) -> pd.DataFrame:
    """Long ``device/time/value`` rows → an hourly UTC frame with one column per probe.

    Every configured probe gets a column even when it has no rows, so a silent
    probe shows up as ``missing`` instead of disappearing.
    """
    index = pd.date_range(
        pd.Timestamp(start).tz_convert("UTC").floor("h"),
        pd.Timestamp(end).tz_convert("UTC").floor("h"),
        freq="h",
    )
    if readings.empty:
        return pd.DataFrame(np.nan, index=index, columns=list(probes))
    df = readings.assign(time=pd.to_datetime(readings["time"], utc=True).dt.floor("h"))
    wide = df.pivot_table(index="time", columns="device", values="value", aggfunc="mean")
    return wide.reindex(index=index, columns=list(probes))


async def load_fused(
    provider: SensorDataProvider,
    config: IrrigationConfig,
    *,
    now: datetime,
) -> FusedMoisture:
    """Fuse the configured probes over their whole history up to *now*.

    The whole history is always loaded, even for a page that shows one season.
    The references come from it, so the scale doesn't change with the window.
    """
    start = datetime.combine(config.history_start, datetime.min.time()).replace(tzinfo=now.tzinfo)
    hours = int((now - start) / timedelta(hours=1)) + 1
    sensors = [config.sensor, config.temperature_sensor]
    readings = await provider.fetch_data(
        sensor_tags=sensors,
        device_names=list(config.probes),
        start=start,
        end=now,
        bucket=timedelta(hours=1),
        agg="avg",
        # One hourly row per probe and sensor; the limit must cover all of them.
        limit=hours * len(config.probes) * len(sensors),
    )

    def frame(sensor: str) -> pd.DataFrame:
        rows = readings[readings["sensor"] == sensor] if not readings.empty else readings
        return to_hourly_frame(rows, config.probes, start, now)

    return fuse(frame(config.sensor), config, soil_temperature=frame(config.temperature_sensor))
