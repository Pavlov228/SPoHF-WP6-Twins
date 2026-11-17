"""Daily soil water balance on the fused, relative soil-moisture scale.

The root zone's state ``S`` is the fused soil moisture (0 dry – 1 wet). One day:

    inputs  = rain_eff · rain + irr_eff · irrigation       (mixed over two days, below)
    et      = et_scale · Kc(day of year) · Ks(S) · ET0
    drain   = drain_rate · max(S − s_fc, 0)
    S'      = S + inputs − et − drain [− exchange]

- **Ks(S)** = clip(S / θ, 0, 1): water use shrinks as the root zone dries below θ.
  This is the flattening dry-down seen in the data (research note §4.3).
- **Kc** changes through the season: kc_slope tilts it around midsummer.
- **The mix.** S is a *daily mean*, so a day's rain is only partly in that
  day's mean, and an evening watering barely is. A step t → t+1 therefore
  takes ``mix · inputs[t+1] + (1 − mix) · inputs[t]``, with mix fitted.
- **Rain and drip are separate inputs** with their own effectiveness. Drip
  barely reaches the probes (research note §4.1), so ``irr_eff`` has a tight
  upper bound and is never negative, which keeps the model controllable.

The **two-bucket** variant adds the unmeasured soil bed ``B``. It exchanges
water with the root zone (``exchange · (S − B)``) and relaxes towards its own
level ``b_eq`` (groundwater, lateral flow). This is the candidate explanation
for the flat, rainless August 2025 (research note §4.3).

Everything is vectorised over *origins*: one call simulates every forecast
origin of a backtest at once.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

#: Midsummer day of year, the pivot of the seasonal crop coefficient.
_KC_PIVOT_DOY = 196.0


@dataclass(frozen=True)
class BucketSpec:
    """Parameter names, starting values and bounds of one model variant."""

    name: str
    params: tuple[str, ...]
    x0: tuple[float, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    #: When set, the drip effect is not fitted but tied to rain: a litre per
    #: plant counts as a millimetre of rain over this many m² (ADR 0009).
    area_per_plant_m2: float | None = None

    @property
    def two_bucket(self) -> bool:
        return "exchange" in self.params


_SINGLE = {
    #          x0     lower  upper
    "rain_eff": (0.02, 0.0, 0.2),     # per mm
    "irr_eff": (0.01, 0.0, 0.1),      # per litre per plant
    "et_scale": (0.02, 0.0, 0.2),     # per mm ET0
    "kc_slope": (0.0, -1.0, 1.0),     # Kc tilt per 100 days from midsummer
    "theta": (0.5, 0.05, 1.5),        # stress onset on the S scale
    "drain_rate": (0.2, 0.0, 1.0),    # per day above field capacity
    "s_fc": (0.8, 0.3, 1.3),          # field-capacity level on the S scale
    "mix": (0.5, 0.0, 1.0),           # share of a step's inputs from the new day
}
_BED = {
    "exchange": (0.05, 0.0, 0.5),     # per day, root zone ↔ bed
    "bed_relax": (0.02, 0.0, 0.5),    # per day, bed towards its own level
    "b_eq": (0.5, -0.5, 1.5),         # the bed's own level on the S scale
}


def _spec(name: str, table: dict[str, tuple[float, float, float]]) -> BucketSpec:
    return BucketSpec(
        name=name,
        params=tuple(table),
        x0=tuple(v[0] for v in table.values()),
        lower=tuple(v[1] for v in table.values()),
        upper=tuple(v[2] for v in table.values()),
    )


SINGLE_BUCKET = _spec("single bucket", _SINGLE)
TWO_BUCKET = _spec("two bucket", _SINGLE | _BED)


def dose_as_rain(spec: BucketSpec, area_per_plant_m2: float) -> BucketSpec:
    """*spec* with the drip effect tied to rain over the plant's area, not fitted.

    The probes sit away from the drippers, so the data can't identify the drip
    effect on its own (research note §4.1, §7). Instead a litre per plant is
    counted as rain spread over ``area_per_plant_m2``.
    """
    keep = [i for i, name in enumerate(spec.params) if name != "irr_eff"]
    return BucketSpec(
        name=f"{spec.name}, dose as rain over {area_per_plant_m2:g} m²",
        params=tuple(spec.params[i] for i in keep),
        x0=tuple(spec.x0[i] for i in keep),
        lower=tuple(spec.lower[i] for i in keep),
        upper=tuple(spec.upper[i] for i in keep),
        area_per_plant_m2=area_per_plant_m2,
    )


def _drip_effect(spec: BucketSpec, p: dict[str, float]) -> float:
    if spec.area_per_plant_m2 is not None:
        return p["rain_eff"] / spec.area_per_plant_m2
    return p["irr_eff"]


@dataclass(frozen=True)
class Inputs:
    """Daily drivers, shaped (days, origins). Row 0 is the origin day itself
    (needed for the mix); rows 1..H drive the H forecast steps."""

    rain: np.ndarray
    et0: np.ndarray
    irrigation: np.ndarray
    doy: np.ndarray

    @property
    def horizon(self) -> int:
        return self.rain.shape[0] - 1


def simulate(
    spec: BucketSpec,
    x: np.ndarray,
    s0: np.ndarray,
    inputs: Inputs,
    b0: np.ndarray | None = None,
) -> np.ndarray:
    """Forecast S for steps 1..H from *s0* at every origin; returns (H, origins)."""
    p = dict(zip(spec.params, x, strict=True))
    s = s0.astype(float).copy()
    b = (s.copy() if b0 is None else b0.astype(float).copy()) if spec.two_bucket else None
    out = np.empty((inputs.horizon, s.size))
    irr_eff = _drip_effect(spec, p)
    for h in range(1, inputs.horizon + 1):
        mix = p["mix"]
        rain = mix * inputs.rain[h] + (1 - mix) * inputs.rain[h - 1]
        irr = mix * inputs.irrigation[h] + (1 - mix) * inputs.irrigation[h - 1]
        kc = np.clip(1.0 + p["kc_slope"] * (inputs.doy[h] - _KC_PIVOT_DOY) / 100.0, 0.05, None)
        ks = np.clip(s / p["theta"], 0.0, 1.0)
        et = p["et_scale"] * kc * ks * inputs.et0[h]
        drain = p["drain_rate"] * np.maximum(s - p["s_fc"], 0.0)
        delta = p["rain_eff"] * rain + irr_eff * irr - et - drain
        if b is not None:
            flow = p["exchange"] * (s - b)
            b = b + flow + p["bed_relax"] * (p["b_eq"] - b)
            delta = delta - flow
        s = s + delta
        out[h - 1] = s
    return out


def fit(
    spec: BucketSpec,
    s0: np.ndarray,
    inputs: Inputs,
    observed: np.ndarray,
    b0: np.ndarray | None = None,
) -> np.ndarray:
    """Least-squares parameters over every (origin, step) with an observation.

    Fitting the multi-step trajectory, not just one step ahead, is what makes the
    parameters good at forecasting several days out.
    """
    mask = np.isfinite(observed)
    if mask.sum() < len(spec.params) * 3:
        raise ValueError("too few observations to fit")

    def residuals(x: np.ndarray) -> np.ndarray:
        return (simulate(spec, x, s0, inputs, b0) - observed)[mask]

    result = least_squares(
        residuals, np.asarray(spec.x0), bounds=(spec.lower, spec.upper),
        loss="soft_l1", f_scale=0.05,
    )
    return result.x


def spin_up_bed(
    spec: BucketSpec, x: np.ndarray, history_s: np.ndarray, history: Inputs,
) -> float:
    """The bed's state at the end of a history, with S forced to observations.

    The bed is never measured, so it is run forward through the days before an
    origin, driven by the *observed* root zone (gaps hold the last value).
    """
    p = dict(zip(spec.params, x, strict=True))
    b = p["b_eq"]
    s_prev = np.nan
    for day in range(len(history_s)):
        s = history_s[day] if np.isfinite(history_s[day]) else s_prev
        if np.isfinite(s):
            flow = p["exchange"] * (s - b)
            b = b + flow + p["bed_relax"] * (p["b_eq"] - b)
            s_prev = s
    return float(b)


def kalman_noise(residuals_by_horizon: np.ndarray) -> tuple[float, float]:
    """Process and observation noise from forecast residuals at each horizon.

    An h-step residual's variance ≈ h·Q + 2·R: model error accumulates each
    day (Q), and the start and end observations each carry their own noise (R).
    A line through variance-vs-horizon gives Q (slope) and R (half the intercept).
    """
    horizons = np.arange(1, residuals_by_horizon.shape[0] + 1, dtype=float)
    variance = np.array([np.nanvar(r) for r in residuals_by_horizon])
    slope, intercept = np.polyfit(horizons, variance, 1)
    q = max(float(slope), 1e-6)
    r = max(float(intercept) / 2.0, 1e-6)
    return q, r


def kalman_filter(
    spec: BucketSpec, x: np.ndarray, observed: np.ndarray, inputs: Inputs, q: float, r: float,
) -> tuple[float, float]:
    """Filtered S and its variance at the last day of a history.

    *inputs* covers the history (row 0 = first day), *observed* holds S per day.
    Each day predicts one step with the model and blends in the observation in
    proportion to the two variances.
    """
    first = int(np.flatnonzero(np.isfinite(observed))[0])
    s, var = float(observed[first]), r
    for day in range(first + 1, len(observed)):
        one = Inputs(
            rain=inputs.rain[day - 1:day + 1, None],
            et0=inputs.et0[day - 1:day + 1, None],
            irrigation=inputs.irrigation[day - 1:day + 1, None],
            doy=inputs.doy[day - 1:day + 1, None],
        )
        s = float(simulate(spec, x, np.array([s]), one)[0, 0])
        var += q
        if np.isfinite(observed[day]):
            gain = var / (var + r)
            s += gain * (observed[day] - s)
            var *= 1.0 - gain
    return s, var
