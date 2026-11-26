"""Prescriptions: what the twin advises for a day, each tied to a greenhouse part.

Every rule reads the day's 15-minute state and its balance, and says what to
do differently and what it is worth in physical units (MJ/m² of pipe heat,
lamp-hours, J/cm² of light, hours humid or dark) — never in euros, which would
rest on assumed prices. A rule that has nothing to say says so ("ok"), so a
quiet day reads as a checked day rather than an empty page. Thresholds live in ``metadata.yaml``
(``greenhouse.advice``); the reasoning behind each rule is in
the red operational-decisions research (kept outside this repository).

``component`` names the part of the 3D greenhouse the advice belongs to:
lamps, pipes, vents, screen, crop, substrate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from wp6_data.red.greenhouse import planner
from wp6_data.red.greenhouse.config import GreenhouseConfig
from wp6_data.shared.i18n import t

STEP_H = 0.25
J_PER_W_HOUR = 3600 / 1e4  # W/m² held for an hour → J/cm²


@dataclass
class Advice:
    id: str
    component: str
    level: str  # "act" | "consider" | "info" | "ok"
    title: str
    detail: str
    impact: dict[str, float] = field(default_factory=dict)
    windows: list[list[str]] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


def _say(
    rule: str,
    variant: str,
    component: str,
    level: str,
    impact: dict[str, float] | None = None,
    windows: list[list[str]] | None = None,
    **params,
) -> Advice:
    """An advice in the current language (catalogue ``greenhouse.advice.<rule>.<variant>``)."""
    key = f"greenhouse.advice.{rule}.{variant}"
    return Advice(
        rule,
        component,
        level,
        t(f"{key}.title", **params),
        t(f"{key}.detail", **params),
        impact or {},
        windows or [],
    )


def _mj(w_per_m2: pd.Series) -> float:
    """W/m² per 15-minute step → MJ/m² over the steps."""
    return float(w_per_m2.fillna(0).sum() * STEP_H * 3600 / 1e6)


def _clock(t: pd.Timestamp, day_start: pd.Timestamp) -> str:
    """HH:MM, with the day's end written 24:00 rather than 00:00."""
    return "24:00" if t >= day_start + pd.Timedelta(days=1) else t.strftime("%H:%M")


def _span(mask: pd.Series, max_segments: int = 4) -> list[list[str]]:
    """The unbroken stretches where ``mask`` holds, longest first, as [start, end]."""
    mask = mask.fillna(False).astype(bool)
    if not mask.any():
        return []
    day_start = mask.index[0].normalize()
    run = (mask != mask.shift()).cumsum()
    segments = []
    for _, g in mask[mask].groupby(run[mask]):
        end = g.index[-1] + pd.Timedelta(minutes=15)
        segments.append((len(g), [_clock(g.index[0], day_start), _clock(end, day_start)]))
    segments.sort(key=lambda x: -x[0])
    return sorted((seg for _, seg in segments[:max_segments]), key=lambda x: x[0])


def hourly(s: pd.DataFrame) -> pd.DataFrame:
    """Hourly sun (J/cm²) and lamp share from the 15-minute state."""
    h = s[["radiation", "lamps"]].resample("1h").mean()
    h["sun_j"] = h["radiation"].fillna(0) * J_PER_W_HOUR
    return h


def lamp_hours_needed(sun_j: np.ndarray, cfg: GreenhouseConfig) -> float | None:
    """The fewest whole lamp-hours that bring ``sun_j`` (J/cm² per hour) to the target.

    The planner's constraints, with every hour costing the same: at least
    ``dark_min_h`` of unbroken dark, no lamps above ``sun_max_w``, and
    ``wake_lamp_h`` of light lost per switch-on. None when no schedule fits.
    """
    a = cfg.advice
    u = planner.plan_day(
        np.asarray(sun_j, dtype=float),
        np.ones(len(sun_j)),
        a.light_target_jcm2,
        cfg.lamps.radiation_eq,
        dark_min=a.dark_min_h,
        sun_max_jcm2=a.sun_max_w * J_PER_W_HOUR,
        wake=a.wake_lamp_h,
    )
    return None if u is None else float(u.sum())


def longest_day_lamp_hours(sun_j: np.ndarray, cfg: GreenhouseConfig) -> float:
    """Lamp-hours under the grower's longest-day rule: every hour the sun is weak, less the rest."""
    a = cfg.advice
    weak = int((np.asarray(sun_j) < a.sun_max_w * J_PER_W_HOUR).sum())
    return float(max(0, weak - a.dark_min_h))


def light_period(f5: pd.DataFrame, day_start: pd.Timestamp) -> dict:
    """The day's light and dark, from the 5-minute frame (with its margins).

    Sun: outside radiation above zero. Lamps only: lamps on with no sun. Dark:
    neither. Hours are counted inside the local day; the longest dark spell
    may run into the evening before or the night after, since a night
    straddles midnight.
    """
    step_h = 5 / 60
    end = day_start + pd.Timedelta(days=1)
    lamps = f5["lamp_on"].clip(0, 1) * f5["lamp_pct"].clip(0, 100) / 100
    rad = f5["radiation"]
    sun = rad > 0
    lamp_only = (lamps > 0.01) & ~sun
    dark = (rad <= 0) & (lamps <= 0.01)
    inside = (f5.index >= day_start) & (f5.index < end)
    longest, frm, to = 0.0, None, None
    run = (dark != dark.shift()).cumsum()
    for _, g in dark[dark].groupby(run[dark]):
        last = g.index[-1] + pd.Timedelta(minutes=5)
        if last <= day_start or g.index[0] >= end:
            continue
        hours = len(g) * step_h
        if hours > longest:
            longest, frm, to = hours, g.index[0], last
    sun_h = float(sun[inside].sum() * step_h)
    lamp_h = float(lamp_only[inside].sum() * step_h)
    return {
        "sun_h": round(sun_h, 2),
        "lamp_only_h": round(lamp_h, 2),
        "light_h": round(sun_h + lamp_h, 2),
        "dark_h": round(float(dark[inside].sum() * step_h), 2),
        "longest_dark_h": round(longest, 2),
        "longest_dark": [frm.strftime("%d %b %H:%M"), to.strftime("%d %b %H:%M")]
        if frm is not None
        else None,
    }


def light_plan(h: pd.DataFrame, cfg: GreenhouseConfig) -> dict | None:
    """The light the day got against the target, and the lamp-hours it needed."""
    if len(h) < 23:
        return None
    a = cfg.advice
    actual = h["lamps"].fillna(0).to_numpy()
    sun = h["sun_j"].to_numpy()
    return {
        "target_jcm2": a.light_target_jcm2,
        "sun_jcm2": float(sun.sum()),
        "light_jcm2": float(sun.sum() + actual.sum() * cfg.lamps.radiation_eq),
        "lamp_h_actual": float(actual.sum()),
        "lamp_h_needed": lamp_hours_needed(sun, cfg),
        "lamp_h_longest_day": longest_day_lamp_hours(sun, cfg),
    }


def advise(
    s: pd.DataFrame,
    flows: pd.DataFrame,
    cfg: GreenhouseConfig,
    screen_closed: pd.DataFrame | None,
    light: dict,
) -> tuple[list[Advice], dict | None]:
    """All rules for one day; returns the advice and the light plan behind it.

    ``light`` is :func:`light_period` for the day.
    """
    a = cfg.advice
    k = cfg.lamps.radiation_eq
    out: list[Advice] = []
    plan = light_plan(hourly(s), cfg)

    # lamps: the light sum against the target
    if plan is None:
        out.append(_say("light_sum", "not_judged", "lamps", "info"))
    elif plan["sun_jcm2"] >= a.light_target_jcm2 and plan["lamp_h_actual"] < 0.5:
        out.append(
            _say(
                "light_sum",
                "sun_enough",
                "lamps",
                "ok",
                sun=round(plan["sun_jcm2"]),
                target=a.light_target_jcm2,
            )
        )
    elif plan["light_jcm2"] < a.light_target_jcm2 - k / 2:
        short = a.light_target_jcm2 - plan["light_jcm2"]
        out.append(
            _say(
                "light_sum",
                "short",
                "lamps",
                "act" if short > 2 * k else "consider",
                {"jcm2": round(-short), "lamp_h": round(short / k, 1)},
                light=round(plan["light_jcm2"]),
                target=a.light_target_jcm2,
                sun=round(plan["sun_jcm2"]),
                lamp_h=plan["lamp_h_actual"],
                k=k,
                dark_min=a.dark_min_h,
                needed=plan["lamp_h_needed"] or 0,
            )
        )
    else:
        needed = plan["lamp_h_needed"] or 0.0
        extra = plan["lamp_h_actual"] - needed
        level = "consider" if extra >= 3 else "info" if extra >= 1 else "ok"
        out.append(
            _say(
                "light_sum",
                "extra" if level != "ok" else "matched",
                "lamps",
                level,
                {"lamp_h": round(extra, 1), "jcm2": round(plan["light_jcm2"] - a.light_target_jcm2)}
                if level != "ok"
                else {},
                extra=extra,
                lamp_h=plan["lamp_h_actual"],
                light=round(plan["light_jcm2"]),
                dark_min=a.dark_min_h,
                wake=a.wake_lamp_h,
                target=a.light_target_jcm2,
                needed=needed,
            )
        )

    # pipes: heat going in while vents throw it out on dry air
    open_ = s["vent"] * 100 > a.vent_open_pct
    dry = s["rh_in"] < a.dry_rh
    warm = s["t_in"] > s["heat_sp"] + 1
    vented = open_ & dry & warm
    mj = _mj(flows["pipes"].where(vented, 0))
    if mj > 0.3:
        out.append(
            _say(
                "heat_vented",
                "wasted",
                "pipes",
                "act" if mj > 1.0 else "consider",
                {"mj": round(mj, 2)},
                _span(vented),
                mj=mj,
                vent_open=a.vent_open_pct,
                dry_rh=a.dry_rh,
            )
        )
    else:
        out.append(_say("heat_vented", "ok", "pipes", "ok"))

    # screen: close it on dark, cold hours
    if not s["screen_logged"].astype(bool).any():
        out.append(_say("screen", "not_logged", "screen", "info"))
    elif screen_closed is not None:
        dark = (s["radiation"] <= 5) & (s["screen"] < 0.5) & (s["t_in"] - s["t_out"] > 6)
        saved = -_mj(screen_closed["pipes_change"].where(dark, 0))
        if saved > 0.2:
            out.append(
                _say(
                    "screen",
                    "save",
                    "screen",
                    "act" if saved > 0.8 else "consider",
                    {"mj": round(saved, 2)},
                    _span(dark),
                    saved=saved,
                    saving=cfg.cover.screen_saving,
                )
            )
        else:
            out.append(_say("screen", "ok", "screen", "ok"))

    # humidity: on target, too humid, or dried at a cost
    humid_h = float((s["rh_in"] >= a.humid_rh).sum() * STEP_H)
    if humid_h > 2 * a.humid_hours_target:
        out.append(
            _say(
                "humidity",
                "humid",
                "crop",
                "act",
                {"humid_h": humid_h},
                _span(s["rh_in"] >= a.humid_rh),
                hours=humid_h,
                rh=a.humid_rh,
            )
        )
    elif humid_h < 0.5 and mj > 1.0:
        out.append(
            _say("humidity", "dried", "crop", "consider", {"humid_h": humid_h}, rh=a.humid_rh)
        )
    else:
        out.append(
            _say(
                "humidity",
                "ok",
                "crop",
                "ok",
                hours=humid_h,
                rh=a.humid_rh,
                target=a.humid_hours_target,
            )
        )

    # dark rule
    longest = light["longest_dark_h"]
    if longest < a.dark_min_h:
        out.append(
            _say(
                "dark",
                "short",
                "lamps",
                "act",
                {"dark_h": longest},
                hours=longest,
                dark_min=a.dark_min_h,
            )
        )
    else:
        out.append(
            _say(
                "dark",
                "ok",
                "lamps",
                "ok",
                hours=longest,
                light_h=light["light_h"],
                sun_h=light["sun_h"],
                lamp_h=light["lamp_only_h"],
                dark_min=a.dark_min_h,
            )
        )

    # CO₂ in lamp hours
    lit = s["lamps"] > 0.5
    if lit.sum() >= 8:
        co2 = float(s.loc[lit, "co2"].mean())
        if co2 < a.co2_lamp_floor_ppm:
            out.append(_say("co2", "low", "crop", "consider", {"ppm": round(co2)}, ppm=round(co2)))
    return out, plan


def irrigation_timing(
    s: pd.DataFrame, five: pd.DataFrame, wake: pd.Timestamp | None
) -> Advice | None:
    """Did the first irrigation follow the crop waking up?"""
    irr = five["irrigation"].diff()
    day = s.index[0].normalize()
    first = irr[(irr > 0.005) & (irr.index >= day) & (irr.index < day + pd.Timedelta(hours=14))]
    if first.empty or wake is None:
        return None
    t0 = first.index[0]
    lag = (t0 - wake).total_seconds() / 3600
    woke, started = wake.strftime("%H:%M"), t0.strftime("%H:%M")
    if lag > 2:
        return _say(
            "irrigation",
            "late",
            "substrate",
            "consider",
            {"lag_h": round(lag, 1)},
            [[woke, started]],
            lag=lag,
            wake=woke,
            start=started,
        )
    if lag < -0.5:
        return _say(
            "irrigation",
            "early",
            "substrate",
            "consider",
            {"lag_h": round(lag, 1)},
            [[started, woke]],
            lag=-lag,
            wake=woke,
            start=started,
        )
    return _say("irrigation", "ok", "substrate", "ok", wake=woke, start=started)
