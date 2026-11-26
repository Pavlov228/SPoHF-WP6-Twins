"""Day-ahead lamp planner: the cheapest lamp hours that still reach a light sum.

Per local calendar day, given hourly sun (J/cm²), day-ahead price (€/MWh) and
a target light sum (outdoor-equivalent J/cm², the grower's unit):

    minimise   Σ_h price_h · P · u_h
    subject to Σ_h sun_h + K · (Σ_h u_h − wake · starts) ≥ target
               one unbroken dark window of ≥ dark_min hours, no sun and no lamp
               u_h = 0 when the sun alone is above sun_max (lamps add little there)
               u_h ∈ {0, 1}  (whole hours, as the grower switches)

K is the grower's J/cm² per full lamp-hour, P the lamp power. ``wake`` is the
light lost at each switch-on while the crop is still waking up — about half a
lamp-hour on the substrate scale (CONTEXT "Crop activity"). Solved exactly by
dynamic programming over (hour, on/off, effective half-lamp-hours, current
dark run, dark window met). Scored against four seasons of what ran in the red operational-decisions
research (kept outside this repository; scripts/red_operations reproduces it).
"""

from __future__ import annotations

import numpy as np

WAKE = 0.5  # default; red configures it as greenhouse.advice.wake_lamp_h


def plan_day(
    sun: np.ndarray,
    price: np.ndarray,
    target: float,
    k: float,
    dark_min: int = 6,
    sun_max_jcm2: float = 108.0,
    wake: float = WAKE,
    prev_on: bool = False,
    dark_run: int = 0,
    dark_ok: bool = False,
    already: float = 0.0,
) -> np.ndarray | None:
    """Lamp on/off per remaining hour, or None when infeasible.

    ``sun``/``price`` cover the hours still to plan; ``already`` is the light
    (J/cm², effective) delivered earlier in the day, and ``dark_run`` /
    ``dark_ok`` carry the dark-window state, so the same solver serves a whole
    day and an hourly re-plan. sun_max_jcm2 108 ≈ 300 W/m² over the hour.
    """
    n = len(sun)
    need = target - already - float(np.nansum(sun))
    # effective light in half lamp-hours: +2 per lit hour, −2·wake per start
    w = int(round(2 * wake))
    need_half = int(np.ceil(max(need, 0) / k * 2 - 1e-9))
    cap = 2 * n + 1
    # state: (prev_on, eff (clipped at need_half), dark_run (0..dark_min), dark_ok)
    best = {(prev_on, 0, min(dark_run, dark_min), dark_ok): (0.0, ())}
    for h in range(n):
        nxt: dict = {}
        night = sun[h] <= 0.5
        for (on, eff, run, ok), (cost, path) in best.items():
            for u in (0, 1):
                if u and sun[h] >= sun_max_jcm2:
                    continue
                e2 = eff
                if u:
                    e2 = eff + 2 - (w if not on else 0)
                e2 = min(max(e2, -cap), need_half)  # clip: more light than needed is not rewarded
                run2 = min(run + 1, dark_min) if (night and not u) else 0
                ok2 = ok or run2 >= dark_min
                c2 = cost + (price[h] if u else 0.0)
                key = (bool(u), e2, run2, ok2)
                if key not in nxt or c2 < nxt[key][0]:
                    nxt[key] = (c2, path + (u,))
        best = nxt
    final = [(c, p) for (on, eff, run, ok), (c, p) in best.items() if ok and eff >= need_half]
    if not final:
        return None
    c, p = min(final, key=lambda t: t[0])
    return np.array(p, dtype=float)


def rolling_plan(
    sun_real: np.ndarray,
    sun_fc: np.ndarray,
    price: np.ndarray,
    target: float,
    k: float,
    prev_on: bool = False,
    dark_min: int = 6,
    wake: float = WAKE,
    sun_max_jcm2: float = 108.0,
) -> np.ndarray:
    """Re-plan every hour: realised sun so far, forecast for the rest (MPC).

    Only the current hour's decision is applied. If no plan reaches the target
    any more, it lights every allowed hour left that keeps the dark window.
    """
    n = len(sun_real)
    u = np.zeros(n)
    on, run, ok, eff = prev_on, 0, False, 0.0
    for h in range(n):
        fc = (
            np.concatenate([[sun_real[h]], sun_fc[h + 1 :]])
            if h + 1 < n
            else np.array([sun_real[h]])
        )
        plan = plan_day(
            fc,
            price[h:],
            target,
            k,
            dark_min=dark_min,
            sun_max_jcm2=sun_max_jcm2,
            wake=wake,
            prev_on=on,
            dark_run=run,
            dark_ok=ok,
            already=eff + float(sun_real[:h].sum()),
        )
        if plan is None:
            # target out of reach: the most light that still keeps the dark rule
            already = eff + float(sun_real[:h].sum())
            t = target
            while plan is None and t > already:
                t -= k / 2
                plan = plan_day(
                    fc,
                    price[h:],
                    t,
                    k,
                    dark_min=dark_min,
                    sun_max_jcm2=sun_max_jcm2,
                    wake=wake,
                    prev_on=on,
                    dark_run=run,
                    dark_ok=ok,
                    already=already,
                )
            if plan is None:  # the dark window itself can no longer fit: lamps off
                plan = np.zeros(n - h)
        x = plan[0]
        u[h] = x
        if x > 0.5:
            eff += k * (1.0 - (wake if not on else 0.0))
        night = sun_real[h] <= 0.5
        run = min(run + 1, dark_min) if (night and x < 0.5) else 0
        ok = ok or run >= dark_min
        on = x > 0.5
    return u


def starts(u: np.ndarray, prev_on: bool = False) -> int:
    """Switch-ons in ``u`` (on after off), counting the first hour against ``prev_on``."""
    on = u > 0.5
    prev = np.concatenate([[prev_on], on[:-1]])
    return int((on & ~prev).sum())
