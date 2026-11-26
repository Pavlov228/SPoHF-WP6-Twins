"""The greenhouse's energy and vapour balance, one time step at a time.

Quasi-steady: at each step the heat going in (sun through the glass, lamps,
heating pipes) is set against the heat going out (through the cover, with the
ventilation air, and as evaporated water), and the vapour the crop transpires
against the vapour the air exchange and condensation remove. Whatever the
measured inputs and the modelled losses do not account for is kept as the
**residual** — the air and structure storing or releasing heat, and model
error — and shown, never hidden.

The same balance answers *what if*: with other vent, screen or lamp settings,
how much pipe heat would have held the temperature LetsGrow measured, and what
humidity would that leave? Each what-if is reported as a change from the
balance under the real settings, so the model's own error cancels.

Units: heat flows in W per m² floor; transpiration and vapour flows in l (kg)
per m² floor per hour; absolute humidity in g/m³; temperatures in °C; vent,
screen and lamp settings as 0-1 shares.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from wp6_data.red.greenhouse.config import GreenhouseConfig

SIGMA = 5.670e-8  # Stefan-Boltzmann, W/m²/K⁴
LATENT_HEAT = 2.45e6  # J per kg water evaporated
RHO_CP = 1200.0  # J/m³/K, volumetric heat capacity of air
PIPE_EMISSIVITY = 0.88
KELVIN = 273.15

#: Flow columns of :func:`balance`: gains positive, losses positive.
GAINS = ("sun", "lamps", "pipes")
LOSSES = ("cover", "ventilation", "evaporation")


def saturated_vapour_density(t_c: pd.Series | np.ndarray | float):
    """Water vapour at saturation, g/m³ (Tetens over water)."""
    t = np.asarray(t_c, dtype=float)
    e_kpa = 0.6108 * np.exp(17.27 * t / (t + 237.3))
    return 2167.0 * e_kpa / (t + KELVIN)


def relative_humidity(ah: pd.Series | np.ndarray, t_c: pd.Series | np.ndarray):
    """RH (%) of air holding ``ah`` g/m³ at ``t_c`` °C, capped at 100."""
    return np.minimum(100.0, 100.0 * np.asarray(ah, float) / saturated_vapour_density(t_c))


def pipe_heat(pipe_c, air_c, diameter: float, length: float):
    """Heat a pipe circuit gives the greenhouse, W/m² floor.

    Free convection after Bot (1983), as in Vanthoor (2011) and GreenLight:
    1.99·π·d·L·ΔT^1.32, plus long-wave radiation ε·σ·π·d·L·(Tp⁴ − Ta⁴). A pipe
    no warmer than the air gives nothing.
    """
    pipe = np.asarray(pipe_c, float)
    air = np.asarray(air_c, float)
    dt = np.clip(pipe - air, 0, None)
    conv = 1.99 * np.pi * diameter * length * dt**1.32
    rad = (
        PIPE_EMISSIVITY
        * SIGMA
        * np.pi
        * diameter
        * length
        * ((pipe + KELVIN) ** 4 - (air + KELVIN) ** 4)
    )
    return conv + np.clip(rad, 0, None)


def ventilation_rate(opening, cfg: GreenhouseConfig):
    """Air exchange, m³ per m² floor per s, for a mean vent opening (0-1)."""
    v = cfg.ventilation
    return np.interp(np.clip(np.asarray(opening, float), 0, 1), v.opening, v.rate)


def cover_u(wind, screen, cfg: GreenhouseConfig):
    """Heat loss through the cover, W/m² floor per K of inside-outside difference."""
    c = cfg.cover
    return (c.u_open + c.u_wind * np.asarray(wind, float)) * (
        1 - c.screen_saving * np.asarray(screen, float)
    )


def sun_inside(radiation, screen, cfg: GreenhouseConfig):
    """Outdoor global radiation that gets through glass and screen, W/m² floor.

    What the crop and floor then reflect back out is booked separately, as
    :func:`sun_absorbed` takes it off.
    """
    c = cfg.cover
    screen = np.asarray(screen, float)
    return (
        c.transmission * (1 - screen * (1 - c.screen_transmission)) * np.asarray(radiation, float)
    )


def sun_absorbed(radiation, screen, cfg: GreenhouseConfig):
    """The part of the sun inside that stays: what is not reflected back out, W/m²."""
    return (1 - cfg.cover.reflected_share) * sun_inside(radiation, screen, cfg)


def steady_humidity(transpiration, ventilation, ah_out, t_in, t_out, screen, cfg: GreenhouseConfig):
    """Absolute humidity (g/m³) at which vapour in equals vapour out.

    In: transpiration. Out: the air exchange carries ``g·(AH − AH_out)``, and
    vapour condenses on a cover colder than the dew point at the cover's mass
    transfer rate. Both are linear in AH, so the steady state is closed-form —
    first assuming no condensation, then with it where the result would
    exceed what the cover can hold.
    """
    c = cfg.cover
    e = np.asarray(transpiration, float) * 1000 / 3600  # g/m²/s
    g = np.asarray(ventilation, float)
    ah_out = np.asarray(ah_out, float)
    t_cover = np.asarray(t_out, float) + c.cover_temperature_share * (
        np.asarray(t_in, float) - np.asarray(t_out, float)
    )
    # a screen shields the cover: less of the house's vapour reaches the glass
    h = c.condensation_rate * (1 - c.screen_saving * np.asarray(screen, float))
    ah_cover = saturated_vapour_density(t_cover)
    dry = ah_out + e / g
    wet = (e + g * ah_out + h * ah_cover) / (g + h)
    return np.where(dry > ah_cover, wet, dry)


def balance(state: pd.DataFrame, cfg: GreenhouseConfig) -> pd.DataFrame:
    """Every heat and vapour flow for each step of ``state``.

    ``state`` columns: t_in, ah_in, t_out, ah_out, wind, radiation, lamps,
    vent (0-1 mean opening), screen (0-1), transpiration (l/m²/h) and one
    column per pipe tag (°C). Returns W/m² flows (``sun``, ``lamps``,
    ``pipes`` and one per pipe; ``cover``, ``ventilation``, ``evaporation``;
    ``residual``) plus ``air_exchange`` (m³/m²/s), ``vapour_out`` (l/m²/h
    carried out by the air) and the modelled steady ``ah_model``.
    """
    s = state
    screen = s["screen"].fillna(0).clip(0, 1)
    out = pd.DataFrame(index=s.index)
    out["sun"] = sun_absorbed(s["radiation"].clip(lower=0), screen, cfg)
    out["lamps"] = s["lamps"].clip(0, 1) * cfg.lamps.power_w
    pipes = 0.0
    for p in cfg.pipes:
        q = pd.Series(pipe_heat(s[p.tag], s["t_in"], p.diameter, p.length), index=s.index)
        out[f"pipe:{p.tag}"] = q
        pipes = pipes + q.fillna(0)
    out["pipes"] = pipes
    d_t = s["t_in"] - s["t_out"]
    g = pd.Series(ventilation_rate(s["vent"].fillna(0), cfg), index=s.index)
    out["air_exchange"] = g
    out["cover"] = cover_u(s["wind"].fillna(0), screen, cfg) * d_t
    out["ventilation"] = RHO_CP * g * d_t
    out["evaporation"] = s["transpiration"] / 3600 * LATENT_HEAT
    out["residual"] = (
        out["sun"]
        + out["lamps"]
        + out["pipes"]
        - out["cover"]
        - out["ventilation"]
        - out["evaporation"]
    )
    out["vapour_out"] = g * (s["ah_in"] - s["ah_out"]) * 3600 / 1000
    out["ah_model"] = steady_humidity(
        s["transpiration"],
        g,
        s["ah_out"],
        s["t_in"],
        s["t_out"],
        screen,
        cfg,
    )
    return out


def heat_needed(state: pd.DataFrame, cfg: GreenhouseConfig) -> pd.Series:
    """Pipe heat that would hold the measured air temperature, W/m² floor.

    Losses minus the free gains (sun and lamps). Negative means surplus: the
    temperature would rise unless more air is let out.
    """
    f = balance(state, cfg)
    return f["cover"] + f["ventilation"] + f["evaporation"] - f["sun"] - f["lamps"]


def what_if(
    state: pd.DataFrame, cfg: GreenhouseConfig, **settings: pd.Series | float
) -> pd.DataFrame:
    """The balance under other settings (``vent``, ``screen``, ``lamps``), as changes.

    Holds the measured air temperature. Returns per step the pipe heat the
    new settings would need (``pipes``: the measured pipe heat shifted by the
    change in what is needed, never below zero), ``pipes_change``, and the
    humidity they would leave (``ah``, ``rh``) — the measured humidity moved by
    the change in the modelled steady humidity, so the model's offset cancels.
    Lamps change the transpiration they drive, per the configured shape.
    """
    base = balance(state, cfg)
    alt_state = state.copy()
    for key, value in settings.items():
        if key not in ("vent", "screen", "lamps"):
            raise ValueError(f"unknown setting {key!r}")
        alt_state[key] = value
    if "lamps" in settings:
        dl = (alt_state["lamps"] - state["lamps"]) * cfg.lamps.power_w
        alt_state["transpiration"] = (
            state["transpiration"] + dl * cfg.transpiration.per_lamp_w / 1000
        ).clip(lower=0)
    alt = balance(alt_state, cfg)
    need_base = (
        base["cover"] + base["ventilation"] + base["evaporation"] - base["sun"] - base["lamps"]
    )
    need_alt = alt["cover"] + alt["ventilation"] + alt["evaporation"] - alt["sun"] - alt["lamps"]
    out = pd.DataFrame(index=state.index)
    out["pipes"] = (base["pipes"] + (need_alt - need_base)).clip(lower=0)
    out["pipes_change"] = out["pipes"] - base["pipes"]
    out["ah"] = (state["ah_in"] + (alt["ah_model"] - base["ah_model"])).clip(lower=0)
    out["rh"] = relative_humidity(out["ah"], state["t_in"])
    out["surplus"] = (-need_alt).clip(lower=0)
    return out
