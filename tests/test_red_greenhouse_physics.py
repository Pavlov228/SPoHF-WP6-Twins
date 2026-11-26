"""The greenhouse balance: physical sanity and what-if consistency."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wp6_data.red.greenhouse import physics as P
from wp6_data.red.greenhouse.config import load_greenhouse

CFG = load_greenhouse(Path(__file__).parents[1] / "src/wp6_data/red/metadata.yaml")
LOWER, GROWTH = (p.tag for p in CFG.pipes)


def _state(n: int = 4, **over) -> pd.DataFrame:
    """A cold, dark night with the pipes on, vents shut, screen open."""
    base = {
        "t_in": 18.0,
        "ah_in": 11.0,
        "t_out": 4.0,
        "ah_out": 5.5,
        "wind": 2.0,
        "radiation": 0.0,
        "lamps": 0.0,
        "vent": 0.0,
        "screen": 0.0,
        "transpiration": 0.04,
        LOWER: 50.0,
        GROWTH: 35.0,
    }
    base.update(over)
    idx = pd.date_range("2026-01-10 00:00", periods=n, freq="15min", tz="Europe/Amsterdam")
    return pd.DataFrame({k: [v] * n for k, v in base.items()}, index=idx)


def test_saturation_and_rh_round_trip():
    ah = P.saturated_vapour_density(20.0)
    assert 17.0 < ah < 17.6  # handbook: 17.3 g/m³ at 20 °C
    assert P.relative_humidity(ah / 2, 20.0) == pytest.approx(50.0)


def test_a_pipe_no_warmer_than_the_air_gives_no_heat():
    pipe = CFG.pipes[0]
    assert P.pipe_heat(18.0, 18.0, pipe.diameter, pipe.length) == pytest.approx(0.0)
    assert P.pipe_heat(10.0, 18.0, pipe.diameter, pipe.length) == pytest.approx(0.0)
    assert P.pipe_heat(60.0, 18.0, pipe.diameter, pipe.length) > P.pipe_heat(
        40.0,
        18.0,
        pipe.diameter,
        pipe.length,
    )


def test_ventilation_follows_the_calibrated_curve():
    v = CFG.ventilation
    assert P.ventilation_rate(0.0, CFG) == pytest.approx(v.rate[0])
    assert P.ventilation_rate(1.0, CFG) == pytest.approx(v.rate[-1])
    rates = P.ventilation_rate(np.linspace(0, 1, 11), CFG)
    assert np.all(np.diff(rates) >= 0)


def test_balance_books_every_flow_and_keeps_the_rest_as_residual():
    f = P.balance(_state(radiation=200.0, lamps=1.0, vent=0.3), CFG)
    gains = f["sun"] + f["lamps"] + f["pipes"]
    losses = f["cover"] + f["ventilation"] + f["evaporation"]
    assert np.allclose(f["residual"], gains - losses)
    assert np.allclose(f["lamps"], CFG.lamps.power_w)
    assert np.allclose(
        f["sun"],
        200.0 * CFG.cover.transmission * (1 - CFG.cover.reflected_share),
    )
    assert np.allclose(f["pipes"], f[f"pipe:{LOWER}"] + f[f"pipe:{GROWTH}"])


def test_what_if_with_the_real_settings_changes_nothing():
    s = _state()
    w = P.what_if(s, CFG, vent=s["vent"], screen=s["screen"])
    assert np.allclose(w["pipes_change"], 0.0)
    assert np.allclose(w["ah"], s["ah_in"])


def test_closing_the_screen_saves_its_share_of_the_cover_loss():
    s = _state()
    w = P.what_if(s, CFG, screen=1.0)
    d_t = s["t_in"] - s["t_out"]
    expected = -CFG.cover.screen_saving * P.cover_u(s["wind"], 0.0, CFG) * d_t
    assert np.allclose(w["pipes_change"], expected)


def test_opening_the_vents_costs_heat_and_dries_the_air():
    s = _state()
    w = P.what_if(s, CFG, vent=0.5)
    assert (w["pipes_change"] > 0).all()
    assert (w["ah"] < s["ah_in"]).all()


def test_switching_lamps_on_gives_free_heat_and_more_transpiration():
    s = _state()
    w = P.what_if(s, CFG, lamps=1.0)
    # the lamp's heat replaces pipe heat, minus what its extra transpiration costs
    extra_e = CFG.lamps.power_w * CFG.transpiration.per_lamp_w / 1000
    expected = -CFG.lamps.power_w + extra_e / 3600 * P.LATENT_HEAT
    assert np.allclose(w["pipes_change"], expected)
    assert (w["ah"] > s["ah_in"]).all()


def test_unknown_setting_is_refused():
    with pytest.raises(ValueError, match="unknown setting"):
        P.what_if(_state(), CFG, heating=1.0)
