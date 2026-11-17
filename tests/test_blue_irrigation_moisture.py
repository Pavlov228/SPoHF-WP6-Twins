"""Fused soil moisture: synthetic probes with known faults fuse back to the truth.

Every probe sees the same soil water through its own offset and gain, which is
the situation the research note describes. The fusion should undo the
offset/gain and drop the faults, and it should report what it dropped and why.
"""

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest
import yaml

from wp6_data.blue.irrigation import moisture
from wp6_data.blue.irrigation.config import (
    CONFIG_PATH,
    IrrigationConfig,
    parse_config,
)
from wp6_data.blue.irrigation.moisture import (
    DISAGREES,
    DROPOUT,
    EXCLUDED,
    FROZEN,
    MISSING,
    OK,
    SPIKE,
    fuse,
    to_hourly_frame,
)

PROBES = ("p-a", "p-b", "p-c", "p-d", "p-e")
#: (offset, gain) per probe: the same water, read through different calibrations.
CALIBRATION = {"p-a": (12.0, 20.0), "p-b": (18.0, 15.0), "p-c": (8.0, 25.0),
               "p-d": (20.0, 18.0), "p-e": (15.0, 22.0)}
DAYS = 120


def _raw_config() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _config(**overrides) -> IrrigationConfig:
    raw = _raw_config() | {"probes": list(PROBES)} | overrides
    return IrrigationConfig(**raw)


def _truth() -> pd.Series:
    """Soil water in 0..1: dry-downs reset by a wetting every 9 days, plus a daily wiggle."""
    index = pd.date_range("2025-04-01", periods=DAYS * 24, freq="h", tz=UTC)
    hours = np.arange(len(index))
    since_wetting = hours % (9 * 24)
    drydown = 0.9 * np.exp(-since_wetting / (24 * 6)) + 0.05
    daily = 0.02 * np.sin(2 * np.pi * hours / 24)
    return pd.Series(drydown + daily, index=index)


def _probes(truth: pd.Series, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        p: off + gain * truth + rng.normal(0, 0.05, len(truth))
        for p, (off, gain) in CALIBRATION.items()
    })


def _scaled_truth(truth: pd.Series, config: IrrigationConfig) -> pd.Series:
    f = config.fusion
    lo, hi = truth.quantile(f.reference_low_quantile), truth.quantile(f.reference_high_quantile)
    return (truth - lo) / (hi - lo)


def test_clean_probes_with_different_calibrations_fuse_to_the_truth():
    config = _config()
    truth = _truth()

    result = fuse(_probes(truth), config)

    error = (result.fused - _scaled_truth(truth, config)).abs()
    assert error.max() < 0.05
    assert (result.coverage == len(PROBES)).all()


def test_a_probe_that_loses_contact_is_dropped_and_padded():
    config = _config()
    truth = _truth()
    raw = _probes(truth)
    lost = slice("2025-06-01", "2025-06-10")
    raw.loc[lost, "p-c"] = 0.0

    result = fuse(raw, config)

    assert (result.reasons.loc[lost, "p-c"] == DROPOUT).all()
    pad = pd.Timedelta(hours=config.fusion.dropout_pad_hours)
    edge = pd.Timestamp("2025-06-01", tz=UTC) - pad
    assert result.reasons.loc[edge, "p-c"] == DROPOUT
    assert (result.coverage.loc[lost] == len(PROBES) - 1).all()
    error = (result.fused - _scaled_truth(truth, config)).abs()
    assert error.loc[lost].max() < 0.05


def test_a_one_hour_spike_is_dropped_but_a_rain_step_is_kept():
    config = _config()
    truth = _truth()
    raw = _probes(truth)
    spike_at = pd.Timestamp("2025-05-15 12:00", tz=UTC)
    raw.loc[spike_at, "p-a"] += 15.0

    result = fuse(raw, config)

    assert result.reasons.loc[spike_at, "p-a"] == SPIKE
    # The wettings in the truth are hour-sharp steps, and none may read as a spike.
    steps = truth.index[truth.diff() > 0.5]
    assert len(steps) > 0
    assert (result.reasons.loc[steps] != SPIKE).all().all()


def test_a_probe_that_moves_against_the_others_is_masked_for_that_stretch():
    config = _config()
    truth = _truth()
    raw = _probes(truth)
    wrong = slice("2025-06-01", "2025-07-15")
    off, gain = CALIBRATION["p-d"]
    raw.loc[wrong, "p-d"] = off + gain * (1 - truth.loc[wrong])

    result = fuse(raw, config)

    middle = slice("2025-06-15", "2025-07-01")
    assert (result.reasons.loc[middle, "p-d"] == DISAGREES).all()
    # Outside the inverted stretch (and its window), p-d is trusted again.
    assert (result.reasons.loc["2025-04-15":"2025-05-15", "p-d"] == OK).all()
    error = (result.fused - _scaled_truth(truth, config)).abs()
    assert error.loc[middle].max() < 0.05


def test_a_manual_exclusion_wins_over_the_mechanical_checks():
    excl = {"probe": "p-b", "start": "2025-05-01T00:00:00Z",
            "end": "2025-05-03T00:00:00Z", "reason": "probe reseated"}
    config = _config(exclude=[excl])
    raw = _probes(_truth())
    raw.loc["2025-05-01":"2025-05-02", "p-b"] = 0.0

    result = fuse(raw, config)

    assert (result.reasons.loc["2025-05-01":"2025-05-02", "p-b"] == EXCLUDED).all()


def test_a_silent_probe_is_reported_missing_not_dropped():
    start = datetime(2025, 4, 1, tzinfo=UTC)
    end = datetime(2025, 4, 3, tzinfo=UTC)
    rows = pd.DataFrame({
        "device": ["p-a", "p-a"],
        "time": [start, end],
        "value": [20.0, 21.0],
    })

    frame = to_hourly_frame(rows, PROBES, start, end)

    assert list(frame.columns) == list(PROBES)
    assert len(frame) == 49
    assert frame["p-b"].isna().all()
    result = fuse(frame, _config())
    assert (result.reasons["p-b"] == MISSING).all()


def test_probes_must_be_declared_in_metadata():
    raw = _raw_config()
    with pytest.raises(ValueError, match="not declared in metadata"):
        parse_config(raw, known_devices=set())


def test_the_shipped_config_names_only_known_devices():
    from wp6_data.blue.deps import metadata

    config = parse_config(_raw_config(), set(metadata.devices))
    assert config.probes


def test_unknown_fusion_knobs_are_rejected():
    raw = _raw_config()
    known = set(raw["probes"]) | {raw["rain"]["station_device"]}
    parse_config(raw, known)  # valid as shipped
    raw["fusion"] = raw["fusion"] | {"flor": 2.0}
    with pytest.raises(ValueError, match="flor"):
        parse_config(raw, known)


@pytest.mark.asyncio
async def test_load_fused_asks_for_hourly_rows_covering_every_probe():
    calls = {}

    class FakeProvider:
        async def fetch_data(self, **kwargs):
            calls.update(kwargs)
            return pd.DataFrame(columns=["device", "sensor", "time", "value"])

    config = _config()
    now = datetime(2025, 4, 2, tzinfo=UTC)

    result = await moisture.load_fused(FakeProvider(), config, now=now)

    assert calls["bucket"] == pd.Timedelta(hours=1)
    assert calls["device_names"] == list(PROBES)
    hours = len(result.fused)
    assert calls["limit"] >= hours * len(PROBES)


def test_quality_table_shares_add_up_and_reasons_are_counted():
    config = _config()
    raw = _probes(_truth())
    raw.loc["2025-06-01":"2025-06-03", "p-c"] = 0.0
    result = fuse(raw, config)
    start, end = result.fused.index[0], result.fused.index[-1]

    table = moisture.quality_table(result, start, end)

    assert table["share"].sum() == pytest.approx(1.0, abs=0.02)
    assert table.loc["p-c", DROPOUT] > 0
    assert (table["used_h"] + table.drop(columns=["used_h", "dry_ref", "wet_ref", "share"])
            .sum(axis=1) == len(result.fused)).all()


@pytest.mark.asyncio
async def test_irrigation_page_renders_the_fused_stream(monkeypatch):
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient

    from wp6_data.blue.routes import irrigation
    from wp6_data.shared.auth import verify_session_user
    from wp6_data.shared.templates.config import configure_dashboard

    configure_dashboard("blue")
    truth = _truth()
    probes = _probes(truth)
    rows = probes.stack().rename("value").reset_index()
    rows.columns = ["time", "device", "value"]
    rows["sensor"] = _config().sensor

    class FakeProvider:
        async def fetch_data(self, **kwargs):
            return rows

    monkeypatch.setattr(irrigation, "_provider", FakeProvider())
    monkeypatch.setattr(irrigation, "load_config", lambda: _config(history_start="2025-04-01"))
    irrigation._fused_cache.clear()

    async def weather_down():
        raise ConnectionError("open-meteo unreachable")

    # Weather is context: its outage must not take the moisture page down.
    monkeypatch.setattr(irrigation, "irrigation_weather", weather_down)

    app = FastAPI()
    app.include_router(irrigation.router)
    app.dependency_overrides[verify_session_user] = lambda: "test-user"
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/irrigation?year=2025")

    irrigation._fused_cache.clear()
    assert response.status_code == 200
    assert "Fused soil moisture" in response.text
    assert "Probe quality 2025" in response.text
    for probe in PROBES:
        assert probe in response.text


def test_spread_band_never_bridges_a_gap():
    """Plotly's tonexty fill drew wedges across missing stretches; each
    stretch must be its own closed polygon, separated by a break."""
    from wp6_data.blue.routes.irrigation import band_polygons

    index = pd.date_range("2025-06-01", periods=10, freq="h", tz=UTC)
    low = pd.Series([0.2, 0.2, 0.2, np.nan, np.nan, 0.3, 0.3, 0.3, 0.3, 0.3], index=index)
    high = low + 0.1

    xs, ys = band_polygons(low, high)

    breaks = [i for i, x in enumerate(xs) if x is None]
    assert len(breaks) == 1
    first, second = xs[:breaks[0]], xs[breaks[0] + 1:]
    # Each polygon walks its stretch forward along the top and back along the bottom.
    assert first == [*index[:3], *index[:3][::-1]]
    assert second == [*index[5:], *index[5:][::-1]]
    assert ys[breaks[0]] is None
    assert np.allclose(ys[:3], high.iloc[:3]) and np.allclose(ys[3:6], low.iloc[:3][::-1])


def test_a_frozen_probe_is_masked_so_a_thaw_does_not_read_as_rain():
    config = _config()
    truth = _truth()
    raw = _probes(truth)
    frost = slice("2025-05-01", "2025-05-04")
    # Ice reads like dry soil: during the frost every probe "dries" sharply.
    raw.loc[frost] = raw.loc[frost] * 0.3
    temperature = pd.DataFrame(10.0, index=raw.index, columns=raw.columns)
    temperature.loc[frost] = config.fusion.frozen_below_c - 1.0
    # A probe with no temperature reading is not presumed frozen.
    temperature.loc[frost, "p-e"] = np.nan

    result = fuse(raw, config, soil_temperature=temperature)

    assert (result.reasons.loc[frost, ["p-a", "p-b", "p-c", "p-d"]] == FROZEN).all().all()
    assert (result.reasons.loc[frost, "p-e"] != FROZEN).all()
    assert (result.coverage.loc[frost] <= 1).all()


def test_probe_view_draws_every_reading_coloured_by_how_it_was_used():
    from wp6_data.blue.routes.irrigation import PROBE_READING_STYLES, probe_figure

    config = _config()
    raw = _probes(_truth())
    raw.loc["2025-06-01":"2025-06-03", "p-c"] = 0.0
    result = fuse(raw, config)
    start, end = result.raw.index[0], result.raw.index[-1]

    fig = probe_figure(result, start, end)

    drawn = {}
    for trace in fig.data:
        drawn[trace.name] = drawn.get(trace.name, 0) + int(np.isfinite(
            np.asarray(trace.y, dtype=float)).sum())
    # Every reading appears exactly once, as used or under its drop reason.
    assert sum(drawn.values()) == int(result.raw.notna().sum().sum())
    assert drawn[PROBE_READING_STYLES[DROPOUT][0]] > 0
    assert drawn[PROBE_READING_STYLES[OK][0]] > 0
    # Each label sits in the legend once, not once per probe.
    legend = [t.name for t in fig.data if t.showlegend]
    assert len(legend) == len(set(legend))


@pytest.mark.asyncio
async def test_probe_readings_load_on_demand_as_json(monkeypatch):
    import json

    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient

    from wp6_data.blue.routes import irrigation
    from wp6_data.shared.auth import verify_session_user

    result = fuse(_probes(_truth()), _config())

    async def fake_fused():
        return result

    monkeypatch.setattr(irrigation, "fused_moisture", fake_fused)
    app = FastAPI()
    app.include_router(irrigation.router)
    app.dependency_overrides[verify_session_user] = lambda: "test-user"
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/irrigation/probes.json?year=2025")

    assert response.status_code == 200
    fig = json.loads(response.text)
    assert {"data", "layout"} <= fig.keys()
    assert len(fig["data"]) >= len(PROBES)
