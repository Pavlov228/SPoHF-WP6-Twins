"""Tests for red.lamp — attenuation and the observed lamp model.

Link 2 predicts natural light only; the lamp part is observed and added back.
These pin that the addition is honest in both directions — nothing invented when
the lamps are off, nothing forgotten when they are on.
"""

import math
from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from wp6_data.red.dli.constants import SECONDS_PER_HOUR, UMOL_TO_MOL
from wp6_data.red.lamp import (
    DAYLIGHT_THRESHOLD,
    LAMP_FULL_ON,
    LAMP_THRESHOLD,
    RECENT_DAYS,
    LampModel,
    LampState,
    LampStatusConfig,
    compute_attenuation,
    derive_lamp_model,
    lamp_level_by_date,
    lamp_on_fraction,
    lamp_state_grid,
    load_lamp_status,
    observed_lamp_par,
)


def _series(index, values):
    return pd.DataFrame({"time": index, "value": values})


class TestCanopyAssembly:
    def _model(self, **kwargs):
        defaults = dict(
            attenuation=0.62, attenuation_days=90, power_par=168.0,
            hours_on=frozenset({23, 0, 1}), observed_days=14,
        )
        return LampModel(**{**defaults, **kwargs})

    def test_lamp_hours_add_the_observed_level(self):
        model = self._model()

        at_midnight = model.canopy_par(100.0, datetime(2026, 1, 1, 0, tzinfo=UTC))

        assert at_midnight == pytest.approx(100.0 * 0.62 + 168.0)

    def test_outside_lamp_hours_only_attenuation_applies(self):
        model = self._model()

        at_noon = model.canopy_par(100.0, datetime(2026, 1, 1, 12, tzinfo=UTC))

        assert at_noon == pytest.approx(62.0)

    def test_a_schedule_crossing_midnight_needs_no_special_case(self):
        """Holding hours as a set is why 23:00-01:00 works without wrap logic."""
        model = self._model()

        for hour in (23, 0, 1):
            assert model.lamp_par_at(datetime(2026, 1, 1, hour, tzinfo=UTC)) == 168.0
        for hour in (2, 22):
            assert model.lamp_par_at(datetime(2026, 1, 1, hour, tzinfo=UTC)) == 0.0

    def test_an_unlit_greenhouse_adds_nothing(self):
        model = self._model(power_par=None, hours_on=frozenset())

        assert model.is_lighting is False
        assert model.canopy_par(100.0, datetime(2026, 7, 1, 0, tzinfo=UTC)) == 62.0

    def test_negative_natural_light_does_not_subtract(self):
        """A linear model can predict below zero; that must not eat the lamps."""
        model = self._model()

        assert model.canopy_par(-50.0, datetime(2026, 1, 1, 0, tzinfo=UTC)) == 168.0


class TestDeriveLampModel:
    def _day(self, days=20, lamp_level=180.0, lamp_hours=(23, 0, 1)):
        index = pd.date_range("2025-12-01", periods=days * 24, freq="h", tz="UTC")
        hour = index.hour.to_numpy()
        natural = np.clip(300 * np.sin(np.pi * (hour - 8) / 8), 0, None)
        canopy = natural * 0.62
        lit = np.isin(hour, list(lamp_hours))
        canopy = np.where(lit, canopy + lamp_level, canopy)
        return _series(index, natural), _series(index, canopy)

    def test_detects_the_level_and_the_hours(self):
        natural, canopy = self._day()

        model = derive_lamp_model(natural, canopy)

        assert model.is_lighting
        assert model.power_par == pytest.approx(180.0, abs=1.0)
        assert model.hours_on == frozenset({23, 0, 1})

    def test_an_unlit_greenhouse_reports_no_lamp(self):
        natural, canopy = self._day(lamp_hours=())

        model = derive_lamp_model(natural, canopy)

        assert model.is_lighting is False
        assert model.hours_on == frozenset()

    def test_twilight_below_the_bar_is_not_mistaken_for_lamps(self):
        """At a 10.0 bar this fired on 20.4 µmol/m²/s of pre-dawn twilight and
        added it to every forecast."""
        natural, canopy = self._day(lamp_level=LAMP_THRESHOLD - 25.0)

        model = derive_lamp_model(natural, canopy)

        assert model.is_lighting is False

    def test_empty_inputs_yield_a_neutral_model_not_a_crash(self):
        empty = pd.DataFrame({"time": [], "value": []})

        model = derive_lamp_model(empty, empty)

        assert model.attenuation == 1.0
        assert model.is_lighting is False
        assert model.observed_days == 0

    def test_only_the_recent_window_sets_the_schedule(self):
        """A schedule that has since been switched off must not persist."""
        natural, canopy = self._day(days=40, lamp_hours=())
        old_index = pd.date_range("2025-10-01", periods=10 * 24, freq="h", tz="UTC")
        old_natural = _series(old_index, np.zeros(len(old_index)))
        old_canopy = _series(old_index, np.full(len(old_index), 200.0))

        model = derive_lamp_model(
            pd.concat([old_natural, natural]), pd.concat([old_canopy, canopy]),
        )

        assert model.is_lighting is False

    def test_daylight_hours_never_count_as_lamp_hours(self):
        natural, canopy = self._day()

        model = derive_lamp_model(natural, canopy)

        daylit = {h for h in range(24) if 8 < h < 16}
        assert not (model.hours_on & daylit)
        assert DAYLIGHT_THRESHOLD < LAMP_THRESHOLD


class TestAttenuationWindow:
    """Attenuation and the schedule want different windows; only one is recent_days.

    Red's real winter schedule runs 23:00-15:00, so a winter window has no
    lamp-free daylight hour and a raw ratio came out at 1.679 — more light at the
    canopy than at the roof above it. `compute_attenuation` now removes the
    measured lamp before taking the ratio, so such a window is identifiable
    again (0.565 on Nov 2025 - Feb 2026); what these pin is that an
    *unidentifiable* one is still reported rather than disguised.
    """

    def _winter(self, *, days=40, lamp_level=190.0, attenuation=0.63, flat=None):
        """Lamps 23:00-15:00, the schedule prod actually runs in midwinter.

        Cloud varies the above-lamp reading day to day, as it does in reality —
        without that spread no slope is identifiable and the fixture would be
        testing a degenerate case rather than a winter.
        """
        rows_above, rows_canopy = [], []
        lit = set(range(0, 16)) | {23}
        for offset in range(days):
            day = date(2026, 1, 20) - timedelta(days=offset)
            cloud = 0.3 + 0.7 * ((offset * 7) % 10) / 9
            for hour in range(24):
                above = (
                    flat[hour]
                    if flat is not None
                    else (
                        max(0.0, 220 * cloud * math.sin(math.pi * (hour - 8.5) / 7.5))
                        if 8.5 <= hour <= 16.0
                        else 0.0
                    )
                )
                canopy = above * attenuation + (lamp_level if hour in lit else 0.0)
                stamp = datetime(day.year, day.month, day.day, hour, tzinfo=UTC)
                rows_above.append({"time": stamp, "value": above})
                rows_canopy.append({"time": stamp, "value": canopy})
        return pd.DataFrame(rows_above), pd.DataFrame(rows_canopy)

    def test_a_lamp_lit_winter_window_is_identifiable_once_the_lamp_is_removed(self):
        """The 1.679 case: lamps run through the day, so the raw ratio exceeds 1."""
        above, canopy = self._winter(attenuation=0.63)

        raw = (canopy["value"].sum() / above["value"].sum())
        assert raw > 1.0, "fixture should reproduce the contaminated ratio"

        ratio, days = compute_attenuation(above, canopy)

        assert days > 0
        assert ratio == pytest.approx(0.63, abs=0.05)

    def test_a_window_with_no_usable_signal_is_refused(self):
        """Nothing to measure from is reported as 0 days, never as 'it is 1.0'."""
        empty = pd.DataFrame({"time": [], "value": []})
        assert compute_attenuation(empty, empty) == (1.0, 0)

    def test_unmeasurable_attenuation_is_labelled_not_disguised(self):
        """A flat window: no spread in the above-lamp reading, so no slope."""
        flat_hours = {h: (400.0 if 9 <= h <= 15 else 0.0) for h in range(24)}
        above, canopy = self._winter(lamp_level=0.0, flat=flat_hours)

        lamp = derive_lamp_model(above, canopy)

        assert lamp.attenuation_source in {"measured", "unknown"}
        if lamp.attenuation_source == "unknown":
            assert lamp.attenuation_days == 0

    def test_a_supplied_attenuation_is_used_and_labelled(self):
        """What the routes do: hand over the value the trained model fitted."""
        above, canopy = self._winter()
        lamp = derive_lamp_model(above, canopy, attenuation=0.629)

        assert lamp.attenuation == 0.629
        assert lamp.attenuation_source == "supplied"
        # ...and supplying it does not disturb the schedule half.
        assert lamp.is_lighting
        assert lamp.power_par == pytest.approx(190.0)

    def test_the_daylight_test_needs_no_attenuation(self):
        """Why the ratio test replaced the residual one: no circularity left.

        The lamp had to be known to measure attenuation, and attenuation was
        needed to find the lamp. Asking only whether the canopy outshone the roof
        breaks that — and, unlike a residual, its error does not grow with
        brightness, so summer mornings stop reading as lamps.
        """
        from wp6_data.red.lamp import _hourly, _schedule_over

        above, canopy = self._winter()
        frame = pd.concat(
            [_hourly(above).rename("above"), _hourly(canopy).rename("canopy")], axis=1
        ).dropna()

        hours, power = _schedule_over(frame)

        assert power == pytest.approx(190.0)
        assert any(h in hours for h in range(9, 16)), "daylight hours must be found"

    def test_a_lamp_free_window_still_measures_it(self):
        above, canopy = self._winter(lamp_level=0.0, attenuation=0.63)
        lamp = derive_lamp_model(above, canopy)
        assert lamp.attenuation_source == "measured"
        assert lamp.attenuation == pytest.approx(0.63, abs=0.01)
        assert not lamp.is_lighting


class TestDaytimeLampDetection:
    """Red runs its lamps through the short winter day (issues/054).

    Dark-hours-only detection found 8 of 17 hours and understated winter lamp DLI
    by 53% (5.43 against 11.55 mol/m²/day).
    """

    ATTENUATION = 0.63
    LAMP = 190.0

    def _day(self, lit_hours, *, days=30, peak=220.0, attenuation=None):
        """Midwinter: daylight 08:30-16:00, lamps on `lit_hours`."""
        attenuation = self.ATTENUATION if attenuation is None else attenuation
        rows_above, rows_canopy = [], []
        for offset in range(days):
            day = date(2026, 1, 20) - timedelta(days=offset)
            for hour in range(24):
                above = (
                    max(0.0, peak * math.sin(math.pi * (hour - 8.5) / 7.5))
                    if 8.5 <= hour <= 16.0
                    else 0.0
                )
                canopy = above * attenuation + (self.LAMP if hour in lit_hours else 0.0)
                stamp = datetime(day.year, day.month, day.day, hour, tzinfo=UTC)
                rows_above.append({"time": stamp, "value": above})
                rows_canopy.append({"time": stamp, "value": canopy})
        return pd.DataFrame(rows_above), pd.DataFrame(rows_canopy)

    def test_finds_lamps_running_through_the_day(self):
        lit = set(range(0, 16)) | {23}
        above, canopy = self._day(lit)

        lamp = derive_lamp_model(above, canopy, attenuation=self.ATTENUATION)

        assert lamp.hours_on == frozenset(lit)
        assert lamp.power_par == pytest.approx(self.LAMP)

    def test_a_night_only_schedule_gains_no_daylight_hours(self):
        """The dangerous direction: daylight must not invent lamps."""
        lit = {23, 0, 1, 2, 3, 4, 5, 6}
        above, canopy = self._day(lit)

        lamp = derive_lamp_model(above, canopy, attenuation=self.ATTENUATION)

        assert lamp.hours_on == frozenset(lit)

    def test_no_false_positive_when_attenuation_is_badly_wrong(self):
        """A wrong attenuation scales with `above`; the bar sits above its reach."""
        lit = {23, 0, 1, 2, 3, 4, 5, 6}
        above, canopy = self._day(lit)

        # 30% adrift in the direction that inflates the residual
        lamp = derive_lamp_model(above, canopy, attenuation=self.ATTENUATION * 0.7)

        assert lamp.hours_on == frozenset(lit), "daylight hours were invented"

    def test_without_a_lamp_level_no_daytime_claim_is_made(self):
        """No dark-hour lamp means no bar to threshold against, so no guess."""
        above, canopy = self._day(set())

        lamp = derive_lamp_model(above, canopy, attenuation=self.ATTENUATION)

        assert not lamp.is_lighting
        assert lamp.hours_on == frozenset()

    def test_the_lamp_dli_recovered_is_the_whole_schedule(self):
        lit = set(range(0, 16)) | {23}
        above, canopy = self._day(lit)

        lamp = derive_lamp_model(above, canopy, attenuation=self.ATTENUATION)
        schedule = lamp.hourly_schedule()
        lamp_dli = sum(schedule.values()) * SECONDS_PER_HOUR / UMOL_TO_MOL

        assert lamp_dli == pytest.approx(len(lit) * self.LAMP * SECONDS_PER_HOUR / UMOL_TO_MOL)


class TestLampStateGridOverlap:
    def test_daylight_hours_with_lamps_get_their_own_state(self):
        lit = set(range(0, 16)) | {23}
        above, canopy = TestDaytimeLampDetection()._day(lit, days=20)

        by_hour = lamp_state_grid(above, canopy).set_index("hour")["state"].to_dict()

        assert by_hour[12] == LampState.DAYLIGHT_LIT   # sun and lamps together
        assert by_hour[3] == LampState.LIT             # lamps alone

    def test_a_lamp_free_day_has_no_lamp_states(self):
        above, canopy = TestDaytimeLampDetection()._day(set(), days=20)

        states = set(lamp_state_grid(above, canopy)["state"])

        assert LampState.DAYLIGHT_LIT not in states
        assert LampState.LIT not in states


# ── Lamp state from LetsGrow ─────────────────────────────────────────────────

_CONFIG = LampStatusConfig(device="lg-lights", switch="on_tag", share="pct_tag")
_ATTENUATION = 0.62


def _status(index, switch, share):
    """LetsGrow's two lamp tags at ``index``, in the ``fetch_data_tsdb`` shape."""
    switch = np.broadcast_to(np.asarray(switch, dtype=float), len(index))
    share = np.broadcast_to(np.asarray(share, dtype=float), len(index))
    return pd.concat([
        pd.DataFrame({"device": _CONFIG.device, "sensor": _CONFIG.switch,
                      "time": index, "value": switch}),
        pd.DataFrame({"device": _CONFIG.device, "sensor": _CONFIG.share,
                      "time": index, "value": share}),
    ])


def _greenhouse(days, lamp_level, lit_hours, start="2025-12-01"):
    """Roof and canopy PAR plus LetsGrow, for lamps on at ``lit_hours`` every day.

    The daylight peak is bright enough that a lamp at ``lamp_level`` cannot lift
    the canopy above the roof — the case the PAR ratio test cannot see.
    """
    index = pd.date_range(start, periods=days * 24, freq="h", tz="UTC")
    hour = index.hour.to_numpy()
    natural = np.clip(1200 * np.sin(np.pi * (hour - 8) / 8), 0, None)
    lit = np.isin(hour, list(lit_hours))
    level = np.broadcast_to(np.asarray(lamp_level, dtype=float), len(index))
    canopy = natural * _ATTENUATION + np.where(lit, level, 0.0)
    status = _status(index, lit.astype(float), 100.0)
    return _series(index, natural), _series(index, canopy), status


class TestLampOnFraction:
    def test_a_whole_hour_at_full_share_is_one(self):
        index = pd.date_range("2026-01-01", periods=12, freq="5min", tz="UTC")

        fraction = lamp_on_fraction(_status(index, 1, 100), _CONFIG)

        assert fraction.iloc[0] == pytest.approx(1.0)

    def test_half_the_installation_is_half(self):
        index = pd.date_range("2026-01-01", periods=12, freq="5min", tz="UTC")

        fraction = lamp_on_fraction(_status(index, 1, 50), _CONFIG)

        assert fraction.iloc[0] == pytest.approx(0.5)

    def test_switched_off_is_zero_whatever_the_share_says(self):
        index = pd.date_range("2026-01-01", periods=12, freq="5min", tz="UTC")

        fraction = lamp_on_fraction(_status(index, 0, 100), _CONFIG)

        assert fraction.iloc[0] == 0.0

    def test_an_hour_letsgrow_did_not_report_is_absent_not_off(self):
        index = pd.date_range("2026-01-01", periods=12, freq="5min", tz="UTC")
        later = pd.date_range("2026-01-01 03:00", periods=12, freq="5min", tz="UTC")
        status = pd.concat([_status(index, 1, 100), _status(later, 1, 100)])

        fraction = lamp_on_fraction(status, _CONFIG)

        assert pd.Timestamp("2026-01-01 01:00", tz="UTC") not in fraction.index

    def test_no_letsgrow_rows_give_an_empty_series(self):
        empty = pd.DataFrame(columns=["device", "sensor", "time", "value"])

        assert lamp_on_fraction(empty, _CONFIG).empty


class TestLampLevel:
    def test_the_level_is_read_on_dark_hours_letsgrow_says_were_lit(self):
        natural, canopy, status = _greenhouse(10, 180.0, (23, 0, 1, 2))

        level = lamp_level_by_date(natural, canopy, lamp_on_fraction(status, _CONFIG))

        assert level.dropna().to_numpy() == pytest.approx(180.0)

    def test_the_level_follows_a_moved_sensor(self):
        """Same lamps at 100%, the sensor now under fewer of them: 188 -> 140."""
        days = 3 * RECENT_DAYS
        index_days = pd.date_range("2026-01-01", periods=days, freq="D")
        per_day = np.where(np.arange(days) < days // 2, 188.0, 140.0)
        natural, canopy, status = _greenhouse(
            days, np.repeat(per_day, 24), (23, 0, 1, 2), start="2026-01-01",
        )

        level = lamp_level_by_date(natural, canopy, lamp_on_fraction(status, _CONFIG))

        assert level[index_days[0].date()] == pytest.approx(188.0)
        assert level[index_days[-1].date()] == pytest.approx(140.0)

    def test_a_part_lit_hour_does_not_set_the_level(self):
        index = pd.date_range("2026-01-01", periods=48, freq="h", tz="UTC")
        dark = np.zeros(len(index))
        canopy = np.full(len(index), 90.0)
        half = _status(index, 1, 100 * (LAMP_FULL_ON - 0.1))

        level = lamp_level_by_date(
            _series(index, dark), _series(index, canopy), lamp_on_fraction(half, _CONFIG),
        )

        assert level.empty


class TestObservedLampPar:
    def test_a_bright_lit_hour_gets_its_lamp_although_the_ratio_test_misses_it(self):
        noon = 12
        natural, canopy, status = _greenhouse(10, 180.0, (23, 0, 1, noon))

        lamp, _ = observed_lamp_par(natural, canopy, lamp_on_fraction(status, _CONFIG))

        at_noon = lamp[lamp.index.hour == noon]
        assert at_noon.to_numpy() == pytest.approx(180.0)

    def test_hours_letsgrow_says_were_off_add_nothing(self):
        natural, canopy, status = _greenhouse(10, 180.0, (23, 0, 1))

        lamp, _ = observed_lamp_par(natural, canopy, lamp_on_fraction(status, _CONFIG))

        assert (lamp[lamp.index.hour.isin([3, 12, 20])] == 0.0).all()

    def test_a_day_with_the_lamps_on_adds_its_whole_lamp_dli(self):
        lit_hours = (23, 0, 1, 2, 12)
        natural, canopy, status = _greenhouse(10, 180.0, lit_hours)

        lamp, _ = observed_lamp_par(natural, canopy, lamp_on_fraction(status, _CONFIG))

        one_day = lamp[lamp.index.date == date(2025, 12, 5)].sum()
        expected = 180.0 * len(lit_hours) * SECONDS_PER_HOUR / UMOL_TO_MOL
        assert one_day * SECONDS_PER_HOUR / UMOL_TO_MOL == pytest.approx(expected)

    def test_hours_after_the_last_delivery_fall_back_to_the_detector(self):
        natural, canopy, status = _greenhouse(10, 180.0, (23, 0, 1, 2))
        delivered_until = pd.Timestamp("2025-12-08", tz="UTC")
        late = status[status["time"] < delivered_until]

        lamp, covered = observed_lamp_par(natural, canopy, lamp_on_fraction(late, _CONFIG))

        assert covered < delivered_until
        after = lamp[(lamp.index >= delivered_until) & (lamp.index.hour == 0)]
        assert not after.empty
        assert after.to_numpy() == pytest.approx(180.0)

    def test_without_letsgrow_the_whole_span_is_read_off_the_sensors(self):
        natural, canopy, _ = _greenhouse(10, 180.0, (23, 0, 1, 2))

        lamp, covered = observed_lamp_par(natural, canopy, pd.Series(dtype=float))

        assert covered is None
        assert lamp[lamp.index.hour == 0].to_numpy() == pytest.approx(180.0)


class TestLampStatusConfig:
    def test_red_declares_where_letsgrow_reports_the_lamps(self):
        from wp6_data.red import deps

        config = load_lamp_status(deps._METADATA_PATH)

        assert config.device.startswith("letsgrow-")

    def test_a_metadata_file_without_the_block_is_refused(self, tmp_path):
        path = tmp_path / "metadata.yaml"
        path.write_text("devices: {}\n", encoding="utf-8")

        with pytest.raises(ValueError, match="lamps"):
            load_lamp_status(path)
