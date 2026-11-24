"""Tests for the per-hour lamp state grid and daily counts behind /dli/lamps."""

from datetime import UTC, date, datetime, timedelta

import pandas as pd
import pytest

from wp6_data.red import deps
from wp6_data.red.lamp import (
    LIT_HOUR_SHARE,
    LampState,
    daily_light_hours,
    lamp_state_grid,
    load_daylight,
    radiation_without_frozen_hours,
)

THRESHOLD = deps.daylight.threshold
SUN = THRESHOLD + 300.0


def _hourly(day: date, by_hour: dict[int, float], days: int = 1) -> pd.Series:
    index, values = [], []
    for d in (day + timedelta(days=o) for o in range(days)):
        for hour, value in by_hour.items():
            index.append(datetime(d.year, d.month, d.day, hour, tzinfo=UTC))
            values.append(value)
    return pd.Series(values, index=pd.DatetimeIndex(index), dtype=float)


def _winter_day() -> tuple[dict[int, float], dict[int, float]]:
    """Sun 09-15; lamps 04-06 and at a bright midday, where a PAR test misses them."""
    radiation = {h: (SUN if 9 <= h <= 15 else THRESHOLD) for h in range(24)}
    lamps = {h: (1.0 if h in (4, 5, 6, 12) else 0.0) for h in range(24)}
    return radiation, lamps


class TestLampStateGrid:
    day = date(2026, 1, 15)

    def test_classifies_every_state(self):
        radiation, lamps = _winter_day()

        grid = lamp_state_grid(_hourly(self.day, radiation), _hourly(self.day, lamps), THRESHOLD)

        by_hour = grid.set_index("hour")["state"].to_dict()
        assert by_hour[5] == LampState.LIT
        assert by_hour[12] == LampState.DAYLIGHT_LIT
        assert by_hour[13] == LampState.DAYLIGHT
        assert by_hour[20] == LampState.DARK

    def test_radiation_at_the_threshold_is_not_daylight(self):
        """The station logs whole W/m², so twilight reads exactly the bar."""
        grid = lamp_state_grid(
            _hourly(self.day, {7: THRESHOLD}), _hourly(self.day, {7: 0.0}), THRESHOLD,
        )
        assert grid["state"].tolist() == [LampState.DARK]

    def test_a_mostly_off_hour_is_not_lit(self):
        grid = lamp_state_grid(
            _hourly(self.day, {5: THRESHOLD}),
            _hourly(self.day, {5: LIT_HOUR_SHARE - 0.1}),
            THRESHOLD,
        )
        assert grid["state"].tolist() == [LampState.DARK]

    def test_hours_either_series_is_missing_are_absent_not_dark(self):
        radiation, lamps = _winter_day()
        lamps_early = {h: v for h, v in lamps.items() if h < 12}

        grid = lamp_state_grid(
            _hourly(self.day, radiation), _hourly(self.day, lamps_early), THRESHOLD,
        )

        assert sorted(grid["hour"]) == list(range(12))

    def test_carries_the_readings_behind_each_verdict(self):
        radiation, lamps = _winter_day()

        grid = lamp_state_grid(_hourly(self.day, radiation), _hourly(self.day, lamps), THRESHOLD)

        row = grid.set_index("hour").loc[12]
        assert row["radiation"] == SUN
        assert row["on_fraction"] == 1.0

    def test_empty_inputs_give_an_empty_grid(self):
        empty = pd.Series(dtype=float)
        assert lamp_state_grid(empty, empty, THRESHOLD).empty


class TestDailyLightHours:
    def test_counts_lamps_with_and_without_sun_and_all_lit_hours(self):
        day = date(2026, 1, 15)
        radiation, lamps = _winter_day()
        grid = lamp_state_grid(_hourly(day, radiation), _hourly(day, lamps), THRESHOLD)

        daily = daily_light_hours(grid).set_index("date")

        lamp_hours = sum(1 for v in lamps.values() if v >= LIT_HOUR_SHARE)
        sun_hours = {h for h, v in radiation.items() if v > THRESHOLD}
        lit_hours = {h for h, v in lamps.items() if v >= LIT_HOUR_SHARE}
        assert daily.loc[day, "lamp_hours"] == lamp_hours
        assert daily.loc[day, "light_hours"] == len(sun_hours | lit_hours)

    def test_a_day_with_hours_missing_is_left_out(self):
        full = date(2026, 1, 15)
        radiation, lamps = _winter_day()
        rad = _hourly(full, radiation, days=2)
        lamp = _hourly(full, lamps, days=2).iloc[:-1]  # second day lacks its 23:00

        daily = daily_light_hours(lamp_state_grid(rad, lamp, THRESHOLD))

        assert list(daily["date"]) == [full]

    def test_empty_grid_gives_empty_frame(self):
        empty = pd.Series(dtype=float)
        assert daily_light_hours(lamp_state_grid(empty, empty, THRESHOLD)).empty


class TestDaylightConfig:
    def test_red_metadata_names_a_letsgrow_device(self):
        config = load_daylight(deps._METADATA_PATH)

        assert config.device.startswith("letsgrow-")

    def test_a_metadata_file_without_the_block_is_refused(self, tmp_path):
        path = tmp_path / "metadata.yaml"
        path.write_text("devices: {}\n", encoding="utf-8")

        with pytest.raises(ValueError, match="daylight"):
            load_daylight(path)


class TestFrozenRadiation:
    def _stats(self, rows: dict[int, tuple[float, float, float]]) -> pd.DataFrame:
        index = pd.DatetimeIndex([datetime(2026, 6, 8, h, tzinfo=UTC) for h in rows])
        return pd.DataFrame(list(rows.values()), index=index, columns=["mean", "min", "max"])

    def test_a_flat_non_zero_hour_is_dropped(self):
        stats = self._stats({2: (146.0, 146.0, 146.0), 12: (400.0, 380.0, 420.0)})

        kept = radiation_without_frozen_hours(stats)

        assert kept.index.hour.tolist() == [12]

    def test_a_flat_zero_hour_is_night_and_kept(self):
        stats = self._stats({2: (0.0, 0.0, 0.0)})

        assert radiation_without_frozen_hours(stats).tolist() == [0.0]

    def test_empty_stats_give_an_empty_series(self):
        empty = pd.DataFrame(columns=["mean", "min", "max"], dtype=float)
        assert radiation_without_frozen_hours(empty).empty


class TestHoursByYearChart:
    def _daily(self) -> pd.DataFrame:
        return pd.DataFrame({
            "date": [date(2025, 1, 15), date(2026, 1, 15)],
            "lamp_hours": [16, 15],
            "light_hours": [18, 17],
        })

    def test_one_line_per_year_of_the_chosen_count(self):
        from wp6_data.red.routes.dli.lamps import LAMP_HOURS, LIGHT_HOURS, _make_hours_by_year

        lamps = _make_hours_by_year(self._daily(), LAMP_HOURS)
        light = _make_hours_by_year(self._daily(), LIGHT_HOURS)

        assert [t.name for t in lamps.data if t.showlegend is not False] == ["2025", "2026"]
        daily_lines = [t for t in lamps.data if t.yaxis != "y2"]
        assert [list(t.y) for t in daily_lines] == [[16], [15]]
        assert [list(t.y) for t in light.data if t.yaxis != "y2"] == [[18], [17]]

    def test_years_share_one_calendar(self):
        from wp6_data.red.routes.dli.lamps import LAMP_HOURS, _make_hours_by_year

        fig = _make_hours_by_year(self._daily(), LAMP_HOURS)

        assert fig.data[0].x == fig.data[1].x


class TestMonthEndTotals:
    def _year(self, first: date, days: int, per_day: int) -> pd.DataFrame:
        dates = [first + timedelta(days=n) for n in range(days)]
        return pd.DataFrame({"lamp_hours": [per_day] * days}, index=pd.Index(dates, name="date"))

    def test_sums_the_year_to_each_month_end_the_data_reaches(self):
        from wp6_data.red.routes.dli.lamps import _month_end_totals

        totals = _month_end_totals(self._year(date(2026, 1, 1), 70, 10), "lamp_hours")

        assert totals["date"].tolist() == [date(2026, 1, 31), date(2026, 2, 28)]
        assert totals["total"].tolist() == [31 * 10, 59 * 10]

    def test_a_late_starting_year_says_how_many_days_it_counts(self):
        from wp6_data.red.routes.dli.lamps import _month_end_totals

        totals = _month_end_totals(self._year(date(2023, 2, 11), 18, 10), "lamp_hours")

        row = totals.iloc[0]
        assert row["date"] == date(2023, 2, 28)
        assert (row["counted"], row["elapsed"]) == (18, 31 + 28)
