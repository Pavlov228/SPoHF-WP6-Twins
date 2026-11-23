"""Tests for the per-hour lamp state grid behind /dli/lamps."""

from datetime import UTC, date, datetime, timedelta

import pandas as pd

from wp6_data.red.lamp import (
    DAYLIGHT_THRESHOLD,
    LAMP_SOURCE_LETSGROW,
    LAMP_SOURCE_SENSORS,
    LAMP_THRESHOLD,
    LIT_HOUR_SHARE,
    LampState,
    lamp_state_grid,
)


def _frame(day: date, values_by_hour: dict[int, float], days: int = 1) -> pd.DataFrame:
    rows = [
        {"time": datetime(d.year, d.month, d.day, hour, tzinfo=UTC), "value": value}
        for d in (day - timedelta(days=o) for o in range(days))
        for hour, value in values_by_hour.items()
    ]
    return pd.DataFrame(rows)


class TestLampStateGrid:
    def _winter_day(self):
        """Short daylight 09-15, lamps 04-06."""
        above = {h: (400.0 if 9 <= h <= 15 else 0.0) for h in range(24)}
        canopy = {h: value * 0.7 for h, value in above.items()}
        canopy.update({h: LAMP_THRESHOLD * 3 for h in (4, 5, 6)})
        return above, canopy

    def test_classifies_lamp_daylight_and_dark(self):
        above, canopy = self._winter_day()
        day = date(2026, 1, 15)
        grid = lamp_state_grid(_frame(day, above), _frame(day, canopy))

        by_hour = grid.set_index("hour")["state"].to_dict()
        assert by_hour[5] == LampState.LIT
        # The sensors alone cannot rule a lamp out under the sun.
        assert by_hour[12] == LampState.DAYLIGHT_UNKNOWN
        assert by_hour[20] == LampState.DARK

    def test_long_day_has_no_lit_hours(self):
        day = date(2026, 6, 21)
        above = {h: (400.0 if 4 <= h <= 20 else 0.0) for h in range(24)}
        canopy = {h: value * 0.7 for h, value in above.items()}
        grid = lamp_state_grid(_frame(day, above), _frame(day, canopy))
        assert not (grid["state"] == LampState.LIT).any()

    def test_twilight_below_the_lamp_bar_is_not_a_lamp(self):
        """The false positive the bar was raised for: pre-dawn glow, not lighting.

        Needs a real day around it — a lone twilight hour is a day the sensor was
        not measuring, which the grid drops rather than classifies.
        """
        day = date(2026, 9, 15)
        above = {h: (500.0 if 8 <= h <= 17 else 0.0) for h in range(24)}
        above[4] = DAYLIGHT_THRESHOLD - 1
        canopy = {h: v * 0.7 for h, v in above.items()}
        canopy[4] = LAMP_THRESHOLD - 1

        grid = lamp_state_grid(_frame(day, above), _frame(day, canopy))

        assert grid.set_index("hour").loc[4, "state"] == LampState.DARK

    def test_hours_with_one_sensor_silent_are_absent(self):
        day = date(2026, 1, 15)
        readings = {h: (500.0 if 9 <= h <= 15 else 0.0) for h in range(24)}
        above = _frame(day, readings)
        canopy = _frame(day, {h: v * 0.7 for h, v in readings.items() if h < 12})

        grid = lamp_state_grid(above, canopy)

        assert sorted(grid["hour"]) == list(range(12))

    def test_a_day_the_sensor_was_not_measuring_is_dropped_not_drawn_dark(self):
        """Red's PAR sensors were out of the greenhouse for six weeks in 2026.

        A switched-off sensor reports zeros, not nothing, so without this the
        calendar draws solid midsummer night where there is simply no reading.
        """
        real_day = date(2026, 6, 21)
        daylight = {h: (600.0 if 5 <= h <= 20 else 0.0) for h in range(24)}
        off = dict.fromkeys(range(24), 0.0)

        grid = lamp_state_grid(
            pd.concat([_frame(real_day, daylight),
                       _frame(real_day - timedelta(days=1), off)]),
            pd.concat([_frame(real_day, {h: v * 0.7 for h, v in daylight.items()}),
                       _frame(real_day - timedelta(days=1), off)]),
        )

        assert set(grid["date"]) == {real_day}

    def test_empty_inputs_give_an_empty_grid(self):
        empty = pd.DataFrame(columns=["time", "value"])
        grid = lamp_state_grid(empty, empty)
        assert grid.empty
        assert list(grid.columns) == [
            "date", "hour", "state", "above_par", "canopy_par", "source",
        ]

    def test_carries_the_readings_behind_each_verdict(self):
        above, canopy = self._winter_day()
        day = date(2026, 1, 15)
        grid = lamp_state_grid(_frame(day, above), _frame(day, canopy))
        lit = grid[grid["state"] == LampState.LIT].iloc[0]
        assert lit["canopy_par"] == LAMP_THRESHOLD * 3
        assert lit["above_par"] == 0.0


def _on_fraction(day: date, fraction_by_hour: dict[int, float]) -> pd.Series:
    """LetsGrow's verdict per hour, in the shape ``lamp_on_fraction`` returns."""
    index = pd.DatetimeIndex(
        [datetime(day.year, day.month, day.day, h, tzinfo=UTC) for h in fraction_by_hour]
    )
    return pd.Series(list(fraction_by_hour.values()), index=index, dtype=float)


class TestLampStateGridFromLetsGrow:
    """Where LetsGrow has reported, the switch decides; the sun still comes from PAR."""

    day = date(2026, 1, 15)

    def _bright_winter_day(self):
        """Lamps on at 04-06 and at a bright midday, where the sun hides them."""
        above = {h: (1200.0 if 9 <= h <= 15 else 0.0) for h in range(24)}
        canopy = {h: value * 0.62 for h, value in above.items()}
        canopy.update({h: 180.0 for h in (4, 5, 6)})
        canopy[12] += 180.0
        return above, canopy

    def test_a_bright_lit_midday_shows_as_daylight_plus_lamps(self):
        above, canopy = self._bright_winter_day()
        lit = {h: (1.0 if h in (4, 5, 6, 12) else 0.0) for h in range(24)}

        grid = lamp_state_grid(
            _frame(self.day, above), _frame(self.day, canopy), _on_fraction(self.day, lit),
        )

        by_hour = grid.set_index("hour")["state"].to_dict()
        assert by_hour[12] == LampState.DAYLIGHT_LIT
        assert by_hour[5] == LampState.LIT
        assert by_hour[13] == LampState.DAYLIGHT
        assert by_hour[20] == LampState.DARK

    def test_letsgrow_off_overrides_a_bright_canopy(self):
        """A canopy brighter than the roof is no lamp if the switch was off."""
        above, canopy = self._bright_winter_day()
        off = dict.fromkeys(range(24), 0.0)

        grid = lamp_state_grid(
            _frame(self.day, above), _frame(self.day, canopy), _on_fraction(self.day, off),
        )

        assert not grid["state"].isin([LampState.LIT, LampState.DAYLIGHT_LIT]).any()

    def test_a_mostly_off_hour_is_not_lit(self):
        above, canopy = self._bright_winter_day()
        brief = {h: 0.0 for h in range(24)} | {5: LIT_HOUR_SHARE - 0.1}

        grid = lamp_state_grid(
            _frame(self.day, above), _frame(self.day, canopy), _on_fraction(self.day, brief),
        )

        assert grid.set_index("hour").loc[5, "state"] == LampState.DARK

    def test_hours_letsgrow_has_not_delivered_keep_the_sensor_reading(self):
        above, canopy = self._bright_winter_day()
        early = {h: 0.0 for h in range(4)}

        grid = lamp_state_grid(
            _frame(self.day, above), _frame(self.day, canopy), _on_fraction(self.day, early),
        )

        by_hour = grid.set_index("hour")
        assert by_hour.loc[2, "source"] == LAMP_SOURCE_LETSGROW
        assert by_hour.loc[5, "source"] == LAMP_SOURCE_SENSORS
        assert by_hour.loc[5, "state"] == LampState.LIT

    def test_a_sunlit_hour_letsgrow_missed_is_unknown_not_off(self):
        """An outage must not draw as lamps switched off at midday."""
        above, canopy = self._bright_winter_day()
        early = {h: 0.0 for h in range(4)}

        grid = lamp_state_grid(
            _frame(self.day, above), _frame(self.day, canopy), _on_fraction(self.day, early),
        )

        by_hour = grid.set_index("hour")
        assert by_hour.loc[13, "source"] == LAMP_SOURCE_SENSORS
        assert by_hour.loc[13, "state"] == LampState.DAYLIGHT_UNKNOWN
        assert by_hour.loc[20, "state"] == LampState.DARK
