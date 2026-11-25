"""Tests for wp6_data.shared.charts."""

from datetime import date, timedelta

import pandas as pd

from wp6_data.shared.charts import (
    make_daily_by_year_chart,
    make_dual_axis_chart,
    make_line_chart,
    prepare_comparison,
    year_point_labels,
    year_to_date,
)


def _sample_df():
    """Create a minimal sensor DataFrame for testing."""
    return pd.DataFrame(
        {
            "device": ["d1", "d1", "d2", "d2"],
            "sensor": ["temp", "temp", "temp", "temp"],
            "time": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-01", "2026-01-02"]),
            "value": [20.0, 21.0, 19.0, 20.5],
        }
    )


def _dual_axis_df():
    """Create a DataFrame with two sensor types."""
    return pd.DataFrame(
        {
            "device": ["d1", "d1", "d1", "d1"],
            "sensor": ["temp", "temp", "humidity", "humidity"],
            "time": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-01", "2026-01-02"]),
            "value": [20.0, 21.0, 55.0, 60.0],
        }
    )


class TestMakeLineChart:
    def test_returns_figure(self):
        fig = make_line_chart(_sample_df())
        assert fig is not None
        assert hasattr(fig, "data")

    def test_default_title(self):
        fig = make_line_chart(_sample_df())
        assert fig.layout.title.text == "Sensor Readings Over Time"

    def test_custom_title(self):
        fig = make_line_chart(_sample_df(), title="My Chart")
        assert fig.layout.title.text == "My Chart"

    def test_series_per_device(self):
        fig = make_line_chart(_sample_df())
        # Two devices → two traces
        assert len(fig.data) == 2

    def test_series_names_contain_device_and_sensor(self):
        fig = make_line_chart(_sample_df())
        names = {t.name for t in fig.data}
        assert "d1 | temp" in names
        assert "d2 | temp" in names

    def test_height_is_600(self):
        fig = make_line_chart(_sample_df())
        assert fig.layout.height == 600

    def test_hovermode_unified(self):
        fig = make_line_chart(_sample_df())
        assert fig.layout.hovermode == "x unified"

    def test_does_not_mutate_input(self):
        df = _sample_df()
        cols_before = list(df.columns)
        make_line_chart(df)
        assert list(df.columns) == cols_before
        assert "series" not in df.columns


class TestMakeDualAxisChart:
    def test_returns_figure(self):
        fig = make_dual_axis_chart(_dual_axis_df(), "temp", "humidity")
        assert fig is not None

    def test_has_traces_for_both_axes(self):
        fig = make_dual_axis_chart(_dual_axis_df(), "temp", "humidity")
        assert len(fig.data) == 2

    def test_trace_names(self):
        fig = make_dual_axis_chart(_dual_axis_df(), "temp", "humidity")
        names = {t.name for t in fig.data}
        assert "d1 | temp" in names
        assert "d1 | humidity" in names

    def test_default_title(self):
        fig = make_dual_axis_chart(_dual_axis_df(), "temp", "humidity")
        assert fig.layout.title.text == "temp vs humidity"

    def test_custom_title(self):
        fig = make_dual_axis_chart(_dual_axis_df(), "temp", "humidity", title="Compare")
        assert fig.layout.title.text == "Compare"

    def test_right_axis_uses_dashed_lines(self):
        fig = make_dual_axis_chart(_dual_axis_df(), "temp", "humidity")
        # The humidity trace (right axis) should be dashed
        right_trace = [t for t in fig.data if "humidity" in t.name][0]
        assert right_trace.line.dash == "dash"

    def test_yaxis_labels(self):
        fig = make_dual_axis_chart(_dual_axis_df(), "temp", "humidity")
        assert fig.layout.yaxis.title.text == "temp"
        assert fig.layout.yaxis2.title.text == "humidity"

    def test_height_is_600(self):
        fig = make_dual_axis_chart(_dual_axis_df(), "temp", "humidity")
        assert fig.layout.height == 600

    def test_empty_side(self):
        """Chart with no data for one axis should still render."""
        df = _dual_axis_df()
        fig = make_dual_axis_chart(df, "temp", "nonexistent")
        # Only temp traces
        assert len(fig.data) == 1

    def test_multiple_devices(self):
        df = pd.DataFrame(
            {
                "device": ["d1", "d1", "d2", "d2", "d1", "d1", "d2", "d2"],
                "sensor": ["temp"] * 4 + ["hum"] * 4,
                "time": pd.to_datetime(["2026-01-01", "2026-01-02"] * 4),
                "value": [20, 21, 19, 20, 55, 60, 50, 52],
            }
        )
        fig = make_dual_axis_chart(df, "temp", "hum")
        # 2 devices × 2 sensors = 4 traces
        assert len(fig.data) == 4


def _left_df():
    return pd.DataFrame({
        "device": ["d1", "d1"],
        "sensor": ["temp", "temp"],
        "time": pd.to_datetime(["2026-01-02", "2026-01-01"]),
        "value": [21.0, 20.0],
    })


def _right_df():
    return pd.DataFrame({
        "device": ["d2", "d2"],
        "sensor": ["humidity", "humidity"],
        "time": pd.to_datetime(["2026-01-01", "2026-01-02"]),
        "value": [55.0, 60.0],
    })


class TestPrepareComparison:
    def test_relabels_sensor_column(self):
        df, ll, rl = prepare_comparison(_left_df(), _right_df(), "Temp", "Humidity")
        assert set(df["sensor"].unique()) == {"Temp", "Humidity"}
        assert ll == "Temp"
        assert rl == "Humidity"

    def test_disambiguates_same_labels(self):
        df, ll, rl = prepare_comparison(_left_df(), _right_df(), "X", "X")
        assert ll == "X (left)"
        assert rl == "X (right)"
        assert set(df["sensor"].unique()) == {"X (left)", "X (right)"}

    def test_sorted_by_time(self):
        df, _, _ = prepare_comparison(_left_df(), _right_df(), "A", "B")
        times = df["time"].tolist()
        assert times == sorted(times)

    def test_does_not_mutate_inputs(self):
        left, right = _left_df(), _right_df()
        left_sensors = left["sensor"].tolist()
        right_sensors = right["sensor"].tolist()
        prepare_comparison(left, right, "A", "B")
        assert left["sensor"].tolist() == left_sensors
        assert right["sensor"].tolist() == right_sensors

    def test_left_empty(self):
        empty = pd.DataFrame(columns=["device", "sensor", "time", "value"])
        df, ll, rl = prepare_comparison(empty, _right_df(), "A", "B")
        assert len(df) == 2
        assert set(df["sensor"].unique()) == {"B"}

    def test_right_empty(self):
        empty = pd.DataFrame(columns=["device", "sensor", "time", "value"])
        df, ll, rl = prepare_comparison(_left_df(), empty, "A", "B")
        assert len(df) == 2
        assert set(df["sensor"].unique()) == {"A"}

    def test_both_empty(self):
        empty = pd.DataFrame(columns=["device", "sensor", "time", "value"])
        df, _, _ = prepare_comparison(empty, empty, "A", "B")
        assert df.empty

    def test_combined_row_count(self):
        df, _, _ = prepare_comparison(_left_df(), _right_df(), "A", "B")
        assert len(df) == 4


# ── Daily value by year ──────────────────────────────────────────────────────

def _year(first: date, days: int, per_day: float) -> pd.DataFrame:
    dates = [first + timedelta(days=n) for n in range(days)]
    return pd.DataFrame({"value": [per_day] * days}, index=pd.Index(dates, name="date"))


class TestYearToDate:
    def test_starts_at_zero_on_new_year_then_each_month_end_then_the_last_day(self):
        totals = year_to_date(_year(date(2026, 1, 1), 70, 10), "value")

        assert totals["date"].tolist() == [
            date(2026, 1, 1), date(2026, 1, 31), date(2026, 2, 28), date(2026, 3, 11),
        ]
        assert totals["total"].tolist() == [0, 31 * 10, 59 * 10, 70 * 10]
        assert totals["added"].tolist() == [0, 31 * 10, 28 * 10, 11 * 10]

    def test_a_late_starting_year_starts_at_zero_where_its_data_does(self):
        totals = year_to_date(_year(date(2023, 2, 11), 18, 10), "value")

        assert totals.iloc[0][["date", "total"]].tolist() == [date(2023, 2, 10), 0]
        last = totals.iloc[-1]
        assert last["date"] == date(2023, 2, 28)
        assert (last["counted"], last["elapsed"]) == (18, 31 + 28)

    def test_fractional_values_total_to_whole_units(self):
        totals = year_to_date(_year(date(2026, 1, 1), 3, 0.4), "value")

        assert totals["total"].tolist() == [0, 1]


class TestYearPointLabels:
    def test_each_point_shows_what_it_added_and_the_last_where_the_year_stands(self):
        totals = year_to_date(_year(date(2026, 1, 1), 70, 10), "value")

        texts, _ = year_point_labels(2026, totals, "h")

        assert texts == ["", "+310", "+280", "2026: 700 h (+110)"]

    def test_a_unitless_value_has_no_dangling_space(self):
        totals = year_to_date(_year(date(2026, 1, 1), 3, 10), "value")

        texts, _ = year_point_labels(2026, totals, "")

        assert texts[-1] == "2026: 30 (+30)"


class TestDailyByYearChart:
    def _daily(self) -> pd.DataFrame:
        return pd.DataFrame({
            "date": [date(2025, 1, 15), date(2026, 1, 15)],
            "value": [16.0, 15.0],
        })

    def _daily_lines(self, fig):
        return [t for t in fig.data if t.showlegend is not False]

    def test_one_daily_line_per_year(self):
        lines = self._daily_lines(make_daily_by_year_chart(self._daily(), "value", "Hours"))

        assert [t.name for t in lines] == ["2025", "2026"]
        assert [list(t.y) for t in lines] == [[16.0], [15.0]]

    def test_years_share_one_calendar(self):
        lines = self._daily_lines(make_daily_by_year_chart(self._daily(), "value", "Hours"))

        assert lines[0].x == lines[1].x

    def test_only_each_years_final_total_names_the_year(self):
        fig = make_daily_by_year_chart(self._daily(), "value", "Hours", "h")
        totals = [t for t in fig.data if t.showlegend is False]

        for trace in totals:
            assert trace.text[-1].startswith(f"{trace.name[:4]}: {trace.y[-1]:,} h")

    def test_a_late_starting_year_can_start_hidden_but_listed(self):
        daily = pd.DataFrame({
            "date": [date(2023, 2, 11), date(2024, 1, 1), date(2026, 1, 1)],
            "value": [1.0, 2.0, 3.0],
        })

        fig = make_daily_by_year_chart(daily, "value", "Hours", hide_late_starts=True)

        visible = {t.legendgroup: t.visible for t in fig.data}
        assert visible == {"2023": "legendonly", "2024": True, "2026": True}
        assert all(t.visible == visible[t.legendgroup] for t in fig.data)

    def test_late_starting_years_are_drawn_unless_asked(self):
        daily = pd.DataFrame({"date": [date(2023, 2, 11)], "value": [1.0]})

        fig = make_daily_by_year_chart(daily, "value", "Hours")

        assert all(t.visible in (None, True) for t in fig.data)
