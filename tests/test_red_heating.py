"""Tests for the heat-output index behind /energy."""

from datetime import UTC, date, datetime, timedelta

import pandas as pd
import pytest

from wp6_data.red import deps
from wp6_data.red.heating import daily_heat_output, load_heat_output


def _hourly(first: date, hours: int, value: float) -> pd.Series:
    start = datetime(first.year, first.month, first.day, tzinfo=UTC)
    index = pd.DatetimeIndex([start + timedelta(hours=h) for h in range(hours)])
    return pd.Series([value] * hours, index=index, dtype=float)


class TestDailyHeatOutput:
    def test_sums_each_complete_day(self):
        daily = daily_heat_output(_hourly(date(2026, 1, 15), 48, 2.5))

        assert daily["date"].tolist() == [date(2026, 1, 15), date(2026, 1, 16)]
        assert daily["heat_index"].tolist() == [24 * 2.5, 24 * 2.5]

    def test_a_day_with_hours_missing_is_left_out_not_summed_short(self):
        daily = daily_heat_output(_hourly(date(2026, 1, 15), 47, 2.5))

        assert daily["date"].tolist() == [date(2026, 1, 15)]

    def test_empty_gives_empty(self):
        assert daily_heat_output(pd.Series(dtype=float)).empty


class TestHeatOutputConfig:
    def test_red_metadata_names_letsgrow_pipes(self):
        config = load_heat_output(deps._METADATA_PATH)

        assert config.device.startswith("letsgrow-")
        assert config.pipes
        # The one fault seen read 150 °C; a ceiling at or above it would let it through.
        assert config.max_pipe < 150

    def test_a_metadata_file_without_the_block_is_refused(self, tmp_path):
        path = tmp_path / "metadata.yaml"
        path.write_text("devices: {}\n", encoding="utf-8")

        with pytest.raises(ValueError, match="heating"):
            load_heat_output(path)
