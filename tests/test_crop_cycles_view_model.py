"""Tests for the pure half of the red crop-cycles view model.

Covers attachment (which cohort a measurement belongs to) and coverage (how much
of a cohort's window the climate actually spans) — the two pieces of logic the
picture depends on and the eye cannot check.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from wp6_data.red.crop_cycles.config import load_crop_cycles, to_cohort_spec, to_cycle_spec
from wp6_data.red.crop_cycles.view_model import build_waterfall, daily_day_length
from wp6_data.red.dli import DEFAULT_PHOTOPERIOD_THRESHOLD, calculate_daily_dli
from wp6_data.shared.cycles import generate_cohorts

RED_METADATA = Path(__file__).parent.parent / "src/wp6_data/red/metadata.yaml"
CONFIG = load_crop_cycles(RED_METADATA)
SPEC = to_cohort_spec(CONFIG)
CYCLE = CONFIG.cycles[0]
COHORTS = generate_cohorts(to_cycle_spec(CYCLE), SPEC)

DEVICES = {"dev-a": "Alpha", "dev-b": "Beta"}


def _daily(start: date, days: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": [start + timedelta(days=i) for i in range(days)],
            "value": [20.0] * days,
        }
    )


def _full_daily() -> pd.DataFrame:
    return _daily(CYCLE.start, (CYCLE.end - CYCLE.start).days)


def _obs(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["date", "device", "value"])


class TestAttachment:
    def test_measurement_lands_on_the_cohort_that_set_a_duration_earlier(self):
        target = COHORTS[5]
        observed = target.end  # first day of its completion window
        view = build_waterfall(
            COHORTS, SPEC, _obs([(observed, "dev-a", 4.2)]), _full_daily(),
            device_labels=DEVICES,
        )
        measured = view.measured
        assert len(measured) == 1
        assert measured[0].cohort.key == target.key
        assert measured[0].cohort.start == observed - SPEC.duration
        assert not view.unattached

    def test_measurement_outside_every_cohort_is_reported_not_dropped(self):
        stray = CYCLE.start  # before any cohort has completed
        view = build_waterfall(
            COHORTS, SPEC, _obs([(stray, "dev-a", 4.2)]), _full_daily(),
            device_labels=DEVICES,
        )
        assert not view.measured
        assert [(u.day, u.device) for u in view.unattached] == [(stray, "dev-a")]

    def test_every_cohort_becomes_a_lane_even_when_unmeasured(self):
        view = build_waterfall(
            COHORTS, SPEC, _obs([]), _full_daily(), device_labels=DEVICES,
        )
        assert len(view.lanes) == len(COHORTS)
        assert not view.measured

    def test_lane_group_is_the_cycle_label(self):
        view = build_waterfall(
            COHORTS, SPEC, _obs([]), _full_daily(), device_labels=DEVICES,
        )
        assert {lane.group for lane in view.lanes} == {CYCLE.label}


class TestSampleAggregation:
    """Several samples of one cohort become one marker, not several chips."""

    def test_repeat_samples_of_one_cultivar_are_averaged(self):
        target = COHORTS[5]
        view = build_waterfall(
            COHORTS, SPEC,
            _obs([(target.end, "dev-a", 4.0),
                  (target.end, "dev-a", 5.0),
                  (target.end + timedelta(days=1), "dev-a", 6.0)]),
            _full_daily(), device_labels=DEVICES,
        )
        markers = view.measured[0].markers
        assert len(markers) == 1
        assert markers[0].value == 5.0
        assert markers[0].samples == 3
        assert markers[0].detail == "mean of 3"

    def test_each_cultivar_is_averaged_separately(self):
        target = COHORTS[5]
        view = build_waterfall(
            COHORTS, SPEC,
            _obs([(target.end, "dev-a", 4.0), (target.end, "dev-a", 6.0),
                  (target.end, "dev-b", 1.0), (target.end, "dev-b", 3.0)]),
            _full_daily(), device_labels=DEVICES,
        )
        markers = {m.series: m for m in view.measured[0].markers}
        assert markers["Alpha"].value == 5.0
        assert markers["Beta"].value == 2.0
        assert all(m.samples == 2 for m in markers.values())

    def test_a_lone_sample_reports_one_sample(self):
        """A mean of eight and a single reading must not look alike on hover."""
        view = build_waterfall(
            COHORTS, SPEC, _obs([(COHORTS[5].end, "dev-a", 4.2)]),
            _full_daily(), device_labels=DEVICES,
        )
        assert view.measured[0].markers[0].samples == 1

    def test_a_total_measure_sums_rather_than_averages(self):
        """A yield picked across several passes is a total, not a per-item mean."""
        target = COHORTS[5]
        view = build_waterfall(
            COHORTS, SPEC,
            _obs([(target.end, "dev-a", 100.0),
                  (target.end + timedelta(days=1), "dev-a", 60.0)]),
            _full_daily(), device_labels=DEVICES, measure_agg="sum",
        )
        marker = view.measured[0].markers[0]
        assert marker.value == 160.0
        assert marker.samples == 2
        assert marker.detail == "sum of 2"  # names the aggregation, not just n

    def test_an_unknown_measure_agg_raises_rather_than_guessing(self):
        with pytest.raises(ValueError, match="unknown measure agg"):
            build_waterfall(
                COHORTS, SPEC, _obs([(COHORTS[5].end, "dev-a", 4.2)]),
                _full_daily(), device_labels=DEVICES, measure_agg="median",
            )

    def test_samples_survive_the_single_series_collapse(self):
        view = build_waterfall(
            COHORTS, SPEC,
            _obs([(COHORTS[5].end, "dev-a", 4.0), (COHORTS[5].end, "dev-a", 6.0)]),
            _full_daily(), device_labels=DEVICES,
        )
        marker = view.measured[0].markers[0]
        assert marker.series == ""  # collapsed: only one cultivar present
        assert marker.samples == 2


class TestVarieties:
    def test_both_cultivars_share_one_lane_as_paired_markers(self):
        observed = COHORTS[5].end
        view = build_waterfall(
            COHORTS, SPEC,
            _obs([(observed, "dev-a", 4.2), (observed, "dev-b", 3.8)]),
            _full_daily(), device_labels=DEVICES,
        )
        assert len(view.measured) == 1
        markers = view.measured[0].markers
        assert {m.series for m in markers} == {"Alpha", "Beta"}
        assert {m.value for m in markers} == {4.2, 3.8}

    def test_a_single_cultivar_drops_the_series_tag(self):
        observed = COHORTS[5].end
        view = build_waterfall(
            COHORTS, SPEC, _obs([(observed, "dev-a", 4.2)]), _full_daily(),
            device_labels=DEVICES,
        )
        assert [m.series for m in view.measured[0].markers] == [""]

    def test_unknown_device_falls_back_to_its_id(self):
        observed = COHORTS[5].end
        view = build_waterfall(
            COHORTS, SPEC, _obs([(observed, "mystery", 4.2)]), _full_daily(),
            device_labels=DEVICES,
        )
        assert view.measured[0].markers[0].label == "mystery"


class TestCoverage:
    def test_full_climate_leaves_every_lane_unfaded(self):
        view = build_waterfall(
            COHORTS, SPEC, _obs([]), _full_daily(), device_labels=DEVICES,
        )
        assert not view.partial

    def test_climate_starting_late_fades_the_early_lanes(self):
        # Climate that only begins once the season is underway, as red's real
        # sensors do relative to the first Sijia season.
        late = _daily(COHORTS[4].start, (CYCLE.end - COHORTS[4].start).days)
        view = build_waterfall(
            COHORTS, SPEC, _obs([]), late, device_labels=DEVICES,
        )
        assert view.partial
        assert view.lanes[0].coverage < view.lanes[-1].coverage

    def test_no_climate_at_all_gives_zero_coverage(self):
        view = build_waterfall(
            COHORTS, SPEC, _obs([]), pd.DataFrame(), device_labels=DEVICES,
        )
        assert all(lane.coverage == 0.0 for lane in view.lanes)
        assert len(view.partial) == len(view.lanes)


class TestDayLength:
    """Day length: hours above the DLI photoperiod threshold, per local day."""

    TZ = "Europe/Amsterdam"

    @staticmethod
    def _readings(day: date, lit: tuple[int, int], tz: str, level: float = 200.0):
        """10-minute PAR for one local day, ``level`` within ``[lit)`` hours."""
        times = pd.date_range(pd.Timestamp(day, tz=tz), periods=144, freq="10min")
        hours = times.hour + times.minute / 60
        values = [level if lit[0] <= h < lit[1] else 0.0 for h in hours]
        return pd.DataFrame({
            "device": "par", "sensor": "par",
            "time": times.tz_convert("UTC"), "value": values,
        })

    def test_counts_the_hours_light_is_on(self):
        df = self._readings(date(2026, 1, 10), (6, 22), self.TZ)
        out = daily_day_length(df, self.TZ)
        assert out["date"].tolist() == [date(2026, 1, 10)]
        assert out["value"].iloc[0] == pytest.approx(16, abs=0.2)

    def test_light_below_the_dli_threshold_is_not_day(self):
        # Under the photoperiod bar, yet integrating to more than the no-data
        # gate: a reported day with zero hours of day in it.
        df = self._readings(
            date(2026, 1, 10), (6, 22), self.TZ,
            level=DEFAULT_PHOTOPERIOD_THRESHOLD / 2,
        )
        out = daily_day_length(df, self.TZ)
        assert out["value"].tolist() == [0.0]

    def test_a_day_of_flat_zeros_is_absent_not_dark(self):
        """Red's sensors read zero for weeks; that is an outage, not night."""
        df = self._readings(date(2026, 6, 1), (0, 0), self.TZ)
        assert daily_day_length(df, self.TZ).empty

    def test_days_split_at_local_midnight(self):
        # Lit 23:00–24:00 local is the *same* local day, though in UTC it is
        # the previous one's evening (Amsterdam is ahead of UTC).
        df = self._readings(date(2026, 1, 10), (23, 24), self.TZ, level=2000.0)
        out = daily_day_length(df, self.TZ)
        assert out["date"].tolist() == [date(2026, 1, 10)]

    def test_matches_the_dli_pages_photoperiod(self):
        """One definition of day length across the twin."""
        df = self._readings(date(2026, 1, 10), (5, 21), "UTC")
        expected = calculate_daily_dli(df)["photoperiod_hours"].iloc[0]
        assert daily_day_length(df, "UTC")["value"].iloc[0] == expected
