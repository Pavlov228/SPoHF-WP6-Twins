"""The schedule chart: the supplemental band and hourly steps."""

import pandas as pd

from wp6_data.shared import make_schedule_chart

LABEL = "Supplemental (predicted)"


def _frame(values):
    return pd.DataFrame({
        "datetime": pd.date_range("2026-11-24", periods=len(values), freq="h", tz="UTC"),
        "par": values,
    })


NATURAL = _frame([0.0, 0.0, 50.0, 100.0])
PREDICTED = _frame([0.0, 140.0, 190.0, 100.0])


def _names(fig):
    return [trace.name for trace in fig.data]


class TestSupplementBand:
    def _fig(self):
        return make_schedule_chart(
            predicted_df=PREDICTED, natural_df=NATURAL, supplement_label=LABEL,
        )

    def test_the_band_sits_at_the_bottom_with_natural_stacked_on_top(self):
        fig = self._fig()
        names = _names(fig)
        band = fig.data[names.index(LABEL)]
        natural = fig.data[names.index("Natural (predicted)")]

        assert names.index(LABEL) < names.index("Natural (predicted)")
        assert band.fill == "tozeroy"
        assert list(band.y) == [0.0, 140.0, 140.0, 0.0]
        # `tonexty` fills down to the band, so the top of the stack is the total.
        assert natural.fill == "tonexty"
        assert list(natural.y) == list(PREDICTED["par"])

    def test_each_band_hovers_its_own_amount(self):
        fig = self._fig()
        natural = fig.data[_names(fig).index("Natural (predicted)")]

        assert list(natural.customdata) == list(NATURAL["par"])

    def test_no_label_no_band(self):
        fig = make_schedule_chart(predicted_df=PREDICTED, natural_df=NATURAL)

        assert LABEL not in _names(fig)

    def test_no_band_without_natural_to_stand_on(self):
        fig = make_schedule_chart(predicted_df=PREDICTED, supplement_label=LABEL)

        assert LABEL not in _names(fig)


class TestHourlySteps:
    def test_predicted_series_are_held_across_the_hour(self):
        fig = make_schedule_chart(
            predicted_df=PREDICTED, natural_df=NATURAL, supplement_label=LABEL,
            actual_df=NATURAL, hourly_steps=True,
        )

        shapes = {trace.name: trace.line.shape for trace in fig.data}
        assert shapes["Natural (predicted)"] == "hv"
        assert shapes[LABEL] == "hv"
        assert shapes["Predicted"] == "hv"
        # Actual readings are measurements, not hourly values: left as drawn.
        assert shapes["Actual"] in (None, "linear")

    def test_off_by_default(self):
        fig = make_schedule_chart(predicted_df=PREDICTED, natural_df=NATURAL)

        assert {trace.line.shape for trace in fig.data} == {"linear"}
