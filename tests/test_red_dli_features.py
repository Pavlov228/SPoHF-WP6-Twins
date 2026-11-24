"""The weather fit carries no seasonal feature, and serving supplies what it was fitted on.

The fit maps a modelled sky onto the sensor above the lamps. Its radiation input
already *is* the season, so a calendar feature beside it gives ridge two
collinear ways to say "it is June"; weight spreads across both and the radiation
coefficient is left too small to tell a bright June day from a dull one.

Measured for the old outdoor-lux stage (skill vs persistence, out of fold):
shortwave alone +0.239, with day-of-year +0.069. And again for the single fit
that replaced the chain, out of fold over Nov 2025 – Nov 2026: roof MAE 2.89
with shortwave alone, 3.13 with the day-of-year pair added (MODEL_PAPER §5.7).
"""

from datetime import UTC, date, datetime

import pytest

from wp6_data.red.dli.model import FEATURES
from wp6_data.red.dli.schedule import predict_natural_dli_from_weather
from wp6_data.shared.weather import DailyForecast, HourlyWeather

SEASONAL = ("day_of_year_sin", "day_of_year_cos", "terrestrial_sum", "clear_sky_index")


def _forecast(day: date, *, direct: float, diffuse: float) -> DailyForecast:
    return DailyForecast(
        date=day,
        hourly=[
            HourlyWeather(
                datetime=datetime(day.year, day.month, day.day, hour, tzinfo=UTC),
                solar_radiation=(direct + diffuse) / 24,
                cloud_cover=50.0, temperature=10.0,
                direct_radiation=direct / 24, diffuse_radiation=diffuse / 24,
            )
            for hour in range(24)
        ],
    )


class _Recording:
    def __init__(self):
        self.calls = []

    def predict_dli(self, direct_radiation_sum=None, **kwargs):
        self.calls.append({"direct_radiation_sum": direct_radiation_sum, **kwargs})
        return 1.0


class TestFittedFeatureSet:
    def test_it_carries_no_seasonal_feature(self):
        assert not set(FEATURES) & set(SEASONAL)

    def test_it_is_global_irradiance_alone(self):
        assert FEATURES == ("shortwave_sum",)


class TestServingSuppliesTheFittedFeature:
    def test_shortwave_sum_reaches_the_model(self):
        """A feature the model was fitted on and never served predicts nothing."""
        model = _Recording()
        forecast = _forecast(date(2026, 6, 21), direct=800.0, diffuse=400.0)

        predict_natural_dli_from_weather(model, [forecast])

        assert model.calls[0]["shortwave_sum"] == pytest.approx(1200.0)

    def test_it_is_the_global_total_not_the_beam(self):
        """Handing the beam sum to a shortwave slot is the v7 bug in reverse."""
        model = _Recording()
        forecast = _forecast(date(2025, 12, 10), direct=3.0, diffuse=272.0)

        predict_natural_dli_from_weather(model, [forecast])

        call = model.calls[0]
        assert call["shortwave_sum"] == pytest.approx(275.0)
        assert call["direct_radiation_sum"] == pytest.approx(3.0)
