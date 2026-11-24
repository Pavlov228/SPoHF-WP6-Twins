"""Daily light model: weather → light above the lamps → light at the canopy.

Link 1 (fitted, Ridge): OpenMeteo daily shortwave → daily PAR integral at
``NATURAL_LIGHT_SENSOR``, the sensor above the lamps. That sensor sees only
natural light, unoccluded by the crop, so it is the clean thing to learn the
weather from.

Link 2 (measured): × the canopy/roof ratio, to reach ``TOTAL_LIGHT_SENSOR``'s
position. The fitted ``attenuation_factor`` is a single number over the whole
training range; callers that know a recent ratio (see
``red.lamp.attenuation_by_date``) pass it instead, because the real ratio moves
with the season and the crop.

The lamps are not modelled here: LetsGrow reports them.

There used to be a third link, through the s1000 outdoor lux station. At
forecast time it only passed a prediction through, and chaining two fits
compounds their errors — the direct fit scores better out of sample than the
chain did in sample (MODEL_PAPER §5.7).
"""

import contextlib
import os
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import structlog

from wp6_data.shared.artifacts import persist_held, read_pickled

if TYPE_CHECKING:
    from wp6_data.shared.blob import BlobStore

from wp6_data.red.dli.aggregation import (
    add_day_of_year_features,
    align_weather_to_indoor_daily,
    encode_day_of_year,
)
from wp6_data.red.dli.constants import (
    DEFAULT_TRAINING_START,
    NATURAL_LIGHT_SENSOR,
    TOTAL_LIGHT_SENSOR,
    UMOL_TO_MOL,
)
from wp6_data.red.fitting import (
    StageStats,
    climatology_baseline,
    climatology_table,
    fit_ridge,
    out_of_fold_predictions,
    persistence_baseline,
    score_holdout,
)
from wp6_data.shared.artifacts import fingerprint

# Bumped for a deliberate break: a change in the pickle layout, or in what the
# stored numbers mean when their shape is unchanged. Unlike earlier revisions
# this is an *equality* check, not a floor — see `load`.
#
# v11: one fit, weather straight to the above-lamp sensor; s1000 dropped.
MODEL_VERSION = 11


# What the weather link is fitted on: global horizontal irradiance and nothing
# else. Chosen by measurement, out of fold (MODEL_PAPER §5.6, §5.7). The
# radiation already *is* the season, so a calendar term beside it gives ridge two
# collinear ways to say "it is June" and leaves the radiation coefficient too
# small to tell a bright day from a dull one. Adding the day-of-year pair that
# the old stage 2 carried scored worse here too (roof MAE 3.13 vs 2.89). What the
# season does to the light *below* the roof is the canopy/roof ratio's job.
FEATURES: tuple[str, ...] = ("shortwave_sum",)


def fit_fingerprint() -> str:
    """Identifies the inputs this model would be fitted from today.

    Not a config object like the climate chain's, because the DLI model's inputs
    are still module constants (moving them into `metadata.yaml` is filed as
    follow-up work). What matters is that they are all here: change the training
    start, swap a sensor, or alter which weather variables are requested, and a
    model fitted before that change is refused rather than served.
    """
    return fingerprint(
        [
            MODEL_VERSION,
            DEFAULT_TRAINING_START.isoformat(),
            NATURAL_LIGHT_SENSOR,
            TOTAL_LIGHT_SENSOR,
            # What `train_model_from_db` asks OpenMeteo for.
            ["shortwave_radiation", "cloud_cover"],
            list(FEATURES),
        ]
    )


def _resolve(store: BlobStore | None, key: str) -> tuple[BlobStore, str]:
    """The store and key to use, defaulting to red's models prefix.

    Imported lazily: ``red.deps`` builds its stores at import, and importing it
    from module scope here would make this module's import order matter.
    Overridable so tests can hand in a ``LocalBlobStore``.
    """
    from wp6_data.red import deps

    return store or deps.MODELS_STORE, key or deps.DLI_MODEL_KEY


def _previous_day(days: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Each row's value from the calendar day before, NaN where there isn't one.

    Built from a date lookup rather than ``shift(1)``: the daily frames drop
    days that failed a quality gate, so the previous *row* is often not the
    previous *day*, and persistence has to mean yesterday.
    """
    lookup = dict(zip(days, values, strict=True))
    return np.array(
        [lookup.get(day - timedelta(days=1), np.nan) for day in days], dtype=float
    )


def _score_stage(
    stats: StageStats,
    days: np.ndarray,
    actual: np.ndarray,
    predicted: np.ndarray,
    mask: np.ndarray,
) -> None:
    """Attach out-of-fold quality, plus skill against the baselines that matter.

    A daily light model posts a high R² for knowing the time of year, so the
    number worth reading is what it removes from a baseline that already knows
    that: ``climatology`` (this month's mean) and ``persistence`` (yesterday).

    Everything is scored on one row set — those with an out-of-fold prediction
    *and* a previous day to carry forward. A skill score computed over
    different rows than the model's own error is not a comparison.

    ``climatology_table`` keys on (month, hour); these rows are daily, so the
    hour is constant and the table is a month-of-year mean, which is what a
    daily climatology should be. When every row has been held out at some point
    the table is necessarily built over all of them, so the baseline is mildly
    optimistic — which makes the reported skill against it conservative.
    """
    previous = _previous_day(days, actual)
    scored = mask & np.isfinite(previous)
    if not scored.any():
        return

    times = pd.Series(pd.to_datetime(list(days[scored]), utc=True))
    rest = ~scored
    table = (
        climatology_table(pd.Series(pd.to_datetime(list(days[rest]), utc=True)), actual[rest])
        if rest.any()
        else climatology_table(times, actual[scored])
    )
    score_holdout(
        stats,
        actual[scored],
        predicted[scored],
        {
            "persistence": persistence_baseline(previous[scored]),
            "climatology": climatology_baseline(
                table, times, float(np.mean(actual[scored]))
            ),
        },
    )


@dataclass
class ModelStats:
    """Statistics from training the weather link, plus the fitted ratio."""

    fit: StageStats  # OpenMeteo daily shortwave → above-lamp PAR integral
    training_date: datetime
    date_range: tuple[date, date]
    indoor_sensor: str = NATURAL_LIGHT_SENSOR
    aggregation: str = "daily"
    model_version: int = MODEL_VERSION
    attenuation_factor: float = 1.0
    attenuation_samples: int = 0

    @property
    def r2_score(self) -> float:
        """Out-of-fold R² of the weather link, in-sample only if it could not fold."""
        if self.fit.holdout_r2 is not None:
            return self.fit.holdout_r2
        return self.fit.r2_score

    @property
    def n_samples(self) -> int:
        """Days the weather link was fitted on."""
        return self.fit.n_samples


class TwoStageLightModel:
    """Weather → above-lamp light (fitted), then × canopy/roof ratio (measured).

    The two stages are the two links in the module docstring. Only the first is
    a regression; the second is a ratio, fitted once as ``attenuation_factor``
    and overridable per call with a recent measured one.
    """

    def __init__(self):
        self.weather_model = None
        self.weather_scaler = None
        self.stats: ModelStats | None = None
        # Overwritten by `train` or `load`; the declared set is the default so an
        # untrained instance cannot describe a shape nothing was fitted to.
        self.features = list(FEATURES)
        # Above-lamp → plant-level ratio, fitted over the whole training range.
        self.attenuation_factor: float = 1.0

    def is_trained(self) -> bool:
        """Check if the weather link is trained."""
        return self.weather_model is not None

    def train(
        self,
        weather_df: pd.DataFrame,
        indoor_df: pd.DataFrame,
        indoor_sensor: str = NATURAL_LIGHT_SENSOR,
        plant_level_df: pd.DataFrame | None = None,
        above_lamp_df: pd.DataFrame | None = None,
    ) -> ModelStats:
        """Fit the weather link on daily data, and the fallback ratio.

        Args:
            weather_df: OpenMeteo hourly rows: datetime, shortwave_radiation, ...
            indoor_df: Above-lamp PAR readings (time, value) — the target
            plant_level_df: Optional canopy readings, for the attenuation factor
            above_lamp_df: Optional above-lamp readings, for the attenuation factor
        """
        data = add_day_of_year_features(
            align_weather_to_indoor_daily(weather_df, indoor_df), date_col="date"
        )
        if len(data) < 10:
            raise ValueError(
                f"Insufficient training data: {len(data)} days "
                f"(need OpenMeteo + {indoor_sensor} overlap)"
            )
        features = [f for f in FEATURES if f in data.columns]
        if not features:
            raise ValueError("No valid features found in weather data")

        # Day-blocked folds, not sklearn's random K-fold: consecutive days share
        # a weather system, so a random split trains on the test set's
        # neighbours and every score comes back flattering.
        X = data[features].to_numpy(dtype=float)
        y = data["par_integral"].to_numpy(dtype=float)
        days = data["date"].to_numpy()
        self.weather_model, self.weather_scaler, fit_stats = fit_ridge(
            X, y, features, days=days
        )
        oof, mask = out_of_fold_predictions(X, y, features, days)
        _score_stage(fit_stats, days, y, oof, mask)
        self.features = features

        attenuation_factor = 1.0
        attenuation_samples = 0
        if plant_level_df is not None and above_lamp_df is not None:
            with contextlib.suppress(Exception):
                attenuation_factor, attenuation_samples = self._compute_attenuation(
                    above_lamp_df, plant_level_df
                )
        self.attenuation_factor = attenuation_factor

        self.stats = ModelStats(
            fit=fit_stats,
            training_date=datetime.now(UTC),
            date_range=(min(days), max(days)),
            indoor_sensor=indoor_sensor,
            aggregation="daily",
            model_version=MODEL_VERSION,
            attenuation_factor=round(attenuation_factor, 4),
            attenuation_samples=attenuation_samples,
        )
        return self.stats

    def _compute_attenuation(
        self, above_lamp_df: pd.DataFrame, plant_level_df: pd.DataFrame
    ) -> tuple[float, int]:
        """Attenuation factor; see :func:`wp6_data.red.lamp.compute_attenuation`.

        Kept as a method for the existing callers and tests; the arithmetic
        moved to ``lamp.py`` so the climate model can use the same number
        instead of computing its own.
        """
        from wp6_data.red.lamp import compute_attenuation

        return compute_attenuation(above_lamp_df, plant_level_df)

    def predict_daily(
        self,
        direct_radiation_sum: float | None = None,
        diffuse_radiation_sum: float | None = None,
        cloud_cover_avg: float | None = None,
        day_of_year: int | None = None,
        *,
        shortwave_sum: float | None = None,
        at_plant_level: bool = True,
        attenuation: float | None = None,
    ) -> float:
        """Predict a daily PAR integral (μmol/m²) from an OpenMeteo daily forecast.

        The fit predicts ``NATURAL_LIGHT_SENSOR``, above the lamps.
        ``at_plant_level`` carries it down to the canopy by the canopy/roof
        ratio: ``attenuation`` when given — a recent measured ratio — else the
        fitted ``attenuation_factor``.

        The two positions differ by the whole ratio, around a third, so the
        caller says which it wants rather than inferring it from the magnitudes.
        Scoring a plant-level prediction against the above-lamp sensor reads as a
        model that underpredicts by a third, when nothing is wrong with it.

        Args:
            direct_radiation_sum, diffuse_radiation_sum, cloud_cover_avg: only
                needed if the model was fitted on them
            day_of_year: 1-365 (defaults to today)
            shortwave_sum: Daily sum of global horizontal irradiance (Wh/m²)
            at_plant_level: Predict at the canopy rather than above the lamps
            attenuation: Canopy/roof ratio to use instead of the fitted one
        """
        if not self.is_trained():
            raise RuntimeError("Model not trained. Call train() or load() first.")

        # Default day_of_year to today. Only right for a prediction *about* today:
        # a caller predicting any other date must say which, or the seasonal term
        # is silently frozen (see predict_natural_dli_from_weather).
        if day_of_year is None:
            day_of_year = datetime.now().timetuple().tm_yday
        day_sin, day_cos = encode_day_of_year(day_of_year)

        # A feature the model was *fitted* on has no sensible default: standing in
        # a number the weather never produced reads like a working prediction.
        # Refuse instead, so a caller that forgets one finds out.
        feature_values = {
            "shortwave_sum": shortwave_sum,
            "direct_radiation_sum": direct_radiation_sum,
            "diffuse_radiation_sum": diffuse_radiation_sum,
            "cloud_cover_avg": cloud_cover_avg,
            "day_of_year_sin": day_sin,
            "day_of_year_cos": day_cos,
        }
        missing = [f for f in self.features if feature_values.get(f) is None]
        if missing:
            raise ValueError(
                f"The model was trained on {missing} but they were not supplied. "
                "Pass them from the forecast rather than letting a default stand in."
            )

        X = np.array([[feature_values[f] for f in self.features]])
        if self.weather_scaler is not None:
            X = self.weather_scaler.transform(X)
        predicted_par_sum = float(self.weather_model.predict(X)[0])

        # Attenuation is the above-lamp -> plant-level conversion, so it applies
        # only when the caller asked for plant level. Applying it unconditionally
        # is what made /dli/performance score an attenuated prediction against
        # the un-attenuated above-lamp sensor.
        if at_plant_level:
            predicted_par_sum *= (
                attenuation if attenuation is not None else self.attenuation_factor
            )

        return max(0.0, round(predicted_par_sum, 1))

    def predict_dli(
        self,
        direct_radiation_sum: float | None = None,
        diffuse_radiation_sum: float | None = None,
        cloud_cover_avg: float | None = None,
        day_of_year: int | None = None,
        readings_per_day: int = 144,  # unused from v10; kept for callers
        *,
        shortwave_sum: float | None = None,
        at_plant_level: bool = True,
        attenuation: float | None = None,
    ) -> float:
        """Predict DLI (mol/m²/day); see :meth:`predict_daily` for the arguments."""
        par_integral = self.predict_daily(
            direct_radiation_sum,
            diffuse_radiation_sum=diffuse_radiation_sum,
            cloud_cover_avg=cloud_cover_avg,
            day_of_year=day_of_year,
            shortwave_sum=shortwave_sum,
            at_plant_level=at_plant_level,
            attenuation=attenuation,
        )
        # The fit predicts an integral in μmol/m², so DLI is a unit conversion.
        return round(par_integral / UMOL_TO_MOL, 2)

    async def save(self, store: BlobStore | None = None, key: str = "") -> str:
        """Hold this model as the process's model, then store it.

        Held first, and the store write is best-effort: a fit during an
        object-store outage still serves from this process instead of being
        discarded. Returns the key it was (or would have been) written to.
        """
        if not self.is_trained():
            raise RuntimeError("No model to save. Train first.")

        data = {
            "weather_model": self.weather_model,
            "weather_scaler": self.weather_scaler,
            "features": self.features,
            "stats": self.stats,
            "attenuation_factor": self.attenuation_factor,
            "version": MODEL_VERSION,
            # What produced this fit. Checked on load, because the model now
            # outlives the pod that trained it — see ADR 0007.
            "fingerprint": fit_fingerprint(),
        }

        store, key = _resolve(store, key)
        global _model
        _model = self
        await persist_held(store, key, data)
        return key

    async def load(self, store: BlobStore | None = None, key: str = "") -> ModelStats | None:
        """Load the saved model, or return ``None`` when there isn't a usable one.

        **Equality on the version, not a floor.** A model from another version is
        refused and refitted rather than migrated: a migrated artifact with a
        guessed field is indistinguishable downstream from a fitted one.
        Refusing costs one refit; the alternative costs wrong numbers with no
        signal.

        Never raises on a *stale* artifact. Anything a pickle from another era can
        throw — a moved class, a removed attribute, a truncated write — means the
        same thing to a caller that can simply retrain, and this one is called
        during startup. The version and fingerprint gates live in
        ``shared.artifacts.read_pickled``; see it for why each exists.
        """
        store, key = _resolve(store, key)
        data = await read_pickled(
            store, key, version=MODEL_VERSION, expect_fingerprint=fit_fingerprint(),
        )
        if data is None:
            return None

        try:
            self.weather_model = data["weather_model"]
            self.weather_scaler = data["weather_scaler"]
            self.features = data["features"]
            self.stats = data["stats"]
            self.attenuation_factor = data["attenuation_factor"]
        except KeyError:
            # A key the current version promises is absent: not a model we can
            # use, and not one to patch up with a default.
            return None

        # Simulation override, deliberately applied after the artifact is read.
        override = os.getenv("WP6_RED_DLI_ATTENUATION_OVERRIDE")
        if override:
            self.attenuation_factor = float(override)

        return self.stats


# Backwards compatible alias
LightCorrelationModel = TwoStageLightModel


# Global model instance
_model: TwoStageLightModel | None = None

log = structlog.get_logger()


async def get_model() -> TwoStageLightModel:
    """Get or create the global model instance, loading it back if one is stored.

    Async because the artifact now lives in the object store rather than on a
    mounted volume, and a read is a network round trip — done synchronously it
    would block the event loop and every other in-flight request with it.

    An unreachable store, with nothing held yet, gives an untrained model rather
    than an error: every page already renders "not trained", and a fit (boot's,
    or the admin's) holds itself through :meth:`TwoStageLightModel.save`. It is
    not held here, so the next call tries the store again.
    """
    global _model

    if _model is None:
        model = TwoStageLightModel()
        try:
            await model.load()
        except Exception as exc:  # noqa: BLE001 - an outage degrades to "not trained"
            log.warning("dli_model_store_unavailable", error=str(exc))
            return model
        # Assigned only once loaded, so a failed load does not leave an empty
        # model cached as if it were the real thing.
        _model = model

    return _model
