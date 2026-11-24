"""POST /dli/model/train — Train the daily light model."""

from datetime import UTC, datetime

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from wp6_data.red import deps
from wp6_data.red.db import MySQLConnection
from wp6_data.red.dli import (
    DEFAULT_TRAINING_START,
    NATURAL_LIGHT_SENSOR,
    TOTAL_LIGHT_SENSOR,
    ModelStats,
    OpenMeteoClient,
    get_model,
)
from wp6_data.red.dli import data as dli_data
from wp6_data.shared import render_page, render_stat_grid

router = APIRouter()


async def train_model_from_db(db: MySQLConnection, weather_client: OpenMeteoClient) -> ModelStats:
    """Train the daily light model from database data.

    Fetches sensor data and weather data, trains the model, and saves it.
    Returns ModelStats on success, raises on failure.
    """
    start_dt = datetime(
        DEFAULT_TRAINING_START.year,
        DEFAULT_TRAINING_START.month,
        DEFAULT_TRAINING_START.day,
        tzinfo=UTC,
    )
    end_dt = datetime.now(UTC)

    above_lamp_df = await db.get_par_readings(
        device_ids=[NATURAL_LIGHT_SENSOR], start=start_dt, end=end_dt
    )
    plant_level_df = await db.get_par_readings(
        device_ids=[TOTAL_LIGHT_SENSOR], start=start_dt, end=end_dt
    )

    if above_lamp_df.empty:
        raise ValueError(f"No indoor PAR data ({NATURAL_LIGHT_SENSOR}) found for training")

    # The weather link is fitted on global horizontal irradiance alone. Asking for the
    # column by name — rather than through the generic radiation slot, which
    # renames whatever it is given to `solar_radiation` — is what stops a beam
    # series being fitted as though it were shortwave.
    weather_df = await weather_client.get_hourly(
        ["shortwave_radiation", "cloud_cover"],
        start=start_dt.date(),
        end=end_dt.date(),
    )
    if weather_df.empty:
        raise ValueError("No OpenMeteo weather data found for training")

    model = await get_model()
    stats = model.train(
        weather_df=weather_df,
        indoor_df=above_lamp_df,
        indoor_sensor=NATURAL_LIGHT_SENSOR,
        plant_level_df=plant_level_df if not plant_level_df.empty else None,
        above_lamp_df=above_lamp_df,
    )
    await model.save()
    return stats

PAGE_TITLE = "SPoHF Red - Train DLI Model"

@router.post("/train", response_class=HTMLResponse)
async def dli_model_train() -> str:
    """Train the daily light model."""
    if not dli_data.is_connected():
        return render_page(
            PAGE_TITLE,
            "<h1>Database not connected</h1>",
            show_back_link=True, back_url="/dli/model",
        )

    try:
        stats = await train_model_from_db(dli_data.require_db(), deps.get_weather_client())
    except Exception as e:
        return render_page(
            PAGE_TITLE,
            f"<h1>Training Failed</h1><p>{e}</p>",
            show_back_link=True, back_url="/dli/model",
        )

    stats_grid = render_stat_grid([
        (f"{stats.r2_score:.3f}", "R² (out-of-fold)"),
        (f"{stats.n_samples:,}", "Days"),
        (f"{stats.attenuation_factor:.3f}", "Attenuation (fallback)"),
    ], cols=3)

    content = f"""
        <article style="border: 2px solid var(--pico-ins-color, green)">
            <h1>Light Model Trained (Daily)</h1>
            {stats_grid}
            <p>OpenMeteo daily shortwave → {NATURAL_LIGHT_SENSOR} (above the lamps)
               → × canopy/roof ratio → {TOTAL_LIGHT_SENSOR} (canopy). The ratio is
               the recent measured one where known, else
               {stats.attenuation_factor:.3f}.</p>
            <p>
                Model saved and will be used for PAR predictions.
                <a href="/dli/model">View full model details</a>
            </p>
        </article>
    """

    return render_page(
        PAGE_TITLE,
        content,
        show_back_link=True, back_url="/dli/model",
    )
