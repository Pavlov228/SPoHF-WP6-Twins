"""GET /dli/model — View model status and training options."""

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from wp6_data.red.dli import (
    DEFAULT_TRAINING_START,
    NATURAL_LIGHT_SENSOR,
    TOTAL_LIGHT_SENSOR,
    get_model,
)
from wp6_data.shared import render_page, render_stat_grid, render_table

router = APIRouter()

PAGE_TITLE = "SPoHF Red - DLI Model Status"


def _verdict(stage) -> str:
    """One stage's out-of-sample verdict, or a note that it has none.

    In-sample R² is kept alongside rather than hidden, because the gap between
    the two is itself the diagnosis: a stage that fits well and generalises
    badly is overfitting, and one that does neither is missing a feature.
    """
    if getattr(stage, "holdout_r2", None) is None:
        return (
            '<p class="warning">No out-of-fold score on this artifact — '
            "retrain to measure the stage.</p>"
        )
    skills = ", ".join(
        f"vs {name} <strong>{value:+.3f}</strong>"
        for name, value in sorted(stage.skill.items())
    )
    beaten = sorted(name for name, value in stage.skill.items() if value <= 0)
    note = (
        f'<br><span class="warning">Beaten by {", ".join(beaten)} — this stage '
        "is not earning its place.</span>"
        if beaten
        else ""
    )
    return (
        f"<p>Out-of-fold R² <strong>{stage.holdout_r2:.3f}</strong> over "
        f"{stage.n_holdout} held-out days (RMSE {stage.holdout_rmse:,.0f}). "
        f"Skill {skills}.{note}</p>"
    )


@router.get("/", response_class=HTMLResponse)
async def dli_model_status() -> str:
    """View model status and training options."""
    model = await get_model()

    if model.is_trained() and model.stats:
        stats = model.stats
        fit = stats.fit
        features = fit.feature_names or list(fit.coefficients)
        coef_table = render_table(
            ["Feature", "Coefficient"],
            [["Intercept", f"{fit.intercept:+.4f}"]]
            + [[name, f"{fit.coefficients.get(name, 0):+.4f}"] for name in features],
            sortable=False,
        )
        skill_tiles = [
            (f"{fit.skill[name]:+.3f}", f"Skill vs {name}")
            for name in ("persistence", "climatology")
            if name in fit.skill
        ]
        stats_grid = render_stat_grid([
            (f"{stats.r2_score:.3f}", "R²", "out-of-fold"),
            *skill_tiles,
            (f"{stats.n_samples:,}", "Days"),
            (f"{stats.attenuation_factor:.3f}", "Attenuation", "fallback"),
        ], cols=5)

        status_html = f"""
            <article>
                <h3 class="success">Light Model Trained (Daily)</h3>
                <p>OpenMeteo weather → {NATURAL_LIGHT_SENSOR} (above the lamps)
                   → × canopy/roof ratio → {TOTAL_LIGHT_SENSOR} (canopy)</p>

                {stats_grid}

                <h4>Weather → light above the lamps (Ridge, daily)</h4>
                <p>In-sample R²={fit.r2_score:.3f}, RMSE={fit.rmse:,.0f} μmol/m²/day</p>
                {_verdict(fit)}
                <small>Features: {', '.join(features)}</small>
                {coef_table}

                <h4>Light above the lamps → canopy</h4>
                <p>Pages use the canopy/roof ratio measured over recent days where
                   they can, because it moves with the season and the crop. The
                   fitted ratio below is the fallback.</p>

                <small>
                    Fitted on: <strong>{stats.indoor_sensor}</strong><br>
                    Attenuation: <strong>{stats.attenuation_factor:.4f}</strong>
                    ({stats.attenuation_samples} days)<br>
                    Trained: {stats.training_date.strftime('%Y-%m-%d %H:%M')} UTC<br>
                    Data range: {stats.date_range[0]} to {stats.date_range[1]}<br>
                    Model version: v{stats.model_version}
                </small>
            </article>
        """
    else:
        status_html = f"""
            <article>
                <h3 class="warning">No Model Trained</h3>
                <p>Training needs OpenMeteo weather and {NATURAL_LIGHT_SENSOR}
                   readings; {TOTAL_LIGHT_SENSOR} as well for the fitted ratio.</p>
            </article>
        """

    train_form = f"""
        <article>
            <h3>Train Model</h3>
            <p>Fits OpenMeteo daily weather to {NATURAL_LIGHT_SENSOR}, and the
               fallback canopy/roof ratio.</p>
            <form method="post" action="/dli/model/train">
                <button type="submit">Train Model</button>
            </form>
            <small>Uses all data since {DEFAULT_TRAINING_START.isoformat()}.</small>
        </article>
    """

    content = f"""
        <h1>Light Prediction Model</h1>
        <p>OpenMeteo forecast → light above the lamps → light at the canopy.
           The lamps come from LetsGrow, not from this model.</p>
        {status_html}
        {train_form}
        <p>
            <a href="/dli/model/diagnostic">View training diagnostic</a> -
            investigate data alignment and correlation issues.
        </p>
    """

    return render_page(
        PAGE_TITLE,
        content,
        show_back_link=True, back_url="/dli",
    )
