"""GET /irrigation/simulator: what-ifs on the 7-day soil-moisture outlook.

"What if the rain comes later, or less? What if I give less water, or water
tomorrow?" Each what-if runs the shipped model (ADR 0009) on the current
outlook with changed drivers, and is shown against the no-watering baseline.

The controls are a plain GET form, so every scenario has a shareable URL.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Annotated

import pandas as pd
import plotly.graph_objects as go
import structlog
from fastapi import APIRouter, Depends, Query
from fastapi.responses import HTMLResponse
from plotly.subplots import make_subplots

from wp6_data.blue.irrigation.advice import (
    EVENING,
    MORNING,
    Advice,
    Levels,
    Outlook,
    advise,
    levels,
    scenario,
    trajectory,
)
from wp6_data.blue.irrigation.config import load_config
from wp6_data.blue.irrigation.model import load_model
from wp6_data.blue.routes.irrigation import (
    FORECAST_COLOUR,
    WATERING_COLOUR,
    band_polygons,
    current_outlook,
    fused_moisture,
    irrigation_events,
    irrigation_weather,
    local_days_to_utc,
)
from wp6_data.shared import render_card, render_page, render_table
from wp6_data.shared.auth import verify_session_user

log = structlog.get_logger()

router = APIRouter(dependencies=[Depends(verify_session_user)])

PAGE_TITLE = "SPoHF Blue - Irrigation simulator"
#: Days of observed soil moisture shown before the outlook, for context.
HISTORY_DAYS = 14
BASELINE_COLOUR = "#94a3b8"

SIMULATOR_CSS = """
    .sim-controls { display: grid; gap: 0.75rem 1rem;
        grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); align-items: end; }
    .sim-controls label { font-size: 0.85rem; margin: 0; }
    .sim-controls input, .sim-controls select { margin: 0; padding: 0.35rem 0.5rem; }
    .sim-actions { display: flex; gap: 0.5rem; flex-wrap: wrap; margin-top: 0.75rem;
        align-items: center; }
    .sim-actions button, .sim-actions [role="button"] { width: auto; margin: 0; }
"""


def _summary(path: pd.DataFrame, lv: Levels) -> dict[str, str]:
    below = path.index[path["mean"] < lv.trigger]
    return {
        "lowest": f"{path['mean'].min():.2f}",
        "days below trigger": str(len(below)),
        "first day below": f"{below[0]:%a %d %b}" if len(below) else "—",
        "day 7": f"{path['mean'].iloc[-1]:.2f}",
    }


def _chart(
    observed: pd.Series,
    baseline: pd.DataFrame,
    what_if: pd.DataFrame,
    lv: Levels,
    rain: tuple[pd.Series, pd.Series],
    dose: tuple[pd.Timestamp, float] | None,
) -> str:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28],
                        vertical_spacing=0.05)
    fig.add_trace(go.Scatter(
        x=observed.index, y=observed, mode="lines", line={"color": "#2563eb", "width": 2},
        name="Soil moisture (measured)", hovertemplate="%{y:.2f}<extra></extra>",
    ), row=1, col=1)
    at = baseline.index + pd.Timedelta(hours=12)
    fig.add_trace(go.Scatter(
        x=at, y=baseline["mean"], mode="lines+markers",
        line={"color": BASELINE_COLOUR, "dash": "dot", "width": 2}, marker={"size": 4},
        name="Forecast, no watering", hovertemplate="%{y:.2f}<extra></extra>",
    ), row=1, col=1)
    band_x, band_y = band_polygons(
        pd.Series(what_if["low"].to_numpy(), index=at),
        pd.Series(what_if["high"].to_numpy(), index=at),
    )
    fig.add_trace(go.Scatter(
        x=band_x, y=band_y, mode="lines", line={"width": 0}, fill="toself",
        fillcolor="rgba(124, 58, 237, 0.15)", name="What-if band (80 %)", hoverinfo="skip",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=at, y=what_if["mean"], mode="lines+markers",
        line={"color": FORECAST_COLOUR, "dash": "dash", "width": 2}, marker={"size": 5},
        name="What-if", hovertemplate="%{y:.2f}<extra></extra>",
    ), row=1, col=1)
    for level, label in ((lv.trigger, "trigger"), (lv.target, "target")):
        fig.add_hline(
            y=level, line={"color": "#dc2626" if label == "trigger" else "#16a34a",
                           "dash": "dot", "width": 1},
            annotation_text=f"{label} {level:.2f}", annotation_position="top left",
            row=1, col=1,
        )
    if dose is not None:
        fig.add_vline(x=dose[0], line={"color": WATERING_COLOUR, "dash": "dot"}, row=1, col=1)
        fig.add_annotation(x=dose[0], y=1, yref="y domain", text=f"{dose[1]:g} L/plant",
                           showarrow=False, yanchor="bottom", row=1, col=1)
    forecast_rain, scenario_rain = rain
    fig.add_trace(go.Bar(
        x=forecast_rain.index + pd.Timedelta(hours=12), y=forecast_rain,
        name="Rain, forecast", marker_color="#cbd5e1",
        hovertemplate="%{y:.1f} mm<extra></extra>",
    ), row=2, col=1)
    fig.add_trace(go.Bar(
        x=scenario_rain.index + pd.Timedelta(hours=12), y=scenario_rain,
        name="Rain, what-if", marker_color="#0ea5e9",
        hovertemplate="%{y:.1f} mm<extra></extra>",
    ), row=2, col=1)
    fig.update_yaxes(title_text="relative (0 dry – 1 wet)", row=1, col=1)
    fig.update_yaxes(title_text="mm / day", rangemode="tozero", row=2, col=1)
    fig.update_layout(
        height=600, margin={"l": 60, "r": 20, "t": 20, "b": 40}, barmode="group",
        legend={"orientation": "h", "y": -0.12}, hovermode="x unified",
    )
    return fig.to_html(include_plotlyjs="cdn", full_html=False)


def _controls(
    o: Outlook, rain_pct: int, rain_delay: int, dose: float, dose_day: str, dose_time: str,
    advice: Advice,
) -> str:
    days = [o.origin.date() + timedelta(days=i) for i in range(len(o.days) + 1)]
    day_options = '<option value="">no watering</option>' + "".join(
        f'<option value="{d}"{" selected" if str(d) == dose_day else ""}>{d:%a %d %b}</option>'
        for d in days
    )
    time_options = "".join(
        f'<option value="{t}"{" selected" if t == dose_time else ""}>{t}</option>'
        for t in (MORNING, EVENING)
    )
    advised = ""
    if advice.needed and advice.water_on is not None:
        advised = (
            f'<a role="button" class="outline" href="/irrigation/simulator?dose_day='
            f'{advice.water_on}&dose={advice.dose_l_per_plant:.0f}&dose_time={MORNING}">'
            f"Try the advice ({advice.dose_l_per_plant:.0f} L on {advice.water_on:%a %d %b})</a>"
        )
    return f"""
        <form method="get" action="/irrigation/simulator">
          <div class="sim-controls">
            <label>Rain, % of forecast
              <input type="number" name="rain_pct" min="0" max="300" step="10"
                     value="{rain_pct}"></label>
            <label>Rain arrives later by (days)
              <input type="number" name="rain_delay" min="0" max="6" value="{rain_delay}"></label>
            <label>Water on
              <select name="dose_day">{day_options}</select></label>
            <label>Dose (L per plant)
              <input type="number" name="dose" min="0" max="200" step="0.1"
                     value="{dose:g}"></label>
            <label>Time of day
              <select name="dose_time">{time_options}</select></label>
          </div>
          <div class="sim-actions">
            <button type="submit">Simulate</button>
            <a role="button" class="secondary outline" href="/irrigation/simulator">Reset</a>
            {advised}
          </div>
        </form>"""


@router.get("/irrigation/simulator", response_class=HTMLResponse)
async def irrigation_simulator(
    rain_pct: Annotated[int, Query(ge=0, le=300)] = 100,
    rain_delay: Annotated[int, Query(ge=0, le=6)] = 0,
    dose: Annotated[float, Query(ge=0, le=200)] = 0.0,
    dose_day: Annotated[str, Query()] = "",
    dose_time: Annotated[str, Query(pattern=f"^({MORNING}|{EVENING})$")] = MORNING,
) -> str:
    back = {"show_back_link": True, "back_url": "/irrigation", "show_source_indicator": False}
    try:
        result = await fused_moisture()
        weather = await irrigation_weather()
        events = await irrigation_events()
    except Exception:
        log.exception("irrigation_simulator_load_failed")
        return render_page(PAGE_TITLE, "<h1>Simulator</h1><p>Data could not be loaded.</p>",
                           **back)
    base = current_outlook(result, weather, events)
    if base is None:
        return render_page(
            PAGE_TITLE,
            "<h1>Simulator</h1><p>No outlook right now: it needs a recent soil reading and "
            "the weather forecast.</p>",
            **back,
        )

    model, config = load_model(), load_config()
    lv = levels(model, config)
    advice = advise(base, model, config)
    try:
        water_on = date.fromisoformat(dose_day) if dose_day else None
    except ValueError:
        water_on = None
    what_if_outlook = scenario(
        base, rain_factor=rain_pct / 100, rain_delay_days=rain_delay,
        dose_l=dose, dose_day=water_on, dose_time=dose_time,
    )
    baseline = local_days_to_utc(trajectory(base, model, config))
    what_if = local_days_to_utc(trajectory(what_if_outlook, model, config))

    def rain_series(o: Outlook) -> pd.Series:
        return local_days_to_utc(pd.DataFrame({"r": o.inputs.rain[1:, 0]}, index=o.days))["r"]

    first = pd.Timestamp(base.origin).tz_localize(config.watering.timezone).tz_convert(UTC)
    observed = result.fused.loc[first - pd.Timedelta(days=HISTORY_DAYS):].dropna()
    dose_at = None
    if water_on is not None and dose > 0:
        hour = 6 if dose_time == MORNING else 19
        dose_at = (pd.Timestamp(datetime.combine(water_on, datetime.min.time()))
                   .tz_localize(config.watering.timezone) + pd.Timedelta(hours=hour)
                   ).tz_convert(UTC)

    rows = [
        ["No watering", *_summary(baseline, lv).values(), "0"],
        ["What-if", *_summary(what_if, lv).values(), f"{dose:g}" if water_on else "0"],
    ]
    table = render_table(
        ["", "Lowest", "Days below trigger", "First day below", "Day 7", "Water (L/plant)"],
        rows, sortable=False,
    )
    explainer = (
        f"<p>Change the forecast or plan a watering, then compare the what-if (purple) "
        f"with doing nothing (grey). The soil should stay above the trigger "
        f"({lv.trigger:.2f}, the red line). The target ({lv.target:.2f}) is just below "
        f"field capacity; watering above it mostly drains away.</p>"
        f"<p><small>One litre per plant counts as 1 mm of rain over "
        f"{model.area_per_plant_m2:g} m², because the probes don't see drip (ADR 0009). The "
        f"model works per day, so time of day only decides which day a dose counts: "
        f"morning on that day, evening on the next. Forecasts start from "
        f"{base.origin:%a %d %b}, the last full day of readings.</small></p>"
    )
    body = (
        "<h1>Irrigation simulator</h1>"
        + explainer
        + render_card("What if…", _controls(base, rain_pct, rain_delay, dose, dose_day,
                                            dose_time, advice))
        + render_card("Outlook", _chart(observed, baseline, what_if, lv,
                                        (rain_series(base), rain_series(what_if_outlook)),
                                        (dose_at, dose) if dose_at is not None else None))
        + render_card("Comparison", table)
    )
    return render_page(PAGE_TITLE, body, extra_css=SIMULATOR_CSS, **back)
