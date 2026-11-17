"""GET /irrigation/simulator: what-ifs ahead, and past seasons replayed.

- **What if…**: "what if the rain comes later, or less? What if I give less
  water, or water tomorrow?" Each what-if runs the shipped model (ADR 0009) on
  the current 7-day outlook with changed drivers, and is shown against the
  no-watering baseline.
- **Season replay**: a past season under different watering policies, anchored
  to the measurements (:mod:`wp6_data.blue.irrigation.replay`).

Both have their controls in plain GET forms, so every scenario has a
shareable URL. Each form carries the other's settings, so neither resets the
other.
"""

from __future__ import annotations

import html
from datetime import UTC, date, datetime, timedelta
from typing import Annotated

import pandas as pd
import plotly.graph_objects as go
import structlog
from fastapi import APIRouter, Depends, Query
from fastapi.responses import HTMLResponse
from plotly.subplots import make_subplots

from wp6_data.blue.irrigation import replay as rp
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
from wp6_data.blue.irrigation.daily import daily_frame
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
    advice: Advice, carry: dict[str, object],
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
          {_hidden(carry)}
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


#: Colours of the replayed policies.
POLICY_COLOURS = {
    rp.AS_LOGGED: "#2563eb", rp.NO_WATERING: "#94a3b8",
    rp.ADVICE_RULE: "#7c3aed", rp.FIXED: "#f59e0b",
}


def _hidden(params: dict[str, object]) -> str:
    return "".join(
        f'<input type="hidden" name="{name}" value="{html.escape(str(value))}">'
        for name, value in params.items()
    )


def _replay_section(
    frame: pd.DataFrame, years: list[int], year: int, every: int, fixed_dose: float,
    max_dose: float, carry: dict[str, object],
) -> str:
    model, config = load_model(), load_config()
    lv = levels(model, config)
    result = rp.run_season(frame, year, model, config, every, fixed_dose, max_dose or None)
    year_options = "".join(
        f'<option value="{y}"{" selected" if y == year else ""}>{y}</option>' for y in years
    )
    form = f"""
        <form method="get" action="/irrigation/simulator#replay">
          {_hidden(carry)}
          <div class="sim-controls">
            <label>Season <select name="replay_year">{year_options}</select></label>
            <label>Fixed schedule: every (days)
              <input type="number" name="every" min="1" max="60" value="{every}"></label>
            <label>Fixed schedule: dose (L per plant)
              <input type="number" name="fixed_dose" min="0" max="200" step="0.1"
                     value="{fixed_dose:g}"></label>
            <label>Advice rule: max dose (L, 0 = none)
              <input type="number" name="max_dose" min="0" max="500" step="1"
                     value="{max_dose:g}"></label>
          </div>
          <div class="sim-actions"><button type="submit">Replay</button></div>
        </form>"""
    if result is None:
        return form + f"<p>No soil readings or weather for {year}.</p>"
    runs, measured, gate = result
    active = pd.Series([gate.active(day) for day in range(len(measured))], index=measured.index)

    fig = go.Figure()
    for start, end in _spans(~active):
        # Label only spans long enough to hold it; short ones would overlap.
        label = "crop inactive" if (end - start).days >= 7 else None
        fig.add_vrect(x0=start, x1=end + pd.Timedelta(days=1), fillcolor="#64748b",
                      opacity=0.18, line_width=0, layer="below",
                      annotation_text=label, annotation_position="top left",
                      annotation_font={"size": 10, "color": "#475569"})
    for name, run in runs.items():
        fig.add_trace(go.Scatter(
            x=run.path.index + pd.Timedelta(hours=12), y=run.path, mode="lines",
            name=name if name != rp.AS_LOGGED else "as logged (measured)",
            line={"color": POLICY_COLOURS[name], "width": 2 if name == rp.AS_LOGGED else 1.5,
                  "dash": None if name == rp.AS_LOGGED else "dash"},
            hovertemplate="%{y:.2f}<extra></extra>",
        ))
    fig.add_hline(y=lv.trigger, line={"color": "#dc2626", "dash": "dot", "width": 1},
                  annotation_text=f"trigger {lv.trigger:.2f}", annotation_position="top left")
    fig.update_yaxes(title_text="relative (0 dry – 1 wet)")
    fig.update_layout(height=460, margin={"l": 60, "r": 20, "t": 20, "b": 40},
                      legend={"orientation": "h", "y": -0.15}, hovermode="x unified")

    labels = {
        "water_l_per_plant": "Water (L/plant)", "waterings": "Waterings",
        "stress_days": "Days below trigger*", "longest_stress": "Longest run*",
        "lowest_active": "Lowest*", "drained_mm": "Drained (mm)",
    }
    rows = []
    for name, run in runs.items():
        summary = rp.summarise(run, lv, gate)
        rows.append([name, *(
            f"{summary[k]:.2f}" if k == "lowest_active" else f"{summary[k]:.0f}" for k in labels
        )])
    logged = frame["irrigation_l"][frame.index.year == year].sum()
    caveat = (
        "" if logged > 0 else
        f"<p><small><strong>No watering log for {year} yet</strong>, so \"as logged\" means "
        f"no recorded watering, and the policies are compared with whatever was actually "
        f"done.</small></p>"
    )
    return (
        form + caveat
        + fig.to_html(include_plotlyjs="cdn", full_html=False)
        + render_table(["Policy", *labels.values()], rows, sortable=False)
        + "<p><small>* Only while the crop is active (unshaded: soil ≥ "
        f"{config.advice.min_soil_temp_c:g} °C and demand ≥ "
        f"{config.advice.min_crop_use_mm_per_day:g} mm/day). Each policy is anchored to the "
        "measurements: the model only predicts its one-day difference from what really "
        "happened, so the replay can't drift over the season. Litres rest on the dose "
        "assumption (ADR 0009).</small></p>"
    )


def _spans(flags: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Contiguous runs of True as (first day, last day)."""
    runs, start, prev = [], None, None
    for day, flag in flags.items():
        if flag and start is None:
            start = day
        if not flag and start is not None:
            runs.append((start, prev))
            start = None
        prev = day
    if start is not None:
        runs.append((start, prev))
    return runs


@router.get("/irrigation/simulator", response_class=HTMLResponse)
async def irrigation_simulator(
    rain_pct: Annotated[int, Query(ge=0, le=300)] = 100,
    rain_delay: Annotated[int, Query(ge=0, le=6)] = 0,
    dose: Annotated[float, Query(ge=0, le=200)] = 0.0,
    dose_day: Annotated[str, Query()] = "",
    dose_time: Annotated[str, Query(pattern=f"^({MORNING}|{EVENING})$")] = MORNING,
    replay_year: Annotated[int | None, Query()] = None,
    every: Annotated[int, Query(ge=1, le=60)] = 10,
    fixed_dose: Annotated[float, Query(ge=0, le=200)] = 2.4,
    max_dose: Annotated[float, Query(ge=0, le=500)] = 0.0,
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
    config = load_config()
    frame = daily_frame(result, weather, events, config.watering.timezone)
    years = sorted({d.year for d in frame["s"].dropna().index}, reverse=True)
    replay_params = {"replay_year": replay_year or (years[0] if years else ""), "every": every,
                     "fixed_dose": fixed_dose, "max_dose": max_dose}
    what_if_params = {"rain_pct": rain_pct, "rain_delay": rain_delay, "dose": dose,
                      "dose_day": dose_day, "dose_time": dose_time}

    base = current_outlook(result, weather, events)
    if base is None:
        what_if_html = render_card(
            "What if…",
            "<p>No outlook right now: it needs a recent soil reading and the weather "
            "forecast.</p>",
        )
    else:
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
        what_if_intro = (
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
        what_if_html = (
            what_if_intro
            + render_card("What if…", _controls(base, rain_pct, rain_delay, dose, dose_day,
                                              dose_time, advice, replay_params))
            + render_card("Outlook", _chart(observed, baseline, what_if, lv,
                                            (rain_series(base), rain_series(what_if_outlook)),
                                            (dose_at, dose) if dose_at is not None else None))
            + render_card("Comparison", table)
        )

    replay_html = (
        _replay_section(frame, years, int(replay_params["replay_year"]), every, fixed_dose,
                        max_dose, what_if_params)
        if years else "<p>No soil readings to replay.</p>"
    )
    explainer = (
        "<p><strong>What if…</strong> changes the next week's forecast or plans a watering, "
        "and compares it with doing nothing. <strong>Season replay</strong> runs a past "
        "season under different watering policies.</p>"
    )
    body = (
        "<h1>Irrigation simulator</h1>"
        + explainer
        + what_if_html
        + '<section id="replay">'
        + render_card("Season replay", replay_html)
        + "</section>"
    )
    return render_page(PAGE_TITLE, body, extra_css=SIMULATOR_CSS, **back)
