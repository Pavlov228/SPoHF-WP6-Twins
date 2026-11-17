"""GET /irrigation: the blue irrigation twin.

A top-level blue feature with its own home-page card, like ``/gdd``, so it
self-guards auth rather than inheriting it from the monitor router.

First slice (issue 065): one fused soil-moisture stream, and how each probe
contributed to it. Forecast, advice and the simulator build on this page.
"""

from __future__ import annotations

import html
from datetime import UTC, date, datetime, time
from typing import Annotated

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import structlog
from cachetools import TTLCache
from fastapi import APIRouter, Depends, Query
from fastapi.responses import HTMLResponse, Response
from plotly.subplots import make_subplots

from wp6_data.blue.irrigation.advice import Advice, Outlook, advise, outlook, trajectory
from wp6_data.blue.irrigation.config import load_config
from wp6_data.blue.irrigation.daily import daily_frame
from wp6_data.blue.irrigation.events import load_events
from wp6_data.blue.irrigation.model import load_model
from wp6_data.blue.irrigation.moisture import (
    DISAGREES,
    DROPOUT,
    EXCLUDED,
    FROZEN,
    OK,
    SPIKE,
    FusedMoisture,
    load_fused,
    quality_table,
)
from wp6_data.blue.irrigation.weather import (
    STATION,
    IrrigationWeather,
    load_weather,
)
from wp6_data.blue.provider import BlueSensorProvider
from wp6_data.blue.routes.year_toggle import YEAR_TOGGLE_CSS, year_toggle
from wp6_data.config import Settings
from wp6_data.shared import render_card, render_page, render_table
from wp6_data.shared.auth import verify_session_user

log = structlog.get_logger()

router = APIRouter(dependencies=[Depends(verify_session_user)])

PAGE_TITLE = "SPoHF Blue - Irrigation"

#: The fused stream is rebuilt from the whole history (so the per-probe scale
#: is stable). Readings arrive every ~20 minutes, so a short cache is fresh
#: enough and keeps paging between seasons cheap.
_fused_cache: TTLCache = TTLCache(maxsize=3, ttl=15 * 60)

_provider = BlueSensorProvider()
settings = Settings()

#: Rain traces: the gauge's own reading, coloured by whether the model trusts
#: it, and Open-Meteo as a line, so the two can always be compared.
RAIN_TRACES = {
    "gauge_used": ("Gauge rain (used)", "#0ea5e9"),
    "gauge_unused": ("Gauge rain (not trusted)", "#fca5a5"),
    "open_meteo": ("Open-Meteo rain", "#475569"),
    "forecast": ("Open-Meteo forecast", "#94a3b8"),
}

IRRIGATION_CSS = YEAR_TOGGLE_CSS



async def irrigation_events() -> pd.DataFrame:
    """Watering events (one per day), cached like the moisture."""
    cached = _fused_cache.get("events")
    if cached is None:
        cached = await load_events(_provider, load_config(), now=datetime.now(UTC))
        _fused_cache["events"] = cached
    return cached


async def irrigation_weather() -> IrrigationWeather:
    """Rain (fused gauge/Open-Meteo), ET0 and the forecast, cached like the moisture."""
    cached = _fused_cache.get("weather")
    if cached is None:
        cached = await load_weather(
            _provider, load_config(),
            latitude=settings.blue_weather_lat, longitude=settings.blue_weather_lon,
            today=datetime.now(UTC).date(),
        )
        _fused_cache["weather"] = cached
    return cached


def current_outlook(
    result: FusedMoisture, weather: IrrigationWeather | None, events: pd.DataFrame | None,
) -> Outlook | None:
    """The 7-day outlook from the latest reading, or None when it can't be made."""
    if weather is None:
        return None
    try:
        config = load_config()
        frame = daily_frame(result, weather, events, config.watering.timezone)
        return outlook(frame, load_model(), datetime.now(UTC).date())
    except Exception:
        log.warning("irrigation_outlook_failed", exc_info=True)
        return None


def local_days_to_utc(frame: pd.DataFrame) -> pd.DataFrame:
    """Daily rows are local days; place them on the UTC axis at local midnight."""
    out = frame.copy()
    out.index = pd.DatetimeIndex(out.index).tz_localize(load_config().watering.timezone)
    out.index = out.index.tz_convert(UTC)
    return out


async def fused_moisture() -> FusedMoisture:
    """The fused stream over the whole history, cached briefly."""
    cached = _fused_cache.get("fused")
    if cached is None:
        cached = await load_fused(_provider, load_config(), now=datetime.now(UTC))
        _fused_cache["fused"] = cached
    return cached


def _weather_traces(fig, weather: IrrigationWeather, start: datetime, end: datetime, row: int):
    actual = weather.actual.loc[start:end]
    forecast = weather.forecast.loc[start:end]
    if actual.empty and forecast.empty:
        return

    if not actual.empty:
        day = actual.index.floor("D")
        daily = actual.groupby(day).agg(
            gauge=("gauge_mm", lambda v: v.sum(min_count=1)),
            open_meteo=("open_meteo_mm", "sum"),
            trusted=("rain_source", lambda v: (v == STATION).mean() >= 0.5),
        )
        for key, trusted in (("gauge_used", True), ("gauge_unused", False)):
            label, colour = RAIN_TRACES[key]
            bars = daily["gauge"].where(daily["trusted"] == trusted)
            fig.add_trace(go.Bar(
                x=daily.index, y=bars, name=label, marker_color=colour,
                hovertemplate="%{y:.1f} mm<extra></extra>",
            ), row=row, col=1)
        label, colour = RAIN_TRACES["open_meteo"]
        fig.add_trace(go.Scatter(
            x=daily.index, y=daily["open_meteo"], name=label, mode="lines",
            line={"color": colour, "width": 1, "shape": "hvh"},
            hovertemplate="%{y:.1f} mm<extra></extra>",
        ), row=row, col=1)

    if not forecast.empty:
        label, colour = RAIN_TRACES["forecast"]
        future = forecast["open_meteo_mm"].groupby(forecast.index.floor("D")).sum()
        fig.add_trace(go.Bar(
            x=future.index, y=future, name=label, marker_color=colour,
            hovertemplate="%{y:.1f} mm<extra></extra>",
        ), row=row, col=1)

    both = pd.concat([actual, forecast])
    et0 = both.groupby(both.index.floor("D"))["et0_mm"].sum()
    fig.add_trace(go.Scatter(
        x=et0.index, y=et0, mode="lines", name="Reference ET0",
        line={"color": "#f59e0b", "width": 1.5}, hovertemplate="%{y:.1f} mm<extra></extra>",
    ), row=row, col=1)
    fig.update_yaxes(title_text="mm / day", rangemode="tozero", row=row, col=1)


def _advice_card(advice: Advice | None) -> str:
    """When and how much, from the 7-day outlook, with a link to the what-ifs."""
    link = '<a href="/irrigation/simulator" role="button" class="outline">What if…?</a>'
    if advice is None:
        return render_card(
            "Watering advice",
            "<p>No advice right now: it needs a recent soil reading and the weather "
            f"forecast.</p>{link}",
        )
    lv = advice.levels
    depletion = 1 - lv.trigger / lv.field_capacity
    context = (
        f"<small>Next 7 days: {advice.rain_forecast_mm:.0f} mm rain forecast, crop uses "
        f"about {advice.crop_use_mm:.0f} mm. Trigger {lv.trigger:.2f} "
        f"({depletion:.0%} of the available water used), target {lv.target:.2f}.</small>"
    )
    if not advice.needed:
        maybe = (
            f" If the weather turns out drier, it could get there by "
            f"<strong>{advice.possibly_on:%a %d %b}</strong>."
            if advice.possibly_on else ""
        )
        body = (
            f"<p><strong>No watering needed in the next 7 days.</strong> The soil stays "
            f"above the trigger on the forecast weather.{maybe}</p>"
        )
    else:
        body = (
            f"<p><strong>Water on {advice.water_on:%A %d %B}, around "
            f"{advice.water_at:%H:%M}.</strong> Without water the soil falls below the "
            f"trigger that day.</p>"
            f"<p>To bring it back to the target: about <strong>{advice.dose_mm:.0f} mm"
            f"</strong>, which is <strong>{advice.dose_l_per_plant:.0f} L per plant</strong> "
            f"if a litre spreads over {advice.area_per_plant_m2:g} m² per plant. The probes "
            f"don't see drip, so treat the litres as an estimate.</p>"
        )
    return render_card("Watering advice", body + context + f"<p>{link}</p>")


#: Colour of the no-watering forecast and its band.
FORECAST_COLOUR = "#7c3aed"


def _forecast_traces(fig, outlook: pd.DataFrame) -> None:
    """The model's daily forecast, assuming no watering, with its 80 % band.

    Daily values are drawn at midday, the middle of the day they average.
    """
    if outlook.empty:
        return
    at = outlook.index + pd.Timedelta(hours=12)
    band_x, band_y = band_polygons(
        pd.Series(outlook["low"].to_numpy(), index=at),
        pd.Series(outlook["high"].to_numpy(), index=at),
    )
    fig.add_trace(go.Scatter(
        x=band_x, y=band_y, mode="lines", line={"width": 0}, fill="toself",
        fillcolor="rgba(124, 58, 237, 0.15)", name="Forecast band (80 %)", hoverinfo="skip",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=at, y=outlook["mean"], mode="lines+markers",
        line={"color": FORECAST_COLOUR, "width": 2, "dash": "dash"}, marker={"size": 5},
        name="Forecast, no watering", hovertemplate="%{y:.2f} (forecast)<extra></extra>",
    ), row=1, col=1)


#: Colour of the watering (drip) markers on the soil-moisture panel.
WATERING_COLOUR = "#16a34a"


def _event_traces(fig, events: pd.DataFrame, top: float) -> None:
    """Each watering as a dotted line over the moisture, with its dose on hover.

    The time is the configured default (the CSV is date-only), so the hover
    says so. The line spans the moisture panel only, since a drip dose has no
    place on the rain axis (litres per plant, not mm).
    """
    if events.empty:
        return
    for when in events.index:
        fig.add_vline(
            x=when, line={"color": WATERING_COLOUR, "width": 1, "dash": "dot"}, row=1, col=1,
        )
    spread = [
        f"{lo:.2f}–{hi:.2f} L across treatments" if hi - lo > 1e-6 else "same for all treatments"
        for lo, hi in zip(events["volume_l_min"], events["volume_l_max"], strict=True)
    ]
    fig.add_trace(go.Scatter(
        x=events.index, y=[top] * len(events), mode="markers", name="Watering (drip)",
        marker={"symbol": "triangle-down", "size": 10, "color": WATERING_COLOUR},
        customdata=list(zip(
            events["volume_l_per_plant"], events["duration_min"], spread, strict=True,
        )),
        hovertemplate=(
            "Drip %{customdata[0]:.2f} L/plant · %{customdata[1]:.0f} min"
            "<br>%{customdata[2]} · time assumed<extra></extra>"
        ),
    ), row=1, col=1)


def band_polygons(low: pd.Series, high: pd.Series) -> tuple[list, list]:
    """A spread band as one closed polygon per stretch without gaps.

    Plotly's ``fill="tonexty"`` between two traces ignores gaps: across a
    missing stretch it joins the fill to whatever point comes next, which draws
    slanted wedges that are not data. Instead, each contiguous stretch becomes
    its own polygon (upper edge forward, lower edge back), and ``None``
    separates them so ``fill="toself"`` closes each one on its own.
    """
    valid = low.notna() & high.notna()
    stretch = (valid != valid.shift()).cumsum()
    xs: list = []
    ys: list = []
    for _, idx in valid[valid].groupby(stretch[valid]).groups.items():
        lo, hi = low.loc[idx], high.loc[idx]
        if xs:
            xs.append(None)
            ys.append(None)
        xs += [*hi.index, *lo.index[::-1]]
        ys += [*hi.to_numpy(), *lo.to_numpy()[::-1]]
    return xs, ys


def _figure(
    result: FusedMoisture,
    weather: IrrigationWeather | None,
    start: datetime,
    end: datetime,
    events: pd.DataFrame | None = None,
    outlook: pd.DataFrame | None = None,
) -> go.Figure:
    fused = result.fused.loc[start:end]
    low = result.low.loc[start:end]
    high = result.high.loc[start:end]
    coverage = result.coverage.loc[start:end]

    rows = 3 if weather is not None else 2
    heights = [0.58, 0.27, 0.15] if weather is not None else [0.78, 0.22]
    fig = make_subplots(
        rows=rows, cols=1, shared_xaxes=True, row_heights=heights, vertical_spacing=0.04,
    )
    band_x, band_y = band_polygons(low, high)
    fig.add_trace(go.Scatter(
        x=band_x, y=band_y, mode="lines", line={"width": 0}, fill="toself",
        fillcolor="rgba(37, 99, 235, 0.15)", name="Probe spread (25–75 %)",
        hoverinfo="skip",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=fused.index, y=fused, mode="lines", line={"color": "#2563eb", "width": 2},
        name="Fused soil moisture", hovertemplate="%{y:.2f}<extra></extra>",
    ), row=1, col=1)
    fig.update_yaxes(title_text="relative (0 dry – 1 wet)", row=1, col=1)
    if outlook is not None and not outlook.empty:
        _forecast_traces(fig, outlook.loc[start:end])
    if events is not None:
        top = float(high.max()) if high.notna().any() else 1.0
        _event_traces(fig, events.loc[start:end], top=top)
    if weather is not None:
        _weather_traces(fig, weather, start, end, row=2)
    fig.add_trace(go.Scatter(
        x=coverage.index, y=coverage, mode="lines", line={"color": "#64748b", "shape": "hv"},
        name="Probes used", hovertemplate="%{y} probes<extra></extra>",
    ), row=rows, col=1)
    fig.update_yaxes(title_text="probes", rangemode="tozero", row=rows, col=1)
    fig.update_layout(
        height=640 if weather is not None else 560,
        margin={"l": 60, "r": 20, "t": 20, "b": 40}, barmode="stack",
        legend={"orientation": "h", "y": -0.12}, hovermode="x unified",
    )
    return fig


def _chart(
    result: FusedMoisture,
    weather: IrrigationWeather | None,
    start: datetime,
    end: datetime,
    events: pd.DataFrame | None = None,
    outlook: pd.DataFrame | None = None,
) -> str:
    fig = _figure(result, weather, start, end, events, outlook)
    return fig.to_html(include_plotlyjs="cdn", full_html=False)


#: How each reading is drawn in the per-probe view: used, or dropped and why.
#: A ``missing`` hour has no reading, so there is nothing to draw.
PROBE_READING_STYLES = {
    OK: ("used", "#2563eb"),
    DROPOUT: ("dropout", "#dc2626"),
    SPIKE: ("spike", "#f97316"),
    DISAGREES: ("disagrees", "#9333ea"),
    FROZEN: ("frozen", "#06b6d4"),
    EXCLUDED: ("excluded", "#111827"),
}
#: Height of one probe's row in the per-probe view.
PROBE_ROW_PX = 150


def _compact(values: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Only a trace's own points, plus one NaN after each run to break the line.

    Every probe-reason trace would otherwise carry the full hourly index (mostly
    NaN), which made a year's view about 13 MB. Times go out as epoch
    milliseconds, which Plotly ships as a packed binary array, not one string
    per point.
    """
    present = values.notna()
    keep = present | present.shift(1, fill_value=False)
    kept = values[keep]
    millis = kept.index.as_unit("ms").asi8.astype("float64")
    return millis, kept.to_numpy(dtype="float64")


def probe_figure(result: FusedMoisture, start: datetime, end: datetime) -> go.Figure:
    """Every probe's hourly readings in its own unit, coloured by how the fusion
    treated each one: used, or dropped and why. One row per probe."""
    raw = result.raw.loc[start:end]
    reasons = result.reasons.loc[start:end]
    probes = list(raw.columns)
    table = quality_table(result, start, end)
    titles = [
        f"{p} · used {int(table.loc[p, 'used_h'])} h · share {table.loc[p, 'share']:.0%}"
        for p in probes
    ]
    fig = make_subplots(
        rows=len(probes), cols=1, shared_xaxes=True, vertical_spacing=0.025,
        subplot_titles=titles,
    )
    in_legend: set[str] = set()
    for row, probe in enumerate(probes, start=1):
        for reason, (label, colour) in PROBE_READING_STYLES.items():
            values = raw[probe].where(reasons[probe] == reason)
            if values.notna().sum() == 0:
                continue
            x, y = _compact(values)
            fig.add_trace(go.Scattergl(
                x=x, y=y, mode="lines", connectgaps=False,
                line={"color": colour, "width": 1.5},
                name=label, legendgroup=label, showlegend=label not in in_legend,
                hovertemplate=f"%{{y:.1f}} %VWC · {label}<extra></extra>",
            ), row=row, col=1)
            in_legend.add(label)
        fig.update_yaxes(title_text="%VWC", row=row, col=1)
    fig.update_annotations(font_size=12, xanchor="left", x=0)
    fig.update_layout(
        height=PROBE_ROW_PX * len(probes) + 110,
        margin={"l": 60, "r": 20, "t": 30, "b": 70},
        legend={"orientation": "h", "y": 0, "yanchor": "bottom", "yref": "container"},
        hovermode="x unified",
    )
    # Times are epoch milliseconds (see _compact); tell every axis they're dates.
    fig.update_xaxes(type="date")
    return fig


@router.get("/irrigation/probes.json")
async def irrigation_probes(
    year: Annotated[int, Query(description="Season to show")],
) -> Response:
    """The per-probe view as Plotly JSON, fetched only when the section is opened.
    Every probe's every hour is a lot of points to put on the page up front."""
    result = await fused_moisture()
    start = datetime.combine(date(year, 1, 1), time.min, tzinfo=UTC)
    end = datetime.combine(date(year, 12, 31), time.max, tzinfo=UTC)
    return Response(probe_figure(result, start, end).to_json(), media_type="application/json")


def _probe_details(year: int) -> str:
    """A collapsed section that loads the per-probe view the first time it opens."""
    return f"""
        <details id="probe-readings" data-src="/irrigation/probes.json?year={year}">
            <summary>All readings per probe {year} (loads on demand)</summary>
            <p><small>Each probe's hourly readings in its own %VWC. Blue is what the
            fusion used; other colours are readings it dropped, by reason.</small></p>
            <div id="probe-readings-chart"><p aria-busy="true">Loading…</p></div>
        </details>
        <script>
        (function () {{
            const details = document.getElementById("probe-readings");
            let loaded = false;
            details.addEventListener("toggle", async function () {{
                if (!details.open || loaded) return;
                loaded = true;
                const target = document.getElementById("probe-readings-chart");
                try {{
                    const response = await fetch(details.dataset.src);
                    if (!response.ok) throw new Error(response.status);
                    const fig = await response.json();
                    target.innerHTML = "";
                    Plotly.newPlot(target, fig.data, fig.layout, {{responsive: true}});
                }} catch (err) {{
                    loaded = false;
                    target.innerHTML = "<p>Could not load the probe readings.</p>";
                }}
            }});
        }})();
        </script>"""


#: Readable headers for the quality table's columns.
QUALITY_HEADERS = {
    "used_h": "Used (h)", "dry_ref": "Dry ref (%VWC)", "wet_ref": "Wet ref (%VWC)",
    "share": "Share",
}


def _quality_html(result: FusedMoisture, start: datetime, end: datetime) -> str:
    table = quality_table(result, start, end)
    headers = ["Probe", *[QUALITY_HEADERS.get(c, f"{c.capitalize()} (h)") for c in table.columns]]

    def cell(column: str, value: object) -> str:
        return f"{value:.0%}" if column == "share" else str(value)

    rows = [
        [html.escape(probe), *[cell(c, v) for c, v in zip(table.columns, row, strict=True)]]
        for probe, row in zip(table.index, table.itertuples(index=False), strict=True)
    ]
    return render_table(headers, rows)


@router.get("/irrigation", response_class=HTMLResponse)
async def irrigation_page(
    year: Annotated[int | None, Query(description="Season to show")] = None,
) -> str:
    try:
        result = await fused_moisture()
    except Exception:
        log.exception("irrigation_fuse_failed")
        return render_page(
            PAGE_TITLE, "<h1>Irrigation</h1><p>Soil moisture could not be loaded.</p>",
            show_back_link=True, back_url="/", show_source_indicator=False,
        )

    try:
        weather: IrrigationWeather | None = await irrigation_weather()
    except Exception:
        # The moisture stands on its own; weather is context, not a dependency.
        log.warning("irrigation_weather_failed", exc_info=True)
        weather = None

    try:
        events: pd.DataFrame | None = await irrigation_events()
    except Exception:
        log.warning("irrigation_events_failed", exc_info=True)
        events = None

    current = current_outlook(result, weather, events)
    forecast_path = advice = None
    if current is not None:
        forecast_path = local_days_to_utc(trajectory(current, load_model(), load_config()))
        advice = advise(current, load_model(), load_config())

    has_data = result.coverage[result.coverage > 0]
    if has_data.empty:
        return render_page(
            PAGE_TITLE, "<h1>Irrigation</h1><p>No soil-moisture readings yet.</p>",
            show_back_link=True, back_url="/", show_source_indicator=False,
        )

    years = sorted({t.year for t in has_data.index}, reverse=True)
    year = year if year in years else years[0]
    start = datetime.combine(date(year, 1, 1), time.min, tzinfo=UTC)
    end = datetime.combine(date(year, 12, 31), time.max, tzinfo=UTC)

    explainer = (
        "<p>The soil probes agree on <em>how</em> the soil wets and dries, but not on the "
        "level, so each is put on its own scale: <strong>0</strong> is its driest "
        "normal reading and <strong>1</strong> its wettest. The line is the median "
        "of the probes that passed the quality checks, and the band shows how far they "
        "spread. Probes are dropped automatically while they read zero (dry peat "
        "losing contact), spike, or stop moving with the others.</p>"
        "<p>Rain: bars are the on-site gauge, and the dark line is Open-Meteo for the "
        "same day. The model uses the gauge, which catches local showers, unless it has "
        "under-read Open-Meteo for weeks (a fault). Its bars then turn red and "
        "Open-Meteo stands in. Gauge hours with impossible values are dropped. ET0 is "
        "the day's reference evaporation demand.</p>"
        "<p>The dashed purple line forecasts the next week's soil moisture if you "
        "<em>don't</em> water, from the weather forecast. The band holds the outcome about "
        "70–80 % of the time.</p>"
        "<p>Green markers are drip watering from the fertigation log. The log has dates "
        "only, so the time shown is assumed. The probes are not next to the drippers, "
        "so a watering barely registers on them; rain wets the whole surface.</p>"
    )

    body = (
        "<h1>Irrigation</h1>"
        + _advice_card(advice)
        + explainer
        + year_toggle(years, year, lambda y: f"/irrigation?year={y}")
        + render_card(
            f"Soil moisture and weather {year}",
            _chart(result, weather, start, end, events, forecast_path),
        )
        + render_card(
            f"Probe quality {year}",
            _quality_html(result, start, end) + _probe_details(year),
            description=(
                "Hours each probe was used or dropped, and why; its own dry and wet "
                "reference (in %VWC); and its share of the fused stream."
            ),
        )
    )
    return render_page(
        PAGE_TITLE, body, show_back_link=True, back_url="/",
        extra_css=IRRIGATION_CSS, show_source_indicator=False,
    )
