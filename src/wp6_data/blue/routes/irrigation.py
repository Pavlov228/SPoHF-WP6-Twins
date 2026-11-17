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

import pandas as pd
import plotly.graph_objects as go
import structlog
from cachetools import TTLCache
from fastapi import APIRouter, Depends, Query
from fastapi.responses import HTMLResponse
from plotly.subplots import make_subplots

from wp6_data.blue.irrigation.config import load_config
from wp6_data.blue.irrigation.moisture import FusedMoisture, load_fused, quality_table
from wp6_data.blue.irrigation.weather import (
    FORECAST,
    OPEN_METEO,
    STATION,
    IrrigationWeather,
    load_weather,
)
from wp6_data.blue.provider import BlueSensorProvider
from wp6_data.config import Settings
from wp6_data.shared import render_card, render_page, render_table
from wp6_data.shared.auth import verify_session_user

log = structlog.get_logger()

router = APIRouter(dependencies=[Depends(verify_session_user)])

PAGE_TITLE = "SPoHF Blue - Irrigation"

#: The fused stream is rebuilt from the whole history (so the per-probe scale
#: is stable). Readings arrive every ~20 minutes, so a short cache is fresh
#: enough and keeps paging between seasons cheap.
_fused_cache: TTLCache = TTLCache(maxsize=2, ttl=15 * 60)

_provider = BlueSensorProvider()
settings = Settings()

#: Rain bar colour per source: the gauge, Open-Meteo standing in for it, forecast.
RAIN_COLOURS = {STATION: "#0ea5e9", OPEN_METEO: "#94a3b8", FORECAST: "#cbd5e1"}
RAIN_LABELS = {
    STATION: "Rain, on-site gauge",
    OPEN_METEO: "Rain, Open-Meteo (gauge not trusted)",
    FORECAST: "Rain, forecast",
}

IRRIGATION_CSS = """
    .season-toggle { display: flex; gap: 0.4rem; flex-wrap: wrap; margin-bottom: 0.5rem; }
    .stats-grid { display: flex; gap: 0.5rem; flex-wrap: wrap; }
    .stats-grid article { flex: 1; min-width: 140px; text-align: center; padding: 0.5rem; }
    .stat-value { font-size: 1.5rem; font-weight: bold; }
"""


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


async def fused_moisture() -> FusedMoisture:
    """The fused stream over the whole history, cached briefly."""
    cached = _fused_cache.get("fused")
    if cached is None:
        cached = await load_fused(_provider, load_config(), now=datetime.now(UTC))
        _fused_cache["fused"] = cached
    return cached


def _season_toggle(years: list[int], active: int) -> str:
    def link(y: int) -> str:
        style = "" if y == active else ' class="outline"'
        return f'<a href="/irrigation?year={y}" role="button"{style}>{y}</a>'

    links = "".join(link(y) for y in years)
    return f'<nav class="season-toggle">{links}</nav>'


def _weather_traces(fig, weather: IrrigationWeather, start: datetime, end: datetime, row: int):
    hours = pd.concat([weather.actual, weather.forecast]).loc[start:end]
    if hours.empty:
        return
    day = hours.index.floor("D")
    rain = hours.groupby([day, "rain_source"])["rain_mm"].sum().unstack(fill_value=0.0)
    for source, colour in RAIN_COLOURS.items():
        if source in rain:
            fig.add_trace(go.Bar(
                x=rain.index, y=rain[source], name=RAIN_LABELS[source],
                marker_color=colour, hovertemplate="%{y:.1f} mm<extra></extra>",
            ), row=row, col=1)
    et0 = hours.groupby(day)["et0_mm"].sum()
    fig.add_trace(go.Scatter(
        x=et0.index, y=et0, mode="lines", name="Reference ET0",
        line={"color": "#f59e0b", "width": 1.5}, hovertemplate="%{y:.1f} mm<extra></extra>",
    ), row=row, col=1)
    fig.update_yaxes(title_text="mm / day", rangemode="tozero", row=row, col=1)


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
) -> str:
    return _figure(result, weather, start, end).to_html(include_plotlyjs="cdn", full_html=False)


def _quality_html(result: FusedMoisture, start: datetime, end: datetime) -> str:
    table = quality_table(result, start, end)
    headers = ["Probe", *[str(c) for c in table.columns]]
    rows = [
        [html.escape(probe), *[str(v) for v in row]]
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

    latest_at = has_data.index[-1]
    probes_now = f"{int(result.coverage.loc[latest_at])} / {len(result.reasons.columns)}"
    stats = f"""
        <div class="stats-grid">
            <article><div class="stat-value">{result.fused.loc[latest_at]:.2f}</div>
                <small>Soil moisture now (0 dry – 1 wet)</small></article>
            <article><div class="stat-value" id="probes-now">{probes_now}</div>
                <small>Probes used now</small></article>
            <article><div class="stat-value">{latest_at:%d %b %H:%M}</div>
                <small>Latest reading (UTC)</small></article>
        </div>"""

    explainer = (
        "<p>The soil probes agree on <em>how</em> the soil wets and dries, but not on the "
        "level, so each is put on its own scale: <strong>0</strong> is its driest "
        "normal reading and <strong>1</strong> its wettest. The line is the median "
        "of the probes that passed the quality checks, and the band shows how far they "
        "spread. Probes are dropped automatically while they read zero (dry peat "
        "losing contact), spike, or stop moving with the others.</p>"
        "<p>Rain comes from the on-site gauge, which catches local showers. Where the "
        "gauge is under-reading for weeks (a fault), Open-Meteo stands in, and its "
        "bars are grey. ET0 is the reference evaporation demand of the day.</p>"
    )

    body = (
        "<h1>Irrigation</h1>"
        + explainer
        + stats
        + _season_toggle(years, year)
        + render_card(f"Soil moisture and weather {year}", _chart(result, weather, start, end))
        + render_card(
            f"Probe quality {year}",
            _quality_html(result, start, end),
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
