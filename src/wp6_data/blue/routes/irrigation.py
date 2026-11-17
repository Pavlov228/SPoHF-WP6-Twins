"""GET /irrigation: the blue irrigation twin.

A top-level blue feature with its own home-page card, like ``/gdd``, so it
self-guards auth rather than inheriting it from the monitor router.

First slice (issue 065): one fused pot-moisture stream, and how each probe
contributed to it. Forecast, advice and the simulator build on this page.
"""

from __future__ import annotations

import html
from datetime import UTC, date, datetime, time
from typing import Annotated

import plotly.graph_objects as go
import structlog
from cachetools import TTLCache
from fastapi import APIRouter, Depends, Query
from fastapi.responses import HTMLResponse
from plotly.subplots import make_subplots

from wp6_data.blue.irrigation.config import load_config
from wp6_data.blue.irrigation.moisture import FusedMoisture, load_fused, quality_table
from wp6_data.blue.provider import BlueSensorProvider
from wp6_data.shared import render_card, render_page, render_table
from wp6_data.shared.auth import verify_session_user

log = structlog.get_logger()

router = APIRouter(dependencies=[Depends(verify_session_user)])

PAGE_TITLE = "SPoHF Blue - Irrigation"

#: The fused stream is rebuilt from the whole history (so the per-probe scale
#: is stable). Readings arrive every ~20 minutes, so a short cache is fresh
#: enough and keeps paging between seasons cheap.
_fused_cache: TTLCache = TTLCache(maxsize=1, ttl=15 * 60)

_provider = BlueSensorProvider()

IRRIGATION_CSS = """
    .season-toggle { display: flex; gap: 0.4rem; flex-wrap: wrap; margin-bottom: 0.5rem; }
    .stats-grid { display: flex; gap: 0.5rem; flex-wrap: wrap; }
    .stats-grid article { flex: 1; min-width: 140px; text-align: center; padding: 0.5rem; }
    .stat-value { font-size: 1.5rem; font-weight: bold; }
"""


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


def _chart(result: FusedMoisture, start: datetime, end: datetime) -> str:
    fused = result.fused.loc[start:end]
    low = result.low.loc[start:end]
    high = result.high.loc[start:end]
    coverage = result.coverage.loc[start:end]

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.78, 0.22], vertical_spacing=0.04,
    )
    fig.add_trace(go.Scatter(
        x=high.index, y=high, mode="lines", line={"width": 0},
        hoverinfo="skip", showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=low.index, y=low, mode="lines", line={"width": 0}, fill="tonexty",
        fillcolor="rgba(37, 99, 235, 0.15)", name="Probe spread (25–75 %)",
        hoverinfo="skip",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=fused.index, y=fused, mode="lines", line={"color": "#2563eb", "width": 2},
        name="Fused pot moisture", hovertemplate="%{y:.2f}<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=coverage.index, y=coverage, mode="lines", line={"color": "#64748b", "shape": "hv"},
        name="Probes used", hovertemplate="%{y} probes<extra></extra>",
    ), row=2, col=1)
    fig.update_yaxes(title_text="relative (0 dry – 1 wet)", row=1, col=1)
    fig.update_yaxes(title_text="probes", rangemode="tozero", row=2, col=1)
    fig.update_layout(
        height=560, margin={"l": 60, "r": 20, "t": 20, "b": 40},
        legend={"orientation": "h", "y": -0.12}, hovermode="x unified",
    )
    return fig.to_html(include_plotlyjs="cdn", full_html=False)


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
            PAGE_TITLE, "<h1>Irrigation</h1><p>Pot moisture could not be loaded.</p>",
            show_back_link=True, back_url="/", show_source_indicator=False,
        )

    has_data = result.coverage[result.coverage > 0]
    if has_data.empty:
        return render_page(
            PAGE_TITLE, "<h1>Irrigation</h1><p>No pot-moisture readings yet.</p>",
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
                <small>Pot moisture now (0 dry – 1 wet)</small></article>
            <article><div class="stat-value" id="probes-now">{probes_now}</div>
                <small>Probes used now</small></article>
            <article><div class="stat-value">{latest_at:%d %b %H:%M}</div>
                <small>Latest reading (UTC)</small></article>
        </div>"""

    explainer = (
        "<p>The pot probes agree on <em>how</em> the pots wet and dry, but not on the "
        "level, so each is put on its own scale: <strong>0</strong> is its driest "
        "normal reading and <strong>1</strong> its wettest. The line is the median "
        "of the probes that passed the quality checks, and the band shows how far they "
        "spread. Probes are dropped automatically while they read zero (dry peat "
        "losing contact), spike, or stop moving with the others.</p>"
    )

    body = (
        "<h1>Irrigation</h1>"
        + explainer
        + stats
        + _season_toggle(years, year)
        + render_card(f"Pot moisture {year}", _chart(result, start, end))
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
