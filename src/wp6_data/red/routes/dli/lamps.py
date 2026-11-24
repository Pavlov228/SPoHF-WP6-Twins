"""GET /dli/lamps — When the lamps are on, as a date × hour calendar heatmap."""

from datetime import date, timedelta
from typing import Annotated

import numpy as np
import plotly.graph_objects as go
import structlog
from fastapi import APIRouter, Query
from fastapi.responses import HTMLResponse

from wp6_data.db.pool import is_initialised
from wp6_data.red import deps
from wp6_data.red.dli import data as dli_data
from wp6_data.red.lamp import (
    LIT_HOUR_SHARE,
    LampState,
    daily_light_hours,
    lamp_state_grid,
)
from wp6_data.shared import pill_row, render_date_filter, render_page, utc_day_bounds
from wp6_data.shared.charts import CATEGORICAL_COLORS

router = APIRouter()
log = structlog.get_logger()

PAGE_TITLE = "SPoHF Red - Lamp Calendar"
BASE_PATH = "/dli/lamps"

# Which daily count the by-year chart draws. One at a time: both, for every
# year, is too many lines to read. A query parameter, like /dli/performance's
# view, so a particular view is a link.
LAMP_HOURS = "lamps"
LIGHT_HOURS = "light"
HOURS_LABELS = {
    LAMP_HOURS: "Lamp-hours",
    LIGHT_HOURS: "Light-hours (sun or lamps)",
}
HOURS_COLUMNS = {LAMP_HOURS: "lamp_hours", LIGHT_HOURS: "light_hours"}
DEFAULT_HOURS = LAMP_HOURS

# Lamps get a hue sunlight never takes, so "artificial" reads before the legend
# does. Night is the recessive state — most of a winter grid is night, and it is
# the least interesting thing on the page.
STATE_COLOURS = {
    LampState.DARK: "#eceff3",
    LampState.DAYLIGHT: "#fbbf24",
    LampState.DAYLIGHT_LIT: "#c026d3",
    LampState.LIT: "#7c3aed",
}
STATE_LABELS = {
    LampState.DARK: "Dark (no lamp)",
    LampState.DAYLIGHT: "Daylight",
    LampState.DAYLIGHT_LIT: "Daylight + lamps",
    LampState.LIT: "Lamps only",
}


def _discrete_colorscale() -> list:
    """A stepped scale, so a state is a block of colour rather than a gradient."""
    steps = []
    n = len(STATE_COLOURS)
    for i, colour in enumerate(STATE_COLOURS.values()):
        steps.append([i / n, colour])
        steps.append([(i + 1) / n, colour])
    return steps


def _make_calendar(grid, start: date, end: date) -> go.Figure:
    """Hours down, days across — the shape a grower already reads a schedule in."""
    dates = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    pivot = grid.pivot_table(index="hour", columns="date", values="state", aggfunc="max")
    pivot = pivot.reindex(index=range(24), columns=dates)

    hover = np.dstack([
        grid.pivot_table(index="hour", columns="date", values=column, aggfunc="mean")
        .reindex(index=range(24), columns=dates)
        .to_numpy(dtype=object)
        for column in ("radiation", "on_fraction")
    ])

    fig = go.Figure(
        go.Heatmap(
            z=pivot.to_numpy().tolist(),
            x=[d.isoformat() for d in dates],
            y=[f"{h:02d}:00" for h in range(24)],
            customdata=hover.tolist(),
            colorscale=_discrete_colorscale(),
            zmin=0.5,
            zmax=len(STATE_COLOURS) + 0.5,
            hoverongaps=False,
            xgap=0,
            ygap=0,
            colorbar={
                "tickvals": [int(s) for s in STATE_COLOURS],
                "ticktext": [STATE_LABELS[s] for s in STATE_COLOURS],
                "thickness": 14,
                "tickfont": {"size": 11},
                "outlinewidth": 0,
            },
            hovertemplate=(
                "%{x} %{y}<br>outdoor radiation = %{customdata[0]:.0f} W/m²"
                "<br>lamps on %{customdata[1]:.0%} of the hour<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        template="plotly_white",
        height=560,
        margin={"l": 60, "r": 20, "t": 20, "b": 40},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis={"title": "", "type": "category", "nticks": 20},
        yaxis={"title": "Hour (UTC)", "autorange": "reversed", "dtick": 2},
    )
    return fig


# Every year is drawn onto one leap year, so 29 February has somewhere to go and
# the axis reads as months rather than as any particular year.
_OVERLAY_YEAR = 2000


def _make_hours_by_year(daily, hours: str) -> go.Figure:
    """One daily count per year, one colour per year, laid on one calendar."""
    column, label = HOURS_COLUMNS[hours], HOURS_LABELS[hours]
    fig = go.Figure()
    years = sorted({d.year for d in daily["date"]})
    for i, year in enumerate(years):
        rows = daily[[d.year == year for d in daily["date"]]].set_index("date")
        # Every day between the year's first and last, so a missing day breaks
        # the line instead of being bridged.
        first, last = min(rows.index), max(rows.index)
        days = [first + timedelta(days=n) for n in range((last - first).days + 1)]
        rows = rows.reindex(days)
        fig.add_trace(go.Scatter(
            x=[d.replace(year=_OVERLAY_YEAR) for d in days],
            y=rows[column],
            customdata=[d.isoformat() for d in days],
            mode="lines",
            name=str(year),
            line={"color": CATEGORICAL_COLORS[i % len(CATEGORICAL_COLORS)], "width": 1.5},
            connectgaps=False,
            hovertemplate=f"%{{customdata}}<br>{label}: %{{y}} h<extra></extra>",
        ))
    fig.update_layout(
        template="plotly_white",
        height=420,
        margin={"l": 60, "r": 20, "t": 20, "b": 40},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="closest",
        xaxis={"title": "", "tickformat": "%b", "dtick": "M1",
               "range": [date(_OVERLAY_YEAR, 1, 1), date(_OVERLAY_YEAR, 12, 31)]},
        yaxis={"title": f"{label} per day", "range": [0, 24], "dtick": 4},
    )
    return fig


@router.get("/lamps", response_class=HTMLResponse)
async def dli_lamps(
    start: Annotated[date | None, Query(description="Start date")] = None,
    end: Annotated[date | None, Query(description="End date")] = None,
    hours: Annotated[
        str, Query(description="Daily count to chart by year (lamps/light)"),
    ] = DEFAULT_HOURS,
) -> str:
    """Show which hours the lamps ran and the sun was up, as LetsGrow recorded them."""
    if not is_initialised():
        return render_page(PAGE_TITLE, "<h1>Database not connected</h1>",
                          show_back_link=True, back_url="/dli")

    if hours not in HOURS_LABELS:
        hours = DEFAULT_HOURS
    # The toggle keeps the dates the user chose — and only those, so "all"
    # stays all rather than freezing at today's first day.
    chosen = {"start": start.isoformat() if start else "", "end": end.isoformat() if end else ""}

    today = date.today()
    if end is None:
        end = today
    if start is not None and start > end:
        start = end

    # The default is all of it. Rather than reach back to a fixed epoch, leave
    # the query unbounded and let LetsGrow's first reading decide where the
    # calendar starts.
    _, end_dt = utc_day_bounds(end)
    start_dt = utc_day_bounds(start)[0] if start is not None else None

    try:
        radiation = await dli_data.get_daylight_radiation(start_dt, end_dt)
        on_fraction = await dli_data.get_lamp_on_fraction(start_dt, end_dt)
    except Exception as e:  # noqa: BLE001 - shown on the page, like the other DLI pages
        log.warning("lamp_calendar_letsgrow_unavailable", error=str(e))
        return render_page(PAGE_TITLE, f"<h1>Error: {e}</h1>",
                          show_back_link=True, back_url="/dli")

    daylight = deps.daylight
    grid = lamp_state_grid(radiation, on_fraction, daylight.threshold)

    if grid.empty:
        return render_page(
            PAGE_TITLE,
            f"<h1>Lamp Calendar</h1>{render_date_filter(start or end, end, {'hours': hours})}"
            "<p>LetsGrow has no lamp and radiation readings in this range.</p>",
            show_back_link=True, back_url="/dli",
        )

    # "All" means all of *this* page's data, on the filter as well as on load.
    data_start = min(grid["date"])
    if start is None:
        start = data_start
    filter_html = render_date_filter(start, end, {"hours": hours}, all_start=data_start)

    chart_html = _make_calendar(grid, start, end).to_html(
        full_html=False, include_plotlyjs="cdn"
    )

    daily = daily_light_hours(grid)
    hours_html = (
        _make_hours_by_year(daily, hours).to_html(full_html=False, include_plotlyjs=False)
        if not daily.empty else "<p>No complete days in this range.</p>"
    )

    toggle_html = pill_row(BASE_PATH, "hours", list(HOURS_LABELS.items()), hours, chosen, "Show")

    content = f"""
        <h1>Lamp Calendar</h1>
        <p>Every hour of every day, coloured by what was lighting the crop: the sun,
        and the lamps, both as LetsGrow, Neurath's climate computer, recorded them.</p>
        {filter_html}
        {chart_html}
        <small>
            <strong>Daylight</strong>: LetsGrow's outdoor weather station
            ({daylight.device}) measured more than {daylight.threshold:g} W/m² of
            radiation over the hour.
            <strong>Lamps</strong>: LetsGrow, which switches them, had them on for
            at least {LIT_HOUR_SHARE:.0%} of the hour — the record rather than an
            inference, including on bright winter middays where the lamp is
            invisible next to the sun.<br/>
            Gaps are hours LetsGrow did not report, not hours that were dark.
            LetsGrow delivers a day or more late, so the newest days are blank.
        </small>
        <h2>Lamp-hours and light-hours per day, by year</h2>
        <p>Each year on the same calendar, one line per year. <strong>Lamp-hours</strong>:
        hours the lamps ran. <strong>Light-hours</strong>: hours with any light on
        the crop — sun, lamps or both. Click a year in the legend to hide it,
        double-click to show only that one.</p>
        {toggle_html}
        {hours_html}
        <small>
            Same hour states as the calendar above, counted per day. Only days with
            every hour reported are drawn, so a LetsGrow gap breaks the line rather
            than dipping it.
        </small>
    """

    return render_page(PAGE_TITLE, content, show_back_link=True, back_url="/dli")
