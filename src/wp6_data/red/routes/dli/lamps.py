"""GET /dli/lamps — When the lamps are on, as a date × hour calendar heatmap."""

from datetime import date, timedelta
from typing import Annotated

import numpy as np
import plotly.graph_objects as go
import structlog
from fastapi import APIRouter, Query
from fastapi.responses import HTMLResponse

from wp6_data.red.dli import NATURAL_LIGHT_SENSOR, TOTAL_LIGHT_SENSOR
from wp6_data.red.dli import data as dli_data
from wp6_data.red.lamp import (
    DAYLIGHT_THRESHOLD,
    LAMP_THRESHOLD,
    LampState,
    lamp_state_grid,
)
from wp6_data.shared import render_date_filter, render_page, utc_day_bounds

router = APIRouter()
log = structlog.get_logger()

PAGE_TITLE = "SPoHF Red - Lamp Calendar"

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
    """A three-step scale, so a state is a block of colour rather than a gradient."""
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

    canopy = grid.pivot_table(index="hour", columns="date", values="canopy_par", aggfunc="mean")
    canopy = canopy.reindex(index=range(24), columns=dates)
    source = grid.pivot_table(index="hour", columns="date", values="source", aggfunc="first")
    source = source.reindex(index=range(24), columns=dates)
    hover = np.dstack([canopy.to_numpy(dtype=object), source.to_numpy(dtype=object)])

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
                "%{x} %{y}<br>canopy PAR = %{customdata[0]:.0f} µmol/m²/s"
                "<br>lamps from %{customdata[1]}<extra></extra>"
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


@router.get("/lamps", response_class=HTMLResponse)
async def dli_lamps(
    start: Annotated[date | None, Query(description="Start date")] = None,
    end: Annotated[date | None, Query(description="End date")] = None,
) -> str:
    """Show which hours the lamps ran, read straight off the two PAR sensors."""
    if not dli_data.is_connected():
        return render_page(PAGE_TITLE, "<h1>Database not connected</h1>",
                          show_back_link=True, back_url="/dli")

    today = date.today()
    if end is None:
        end = today
    if start is not None and start > end:
        start = end

    # The default is all of it. Rather than reach back to a fixed epoch — which
    # lands on years the sensors did not exist for — leave the query unbounded
    # and let the first reading decide where the calendar starts.
    _, end_dt = utc_day_bounds(end)
    start_dt = utc_day_bounds(start)[0] if start is not None else None

    try:
        par_df = await dli_data.get_par_readings(
            device_ids=[NATURAL_LIGHT_SENSOR, TOTAL_LIGHT_SENSOR], start=start_dt, end=end_dt
        )
    except Exception as e:
        return render_page(PAGE_TITLE, f"<h1>Error: {e}</h1>",
                          show_back_link=True, back_url="/dli")

    if par_df.empty:
        return render_page(
            PAGE_TITLE,
            f"<h1>Lamp Calendar</h1>{render_date_filter(start or end, end)}"
            "<p>No PAR readings in this range.</p>",
            show_back_link=True, back_url="/dli",
        )

    above_df = par_df[par_df["device"] == NATURAL_LIGHT_SENSOR]
    canopy_df = par_df[par_df["device"] == TOTAL_LIGHT_SENSOR]

    # LetsGrow switches the lamps, so where it has reported, it says whether they
    # ran; the PAR sensors still say whether the sun was up. Its absence (down,
    # or not yet delivered) leaves those hours to the sensors, labelled as such.
    try:
        on_fraction = await dli_data.get_lamp_on_fraction(start_dt, end_dt)
    except Exception as e:  # noqa: BLE001 - LetsGrow down degrades to the PAR sensors
        log.warning("lamp_status_unavailable", error=str(e))
        on_fraction = None

    grid = lamp_state_grid(above_df, canopy_df, on_fraction)

    if grid.empty:
        return render_page(
            PAGE_TITLE,
            f"<h1>Lamp Calendar</h1>{render_date_filter(start or end, end)}"
            "<p>Both PAR sensors are needed to tell lamp light from daylight, "
            "and they do not overlap in this range.</p>",
            show_back_link=True, back_url="/dli",
        )

    # "All" means all of *this* page's data, on the filter as well as on load.
    data_start = min(grid["date"])
    if start is None:
        start = data_start
    filter_html = render_date_filter(start, end, all_start=data_start)

    chart_html = _make_calendar(grid, start, end).to_html(
        full_html=False, include_plotlyjs="cdn"
    )

    content = f"""
        <h1>Lamp Calendar</h1>
        <p>Every hour of every day, coloured by what was lighting the crop: the sun,
        read off the roof PAR sensor, and the lamps, as LetsGrow switched them.</p>
        {filter_html}
        {chart_html}
        <small>
            <strong>Daylight</strong>: the above-lamp sensor ({NATURAL_LIGHT_SENSOR})
            reads at least {DAYLIGHT_THRESHOLD:.0f} µmol/m²/s.
            <strong>Lamps</strong>: LetsGrow, Neurath's climate computer, had them on
            for most of the hour. It switches the lamps, so this is the record rather
            than an inference — including on bright winter middays, where the lamp is
            invisible next to the sun. LetsGrow arrives about two days late; hours
            after its last delivery fall back to the PAR sensors (canopy above
            {LAMP_THRESHOLD:.0f} µmol/m²/s with no sun, or brighter than the roof
            sensor in daylight). Hover a cell to see which applied.<br/>
            Gaps are hours with no usable PAR reading, not hours that were dark.
        </small>
    """

    return render_page(PAGE_TITLE, content, show_back_link=True, back_url="/dli")
