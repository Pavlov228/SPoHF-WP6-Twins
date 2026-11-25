"""GET /energy — lamp-hours and heat output over the years, as LetsGrow recorded them.

The two things red spends energy on, each as a daily line per year on one
Jan-Dec calendar with its running total below (``make_daily_by_year_chart``),
so a season compares against the same season of other years. Lamp-hours are
counted exactly as on /dli/lamps; the heating is the relative heat-output index
of ``red.heating``, because LetsGrow has no gas meter.
"""

import pandas as pd
import structlog
from fastapi import APIRouter, Depends
from fastapi.responses import HTMLResponse

from wp6_data.db.pool import is_initialised
from wp6_data.red import deps
from wp6_data.red.dli import data as dli_data
from wp6_data.red.heating import daily_heat_output
from wp6_data.red.lamp import LIT_HOUR_SHARE, daily_light_hours, lamp_state_grid
from wp6_data.red.tsdb import fetch_heat_output
from wp6_data.shared import render_page
from wp6_data.shared.auth import verify_session_user
from wp6_data.shared.charts import make_daily_by_year_chart

router = APIRouter(dependencies=[Depends(verify_session_user)])
log = structlog.get_logger()

PAGE_TITLE = "SPoHF Red - Energy"


async def _daily_lamp_hours() -> pd.DataFrame:
    """Per day, the hours the lamps ran — the same count /dli/lamps draws."""
    radiation = await dli_data.get_daylight_radiation(None, None)
    on_fraction = await dli_data.get_lamp_on_fraction(None, None)
    grid = lamp_state_grid(radiation, on_fraction, deps.daylight.threshold)
    return daily_light_hours(grid)


async def _daily_heat_index() -> pd.DataFrame:
    config = deps.heat_output
    hourly = await fetch_heat_output(config.device, config.air, config.pipes, config.exponent)
    return daily_heat_output(hourly)


class _Plotly:
    """Hands plotly.js to the first chart that renders, and to no other.

    Whichever section that is: one failing must not leave the other without it.
    """

    def __init__(self) -> None:
        self.loaded = False

    def include(self) -> str | bool:
        if self.loaded:
            return False
        self.loaded = True
        return "cdn"


async def _section(
    plotly: _Plotly,
    title: str,
    load,
    column: str,
    label: str,
    unit: str,
    caption: str,
    **chart_kwargs,
) -> str:
    """One chart with its heading and caption; a failure stays inside its section."""
    try:
        daily = await load()
    except Exception as e:  # noqa: BLE001 - one source down must not blank the page
        log.warning("energy_section_unavailable", section=title, error=str(e))
        return f"<h2>{title}</h2><p>Could not load: {e}</p>"
    if daily.empty:
        return f"<h2>{title}</h2><p>No complete days yet.</p>"
    chart = make_daily_by_year_chart(daily, column, label, unit, **chart_kwargs)
    return (
        f"<h2>{title}</h2>"
        f"{chart.to_html(full_html=False, include_plotlyjs=plotly.include())}"
        f"<small>{caption}</small>"
    )


@router.get("/energy", response_class=HTMLResponse)
async def energy() -> str:
    """Lamp-hours and heat output per day, every year LetsGrow has, side by side."""
    if not is_initialised():
        return render_page(PAGE_TITLE, "<h1>Database not connected</h1>")

    heating = deps.heat_output
    plotly = _Plotly()
    lamps_html = await _section(
        plotly, "Lamp-hours", _daily_lamp_hours, "lamp_hours", "Lamp-hours", "h",
        f"Hours a day the lamps ran for at least {LIT_HOUR_SHARE:.0%} of the hour, from "
        "LetsGrow's lamp switch — the same count as the "
        '<a href="/dli/lamps">Lamp Calendar</a>.',
        daily_range=(0, 24),
    )
    heat_html = await _section(
        plotly, "Heat output", _daily_heat_index, "heat_index", "Heat-output index", "",
        "<strong>How it is calculated:</strong> for every LetsGrow reading, how much "
        f"warmer each heating pipe ({', '.join(heating.pipes)}) is than the greenhouse "
        f"air ({heating.air}), raised to the power {heating.exponent:g} and 0 when the "
        "pipe is not warmer; averaged per hour, added up over the pipes, and summed "
        "per day. A heating pipe gives off heat roughly in proportion to that, so the "
        "index follows gas use. It is <strong>relative</strong>: both pipes count "
        "equally, and no pipe size, boiler efficiency or gas price is in it, so "
        "compare it between days and years, not as kWh or euros. Heating hours "
        "are not shown because the pipes stay warm almost around the clock all year.",
    )

    content = f"""
        <h1>Energy</h1>
        <p>What red spends energy on — the lamps and the heating — per day, with every
        year LetsGrow has laid on the same calendar. Below each chart, the year's
        running total, marked at every month end with what that month added. Click a
        year in a legend to hide it, double-click to show only that one.</p>
        {lamps_html}
        {heat_html}
        <small>Only days LetsGrow reported in full are drawn; it delivers a day or
        more late, so the newest days are missing.</small>
    """
    return render_page(PAGE_TITLE, content, show_back_link=True, back_url="/")
