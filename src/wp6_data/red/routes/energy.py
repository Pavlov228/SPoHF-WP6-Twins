"""GET /energy — lamp-hours and heat output over the years, as LetsGrow recorded them.

The two things red spends energy on, as running totals through each year, every
year on one Jan-Dec calendar (``make_year_to_date_chart``), so a season compares
against the same season of other years. The per-day lines are behind a closed
section: the year-on-year totals are the question this page answers. Lamp-hours are
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
from wp6_data.shared.charts import make_daily_panels_chart, make_year_to_date_chart

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
    hourly = await fetch_heat_output(
        config.device, config.air, config.pipes, config.exponent, config.max_pipe,
    )
    return daily_heat_output(hourly)


async def _load(title: str, load) -> tuple[pd.DataFrame | None, str]:
    """A source's daily frame, or ``None`` and why; a failure stays with its source."""
    try:
        daily = await load()
    except Exception as e:  # noqa: BLE001 - one source down must not blank the page
        log.warning("energy_source_unavailable", source=title, error=str(e))
        return None, f"<p>{title}: could not load ({e}).</p>"
    if daily.empty:
        return None, f"<p>{title}: no complete days yet.</p>"
    return daily, ""


# Plotly sizes a chart when it is drawn, and a chart inside a closed <details>
# is drawn at no width; resize them once the section is opened.
_RESIZE_ON_OPEN = """
<script>
document.querySelectorAll("details.energy-daily").forEach(function (section) {
  section.addEventListener("toggle", function () {
    if (!section.open) return;
    section.querySelectorAll(".plotly-graph-div").forEach(function (div) {
      Plotly.Plots.resize(div);
    });
  });
});
</script>
"""


@router.get("/energy", response_class=HTMLResponse)
async def energy() -> str:
    """Lamp-hours and heat output, year against year, every year LetsGrow has."""
    if not is_initialised():
        return render_page(PAGE_TITLE, "<h1>Database not connected</h1>")

    heating = deps.heat_output
    lamps, lamps_note = await _load("Lamp-hours", _daily_lamp_hours)
    heat, heat_note = await _load("Heat output", _daily_heat_index)

    sources = [
        (lamps, "lamp_hours", "Lamp-hours", "h"),
        (heat, "heat_index", "Heat-output index", ""),
    ]
    present = [s for s in sources if s[0] is not None]

    totals_html = ""
    daily_html = ""
    if present:
        totals = make_year_to_date_chart(present, hide_late_starts=True)
        # The totals come first, so they bring plotly.js for both charts.
        totals_html = totals.to_html(full_html=False, include_plotlyjs="cdn")
        daily = make_daily_panels_chart(present, hide_late_starts=True)
        daily_html = daily.to_html(full_html=False, include_plotlyjs=False)

    content = f"""
        <h1>Energy</h1>
        <p>What red spends energy on — the lamps and the heating — as a running total
        through each year, every year LetsGrow has laid on the same calendar. Each
        point is a month end, marked with what that month added; the last is labelled
        with where the year stands. Click a year in the legend to hide it in both,
        double-click to show only that one. A year whose data starts after January is
        hidden to begin with — its total would read low — and is one click away.</p>
        {lamps_note}{heat_note}
        {totals_html}
        <small>
            <strong>Lamp-hours</strong>: hours the lamps ran for at least
            {LIT_HOUR_SHARE:.0%} of the hour, from LetsGrow's lamp switch — the same
            count as the <a href="/dli/lamps">Lamp Calendar</a>.<br/>
            <strong>Heat-output index</strong>: for every LetsGrow reading, how much
            warmer each heating pipe ({", ".join(heating.pipes)}) is than the greenhouse
            air ({heating.air}), raised to the power {heating.exponent:g} and 0 when the
            pipe is not warmer; averaged per hour, added up over the pipes, and summed per
            day. A heating pipe gives off heat roughly in proportion to that, so the index
            follows gas use. It is <strong>relative</strong>: both pipes count equally, and
            no pipe size, boiler efficiency or gas price is in it, so compare it between
            years, not as kWh or euros. Pipe readings above {heating.max_pipe:g} °C are
            sensor faults and are left out. Heating hours are not shown because the pipes
            stay warm almost around the clock all year.<br/>
            Only days LetsGrow reported in full count; it delivers a day or more late, so
            the newest days are missing.
        </small>
        <details class="energy-daily">
            <summary>Per day</summary>
            {daily_html}
        </details>
        {_RESIZE_ON_OPEN if daily_html else ""}
    """
    return render_page(PAGE_TITLE, content, show_back_link=True, back_url="/")
