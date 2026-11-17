"""GET /irrigation/model: the shipped forecast model and the assumptions it rests on.

Read-only (ADR 0009: the model is fitted locally and shipped in code, and
production never trains it). It shows:

- what the model is, when it was fitted, and on which data
- each fitted parameter, with what it means physically
- how it scored in the backtest
- the **assumption register** (``assumptions.yaml``), with each assumption's
  current value looked up live from the config or the model

Open to every signed-in user, not just admins: the assumptions are what a
grower needs, to judge how far to trust the advice.
"""

from __future__ import annotations

import html

from fastapi import APIRouter, Depends
from fastapi.responses import HTMLResponse

from wp6_data.blue.irrigation.assumptions import format_value, load_assumptions, resolve
from wp6_data.blue.irrigation.config import load_config
from wp6_data.blue.irrigation.model import load_model, production_spec
from wp6_data.blue.irrigation.water_balance import PARAM_DOCS
from wp6_data.shared import render_card, render_page, render_table
from wp6_data.shared.auth import verify_session_user

router = APIRouter(dependencies=[Depends(verify_session_user)])

PAGE_TITLE = "SPoHF Blue - Irrigation model"

#: Readable labels for the backtest scores that calibrate stores.
SCORE_LABELS = {
    "skill_h1": ("Skill vs persistence, 1 day ahead", "{:+.0%}"),
    "skill_h7": ("Skill vs persistence, 7 days ahead", "{:+.0%}"),
    "band80_cover": ("Outcomes inside the 80 % band", "{:.0%}"),
    "cf_irrigation_ok": ("Watering never lowers the forecast", "{:.0%}"),
    "cf_dry_week_ok": ("A dry week never raises the forecast", "{:.0%}"),
    "cf_dose_gain_h2": ("Effect of a 2.4 L dose on day 2", "{:+.3f} S"),
}

MODEL_CSS = """
    .assumption { border-top: 1px solid var(--pico-muted-border-color); padding: 0.75rem 0; }
    .assumption h3 { font-size: 1rem; margin: 0 0 0.25rem; }
    .assumption .value { font-weight: bold; }
    .assumption dl { display: grid; grid-template-columns: 7rem 1fr; gap: 0.2rem 0.75rem;
        margin: 0.5rem 0 0; font-size: 0.9rem; }
    .assumption dt { color: var(--pico-muted-color); }
    .assumption dd { margin: 0; }
"""


def _assumptions_html() -> str:
    config, model = load_config(), load_model()
    parts = []
    for a in load_assumptions():
        value = ""
        if a.value:
            value = (
                f'<p>Current value: <span class="value">'
                f"{html.escape(format_value(resolve(a.value, config, model)))}</span> "
                f"{html.escape(a.unit)} <small>(<code>{html.escape(a.value)}</code>)</small></p>"
            )
        refs = ", ".join(a.refs)
        parts.append(
            f'<div class="assumption" id="{html.escape(a.id)}">'
            f"<h3>{html.escape(a.title)}</h3>"
            f"<p>{html.escape(a.statement)}</p>{value}"
            f"<dl><dt>Why</dt><dd>{html.escape(a.basis)}</dd>"
            f"<dt>If wrong</dt><dd>{html.escape(a.impact)}</dd>"
            f"<dt>How to test</dt><dd>{html.escape(a.test)}</dd>"
            + (f"<dt>See</dt><dd>{html.escape(refs)}</dd>" if refs else "")
            + "</dl></div>"
        )
    return "".join(parts)


@router.get("/irrigation/model", response_class=HTMLResponse)
async def irrigation_model() -> str:
    config, model = load_config(), load_model()
    spec = production_spec(config)

    overview = (
        f"<p><strong>{html.escape(spec.name)}</strong>: a daily water balance on the "
        f"fused soil moisture, forecasting {model.horizon} days ahead (ADR 0009).</p>"
        f"<p>Fitted <strong>{html.escape(model.fitted_at)}</strong> on "
        f"{model.origins} forecast starts, {model.data_first_day} to "
        f"{model.data_last_day}. It is fitted locally with "
        f"<code>wp6-blue-irrigation calibrate</code> and committed; production never "
        f"retrains it.</p>"
    )
    params = render_table(
        ["Parameter", "Value", "Meaning", "Unit"],
        [
            [f"<code>{name}</code>", f"{value:.4g}", html.escape(PARAM_DOCS[name][0]),
             html.escape(PARAM_DOCS[name][1])]
            for name, value in model.params.items()
        ],
        sortable=False,
    )
    scores = render_table(
        ["Backtest (rolling origin, weather as observed)", "Score"],
        [
            [html.escape(label), fmt.format(model.backtest[key])]
            for key, (label, fmt) in SCORE_LABELS.items() if key in model.backtest
        ],
        sortable=False,
    )
    body = (
        "<h1>Irrigation model</h1>"
        + render_card("Model", overview)
        + render_card("Fitted parameters", params)
        + render_card("How well it forecasts", scores)
        + render_card(
            "Assumptions",
            _assumptions_html(),
            description=(
                "Everything the forecast and advice take as given. Each value is read "
                "live from the config or the shipped model."
            ),
        )
    )
    return render_page(PAGE_TITLE, body, show_back_link=True, back_url="/irrigation",
                       extra_css=MODEL_CSS, show_source_indicator=False)
