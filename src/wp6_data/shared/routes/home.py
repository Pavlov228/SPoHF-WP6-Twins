"""Shared dashboard home page with pluggable hero cards."""

import inspect
from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import HTMLResponse

from wp6_data.shared import render_card, render_page
from wp6_data.shared.auth import verify_session_user
from wp6_data.shared.i18n import th
from wp6_data.shared.routes.deps import get_twin_config
from wp6_data.shared.twin import TwinConfig

router = APIRouter(dependencies=[Depends(verify_session_user)])


@router.get("/", response_class=HTMLResponse)
async def home(
    config: Annotated[TwinConfig, Depends(get_twin_config)],
) -> str:
    """Dashboard home page — entry cards; the explorer lives on ``/explore``."""

    # Render hero cards (can be sync or async callables)
    hero_html_parts = []
    for card_fn in config.hero_cards:
        result = card_fn()
        if inspect.isawaitable(result):
            result = await result
        hero_html_parts.append(result)
    hero_html = "\n".join(hero_html_parts)
    extra = config.status_card_extra_html
    status_extra = extra() if callable(extra) else extra

    content = f"""
        <h1>{config.title}</h1>

        {render_card(
            th("home.dashboard.title"),
            '<div style="display:flex;gap:0.5rem">'
            f'<a href="/dashboard" role="button">{th("home.dashboard.mine")}</a>'
            '<a href="/chart" role="button" class="outline">'
            f'{th("home.dashboard.new_chart")}</a>'
            '<a href="/explore" role="button" class="outline">'
            f'{th("home.dashboard.explore")}</a></div>',
            description=th("home.dashboard.description"),
            card_class="card card-bg card-bg-chart",
        )}

        {hero_html}

        {render_card(
            th("home.status.title"),
            '<div style="display:flex;gap:0.5rem;flex-wrap:wrap">'
            f'<a href="/status" role="button">{th("home.status.button")}</a>'
            f'{status_extra}</div>',
            description=th("home.status.description"),
            card_class="card-bg card-bg-status",
        )}
    """

    return render_page(config.title, content)
