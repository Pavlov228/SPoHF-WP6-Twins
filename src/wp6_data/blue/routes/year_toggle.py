"""The segmented year selector shared by blue's season pages (GDD, irrigation)."""

from __future__ import annotations

from collections.abc import Callable

#: Styles for :func:`year_toggle`; include in the page's ``extra_css``.
YEAR_TOGGLE_CSS = """
    .year-toggle {
        display: flex; gap: 0; margin-bottom: 0.5rem;
        border-radius: 6px; overflow: hidden;
        border: 1px solid var(--dashboard-primary);
        width: fit-content;
    }
    .year-btn {
        font-size: 0.8rem; padding: 0.3rem 0.8rem;
        cursor: pointer; border: none; border-radius: 0;
        background: transparent;
        color: var(--dashboard-primary);
        text-decoration: none;
        transition: background 0.15s, color 0.15s;
    }
    .year-btn:not(:last-child) {
        border-right: 1px solid var(--dashboard-primary);
    }
    .year-btn.active {
        background: var(--dashboard-primary); color: #fff;
    }
    .year-btn:not(.active):hover {
        background: color-mix(
            in srgb, var(--dashboard-primary) 12%, transparent);
    }
"""


def year_toggle(years: list[int], active: int, href: Callable[[int], str]) -> str:
    """One link per year, joined into a segmented control; *href* builds each link."""
    buttons = "".join(
        f'<a href="{href(y)}" class="year-btn{" active" if y == active else ""}"'
        f'{" aria-current=\"page\"" if y == active else ""}>{y}</a>'
        for y in years
    )
    return f'<nav class="year-toggle" aria-label="Season">{buttons}</nav>'
