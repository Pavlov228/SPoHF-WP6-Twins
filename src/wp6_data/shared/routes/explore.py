"""Sensor explorer page: device, sensor and manual-measurement tables."""

import asyncio
from typing import Annotated, Any

from fastapi import APIRouter, Depends
from fastapi.responses import HTMLResponse

from wp6_data.shared import (
    build_explore_tabs,
    render_card,
    render_explore_tabs,
    render_page,
)
from wp6_data.shared.auth import verify_session_user
from wp6_data.shared.export import get_export_metadata
from wp6_data.shared.routes.deps import get_provider, get_twin_config
from wp6_data.shared.twin import SensorDataProvider, TwinConfig

router = APIRouter(dependencies=[Depends(verify_session_user)])


async def load_explore_data(
    config: TwinConfig, provider: SensorDataProvider,
) -> tuple[dict[str, dict], dict[str, Any], dict[str, str]]:
    """Everything the explorer reads: devices, manual freshness, exports.

    The three are independent, so they run concurrently and the page waits for
    the slowest one rather than their sum. Also called at startup to warm the
    caches behind them (see ``app_factory``).
    """
    device_data, manual_metadata, export_meta = await asyncio.gather(
        provider.fetch_device_data(),
        provider.fetch_manual_metadata(),
        get_export_metadata(config.export_store),
    )
    available_exports = export_meta.get("devices", {}) if export_meta else {}
    return device_data, manual_metadata, available_exports


@router.get("/explore", response_class=HTMLResponse)
async def explore(
    config: Annotated[TwinConfig, Depends(get_twin_config)],
    provider: Annotated[SensorDataProvider, Depends(get_provider)],
    tab: str = "devices",
) -> str:
    """Sensor explorer — every device, sensor and manual upload in one table set."""
    device_data, manual_metadata, available_exports = await load_explore_data(
        config, provider,
    )

    explore_tabs = build_explore_tabs(
        config.metadata, device_data, available_exports,
        manual_metadata=manual_metadata,
        manual_sources=config.manual_sources,
    )

    content = f"""
        <h1>Explore</h1>

        {render_card("Devices &amp; sensors", render_explore_tabs(
            explore_tabs, active=tab, extra_tabs=config.explore_extra_tabs,
        ))}
    """

    return render_page(f"Explore — {config.title}", content, show_back_link=True)
