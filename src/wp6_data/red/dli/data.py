"""DLI data-access seam over the live MySQL feed.

The single place (outside the composition root in ``red.dashboard``) that
touches the ``wp6_data.red.deps.db`` singleton on behalf of the DLI and
DLI-model routes. Routes import this module instead of reaching for
``deps.db`` directly, keeping red's direct-MySQL surface narrow ahead of
the TSDB migration.

Deliberately NOT re-exported via ``wp6_data.red.dli`` — that facade is the
model/calculator API; data access stays an explicit separate import.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from wp6_data.red import deps

if TYPE_CHECKING:
    from datetime import datetime

    import pandas as pd

    from wp6_data.red.db import MySQLConnection


def is_connected() -> bool:
    """Whether the live MySQL feed is available (the routes' guard pattern)."""
    return deps.db is not None


def require_db() -> MySQLConnection:
    """The live MySQL connection, raising when the app started without one."""
    if deps.db is None:
        raise RuntimeError("Database not connected")
    return deps.db


async def get_par_readings(
    device_ids: list[str] | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    limit: int = 500_000,
) -> pd.DataFrame:
    """PAR readings from the live feed.

    Thin delegation mirroring ``MySQLConnection.get_par_readings``; raises
    ``RuntimeError`` when the database is not connected.
    """
    return await require_db().get_par_readings(
        device_ids=device_ids, start=start, end=end, limit=limit,
    )


async def get_weather_station_readings(
    start: datetime | None = None,
    end: datetime | None = None,
    limit: int = 500_000,
) -> pd.DataFrame:
    """Weather-station readings from the live feed.

    Thin delegation mirroring ``MySQLConnection.get_weather_station_readings``;
    raises ``RuntimeError`` when the database is not connected.
    """
    return await require_db().get_weather_station_readings(
        start=start, end=end, limit=limit,
    )


async def get_lamp_on_fraction(
    start: datetime | None,
    end: datetime | None,
) -> pd.Series:
    """Share of each hour the lamps ran, per LetsGrow — see ``red.tsdb.fetch_lamp_on_fraction``.

    From red's TimescaleDB, where the LetsGrow sync lands — not the MySQL feed
    the PAR readings come from. Which device and tags is ``deps.lamp_status``.
    """
    from wp6_data.red.tsdb import fetch_lamp_on_fraction

    config = deps.lamp_status
    return await fetch_lamp_on_fraction(
        config.device, config.switch, config.share, start, end,
    )


async def get_daylight_radiation(
    start: datetime | None,
    end: datetime | None,
) -> pd.Series:
    """LetsGrow's outdoor radiation, W/m² per UTC hour, frozen hours left out.

    Which tag is ``deps.daylight``; what counts as frozen is
    ``red.lamp.radiation_without_frozen_hours``.
    """
    from wp6_data.red.lamp import radiation_without_frozen_hours
    from wp6_data.red.tsdb import fetch_hourly_stats

    config = deps.daylight
    stats = await fetch_hourly_stats(config.device, config.radiation, start, end)
    return radiation_without_frozen_hours(stats)
