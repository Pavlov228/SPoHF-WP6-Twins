"""E2E: /irrigation fuses probe readings straight out of TimescaleDB."""

from datetime import UTC, datetime, timedelta

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from tests.e2e.conftest import E2E_PREFIX, TSDB_DSN
from wp6_data.blue.irrigation.config import load_config
from wp6_data.blue.routes import irrigation
from wp6_data.db import close_pool, init_pool, upsert_readings
from wp6_data.shared.auth import verify_session_user
from wp6_data.shared.templates.config import configure_dashboard

pytestmark = pytest.mark.e2e

PROBES = tuple(f"{E2E_PREFIX}irr-probe-{i}" for i in range(3))


@pytest.mark.e2e
async def test_irrigation_page_fuses_readings_from_the_database(monkeypatch):
    configure_dashboard("blue")
    base = load_config()
    now = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    start = now - timedelta(days=3)
    config = base.model_copy(update={"probes": PROBES, "history_start": start.date()})
    monkeypatch.setattr(irrigation, "load_config", lambda: config)
    irrigation._fused_cache.clear()

    readings = [
        {
            "device_name": probe,
            "sensor_tag": config.sensor,
            "value": str(10.0 + 5 * i + h / 10),
            "datetime_measure": start + timedelta(hours=h),
        }
        for i, probe in enumerate(PROBES)
        for h in range(72)
    ]
    # The page reads through the blue provider, which uses the global pool.
    pool = await init_pool(TSDB_DSN)
    try:
        async with pool.connection() as conn:
            await upsert_readings(conn, readings)
            await conn.commit()

        app = FastAPI()
        app.include_router(irrigation.router)
        app.dependency_overrides[verify_session_user] = lambda: "e2e-user"
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.get("/irrigation")
    finally:
        irrigation._fused_cache.clear()
        await close_pool()

    assert response.status_code == 200
    assert "Fused pot moisture" in response.text
    assert f'id="probes-now">{len(PROBES)} / {len(PROBES)}<' in response.text
    for probe in PROBES:
        assert probe in response.text
