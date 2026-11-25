"""E2E: the red export job's TSDB leg writes one wide CSV per readings device.

The MySQL legs (sensor tables, wire heights) are covered by the job's own
shape; what needed a real database is the TSDB leg: that a LetsGrow row in
`readings` ends up in `letsgrow-climate-b4.csv` as a pivoted column, and that
the device lands in the export metadata the explorer's Exports tab reads.
"""

import json
from datetime import UTC, datetime

import pandas as pd
import pytest
from psycopg_pool import AsyncConnectionPool

from tests.e2e.conftest import RED_TSDB_DSN, remove_cagg_refresh_policy
from wp6_data.api.models import LetsGrowReading
from wp6_data.db.pool import close_pool, init_pool
from wp6_data.db.queries import upsert_readings
from wp6_data.red import letsgrow
from wp6_data.red.export import export_tsdb_device, get_tsdb_device_names
from wp6_data.red.tsdb import ensure_schema_red
from wp6_data.shared.blob import LocalBlobStore
from wp6_data.shared.export import METADATA_KEY, csv_key, get_export_metadata

pytestmark = pytest.mark.e2e

ROW_TEMP = LetsGrowReading.model_validate({
    "module_id": 39887,
    "module_name": "Climate B4",
    "col_id": 1467902,
    "description": "Temperature greenhouse - °C - 5 min",
    "timestamp": "2026-10-15T14:00:00",
    "value": 21.5,
})
ROW_VPD = LetsGrowReading.model_validate({
    "module_id": 39887,
    "module_name": "Climate B4",
    "col_id": 1467551,
    "description": "Greenhouse: VPD - kPa - 5 min",
    "timestamp": "2026-10-15T14:00:00",
    "value": 0.9,
})


async def _drop_red_schema(conn) -> None:
    """Same teardown as test_letsgrow_ingest: cagg before the hypertable."""
    await remove_cagg_refresh_policy(conn)
    async with conn.cursor() as cur:
        await cur.execute("DROP MATERIALIZED VIEW IF EXISTS sensors_daily_summary")
        await cur.execute("DROP TABLE IF EXISTS sync_metadata CASCADE")
        await cur.execute("DROP TABLE IF EXISTS sync_run_history CASCADE")
        await cur.execute("DROP TABLE IF EXISTS readings CASCADE")
        await cur.execute("DROP TABLE IF EXISTS manual_uploads CASCADE")
    await conn.commit()


@pytest.fixture()
async def red_pool(red_tsdb_conn, tmp_path):
    await _drop_red_schema(red_tsdb_conn)
    pool = AsyncConnectionPool(RED_TSDB_DSN, min_size=1, max_size=2, open=False)
    await pool.open()
    await ensure_schema_red(pool)
    # The export module reads the global pool; point it at the e2e database.
    await close_pool()
    await init_pool(RED_TSDB_DSN, min_size=1, max_size=2)
    yield pool, LocalBlobStore(tmp_path)
    await close_pool()
    await pool.close()
    await _drop_red_schema(red_tsdb_conn)


async def test_tsdb_device_is_exported_wide(red_pool):
    pool, store = red_pool
    async with pool.connection() as conn:
        await upsert_readings(
            conn,
            [letsgrow.to_params(ROW_TEMP), letsgrow.to_params(ROW_VPD)],
            source=letsgrow.SOURCE,
        )
        await conn.commit()

    assert await get_tsdb_device_names() == ["letsgrow-climate-b4"]

    key = await export_tsdb_device("letsgrow-climate-b4", store)
    assert key == csv_key("letsgrow-climate-b4")

    raw = await store.try_get(key)
    df = pd.read_csv(pd.io.common.BytesIO(raw))
    # One row per timestamp, one column per lg_* tag.
    assert list(df.columns) == ["time", "lg_temp_greenhouse", "lg_vpd"]
    assert len(df) == 1
    assert df.loc[0, "lg_temp_greenhouse"] == 21.5
    assert df.loc[0, "lg_vpd"] == 0.9
    # 14:00 Amsterdam summer time -> 12:00 UTC.
    assert df.loc[0, "time"] == "2026-10-15 12:00:00+00:00"


async def test_tsdb_device_without_rows_exports_nothing(red_pool):
    pool, store = red_pool
    async with pool.connection() as conn:
        await upsert_readings(
            conn, [letsgrow.to_params(ROW_TEMP)], source=letsgrow.SOURCE
        )
        await conn.commit()

    assert await export_tsdb_device("letsgrow-water-b4", store) is None
    assert await store.try_get(csv_key("letsgrow-water-b4")) is None


async def test_export_metadata_lists_tsdb_devices_for_the_explorer(red_pool):
    """The Exports tab reads metadata.json; a TSDB device must appear in it."""
    pool, store = red_pool
    async with pool.connection() as conn:
        await upsert_readings(
            conn, [letsgrow.to_params(ROW_TEMP)], source=letsgrow.SOURCE
        )
        await conn.commit()

    key = await export_tsdb_device("letsgrow-climate-b4", store)
    assert key is not None
    exported_at = datetime.now(UTC).isoformat()
    await store.put(
        METADATA_KEY,
        json.dumps({
            "exported_at": exported_at,
            "devices": {"letsgrow-climate-b4": exported_at},
        }).encode(),
    )

    meta = await get_export_metadata(store)
    assert meta["devices"] == {"letsgrow-climate-b4": exported_at}


async def test_tsdb_export_reads_sijia_rows_too(red_pool):
    """Every readings device is exported, not just letsgrow's."""
    pool, store = red_pool
    sijia_row = {
        "device_name": "neurath-B-2034-strabelina",
        "sensor_tag": "chlorophyll",
        "value": "12.3",
        "datetime_measure": "2026-10-15T12:00:00+00:00",
    }
    async with pool.connection() as conn:
        await upsert_readings(conn, [sijia_row], source="sijia")
        await conn.commit()

    assert "neurath-B-2034-strabelina" in await get_tsdb_device_names()
    key = await export_tsdb_device("neurath-B-2034-strabelina", store)
    raw = await store.try_get(key)
    df = pd.read_csv(pd.io.common.BytesIO(raw))
    assert list(df.columns) == ["time", "chlorophyll"]
    assert df.loc[0, "chlorophyll"] == 12.3
