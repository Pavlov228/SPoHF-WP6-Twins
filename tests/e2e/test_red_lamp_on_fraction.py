"""E2E: LetsGrow's hourly lamp on-fraction, computed in red's TimescaleDB.

The rule (share/100 while switched on, else 0, averaged per hour) lives only in
the SQL, so only a real database can test it. The regression it guards:
pulling the raw rows instead hit a 500k cap and silently lost everything after
mid-2025, which sent /dli/lamps back to the PAR sensors for every hour it drew.
"""

from datetime import UTC, datetime, timedelta

import pytest
from psycopg_pool import AsyncConnectionPool

from tests.e2e.conftest import E2E_PREFIX, RED_TSDB_DSN, remove_cagg_refresh_policy
from wp6_data.red.tsdb import ensure_schema_red, fetch_lamp_on_fraction

pytestmark = pytest.mark.e2e

DEVICE, SWITCH, SHARE = f"{E2E_PREFIX}grow-light", "lg_assim_light_on", "lg_assim_light_pct"
T0 = datetime(2026, 1, 1, tzinfo=UTC)
TICK = timedelta(minutes=5)


async def _purge(conn) -> None:
    async with conn.cursor() as cur:
        await cur.execute("DELETE FROM readings WHERE device_name = %s", (DEVICE,))
    await conn.commit()


@pytest.fixture()
async def red_pool(red_tsdb_conn):
    """Keep the schema, delete only this file's rows.

    Dropping and rebuilding the schema per test, as the schema suite does, gives
    the cagg refresh worker a fresh drop to race every test. The policy
    `ensure_schema_red` re-arms is disarmed straight away for the same reason.
    """
    pool = AsyncConnectionPool(RED_TSDB_DSN, min_size=1, max_size=2, open=False)
    await pool.open()
    await ensure_schema_red(pool)
    await remove_cagg_refresh_policy(red_tsdb_conn)
    await _purge(red_tsdb_conn)
    try:
        yield pool
    finally:
        await pool.close()
        await _purge(red_tsdb_conn)


async def _hour(pool, start: datetime, switch: list[float], share: list[float]) -> None:
    """One LetsGrow hour: twelve 5-minute ticks of both tags."""
    rows = []
    for i, (on, pct) in enumerate(zip(switch, share, strict=True)):
        t = start + i * TICK
        rows += [(t, DEVICE, SWITCH, on), (t, DEVICE, SHARE, pct)]
    async with pool.connection() as conn, conn.cursor() as cur:
        await cur.executemany(
            "INSERT INTO readings (time, device_name, sensor_tag, value, source) "
            "VALUES (%s, %s, %s, %s, 'letsgrow')",
            rows,
        )
        await conn.commit()


async def _fraction(pool):
    return await fetch_lamp_on_fraction(DEVICE, SWITCH, SHARE, pool=pool)


async def test_a_whole_hour_at_full_share_is_one(red_pool):
    await _hour(red_pool, T0, [1] * 12, [100] * 12)

    assert (await _fraction(red_pool)).to_dict() == {T0: pytest.approx(1.0)}


async def test_half_the_installation_is_half(red_pool):
    await _hour(red_pool, T0, [1] * 12, [50] * 12)

    assert (await _fraction(red_pool)).iloc[0] == pytest.approx(0.5)


async def test_half_an_hour_on_is_half(red_pool):
    await _hour(red_pool, T0, [1] * 6 + [0] * 6, [100] * 6 + [0] * 6)

    assert (await _fraction(red_pool)).iloc[0] == pytest.approx(0.5)


async def test_switched_off_is_zero_whatever_the_share_says(red_pool):
    await _hour(red_pool, T0, [0] * 12, [100] * 12)

    assert (await _fraction(red_pool)).iloc[0] == 0.0


async def test_an_hour_letsgrow_did_not_report_is_absent_not_off(red_pool):
    await _hour(red_pool, T0, [1] * 12, [100] * 12)
    await _hour(red_pool, T0 + timedelta(hours=3), [1] * 12, [100] * 12)

    fraction = await _fraction(red_pool)

    assert list(fraction.index) == [T0, T0 + timedelta(hours=3)]


async def test_returns_one_row_per_hour_up_to_the_newest(red_pool):
    """Aggregated in the database: a month of ticks comes back as its hours.

    The regression this guards is a row cap, not a size: the raw-row fetch hit
    500k and silently dropped the newest data. One row per hour, with no LIMIT,
    leaves nothing to cut — shown on a month rather than by inserting 500k rows,
    which fills a developer's disk with WAL.
    """
    days = 30
    async with red_pool.connection() as conn, conn.cursor() as cur:
        await cur.execute(
            """
            INSERT INTO readings (time, device_name, sensor_tag, value, source)
            SELECT t, %(device)s, tag, CASE WHEN tag = %(switch)s THEN 1 ELSE 100 END,
                   'letsgrow'
            FROM generate_series(%(start)s::timestamptz,
                                 %(start)s::timestamptz + %(span)s - interval '5 minutes',
                                 interval '5 minutes') AS t,
                 unnest(ARRAY[%(switch)s, %(share)s]) AS tag
            """,
            {"device": DEVICE, "switch": SWITCH, "share": SHARE,
             "start": T0, "span": timedelta(days=days)},
        )
        await conn.commit()

    fraction = await _fraction(red_pool)

    assert len(fraction) == days * 24
    assert fraction.index.max() == T0 + timedelta(days=days) - timedelta(hours=1)
    assert (fraction == 1.0).all()
