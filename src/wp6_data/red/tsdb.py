"""TimescaleDB schema bootstrap and read helpers for the red twin.

Red's `readings` has a different column shape than blue's: a `source` column
(mutually exclusive data origins — manual / Neurath / legacy — see
``project_blue_project_vs_red_source`` memo, not a rename of blue's `project`)
plus a nullable `upload_id` FK to `manual_uploads` for rows that came from a
manual Excel upload.

The cagg + `sync_metadata` infrastructure is twin-agnostic and lives in
`wp6_data.db.schema`/`queries`; `ensure_schema_red` delegates to
`ensure_aggregates(pool)` after creating red's readings hypertable.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import structlog
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from wp6_data.config import Settings
from wp6_data.db.queries import refresh_sensor_summary
from wp6_data.db.schema import MANUAL_UPLOADS_SQL, ensure_aggregates
from wp6_data.shared.aggregation import CHART_AGG_FUNCS

logger = structlog.get_logger()

# `manual_uploads` is now the shared, twin-agnostic constant; red's readings
# DDL (with its `source` column + `upload_id` FK + source-aware dedup index)
# stays red-owned. Composed so the emitted bootstrap SQL is unchanged.
_READINGS_RED_SQL = """
CREATE TABLE IF NOT EXISTS readings (
    time        TIMESTAMPTZ      NOT NULL,
    device_name TEXT             NOT NULL,
    sensor_tag  TEXT             NOT NULL,
    value       DOUBLE PRECISION,
    raw_value   TEXT,
    source      TEXT             NOT NULL DEFAULT 'unknown',
    synced_at   TIMESTAMPTZ      NOT NULL DEFAULT NOW(),
    upload_id   BIGINT           REFERENCES manual_uploads(id)
);

SELECT create_hypertable('readings', 'time', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_readings_device_tag
    ON readings (device_name, sensor_tag, time DESC);

CREATE UNIQUE INDEX IF NOT EXISTS idx_readings_dedup
    ON readings (source, device_name, sensor_tag, time);
"""

# Prescriptive-risk cache (red ADR 0002, issue 015). A *rebuildable* cache, not
# a system of record: `risk_episodes` is the recomputable log (each row stamped
# with the threshold-set that produced it), `risk_state` is the latest per-section
# verdict the page reads cheaply. Plain tables (not hypertables) — they are
# rewritten wholesale per build, not time-series-ingested.
_RISK_CACHE_SQL = """
CREATE TABLE IF NOT EXISTS risk_episodes (
    id          BIGSERIAL        PRIMARY KEY,
    wire        TEXT             NOT NULL,
    height      INTEGER          NOT NULL,
    label       TEXT             NOT NULL,
    risk        TEXT             NOT NULL,
    start_time  TIMESTAMPTZ      NOT NULL,
    end_time    TIMESTAMPTZ,
    peak        DOUBLE PRECISION NOT NULL,
    thresholds  JSONB            NOT NULL,
    built_at    TIMESTAMPTZ      NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_risk_episodes_wire_time
    ON risk_episodes (wire, start_time);

CREATE TABLE IF NOT EXISTS risk_state (
    wire             TEXT             NOT NULL,
    height           INTEGER          NOT NULL,
    label            TEXT             NOT NULL,
    height_dli       DOUBLE PRECISION,
    vpd_latest       DOUBLE PRECISION,
    vpd_in_band      BOOLEAN,
    wet_hours_latest DOUBLE PRECISION,
    fungal_active    BOOLEAN,
    co2_latest       DOUBLE PRECISION,
    co2_depleted     BOOLEAN,
    canopy_deficit   BOOLEAN,
    built_at         TIMESTAMPTZ      NOT NULL DEFAULT NOW(),
    PRIMARY KEY (wire, height)
);

-- The CO₂ columns were added after risk_state shipped; on an existing prod
-- table the CREATE above is a no-op, so add them idempotently. risk_state is a
-- rebuildable cache, so a one-time NULL default until the next build is fine.
ALTER TABLE risk_state ADD COLUMN IF NOT EXISTS co2_latest   DOUBLE PRECISION;
ALTER TABLE risk_state ADD COLUMN IF NOT EXISTS co2_depleted BOOLEAN;
"""

# Shared audit table + red readings DDL + risk cache. Concatenation preserves the
# exact statement set red bootstrapped before promotion.
SCHEMA_SQL = MANUAL_UPLOADS_SQL + _READINGS_RED_SQL + _RISK_CACHE_SQL


async def ensure_schema_red(pool: AsyncConnectionPool) -> None:
    """Bootstrap red's TSDB schema and aggregates idempotently.

    Order matters: readings hypertable first, then the shared aggregates helper
    which creates `sync_metadata` and the `sensors_daily_summary` continuous
    aggregate (with a background refresh policy installed by `ensure_aggregates`).

    The cagg is created WITH NO DATA and its background policy only covers the
    last 7 days, so on first run against a DB that already has readings (e.g.
    historical Sijia uploads) it holds nothing for those older days. Coverage is
    read from the cagg, so materialise it whole-history once when it is empty but
    readings exist. Steady-state refresh is out-of-band (the 7-day background
    policy + a whole-history refresh after WP6_SYNC_MODE=full).
    """
    async with pool.connection() as conn:
        await conn.execute(SCHEMA_SQL)
        await conn.commit()

    await ensure_aggregates(pool)

    async with pool.connection() as conn, conn.cursor() as cur:
        await cur.execute("SELECT count(*) FROM sensors_daily_summary")
        cagg_row = await cur.fetchone()
        cagg_rows = cagg_row[0] if cagg_row else 0
        await cur.execute("SELECT count(*) FROM readings")
        readings_row = await cur.fetchone()
        readings_rows = readings_row[0] if readings_row else 0
    if cagg_rows == 0 and readings_rows > 0:
        await refresh_sensor_summary(pool)
        logger.info("cagg_backfilled_whole_history")

    logger.info("red_tsdb_schema_ensured")


_EMPTY_READINGS = pd.DataFrame(columns=["device", "sensor", "time", "value"])


async def fetch_data_tsdb(
    sensor_tags: list[str] | None = None,
    device_names: list[str] | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    limit: int = 500_000,
    *,
    bucket: timedelta | None = None,
    agg: str | None = None,
) -> pd.DataFrame:
    """Fetch readings from the red TSDB `readings` table.

    Returns a DataFrame with columns ``device, sensor, time, value`` (plus
    ``count`` when ``bucket`` + ``agg`` push aggregation into a ``time_bucket``
    GROUP BY; see :func:`wp6_data.shared.aggregation.bucket_and_aggregate`).
    """
    from wp6_data.db.pool import get_pool

    if not sensor_tags:
        return _EMPTY_READINGS.copy()

    conditions = ["sensor_tag = ANY(%(tags)s)"]
    params: dict[str, Any] = {"tags": sensor_tags}
    if device_names:
        conditions.append("device_name = ANY(%(devices)s)")
        params["devices"] = device_names
    if start:
        conditions.append("time >= %(start)s")
        params["start"] = start
    if end:
        conditions.append("time <= %(end)s")
        params["end"] = end
    params["limit"] = limit

    where = " AND ".join(conditions)
    columns = ["device", "sensor", "time", "value"]

    if bucket is not None and agg is not None:
        if agg not in CHART_AGG_FUNCS:
            raise ValueError(f"Unknown aggregation {agg!r}")
        params["bucket"] = bucket
        params["tz"] = Settings().display_timezone
        # GROUP BY / ORDER BY the real columns + the time_bucket expression,
        # NOT the `AS time` alias: it collides with readings.time, and
        # Postgres resolves that ambiguity differently in GROUP BY (input
        # column) vs ORDER BY (output alias), which silently defeats the
        # bucketing and disorders ties.
        query = f"""
            SELECT device_name AS device, sensor_tag AS sensor,
                   time_bucket(%(bucket)s, time, %(tz)s) AS time,
                   {agg}(value) AS value,
                   min(value) AS value_min,
                   max(value) AS value_max,
                   count(value) AS count
            FROM readings
            WHERE {where}
            GROUP BY device_name, sensor_tag, time_bucket(%(bucket)s, time, %(tz)s)
            ORDER BY time_bucket(%(bucket)s, time, %(tz)s)
            LIMIT %(limit)s
        """
        columns = ["device", "sensor", "time", "value", "value_min", "value_max", "count"]
    else:
        query = f"""
            SELECT device_name AS device, sensor_tag AS sensor, time, value
            FROM readings
            WHERE {where}
            ORDER BY time
            LIMIT %(limit)s
        """

    pool = get_pool()
    async with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        await cur.execute(query, params)
        records = await cur.fetchall()

    if not records:
        return pd.DataFrame(columns=columns)
    df = pd.DataFrame(records)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    for col in ("value", "value_min", "value_max"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values("time")


async def fetch_sensors_from_cagg(
    source: str | None = None,
) -> list[dict[str, Any]]:
    """Per-sensor reading-count + first/last seen from the cagg.

    Mirrors `blue.deps._fetch_sensors_from_cagg`. The cagg aliases red's
    `source` column to ``project`` in its output (uniform shape across twins);
    we filter on it when `source` is given.
    """
    from wp6_data.db.pool import get_pool

    if source is not None:
        where = "WHERE source = %(source)s"
        params: dict[str, Any] = {"source": source}
    else:
        where = ""
        params = {}

    query = f"""
        SELECT device_name AS device, sensor_tag AS sensor,
               sum(reading_count)  AS readings,
               min(first_reading)  AS earliest,
               max(last_reading)   AS latest
        FROM sensors_daily_summary
        {where}
        GROUP BY device_name, sensor_tag
        ORDER BY readings DESC
    """

    pool = get_pool()
    async with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        await cur.execute(query, params)
        return list(await cur.fetchall())


async def fetch_manual_summary_tsdb() -> dict[str, Any]:
    """Aggregate metadata for manually-uploaded measurements.

    Returns ``{"uploads": {source: last_uploaded_at},
              "measurements": {sensor_tag: last_measure_time}}``.
    Drives the Manual tab's *Last upload* (per-source) and
    *Last measure* (per measurement type) columns. Delegates to the shared,
    twin-agnostic query (promoted out of red so blue shares one provenance
    model); behaviour is unchanged for red.
    """
    from wp6_data.db.pool import get_pool
    from wp6_data.db.queries import fetch_manual_summary

    return await fetch_manual_summary(get_pool())


async def fetch_daily_coverage_from_cagg() -> list[dict[str, Any]]:
    """Distinct (device, sensor, day, source) rows from `sensors_daily_summary`.

    Red's TSDB-side coverage (manual Sijia uploads + any risk/TSDB readings) is
    derived from the cagg over `readings`, which refreshes as readings change —
    so a deleted day stops reporting, unlike the retired `daily_coverage` table.
    The `source` column is carried through so the provider can tag manual-upload
    rows (see `red.manual_sources`); this function stays source-agnostic. The
    live MySQL/wire legs are merged separately by the provider.
    """
    from wp6_data.db.pool import get_pool

    pool = get_pool()
    async with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        await cur.execute(
            "SELECT DISTINCT device_name AS device, sensor_tag AS sensor, "
            "bucket::date AS day, source "
            "FROM sensors_daily_summary "
            "ORDER BY device, sensor, day"
        )
        return list(await cur.fetchall())


async def fetch_sync_metrics_tsdb() -> list[dict[str, Any]]:
    """Return red's enriched job-run audit rows from `sync_metadata`.

    Shape matches `blue.deps.fetch_sync_metrics` so `shared.routes.status`
    renders them uniformly. Most of red's rows (``sijia``, ``red-export``) are
    event/job-driven and get no freshness budget; the LetsGrow sync is a
    continuous feed, judged against its once-a-day delivery rather than
    yookr's 15-minute one.
    """
    from wp6_data.db.pool import get_pool
    from wp6_data.db.queries import fetch_sync_metrics_rows
    from wp6_data.red import letsgrow
    from wp6_data.red.manual_sources import MANUAL_SOURCE_VALUES

    rows = await fetch_sync_metrics_rows(get_pool())
    for r in rows:
        r["source_type"] = "manual" if r["endpoint"] in MANUAL_SOURCE_VALUES else "synced"
        r["freshness_budget"] = (
            letsgrow.FRESHNESS_BUDGET_HOURS if r["endpoint"] == letsgrow.ENDPOINT else None
        )
    return rows


async def fetch_lamp_on_fraction(
    device: str,
    switch: str,
    share: str,
    start: datetime | None = None,
    end: datetime | None = None,
    *,
    pool: AsyncConnectionPool | None = None,
) -> pd.Series:
    """Share of each UTC hour the lamps ran, from LetsGrow's switch and share tags.

    Per LetsGrow tick: ``share / 100`` while the switch is on, else 0; then the
    hour's mean. 1.0 is a whole hour at full output, 0.5 half an hour or half the
    installation. Hours LetsGrow did not report are absent, never zero.

    Aggregated here rather than in pandas because the raw rows outgrow any
    reasonable cap: two tags every five minutes since February 2023 was ~790k
    rows, and ``fetch_data_tsdb``'s oldest-first 500k cap silently returned
    only up to mid-2025 — so /dli/lamps fell back to the PAR sensors for every
    hour it actually drew. One row per hour cannot be truncated that way.
    """
    from wp6_data.db.pool import get_pool

    conditions = ["s.device_name = %(device)s", "s.sensor_tag = %(switch)s"]
    params: dict[str, Any] = {"device": device, "switch": switch, "share": share}
    if start is not None:
        conditions.append("s.time >= %(start)s")
        params["start"] = start
    if end is not None:
        conditions.append("s.time <= %(end)s")
        params["end"] = end

    query = f"""
        SELECT time_bucket('1 hour', s.time) AS hour,
               avg(CASE WHEN s.value >= 0.5
                        THEN LEAST(GREATEST(p.value, 0), 100) / 100.0
                        ELSE 0 END) AS on_fraction
        FROM readings s
        JOIN readings p
          ON p.device_name = s.device_name
         AND p.time = s.time
         AND p.sensor_tag = %(share)s
        WHERE {" AND ".join(conditions)}
          AND s.value IS NOT NULL AND p.value IS NOT NULL
        GROUP BY 1
        ORDER BY 1
    """
    async with (pool or get_pool()).connection() as conn, conn.cursor() as cur:
        await cur.execute(query, params)
        rows = await cur.fetchall()

    if not rows:
        return pd.Series(dtype=float)
    index = pd.to_datetime([r[0] for r in rows], utc=True)
    return pd.Series([float(r[1]) for r in rows], index=index, dtype=float)


async def fetch_hourly_stats(
    device: str,
    tag: str,
    start: datetime | None = None,
    end: datetime | None = None,
    *,
    pool: AsyncConnectionPool | None = None,
) -> pd.DataFrame:
    """One tag's ``mean``, ``min`` and ``max`` per UTC hour, aggregated in SQL.

    In SQL for the reason :func:`fetch_lamp_on_fraction` gives: years of
    five-minute rows outgrow ``fetch_data_tsdb``'s cap, one row per hour does
    not. Min and max ride along so a caller can tell a sensor that froze — the
    same value all hour — from a quiet one. Hours with no reading are absent.
    """
    from wp6_data.db.pool import get_pool

    conditions = ["device_name = %(device)s", "sensor_tag = %(tag)s", "value IS NOT NULL"]
    params: dict[str, Any] = {"device": device, "tag": tag}
    if start is not None:
        conditions.append("time >= %(start)s")
        params["start"] = start
    if end is not None:
        conditions.append("time <= %(end)s")
        params["end"] = end

    query = f"""
        SELECT time_bucket('1 hour', time) AS hour, avg(value), min(value), max(value)
        FROM readings
        WHERE {" AND ".join(conditions)}
        GROUP BY 1
        ORDER BY 1
    """
    async with (pool or get_pool()).connection() as conn, conn.cursor() as cur:
        await cur.execute(query, params)
        rows = await cur.fetchall()

    index = pd.to_datetime([r[0] for r in rows], utc=True)
    return pd.DataFrame(
        [[float(v) for v in r[1:]] for r in rows],
        index=index, columns=["mean", "min", "max"], dtype=float,
    )


async def fetch_heat_output(
    device: str,
    air: str,
    pipes: list[str],
    exponent: float,
    start: datetime | None = None,
    end: datetime | None = None,
    *,
    pool: AsyncConnectionPool | None = None,
) -> pd.Series:
    """Heat-output index per UTC hour: Σ circuits mean(max(pipe − air, 0) ** exponent).

    Each pipe reading is paired with the air reading of the same LetsGrow tick,
    so the difference is never taken across time. An hour counts only when
    every circuit reported in it; otherwise it would sum short and read as a
    mild hour. See ``red.heating`` for what the index means. In SQL for the
    row-count reason :func:`fetch_lamp_on_fraction` gives.
    """
    from wp6_data.db.pool import get_pool

    conditions = ["a.device_name = %(device)s", "a.sensor_tag = %(air)s"]
    params: dict[str, Any] = {
        "device": device, "air": air, "pipes": list(pipes),
        "exponent": exponent, "circuits": len(pipes),
    }
    if start is not None:
        conditions.append("a.time >= %(start)s")
        params["start"] = start
    if end is not None:
        conditions.append("a.time <= %(end)s")
        params["end"] = end

    query = f"""
        WITH per_circuit AS (
            SELECT time_bucket('1 hour', a.time) AS hour, p.sensor_tag,
                   avg(power(GREATEST(p.value - a.value, 0), %(exponent)s)) AS output
            FROM readings a
            JOIN readings p
              ON p.device_name = a.device_name
             AND p.time = a.time
             AND p.sensor_tag = ANY(%(pipes)s)
            WHERE {" AND ".join(conditions)}
              AND a.value IS NOT NULL AND p.value IS NOT NULL
            GROUP BY 1, 2
        )
        SELECT hour, sum(output)
        FROM per_circuit
        GROUP BY hour
        HAVING count(*) = %(circuits)s
        ORDER BY hour
    """
    async with (pool or get_pool()).connection() as conn, conn.cursor() as cur:
        await cur.execute(query, params)
        rows = await cur.fetchall()

    if not rows:
        return pd.Series(dtype=float)
    index = pd.to_datetime([r[0] for r in rows], utc=True)
    return pd.Series([float(r[1]) for r in rows], index=index, dtype=float)
