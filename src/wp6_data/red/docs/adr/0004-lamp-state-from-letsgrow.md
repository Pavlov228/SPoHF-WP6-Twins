# Lamp state comes from LetsGrow; the PAR sensors are the fallback

## Status

accepted

## Context

Red's canopy light is part sun, part lamps. Every DLI page has to know which
hours the lamps ran. Until November 2026 that was **inferred from the two PAR
sensors**:
- **at night:** the canopy reads light while the roof sensor reads none;
- **in daylight:** the canopy outshines the roof.

That detector cannot see a lamp under a bright sun: 140–190 µmol/m²/s of lamp
light is hidden next to 600+ of daylight. Its schedule was also read once, from
the fortnight *before* a window, and applied to all of it.

Measured over Oct 2025 – Nov 2026:

- **`/dli/performance` (total) was ~3 mol/m²/day low** through Oct–Nov 2026. The
  schedule read before the window gave 3.0 mol/day of lamp light; the lamps
  actually gave 6.3 as the lit period grew.
- **LetsGrow reports the lamps directly.** It is the climate computer that
  switches them: `lg_assim_light_on` (0/1) and `lg_assim_light_pct` (share, %)
  on `letsgrow-grow-light-b4`, every 5 minutes, back to 2023-02. It agrees with
  the PAR detector on 93% of hours. Most disagreements are hours LetsGrow says
  were lit and the detector missed (463 vs 86): bright middays.
- **The grower switches the lamps off in bright sun.** Inside the lit period,
  hours with lamps off rise from 0% (roof under 200 µmol/m²/s) to 20% (above
  600). The schedule is not a fixed block, so a fixed hour set is wrong in both
  directions.
- **LetsGrow arrives about two days late**, and had one outage with no data at
  all: 26 Oct – 2 Nov 2026.

## Decision

**Where LetsGrow has reported, its switch decides whether the lamps ran.** The
PAR sensors still decide whether it was daylight, and they measure the lamp
**level** on dark hours when LetsGrow says the lamps were fully on.

- The hourly on-fraction (share/100 while switched on, else 0, averaged per
  hour) is computed **in SQL** (`red.tsdb.fetch_lamp_on_fraction`). Pulling raw
  rows hit `fetch_data_tsdb`'s 500k oldest-first cap and silently dropped
  everything after mid-2025.
- **Past hours:** lamp light = level × on-fraction (`observed_lamp_par`). Used by
  `/dli/performance`, `/dli/lamps`, past days on `/dli/forecast`, and the
  climate model's training.
- **Hours ahead:** the per-hour median of LetsGrow's last 3 complete days
  (MODEL_PAPER §5.7).
- **Where LetsGrow has nothing** (its lag, an outage, unreachable), the PAR
  detector stands in, and pages say so. On `/dli/lamps` a sunlit hour it cannot
  judge is drawn as **"Daylight, lamps unknown"**, not as lamps off.
- The device and tags are declared in `metadata.yaml` (`lamps:`), not in code.

## Consequences

- `/dli/performance` (total) scores the natural-light model, not a guessed
  schedule. The Oct–Nov 2026 −3 mol/day bias is gone.
- `/dli/lamps` shows the real schedule, including bright-midday switch-offs.
- A LetsGrow outage is visible: grey "unknown" daylight rather than a week of
  lamps apparently switched off.
- Every lamp-aware page now depends on two sources. A LetsGrow outage degrades
  them to the detector; it does not break them.
- The lamp level is measured, not configured. It follows a moved sensor:
  188 → 140 µmol/m²/s at the July 2026 rotation, with the lamps at 100% both
  times.

## Revisit when

- **LetsGrow stops being the switch** (another climate computer, or a second
  lamp group). `lg_led1_on` has never been on without the assimilation lighting;
  if that changes, it needs its own level.
- **The two-day lag shrinks or grows.** `FORECAST_RATIO_LAG_DAYS` and the
  pages' fallback notes assume about two days.
