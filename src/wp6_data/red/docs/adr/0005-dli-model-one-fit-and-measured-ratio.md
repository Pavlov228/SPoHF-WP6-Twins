# DLI model: one weather fit, a measured canopy/roof ratio, lamps from LetsGrow

## Status

accepted (model v11, 2026-11-24). Supersedes the v7–v10 two-stage chain
described in [`dli/MODEL_PAPER.md`](../../dli/MODEL_PAPER.md) §3 of those versions.

## Context

v10 predicted canopy DLI in three steps:
1. OpenMeteo → s1000 outdoor lux (Ridge);
2. lux → above-lamp PAR (Ridge);
3. × one fitted attenuation (0.642).

`/dli/performance` (total) overshot in late winter and spring, by up to +5.4
mol/m²/day in March, and undershot in autumn. The user asked whether a bigger
model would help, and whether all the steps were still needed now that the
lamps come from LetsGrow ([ADR 0004](0004-lamp-state-from-letsgrow.md)).

Measured over the 339 days with a canopy sensor, out of fold, lamps as they ran
(MODEL_PAPER §5.7):

- **The swing was the attenuation.** The real canopy/roof ratio runs from
  ~0.4 in February to 0.7–0.8 in autumn, most likely the crop growing over the
  canopy sensor after a rotation, plus screens. A single constant overshoots
  half the year and undershoots the other.
- **The s1000 step added error, not information.** At forecast time it only
  passes a prediction through, and the station has no more history than the PAR
  sensors. Roof MAE was 2.89 fitted directly (out of fold) against 3.21 through
  s1000 (in sample).
- **More capacity did not help.** Ridge on 12 weather features scored 2.84;
  boosted trees scored 3.04, worse than v10's 2.88. With one year of canopy data
  a flexible model cannot learn a month it has held out. Fitting takes under two
  seconds, so hardware was never the limit.

## Decision

- **Link 1, fitted:** OpenMeteo daily shortwave → daily PAR integral at the
  above-lamp sensor. One Ridge fit, shortwave alone (day-of-year scored worse).
  Refitted nightly on prod, as before (root [ADR 0007](../../../../../docs/adr/0007-trained-models-persist-and-are-fingerprinted.md)).
- **Link 2, measured:** × the canopy/roof ratio from
  `red.lamp.attenuation_by_date`, i.e. daily (canopy − LetsGrow lamps) / roof,
  median over 14 days.
  - Days ahead on `/dli/forecast`: the newest ratio.
  - Past days (`/dli/forecast`, `/dli/performance`): the ratio a forecast would
    have had, the 14 days ending 3 days earlier.
  - The model's fitted `attenuation_factor` is kept **only as the fallback**.
- **Lamps:** not modelled. They come from LetsGrow ([ADR 0004](0004-lamp-state-from-letsgrow.md)).
- **`TwoStageLightModel` keeps its name.** The two stages are now these two
  links. `MODEL_VERSION` is 11; the s1000 sensor left the fit fingerprint.
- **Training stays on prod.** Fitting locally and committing the parameters
  (as blue's irrigation model does, root [ADR 0009](../../../../../docs/adr/0009-irrigation-model-single-bucket-shipped-in-code.md)) was considered and declined for
  now. It buys reviewability, not accuracy, and a nightly refit keeps the fit
  current without anyone remembering to.

## Consequences

- **The seasonal swing is mostly gone:**
  - worst monthly bias +5.4 → +1.4 mol/m²/day;
  - daily MAE 2.88 → 2.71;
  - on the live page (1 Oct 2025 – 22 Nov 2026), bias +1.60 → +0.34.
- **The fit is better on its own:** out-of-fold R² 0.950 (v10's chain: 0.926),
  skill vs persistence +0.605 (was +0.517).
- **Predictions now depend on recent sensor data, not only the stored model.**
  The ratio needs both PAR sensors and LetsGrow. Without them it falls back to
  the constant and its seasonal error.
- **A trailing ratio lags a real change by up to a week.**
- **The largest remaining error is OpenMeteo's sun against the greenhouse's own:**
  ~3 mol/m²/day above the lamps. LetsGrow's radiation sensor gets ~2.2, but only
  for past days. That is the planned "measured inputs" view on
  `/dli/performance`.
- s1000 is no longer a DLI input. It stays in `metadata.yaml`, on the model
  diagnostic page's raw counts, and in the climate model, whose link 1 still targets it.

## Revisit when

- **Two or three seasons of canopy data exist.** Re-run the more-parameters
  comparison: a model that can see a second March may finally earn its
  complexity. If training then becomes heavy, fit offline and commit the
  parameters.
- **A screen position becomes available** (from LetsGrow or elsewhere). It may
  explain part of the ratio's movement directly.
- **The canopy sensor is moved or replaced.** The ratio absorbs that within a
  window, but the fitted fallback does not.
