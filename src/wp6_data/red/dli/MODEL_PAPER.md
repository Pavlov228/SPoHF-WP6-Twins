# Light Prediction Model for Greenhouse DLI Forecasting

## Abstract

This paper describes the model (v11) that predicts the Daily Light Integral (DLI) at the canopy of red's tomato greenhouse from freely available weather forecasts. It has two links. A Ridge regression maps OpenMeteo's daily shortwave radiation to the PAR integral at the sensor **above** the lamps, which sees natural light only. A **measured, rolling** canopy/roof ratio then carries that down to the canopy. The lamps are not modelled: LetsGrow, the climate computer that switches them, reports when they ran, and the forecast assumes their last few days' pattern continues.

The paper also records how the model got here. Training on the plant-level sensor directly failed because it mixes lamp and natural light (§5.1–5.2). Polynomial features overfitted (§5.3). Daily totals were once sums of rows rather than integrals (§5.5). Calendar features crowded out the weather (§5.6). And v11 removed the outdoor-lux stage and replaced the constant attenuation with a measured ratio. Doing so removed a seasonal swing of up to +5 mol/m²/day (§5.7).

## 1. Introduction

### 1.1 Problem Statement

Predicting indoor light levels in greenhouses is critical for:
- Supplemental lighting scheduling to meet plant DLI targets
- Energy cost optimization
- Crop growth modeling and yield prediction

The challenge: We want to predict indoor Photosynthetically Active Radiation (PAR) using only freely available weather forecasts (OpenMeteo API), without requiring expensive on-site weather stations for prediction (only for training).

### 1.2 Data Sources

| Source | Variables | Resolution | Role |
|--------|-----------|------------|------|
| OpenMeteo API | shortwave_radiation (+ hourly shape for the intraday curve) | Hourly | Predictor (available for forecasts) |
| s2100-01-par | PAR (μmol/m²/s), above lamps | ~10 min | Training target (natural light) |
| s2100-02-par | PAR (μmol/m²/s), under lamps | ~10 min | Canopy/roof ratio, lamp level, and what pages score against |
| LetsGrow (`lg_assim_light_on`, `lg_assim_light_pct`) | Lamp switch and share | 5 min, ~2 days late | When the lamps ran |
| s1000 weather station | lux | ~10 min | **No longer used** (v4–v10 intermediate target; §5.7) |

## 2. Sensor Configuration & Investigation

### 2.1 Physical Setup

| Sensor | Position | What It Sees |
|--------|----------|-------------|
| `s2100-01-par` | **Above** lamps | Natural light only (no lamp contamination) |
| `s2100-02-par` | **Under** lamps | Natural light + lamp light (what plants receive) |

### 2.2 The Overestimation Problem

DLI schedule predictions consistently **overestimated** compared to actual plant-level readings:
- Predicted values formed a smooth curve
- Actual values showed "drops" and lower overall readings
- Initial hypothesis: sensor mismatch between training and evaluation

### 2.3 Key Observation: Ratio > 1

During daylight hours (08:00-16:00), `s2100-02-par` receives **more** light than `s2100-01-par`:

| Metric | Value |
|--------|-------|
| Mean ratio (s2100-02 / s2100-01) | ~2.2x |
| Correlation | High (sensors track together) |

During daytime with lamps ON:
- `s2100-01-par` (above lamps): natural light only
- `s2100-02-par` (under lamps): natural light + lamp light

### 2.4 Ratio Varies by Conditions

| Condition | Ratio | Interpretation |
|-----------|-------|----------------|
| Sunny (PAR >= 200) | ~0.8x | Natural light dominates, structural attenuation visible |
| Cloudy (PAR < 200) | ~2.95x | Lamp light dominates for s2100-02 |

This condition-dependent ratio made direct training on s2100-02 problematic — the model cannot separate natural from artificial light.

## 3. Model Architecture

### 3.1 Two links, and the lamps beside them

```
┌────────────┐   Ridge (daily)   ┌──────────────┐  × canopy/roof ratio  ┌──────────────┐
│ OpenMeteo  │ ────────────────► │ s2100-01-par │ ────────────────────► │  natural at  │
│ shortwave  │                   │ (above lamps)│  measured, rolling    │  the canopy  │
└────────────┘                   └──────────────┘                       └──────┬───────┘
                                                                               │ +
                                          LetsGrow: lamps as they ran (past) ──┤
                                          or their last 3 days' pattern (ahead)│
                                                                               ▼
                                                                     total at the canopy
```

The split follows what drives each part:

- **Weather → roof** is physics with noise, so it is fitted. The above-lamp
  sensor is the clean target: no lamp light, no crop shading it (§5.4).
- **Roof → canopy** is a ratio that moves with the season and the crop (§3.3),
  so it is measured rather than fitted once.
- **The lamps** are an operator's decision, which LetsGrow records. Recent
  behaviour predicts them better than any weather regression could.

### 3.2 Link 1: weather → light above the lamps

**Feature:** `shortwave_sum`, the daily sum of OpenMeteo's global horizontal
irradiance, and nothing else (§5.6, §5.7).

**Target:** the daily PAR **integral** at `s2100-01-par`, in μmol/m² (§5.5).

**Fit:** `RidgeCV` on standardised features, scored out of fold on day-blocked
folds (`red/fitting.py`). Refitted nightly on prod and at boot when the stored
artifact's version or fingerprint differs (root ADR 0007). Why the model is
shaped this way: red [ADR 0005](../docs/adr/0005-dli-model-one-fit-and-measured-ratio.md).

### 3.3 Link 2: the canopy/roof ratio

The share of natural light that reaches the canopy sensor. Per day, for the
forecast and `/dli/performance`:

```
ratio = (canopy − lamp) / predicted roof        (daily DLI)
```

The denominator is what link 1 *predicted* for the roof from OpenMeteo, not
what the roof sensor measured (since 2026-11-24, §5.9). The ratio then corrects
link 1's own seasonal miss along with the crop's. Against the measured roof
(`predicted_roof` not passed) it is the physical canopy/roof share.

`lamp` is what `red.lamp.observed_lamp_par` says ran: LetsGrow's switch × the
lamp level measured on dark hours. The ratio is the **median over 14 days**
(`red.lamp.attenuation_by_date`). Days outside (0, 1] are dropped, because there
the lamp estimate was wrong; against the predicted roof the bound is (0, 1.5],
since a day OpenMeteo thought dull can legitimately exceed 1. A window needs at
least 5 days before it says anything.

Which window depends on what is being predicted:

| Use | Window | Why |
|---|---|---|
| `/dli/forecast`, days ahead | the newest 14 days | the latest thing known |
| `/dli/forecast` and `/dli/performance`, past days | 14 days ending **3 days before** each day | what a forecast would have known: nothing from today, and LetsGrow lands ~2 days late |
| a day with no ratio (sensor outage) | the model's fitted `attenuation_factor` | `compute_attenuation` over the whole training range |

Over Oct 2025 – Nov 2026 the ratio ran from **~0.4 in February to 0.7–0.8 in
autumn**. The likely cause is the crop growing over the canopy sensor after each
rotation, plus screens. LetsGrow reports no screen position, so screens cannot
be checked directly.

### 3.4 The lamps

- **Days that have happened:** as they ran. LetsGrow's hourly on-fraction
  (share/100 while switched on, averaged in SQL by
  `red.tsdb.fetch_lamp_on_fraction`) × the level from `lamp_level_by_date`.
  Hours after LetsGrow's last delivery fall back to the PAR-sensor detector.
- **Hours ahead:** the per-hour **median of LetsGrow's last 3 complete days**,
  at the latest level (§5.7). A short median keeps the switch-on edge sharp, and
  it follows a schedule that moves week by week.

## 4. Design Choices

### 4.1 Daily vs. Hourly Aggregation

We train on **daily aggregates** rather than hourly data.

| Aggregation | Correlation (r) | Samples |
|-------------|-----------------|---------|
| Hourly | 0.685 | 579 |
| Daily | 0.906 | 88 |

**Justification:**
- Hourly data has high noise from transient clouds, sensor lag, timestamp misalignment
- Daily totals smooth out noise while preserving the signal
- DLI is inherently a daily metric — we care about total light per day
- Fewer samples but much higher signal-to-noise ratio

### 4.2 Cyclical Day-of-Year Encoding (v5–v10, not in v11)

Day-of-year was encoded as a sine/cosine pair, so that day 1 and day 365 are
adjacent:

```python
day_of_year_sin = sin(2π × day / 365)
day_of_year_cos = cos(2π × day / 365)
```

It was dropped from stage 1 in v9 (§5.6). In v11 it was tried on the single fit
and scored worse (§5.7). The season's effect on the light reaching the canopy
is now the measured ratio's job, not a calendar term's.

### 4.3 Direct + Diffuse Radiation (superseded)

Beam and diffuse radiation transmit through glass differently, so earlier
versions fed them separately. Measured, a single global-horizontal sum did as
well or better at every stage (§5.6). v11 uses **shortwave alone**.

### 4.4 Ridge Regression with Cross-Validation

We use `RidgeCV` rather than ordinary least squares (OLS) or other models.

**Why Ridge over OLS:**
- Features are correlated (direct/diffuse radiation, sin/cos pairs)
- Ridge regularization prevents coefficient explosion from multicollinearity
- Small dataset benefits from regularization to prevent overfitting

**Why Ridge over Lasso:**
- We don't need feature selection (all features are meaningful)
- Ridge keeps all features with shrunk coefficients
- More stable predictions

**Why Linear over Complex Models:**
- Limited training data is insufficient for deep learning or ensemble methods
- Linear relationships are physically plausible (light transmission is roughly linear)
- Interpretable coefficients aid debugging
- Polynomial features were tested but caused overfitting (see Section 5.3)

**Cross-Validation:**
- `RidgeCV` selects the regularization strength over a log-spaced alpha grid
- Folds are **day-blocked**, never random: consecutive days share a weather
  system, so a random split trains on the test day's neighbours and flatters
  every score
- The same folds give the out-of-fold scores in §6

### 4.5 Feature Scaling

All features are standardized (zero mean, unit variance) before Ridge regression.

**Justification:**
- Ridge penalty is scale-dependent; unscaled features would bias regularization
- Features have vastly different scales (radiation in thousands, cloud cover 0-100)
- Standardization ensures equal regularization pressure across features

## 5. Experiments & Lessons Learned

### 5.1 Training on s2100-02-par Directly (Failed)

**Approach:** Train Stage 2 on raw s2100-02-par (plant-level sensor).

**Result:** Poor correlation (r ≈ 0.523). The model could not separate natural light from lamp light in the training signal, leading to systematic overestimation during prediction (when lamps aren't modeled).

**Lesson:** When the training target contains mixed signals, the model learns the wrong relationship.

### 5.2 Lamp-Corrected s2100-02-par (Partial Success)

**Approach:** Derive lamp power from pre-sunrise/post-sunset readings of s2100-02, subtract from all readings, then train on the corrected signal.

**Algorithm:**
1. Use s2100-01-par to define daylight hours (clean reference)
2. Measure lamp PAR during lamp-only hours (before sunrise / after sunset)
3. Subtract lamp power from s2100-02-par during lamp-on hours
4. Aggregate corrected values to daily sums for training

**Result:** Improved correlation compared to raw, but still noisier than s2100-01. Plant-level sensor experiences variable occlusion from canopy growth, moving equipment, and other greenhouse operations that introduce irreducible noise.

**Lesson:** Lamp correction improves the signal but cannot remove physical occlusion noise.

### 5.3 Polynomial Features (Failed)

**Approach (v6):** Add degree-2 polynomial interaction features to capture non-linear relationships.

**Result:** R² jumped to 0.978 on training data — classic overfitting. Predictions degraded significantly on held-out data. With only ~88 training samples, polynomial features create far too many parameters relative to observations.

**Lesson:** At small sample sizes, model complexity is the enemy. Linear models with regularization outperform complex models that memorize training noise.

### 5.4 Retargeting to s2100-01-par (v7, still the target in v11)

**Approach:** Train Stage 2 on s2100-01-par (above-lamp, natural light only), then apply a constant attenuation factor to reach plant-level estimates.

**Why this works:**
- s2100-01-par provides a clean signal — no lamp contamination, no canopy occlusion
- The above-lamp to plant-level ratio is stable once low-ratio (occluded) days are filtered
- Splitting transmission and attenuation into separate steps matches physical reality
- The attenuation factor can be independently validated against the ratio time series

**Result:** The above-lamp sensor has been the training target ever since. The constant attenuation factor was replaced in v11 by a measured, rolling ratio (§5.7).

### 5.5 Summing readings instead of integrating time (bug, fixed — v10)

**Approach (v4-v9):** a daily total was `groupby("date").sum()` over the raw
sensor rows — `lux_sum` for stage 1's target, `par_sum` for stage 2's.

**Result:** the target measured how often the sensor reported, not how much light
there was. Red's s1000 fell from ~270 readings a day to **97 in June 2026** while
its mean lux was the highest of the year; its summed daily lux collapsed to a
third on the brightest days of the year. Stage 1 was being asked to predict a
sampling schedule from the weather, which is why it had no out-of-fold skill at
all (R² −0.04, worse than its own mean).

It also flattered stage 2. Both its input and its target were reading-sums from
sensors behind the same gateway, so the cadence artefact partly cancelled
between them: stage 2's honest out-of-fold R² is 0.978, but it reached that only
once *both* sides were integrals. Fixing one side alone dropped it to 0.852.

**Fix:** `calculator.integrate_over_time` — a trapezoidal integral over the
timestamps, in one place, used by `calculate_daily_dli` and by both daily
aggregates. Stage 1's target is now lux-hours and stage 2's is a PAR integral in
μmol/m², which also removes the last reading-cadence assumption from
`predict_dli`: a DLI is that integral divided by 10⁶, with no `readings_per_day`
in it.

**Lesson:** a daily total of a sampled quantity is an integral. A sum of rows is
a measurement of the sampler.

### 5.6 Seasonal features in stage 1 (rejected — measured, not argued)

Stage 1 carried `day_of_year_sin/cos` from v5. Holding the weather constant and
moving only the date, the fitted chain predicted **×2.15** more light in June
than December — while a **16×** change in actual beam radiation moved it only
×1.74. The calendar outweighed the weather, so within any month the model could
barely tell a brilliant day from a dull one, and it regressed to the seasonal
mean. That is what clipped the summer peaks.

The cause is collinearity, not a bad approximation: radiation and day-of-year say
the same thing over a year, ridge spreads weight across both, and the radiation
coefficient is left too small. Every candidate was scored end to end, out of
fold, across 4-10 fold counts (chain skill against persistence, mean):

| stage 1 features | mean | min |
|---|---|---|
| shortwave alone | **+0.239** | +0.213 |
| direct + diffuse + cloud | +0.232 | +0.198 |
| shortwave + cloud | +0.227 | +0.209 |
| clear-sky index + TOA envelope + beam + cloud | +0.185 | +0.152 |
| clear-sky index + TOA envelope + cloud | +0.124 | +0.106 |
| direct + diffuse + cloud + day-of-year (v5-v9) | +0.069 | −0.018 |

An **exact** top-of-atmosphere envelope from OpenMeteo's `terrestrial_radiation`
scored *worse* than no seasonal feature at all, so this is not about the harmonic
being crude — it is that stage 1 must carry no calendar term of any kind. The
radiation already is the season.

Beam and diffuse were split in §4.3 because they transmit through glass
differently. That is stage 2's concern: stage 1 maps a modelled sky onto a lux
sensor standing in the open, where no glass is involved, and splitting them there
only costs it. Stage 1 is now **global horizontal irradiance alone**.

Stage 2 keeps its day-of-year term — the one place a seasonal feature earns its
place, standing in for the angle sunlight strikes the glass at. Dropping it, and
swapping it for a clear-sky index, each made the *chain* worse even where they
improved the stage in isolation.

### 5.7 One fit instead of the chain, and a moving ratio (v11 — measured)

Measured over the 339 days with a canopy sensor (Oct 2025 – Nov 2026), out of
fold, with the lamps as LetsGrow ran them. Daily total-DLI MAE:

| model | MAE (mol/m²/day) | worst monthly bias |
|---|---|---|
| v10: OpenMeteo → s1000 → above-lamp, × 0.642 | 2.88 (in-sample) | +5.4 (Mar) |
| one fit OpenMeteo → above-lamp, × 0.642 | 2.78 | +5.4 (Mar) |
| one fit, × trailing 14-day ratio, 3 days old (forecast) | **2.71** | **+1.4** (Feb) |
| one fit, × centred 14-day ratio (past days) | 2.16 | +1.2 |
| Ridge on 12 weather features / boosted trees | 2.84 / 3.04 | |
| measured roof × centred ratio (floor) | 1.35 | |

- **The s1000 stage only passed a prediction through.** No future day has a
  recorded lux, and the station has no more history than the PAR sensors.
  Chained fits compound their errors: roof MAE 2.89 direct (out of fold) against
  3.21 through s1000 (in sample).
- **Shortwave alone, again.** Adding day-of-year to the single fit scored worse
  (roof MAE 3.13 vs 2.89), as §5.6 found for stage 1.
- **The swing was the ratio.** Canopy/roof runs from ~0.4 in February to
  0.7–0.8 in autumn — the crop growing over the canopy sensor after a
  rotation, and probably screens. One constant overshoots half the year and
  undershoots the other. `red.lamp.attenuation_by_date` measures it per day
  as (canopy − LetsGrow lamps) / roof, median over 14 days; forecasts use the
  newest, `/dli/performance` the one a forecast would have had (3 days old).
- **More parameters did not help.** With one year of canopy data a flexible
  model cannot learn a month it has not seen; the fit takes under two seconds,
  so hardware was never the limit. Revisit with 2–3 seasons.
- **Lamp pattern for the forecast:** per-hour median of LetsGrow's last 3
  complete days (lamp-hours/day MAE 1.40), not the 14-day mean share (1.91),
  which smeared the moving switch-on time into a ramp. The last day alone
  scored 1.29, but one odd day (a fault, a manual override) would then set
  tomorrow. Past hours on the forecast page use the lamps as they ran; one
  pattern for the whole range had drawn November's lamps onto a summer week.
- **On the live page:** `/dli/performance`, total, 1 Oct 2025 – 22 Nov 2026:
  bias +1.60 → **+0.34** mol/m²/day, mean absolute error 2.79 → 2.69, mean
  absolute % error 43.3% → 39.8%.

### 5.8 Where the forecast's error comes from (measured, not shipped)

A diagnostic replay, not a page: the model's structure fed what the greenhouse
measured on the day, to see which forecast input costs the most. LetsGrow has no
forecast, so this cannot replace OpenMeteo; `/dli/performance` benchmarks on
OpenMeteo, as the forecast runs.

- **Weather:** LetsGrow's outdoor radiation sensor (`lg_rad`, W/m², daily mean
  over days with 22+ reported hours) × one slope to the above-lamp sensor,
  fitted over the same days as the weather link: **0.193** mol/m²/day per W/m².
- **Ratio:** the centred canopy/roof ratio, from around each day rather than
  before it.

Replayed out of fold (333–336 days, 1 Oct 2025 – 22 Nov 2026, lamps as run),
daily total-DLI MAE:

| | trailing ratio (as forecast) | centred ratio |
|---|---|---|
| OpenMeteo | 2.71 | 2.16 |
| LetsGrow radiation | 2.56 | 1.92 |
| measured roof (floor, §5.7) | | 1.35 |

- **The ratio's lag costs more than the weather:** 0.55–0.64 against
  0.15–0.24. At the roof OpenMeteo is clearly worse than the greenhouse's
  sensor (MAE ~2.9 vs ~2.2 mol/m²/day), but at the canopy that is scaled by a
  ratio of ~0.6 and partly cancels against ratio errors. Part of the centred
  ratio's gain is a mild leak: its 14 days include the day being scored.
- **Link 1 has a season the ratio cannot see.** Per month the radiation → roof
  factor runs from 0.144 (December) and 0.154 (November) to 0.199 (July), most
  likely the roof glass and structure at a low sun. A year-wide fit is dominated
  by bright summer days, so it overshoots the roof by up to a quarter in
  midwinter. The OpenMeteo fit has the same one-coefficient shape, so it carries
  the same error, and the ratio, taken against the *measured* roof, passes it
  straight through. (The leave-one-month-out slope moves only 0.192–0.194: the
  year-wide fit is stable, it is just the wrong shape.)
- **What to do about it:** take the ratio against the *predicted* roof light, so
  it absorbs link 1's season as well. Adopted in §5.9.
- **Considered and dropped:** shipping this as an opt-in "measured inputs" view
  on `/dli/performance`. It is a second, unforecastable prediction beside the
  real one, and the finding above is all it had to say.

### 5.9 The ratio against the predicted roof (adopted 2026-11-24 — measured)

`attenuation_by_date(..., predicted_roof=...)`, fed by
`predict_natural_dli_from_weather(model, ..., at_plant_level=False)` over the
ratio's history window. Both pages now fetch OpenMeteo from the start of that
window, not only for the days shown.

Live `/dli/performance` (total), same model, measured-roof ratio → predicted-roof
ratio:

| window | MAE | bias | MAPE |
|---|---|---|---|
| 1 Oct 2025 – 22 Nov 2026 | 2.67 → **2.53** | +0.35 → +0.45 | 37.7% → 37.1% |
| Nov 2025 – Mar 2026 | 2.50 → 2.44 | +0.85 → +0.82 | 27.4% → 27.1% |
| Sep – Nov 2026 | 2.04 → **1.82** | +0.46 → +0.02 | 14.1% → 12.5% |

Monthly bias, worst month +1.35 (Nov 2026) → +1.13 (Feb 2026). Spring and
summer moved most: May −1.01 → +0.36, Sep +0.92 → +0.02.

- **Winter still overshoots** by +0.8 to +1.1 each month Oct–Feb. That is not
  link 1's season (the ratio now absorbs it) but its noise: on a dull winter
  day the canopy is mostly lamp, so the day's ratio is mostly the lamp
  estimate's error (§7.3).
- **The bias went up overall**, +0.35 → +0.45, while the error went down. The
  old figure was the closer average only because summer undershoot cancelled
  winter overshoot; per month the new one is smaller nearly everywhere.
- **Bounded at 1.5.** The replay with the cap at 1.5 is what was measured; 1.0
  would throw away exactly the days that teach the ratio about OpenMeteo's
  dull-day misses.
- The past days use OpenMeteo's *archive* weather, not the forecast that was
  issued then; the archive is the better of the two, so the past-day ratio is
  mildly optimistic.

## 6. Performance

### 6.1 Version Evolution

Everything from v8 is **out-of-fold** on day-blocked folds, scored through
`red/fitting.py`. Earlier figures are in-sample and not comparable; they are kept
to show what the gap was.

| Version | Stage 1 | Stage 2 | Chain | Changes |
|---------|---------|---------|-------|---------|
| v5-v7 | 0.786* | 0.914* | 0.719* | *in-sample; chain was the product of the two |
| v8 | −0.036 | 0.977 | 0.655 | first out-of-fold scores; chain measured, not derived |
| v9 | 0.213 | 0.977 | 0.805 | stage 1 loses its day-of-year term (§5.6) |
| v10 | 0.917 | 0.978 | 0.926 | daily totals are time integrals (§5.5) |
| **v11** | — | — | **0.950** | one fit, weather → above-lamp PAR; s1000 removed (§5.7) |

For v11 the "chain" column is the single fit's out-of-fold R² (338 days,
1 Nov 2025 – 24 Nov 2026). There is no longer a stage to score separately.

Skill against the baselines, out-of-fold, weather → above-lamp PAR:

| Version | vs persistence | vs climatology |
|---|---|---|
| v10 chain (via s1000 lux) | +0.517 | +0.688 |
| **v11 single fit** | **+0.605** | **+0.742** |

For reference, v10's stages on their own: stage 1 (weather → s1000 lux) +0.532 /
+0.684, stage 2 (lux → indoor PAR) +0.735 / +0.829.

At v7 the chain was level with "same as yesterday" (−0.018 against persistence)
while reporting a combined R² of 0.719. The product was never a bound on
anything: it assumed an error propagation nobody had measured, and credited
stage 2 with an input it never receives. The chain figure above is obtained by
fitting stage 2 on measured lux and predicting held-out days from stage 1's
*predicted* lux, which is what serving does.

### 6.2 Summer peaks

The symptom that started the v9/v10 work: on the brightest days the measured DLI
ran far above the prediction. Over 2026-07-13 → 09-10, above-lamp, the eight
brightest days:

| | mean bias on the 8 brightest | MAPE, all 60 days |
|---|---|---|
| v8 | −30.6% | 23.4% |
| v9 (no seasonal term in stage 1) | −19.9% | 19.0% |
| **v10 (+ time integrals)** | **−12.3%** | **18.9%** |

A residual compression remains — the model still underpredicts the top of the
range and overpredicts the bottom. Since v11 the fit is a single linear map from
shortwave to roof PAR, so this is where to look next. It sits under the larger,
day-to-day error of OpenMeteo's sun against the greenhouse's own (§7.5).

### 6.3 Correlation Analysis

| Level | Pearson r | Notes |
|-------|-----------|-------|
| Hourly (raw) | 0.685 | High noise |
| Daily (aggregated) | 0.906 | Used for training |

## 7. Limitations

### 7.1 Sample Size
- 338 days of training (from 2025-11-01), one full seasonal cycle and no more
- Too few for anything richer than a regularised linear fit: 12-feature Ridge
  and boosted trees did no better or worse out of fold (§5.7). A flexible model
  cannot learn a March it has held out when there is only one March
- Revisit with 2–3 seasons; fitting takes under two seconds, so compute is not
  the constraint

### 7.2 Sensor Specificity
- Fitted to these sensor positions. The canopy sensor most likely moved at the
  July 2026 crop rotation: the lamp level fell from 188 to 140 µmol/m²/s with
  the lamps at 100%. The rolling level and ratio absorb such a move within a
  window; a fitted constant would not
- Another greenhouse would need its own fit and its own ratio

### 7.3 The canopy/roof ratio
- **Trailing windows lag, and that is the forecast's largest canopy error.** A
  step in the ratio takes up to a week to show in the forecast's 14-day median;
  replacing it with the centred one saves more than perfect weather does (§5.8)
- **Winter ratios are noisy.** On a dull winter day the canopy is mostly lamp, so
  the day's ratio is mostly the lamp estimate's error. The median and the (0, 1]
  gate contain this; they do not remove it
- **Needs both PAR sensors and LetsGrow.** Without them the fitted constant
  stands in, with the seasonal error it brings (§5.7)
- **It covers link 1's season only because it is taken against the predicted
  roof** (§5.9). Against the measured roof, the weather link's seasonal miss
  (one coefficient for a roof that transmits ~25% less at a December sun)
  would pass straight through (§5.8)
- The ratio follows the *sensor*, not necessarily the crop: if the crop shades
  the canopy sensor, the predicted canopy DLI is what that sensor would read

### 7.4 Lamps
- **Ahead of now, the forecast assumes the last 3 days repeat.** It does not
  predict the grower switching lamps off in bright sun. Inside the lit period,
  9% of daytime hours with the roof at 400–600 µmol/m²/s had the lamps off,
  and 20% above 600; below 200, none did. A radiation-aware
  share was measured to cut lamp-DLI MAE from 1.29 to 1.11 in autumn
  (issue 075); not adopted yet
- **LetsGrow gaps:** where LetsGrow has no data (its two-day delivery lag, or the
  outage of 26 Oct – 2 Nov 2026), the PAR detector stands in. It cannot see
  lamps under a bright sun

### 7.5 Weather Forecast Accuracy
- **The largest error above the lamps.** OpenMeteo's daily shortwave misses the greenhouse's
  sun by an MAE of ~3 mol/m²/day above the lamps. LetsGrow's own radiation
  sensor gets ~2.2, but only for days that have happened
- **At the canopy it is not the largest.** There the ratio's 3-day lag costs
  more than the weather (§5.8)
- Prediction accuracy degrades with forecast horizon
- Fitted on OpenMeteo's historical weather and served its forecasts, which are
  less accurate than the history

## 8. Conclusion

Training on the cleanest signal, the above-lamp sensor, still holds (§5.4). What
v11 changed is everything around that fit:
- an intermediate stage that added error but no information is gone;
- a constant that hid a seasonal swing is replaced by a measured, rolling ratio;
- the lamps are read from the system that switches them, not inferred.

The model has *fewer* fitted parameters than v10, and it is more accurate day to
day.

**Data quality beats model complexity at small sample sizes**, and v11 adds a
second lesson: **where a quantity can be measured as it moves, measure it rather
than fitting it once.**

---

*Model Version: 11 (one weather fit to above-lamp PAR, rolling canopy/roof ratio, lamps from LetsGrow)*
*v11 results, §5.8 and §5.9 measured 2026-11-24; §6.2 measured 2026-09-18*
*Data Range: November 2025 – November 2026*
