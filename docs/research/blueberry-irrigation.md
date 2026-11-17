# Blueberry irrigation: literature and data basis

Research note for the blue irrigation twin (issues 064–073). It sets the
thresholds, the watering window and the model assumptions that the advice and
the simulator rest on, and it records where each number comes from.

**Status:** draft (2026-11-17). Literature and a first data profile are in; the event log is confirmed real; drip doses are not visible to the probes (§4.1).

## Setting

- Highbush blueberry (*Vaccinium corymbosum*), **young**: first fruit in 2024.
- Grown in **peat substrate set into a soil bed**, about 1–2 m apart, drip fertigated, **outdoors**.
- **The roots have grown out of the peat into the surrounding bed.** The root zone is therefore peat plus bed, not a closed volume.
- **The soil probes sit in the peat**, the substrate the drip wets. The bed is unmeasured: a buffer that takes up drainage and can feed water back by capillary rise.
- In 2025 the farm watered on **11 days**, from 29 Apr to 2 Oct. Every treatment
  got the same dose on the same day, starting at 1333 ml/plant (10 min) and
  rising to 2400 ml/plant (18 min), a fixed rate of about 133 ml/min/plant. Rain
  is the dominant water input. The working assumption, which the farm is
  confirming, is that the CSV lists every watering.
- The farm's practice is treated as **adequate**: no visible drought damage was
  reported. It anchors the empirical band below.

## 1. How dry is too dry

The literature gives no single, universal threshold. What it does agree on:

| Context | Finding | Source |
|---|---|---|
| Field, loam + 30 % peat | Growth and productivity peak at **−4 to −8 kPa**. Irrigate before **−15 kPa**. | [FAO Agris record](https://agris.fao.org/search/fr/records/647369bae17b74d22254f64b) |
| Peat/coir/perlite substrate (containers) | Best growth with irrigation triggered at **−2 kPa** (single pulses) | [ASHS 2025](https://ashs2025.sched.com/event/247yQ/vsf-1-drip-irrigation-strategies-for-improved-blueberry-production-in-substrate), [ISHS 1440_40](https://ishs.org/ishs-article/1440_40/) (abstract only) |
| Sandy soils, extension | Set point **10–15 kPa** | [UGA](https://site.caes.uga.edu/blueberry/?p=639) |
| Grower guide | Never let the soil go drier than **30–40 kPa** | [Rivulis](https://www.rivulis.com/expert-guide-to-growing-blueberries-in-soil/) |

**Duration and timing matter more than the depth of any one dry spell:**

- **Fruit growth through ripening** (late May to late June at the INRAE site) is
  the most drought-sensitive window. Stress then cut fruit size, and cut yield
  by about 40 % in a container study
  ([INRAE](https://hal.inrae.fr/view/index/identifiant/hal-02699098),
  [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S030442380000217X)).
- **Two-week deficits** at single stages had little effect on yield in an Oregon
  five-cultivar study, but did lower firmness and, in several cultivars, berry weight
  ([USDA ARS](https://www.ars.usda.gov/research/publications/publication/?seqNo115=378503)).
- A severe regulated deficit (**50 % of ETa**) lowered photosynthesis and berry
  size and weight
  ([RDI, cv. Brigitta](https://www.researchgate.net/publication/321074223_Regulated_deficit_irrigation_effects_on_physiological_parameters_yield_fruit_quality_and_antioxidants_of_Vaccinium_corymbosum_plants_cv_Brigitta)).
- After-effects were small, except for stress during **flower-bud induction**
  (late summer), which carries over into the next season's crop.

**Peat in a soil bed.** The probes read the peat substrate, so the substrate rows above apply most directly, though the bed softens the drying. Substrate trials trigger irrigation much wetter (**−2 to −4 kPa**) than field soil (−10 to −15 kPa). A small peat volume is a small reservoir that dries quickly; here the bed adds some reach.

**Working rule for the twin (to validate against the data in §4):**
- Keep the substrate wetter than about −4 kPa (container range). The trigger in practice comes from the empirical band.
- Treat each **hour** drier than the trigger as stress. A **cumulative**
  stress-hours budget is the score, not a single crossing. This follows the
  memory note that risk is scored in physical units, not episodes.
- Weight the fruit-growth window (roughly June–July in NL) and the bud-induction
  window (Aug–Sep) more heavily.

## 2. kPa → our sensors' %VWC

The literature gives thresholds as tension (kPa). Our sensors report %VWC, so
the two have to be bridged:

- Peat holds far more water than mineral soil at the same tension. Reported field
  capacities for cultivated peat are about **0.5–0.7 m³/m³**
  ([Hydrus peat study](https://informahealthcare.com/doi/abs/10.1080/02626667.2020.1853132)).
  The van Genuchten (high water-holding peat) or Campbell (low) retention models
  are the usual fits
  ([water retention curve](https://en.wikipedia.org/wiki/Water_retention_curve)).
- **Our sensors read 15–35 %VWC on peat**, about half the expected level. They
  are very likely calibrated for mineral soil. **Absolute VWC is not
  trustworthy. Only relative levels are.**
- **Consequence:** the twin works on a **normalised** moisture scale per
  sensor, where 1 = its post-drainage plateau (≈ field capacity) and 0 = its
  driest seen level. The kPa thresholds are mapped onto that scale through
  the *empirical* band (§4), not through a retention curve we cannot calibrate
  without tensiometers. A tensiometer at the field would close that gap and is
  worth suggesting to the farm.

## 3. When to water (drip)

With drip, the foliage stays dry, so the time-of-day arguments about leaf
wetness and disease ([MSU](https://extension.msstate.edu/publications/the-plant-doctor-watering-and-plant-disease),
[Clemson](https://hgic.clemson.edu/factsheet/landscape-irrigation-management-part-5-irrigation-time-of-day/))
**do not apply**. What remains:

- **No consistent efficiency difference** between morning and afternoon drip
  ([Penn State](https://extension.psu.edu/does-it-matter-what-time-of-day-you-irrigate-with-drip-irrigation)).
  Evaporation from a drip-wetted spot is small.
- Plant water stress peaks at **midday to early afternoon**, which is when
  midday stem water potential is the indicator that tracks fruit size
  ([Lakso 2022](https://docs.florapulse.com/research/lakso-2022-swp-irrigation-scheduling-fruit.pdf)).
  Water placed **before** that peak is available when demand is highest. Water
  given in the late afternoon refills the soil after the day's stress has
  already happened.
- In a tomato greenhouse study, irrigating at 08:00 beat irrigating at 20:00 by
  11 % in yield
  ([greenhouse timing study, cited work](https://beid.kau.edu.sa/Files/155/Researches/59827_32085.pdf)).
  That is indicative only: a different crop and setting.

**Recommendation:** water in the **early morning** (before about 09:00). A late
evening slot is the fallback, since it also has the soil wet before the next
day's demand. The late afternoon is the least useful slot, though with drip on
peat the modelled difference is expected to be **small**. The simulator should
show the size of that effect rather than assume it.

## 4. What the data shows (profiled 2026-11-17)

These numbers come from throwaway scripts, not shipped. Sources:
- Sensors: `exports-blue/` for 2025, plus the local sync for 2025-10-21 onwards.
- Weather: the Open-Meteo archive at blue's configured coordinates.
- Waterings: the prod fertigation CSV.

### 4.1 The logged waterings barely show in the soil probes

For each of the 11 logged days, I took the biggest 3-hour rise per sensor
between day −2 and day +3.

- **On rain-free event days the rise is at most 0.6 %VWC.** A 1.3–2.4 L dose
  should be clearly visible in the root zone.
- **Every large rise in the event windows coincides with rain** (5–10 mm).
- There are a few jumps of 1.5–5 %VWC, with two or more sensors rising together,
  on days that look nearly rain-free in Open-Meteo: 2025-05-31, 08-23, 08-28,
  09-03 and 10-05. **The on-site rain gauge recorded 13, 1.3, 3.3, 5.5 and
  4.3 mm on those days, against Open-Meteo's 6.4, 0.1, 0.9, 1.4 and 2.3 mm.**
  These were local showers that the gridded reanalysis underestimates. They
  were not unlogged waterings.

**Reading (confirmed with the farm 2026-11-17):**
- The CSV **is** the actual watering log.
- The probes are in the same peat as the drippers but **not next to them**, and
  each plant has several drip points.
- So a 1.3–2.4 L drip dose wets bulbs the probes don't see. The water drains
  locally and quickly down into the bed, while rain wets the whole surface,
  probes included.

**Consequences for the model:**
- **Irrigation can't be calibrated from steps in the sensor data.** The
  effect of a dose has to come from a **physical prior**: dose ÷ effective
  capacity, scaled by the share of the dose that reaches the measured root
  zone. That share is fitted only weakly, from the slower decline in the days
  after an event, and needs a tight prior.
- **Drip and rain enter the bucket differently.** The bake-off must keep them
  as separate inputs, each with its own effectiveness parameter.
- **The probes behave like a "bulk root-zone" signal**, not a dripper-bulb signal.
  This is a caveat for advice: the plant's roots near the drippers are wetter
  than the probes report.

### 4.2 Sensor coverage: both seasons are complete (from the DB)

The first pass used `exports-blue/`, which turned out to be incomplete for 2025.
After a full local sync, **all 10 soilMoisture sensors have about 5,000 h in
Apr–Oct 2025**, and they do in 2026 too. With 10 sensors, §4.1 still holds: on dry
event days the largest rise of any sensor is about 1 %VWC.

| Zero dropouts (hours < 1 %VWC) | Apr–Oct 2025 | Apr–Oct 2026 |
|---|---|---|
| `rij4` | 1698 | 2829 |
| `04` / `03` / `01` | 0 | 1019 / 412 / 190 |

- `rij4` is broken in both years. The zeros in the other sensors appear only in 2026, in the summer lows.
- Peat **shrinks and turns water-repellent when it dries**, pulling away from the
  probe, so very low or zero readings are probably loss of contact, not truly dry
  substrate. The quality mask (065) must treat "about 0" as missing, not as a
  reading. Dry peat is also slow to rewet, which bears on advice: don't let
  it get that dry.

### 4.3 Dry-down curves: the expected shape is visible

Rain- and watering-free stretches of 5 days or more, as the slope per sensor:

- **When wet, drying tracks ET0.** At ET0 2.5 mm/d the slope is about
  −0.4 to −1.7 %VWC/d (Apr 2026). At ET0 5.0 mm/d it is −1.3 to −3.0 %VWC/d
  (21–28 May 2026).
- **When dry, drying stalls despite high ET0.** At readings below about 10 %VWC,
  the slope is about 0 even at ET0 5.2 mm/d (6–19 Jul 2026). This is the
  stress-reduction curve Ks(S) from FAO-56, possibly helped by capillary supply
  from the bed, and partly the sensor floor (§4.2). **This is the "curve"
  the model has to carry.**
- **But 2025 behaves differently.** In 6–28 Aug 2025 (23 days, no rain, ET0 3.8 mm/d,
  no logged watering until 2 Sep), the six sensors with full data stayed **almost
  flat at 22–34 %VWC** (slopes between −0.44 and +0.05 %VWC/d). That is not
  stress-stall: the levels are mid-range. In 2026 the same sensors fell at
  1–3 %VWC/d from similar levels.
  - **Unexplained so far.** Candidates: capillary or groundwater supply
    through the bed (a wetter bed in 2025?), a slower sensor response, or
    watering outside the log.
  - **This is the main thing the 2026 watering log must clarify.** If 2026
    had frequent watering and still fell faster than 2025, the bed is the
    likely explanation.
- **Levels differ per sensor (offsets), but the slopes are consistent.** This
  supports normalising each sensor, then fusing.

### 4.4 Seasonal water demand

| Apr–Sep | Rain | ET0 | ET0 − rain |
|---|---|---|---|
| 2025 | 351 mm | 655 mm | 305 mm |
| 2026 | 361 mm | 656 mm | 295 mm |

The two years are nearly identical in weather. With a young-plant Kc of about
0.3–0.5, crop demand beyond rain is on the order of 100–150 mm over the season.
That is worth comparing against the logged doses once the root-zone or bed area is known.

### 4.5 Empirical band (provisional)

The planned "moisture just before each watering → trigger" method is **not
usable yet**, because the logged events don't line up with the sensor response
(§4.1). Interim proxy: the **knee of the dry-down curve**, the level below
which drying stalls (§4.3), sits at the low end of the normalised scale. Treat
it as the stress onset, and set the trigger with a margin above it. Revisit
once the event log is confirmed.

### 4.6 Rain vs soil response (checked 2026-11-17)

- **Gauge and Open-Meteo agree where the gauge is healthy.** Mar–Nov 2025:
  daily correlation 0.89, totals 435 vs 484 mm. Only 1 day had gauge rain and
  no Open-Meteo rain; 10 days the reverse. Open-Meteo stamps hourly sums at the
  **end** of the hour and the gauge at the start, so Open-Meteo rain and ET0 are
  moved back an hour (the cross-correlation then peaks at lag 0).
- **Of 113 sharp rises in fused soil moisture, 104 are explained by the rain the
  twin uses.** 19 of those are only explained because Open-Meteo stood in for a
  failing gauge. The soil probes act as a third rain gauge.
- The rest:
  - **Thaws (Jan 2026).** The soil was at 0 °C. Ice reads like dry soil to a
    permittivity probe, so a thaw looks like rain, and snow only reaches a rain
    gauge once it melts. These hours are now masked as `frozen`
    (soil temperature ≤ 1 °C).
  - **The gauge under-reading briefly** (late Nov–Dec 2024, 2 Jun 2026):
    Open-Meteo had 3–6 mm, but not for long enough for the 14-day check to
    switch over.
  - **Coverage collapse (15 and 17 May 2025).** All probes went silent for about
    a day, and the few that came back were noisy. A median over a changing
    handful of probes jumps without any water. → **The model must weight or
    skip low-coverage hours.**
  - Probe start-up (8 May 2024).
- **Gauge data quality:**
  - Commissioning junk on 2024-03-26 (readings of ±85 mm). Hours with a
    negative reading or more than 60 mm are now dropped.
  - Probe `03` reported −3.7 °C in Nov 2026 while every other probe stayed
    above 1.3 °C and the air never went below 0 °C. That is a fault, which the
    `frozen` mask happens to catch.

## 5. Model inputs from the literature

- **Crop coefficient (Kc).** The literature has no clean FAO-56 value for
  highbush. The generic FAO-56 "berries (bushes)" curve is the starting point
  (to verify: about 0.3 initial / 1.05 mid / 0.5 end). Young or sparse plantings
  measure far lower: **0.27** for a 4-year-old Bluecrop, and 0.11–0.41 from
  June to September for a first-year planting in Oregon
  ([lysimeter Kc](https://agris.fao.org/search/en/records/65dfd59d4c5aef494fe5dcb8),
  [ASHS 2025 Oregon](https://ashs2025.sched.com/event/247yK/vsf-1-water-requirements-for-establishing-a-new-planting-of-northern-highbush-blueberry-in-western-oregon)).
  **Our plants are young**, so Kc starts low (about 0.1–0.4, rising through the season and year on year). **Kc is fitted, with the literature value as the prior.**
- **Root zone = the measured peat plus an unmeasured bed.** The peat volume is only a lower bound. The **effective** water-holding capacity is fitted from the data: the jump per ml dose, and the dry-down rate. Water moves between peat and bed through a fitted exchange term (drainage out when wet, capillary return when dry), instead of free drainage. The bake-off compares a single "leaky" bucket with a two-bucket root-zone+bed model.
- **Units.** Drip is given in ml/plant. Rain reaches the root zone directly and through the bed around it, so the **rain-catch fraction is fitted**, not derived from geometry. Peat dimensions are a prior, not required.
- **ET0** comes from Open-Meteo `et0_fao_evapotranspiration` (FAO-56 Penman-Monteith).
- **Rain: the on-site gauge is better when healthy, but not reliable on its own.**
  - In Apr–Oct 2025 its monthly totals match Open-Meteo within about 10 %
    (Aug: 22 vs 46 mm), and on shower days it catches local rain that
    Open-Meteo misses (§4.1).
  - From Dec 2025 it reads **0 or far too low for whole months** (Dec, Feb,
    Jul, Aug 2026): a clogged or faulty gauge.
  - Use a fused rain input: **the station where it is plausible, and Open-Meteo
    where the station is flat while Open-Meteo shows rain**. Flag the
    substitution in the coverage. This is the same idea as the moisture
    quality mask.
  - Implemented (issue 066) as a centred 14-day check: if Open-Meteo saw at
    least **5 mm** and the gauge less than **30 %** of it, the gauge is
    replaced for those days. The first threshold, 10 mm, missed the gauge's
    death around **21 Jun 2026**: July's showers were only 1–3 mm. Replaced
    so far: Dec 2025, most of Feb 2026, and Jul–Oct 2026 (in part).
  - **Open:** the gauge read **47.3 mm on 11 Nov 2026**, right after its
    outage, against Open-Meteo's 4.5 mm. That is either a cleaned or repaired
    gauge emptying what it collected, or a genuinely local downpour. The
    window check can't tell these apart, so the reading is kept. Worth asking
    the farm.

## Open questions

- Were there waterings in 2026? 2026 is the best-instrumented season but has no CSV yet.
- Can the farm check or clean the rain gauge? It has under-read since about Dec 2025.
- Sensor installation depth, and distance to the dripper.
- Peat volume per plant (optional prior). The effective capacity is fitted.
- Cultivar. (Planting age: young, first fruit 2024.)
- Does canopy overhang block rain from the root zone? This lowers the rain it catches.

## 7. Model bake-off (issue 068, run 2026-11-17)

**Setup:**
- **Data and resolution:** daily, on the fused soil moisture. A day counts only
  if it had at least 3 probes and at least 18 usable hours.
- **Origins:** rolling, every in-season day (Apr–Oct) with a reading,
  Apr 2025 – Oct 2026: 413 origins, forecasting 1–7 days ahead.
- **Refitting:** each model is refitted monthly, on targets before that month.
- **Weather** is as observed (a perfect forecast), so the scores rate the
  models; in operation, forecast error adds to them.
- **Code:** `blue/irrigation/backtest.py`. Model E runs outside the project
  environment through `scripts/irrigation_foundation_models.py`, so torch never
  enters the image.

| Model | MAE 1 d | MAE 7 d | Skill 1 d | Skill 7 d | 80 % band catches | Water never dries | Dry week never wets |
|---|---|---|---|---|---|---|---|
| E Chronos-2 (zero-shot, covariates) | 0.033 | **0.086** | +5 % | **+21 %** | **74 %** | **6 %** ✗ | **34 %** ✗ |
| **A single bucket** | **0.030** | 0.088 | **+11 %** | +19 % | 56 % | 100 % | 100 % |
| B A + Kalman | 0.035 | 0.091 | 0 % | +17 % | 55 % | 100 % | 100 % |
| A2 two bucket (bed) | 0.032 | 0.097 | +7 % | +11 % | 50 % | 98 % | 81 % |
| C A + GBM residual | 0.035 | 0.107 | −2 % | +2 % | 57 % | 100 % | 4 % ✗ |
| persistence | 0.034 | 0.109 | — | — | 64 % | — | — |
| E Chronos-Bolt (univariate) | 0.035 | 0.111 | −3 % | −2 % | 74 % | n/a | n/a |
| D GBM direct | 0.069 | 0.132 | −102 % | −21 % | 53 % | 45 % ✗ | 2 % ✗ |

Skill is relative to persistence. Per season, A wins 2025 (+22 % at 7 days)
and Chronos-2 edges 2026 (+21 % vs +17 %).

**Findings:**
- **A, the single bucket, is the best controllable model.** It has the best
  1-day error, a 7-day error within 0.002 of the best, and it passes both
  physical checks.
  - The extras don't pay off with about 400 training days:
    - **Kalman (B):** it smooths away a real reading at the origin.
    - **Bed (A2):** it overfits.
    - **GBM residual (C):** it breaks the physics, wetting during a dry
      week 96 % of the time.
- **Chronos-2 forecasts well zero-shot, but it is not controllable.** An
  extra watering *lowers* its forecast in 94 % of cases, and a dry week wets
  in 66 %. It learned the correlation "watering happens when dry", exactly as
  the plan anticipated. Its bands are the best calibrated, though, which is
  a useful benchmark for A's.
- **D (gradient boosting) overfits badly:** it is worse than persistence.
- **Probe set:** dropping `rij4` and `01` changes A's MAE by at most 0.002.
  All probes stay.

**Model A fitted on all data reads physically** (S units per unit):
- `rain_eff` 0.0065/mm: a full 0→1 swing is about 150 mm of effective rain.
- `et_scale` 0.0056/mm of ET0, with Ks = S/θ and θ ≈ 1.0. A summer day lowers S
  by about 0.02, and drying slows as the soil dries.
- `kc_slope` −0.47: water use falls from early summer into autumn.
- `drain_rate` 0.63/day above `s_fc` 0.77: fast drainage when overfull.
- `mix` 1.0: a day's inputs show in that day's mean.
- **`irr_eff` 0.0030/L.** It is 0 in most monthly refits, because the doses
  don't show on the probes (§4.1), so the data can't pin it down. The pooled
  ratio `rain_eff / irr_eff` ≈ **2.2 m²** means 1 L of drip ≈ 1 mm of rain over
  2.2 m². That matches a plant every 1–2 m (1–4 m²). → **Proposed prior:**
  `irr_eff = rain_eff / area_per_plant`, with the area in config.

**Open, for the user:**
1. Choose A as the production model.
2. Tie the dose response to rain over the plant's area (the proposed prior above).
3. Calibrate A's bands: they catch 56 % instead of 80 %. Proposal: widen
   them using held-out residuals, as C and D already do.

## 8. Advice levels (issue 071, decided 2026-11-17)

The literature gives no threshold on our relative scale, so the trigger is set
through **allowable depletion**: the share of the available water (field
capacity → dry limit) that may be used up before watering.

| Source | Allowable depletion |
|---|---|
| MSU extension: the default for blueberry, though "not studied adequately" ([MSU](https://bluedynamo.msu.edu/uploads/files/Irrigating%20Blueberries-may%202012.pdf), [MSU news](https://www.canr.msu.edu/news/using_irrigation_to_maintain_high_yields_of_blueberries)) | 50 % |
| Blueberry in substrate: growth peaked at 25 %, and performance fell clearly past 50 % | 25 % |
| Young plants have shallow roots (MSU: about 30 cm young, 45 cm older), so less reserve | favours the low end |

**Decision: 33 %.**
- On our scale, field capacity is `s_fc` ≈ 0.77 (fitted), and 0, the driest
  reading on record (July 2026), is taken as the dry limit. That gives
  **trigger = 0.77 × (1 − 0.33) ≈ 0.52**. The 25 % and 50 % readings give 0.58 and 0.39.
- It matches practice: in 2025 the farm watered at a median soil level of
  0.52 (range 0.31–0.74).
- **Target** = field capacity − 0.07 ≈ 0.70. Water above it mostly drains.
- **Time:** early morning (§3).

**How much.**
- The dose is the rain-equivalent depth that brings the no-watering path back
  to the target on the advised day. It is converted to L/plant through the
  2.2 m² dose prior (ADR 0009).
- At that prior, refilling from the trigger to the target is about 25–30 mm,
  or about **55–65 L/plant**. A summer day's crop use is about 1.8 mm, or about
  4 L/plant. The farm's 1.3–2.4 L doses are small against that. This is
  consistent with July 2026 being dry despite watering (≈143 mm ET0 against
  ≈12 mm rain).
- But the litres rest on the area assumption, so the advice labels them as an
  estimate, and the simulator lets the farm try smaller doses.

**What-ifs** run on the dedicated `/irrigation/simulator` page: rain scaled or
delayed, a dose on a chosen day, morning or evening. The model is daily, so the
time of day only decides which day a dose counts (an evening dose counts the
next day).

## 9. When the crop is active (decided 2026-11-17)

A reading below the trigger in autumn is not a thirsty plant: water use
collapses as the season ends (2025 potential crop demand: June 4.6, September
1.6, October 0.6, November 0.3 mm/day). Advice, and the replay's advice rule,
apply only while **both** hold:

- **The soil is warm enough for roots to work: the 7-day mean is at least 12 °C.**
  Oregon State extension puts the start of blueberry root growth at about 13 °C
  (55 °F) soil temperature, with growth continuing into early November. Root
  production peaks in late May and again in **September, about a month before
  dormancy** ([OSU](https://extension.oregonstate.edu/crop-production/berries/how-blueberry-plants-develop-grow),
  [ISHS 1180_23](https://ishs.org/ishs-article/1180_23/)). So September
  watering still matters. The probes' median soil temperature is used, since
  it is robust to one faulty probe.
- **There is real demand: the mean *potential* crop water use (ET0 × Kc, before
  drought stress) over the coming week is at least 1 mm/day.** Potential, not
  actual use: in a drought, actual use drops because the soil is dry, and
  gating on it would hide exactly the plants that need water.

Rejected: **GDD**. It keeps accumulating through autumn (2025: 2,240 by the end
of September, 2,611 by the end of the year), so it can't mark the end of the
season. Also rejected: a seasonal (deeper) autumn trigger, which hard-codes a
calendar the plants don't follow.

Result: the active season was 6 Apr – 29 Sep 2025 and 13 Apr – 2 Oct 2026. In the
2025 replay, the advice rule's 16 October watering disappears and the
June–July ones remain.

## 10. Simulating from a past day (issue 072)

The user found a whole-season policy replay unhelpful ("stupid"), and it was
removed. Instead, the **what-if simulator can start from any past day** (the
"last known day"). The following week's actual weather stands in for the
forecast. The what-ifs (rain less or later, a different dose or day) run on it,
and the measured soil moisture and logged watering show what really happened,
so a what-if can be held against reality.

Kept from the replay work: the crop-activity gate (§9), now used by the live
advice. Learned: an open-loop season simulation drifts badly (mean error 0.10 in
2025 and 0.26 in 2026), so the model should only be trusted over its 7-day
horizon from a measured start, which is exactly what the simulator does.
