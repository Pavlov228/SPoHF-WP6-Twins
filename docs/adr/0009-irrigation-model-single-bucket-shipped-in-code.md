# The irrigation twin forecasts with a single-bucket water balance, fitted locally and shipped in code

Blue's irrigation twin must answer "when, and how much, should I water?". So
its soil-moisture model has to respond correctly to water: an extra dose must
never dry the soil, and a week without rain or watering must never wet it. A
rolling-origin bake-off (issue 068, research note §7) scored eight candidates
on forecast error *and* on those two counterfactual checks.

We use **model A, a daily single-bucket water balance** on the fused,
relative soil moisture: rain and drip in, ET0 × seasonal Kc × stress curve
Ks(S) out, drainage above field capacity.

- **The drip dose effect is tied to rain**, as `irr_eff = rain_eff /
  area_per_plant`, with the area in config. The probes sit away from the
  drippers, so the data can't identify the dose effect on its own: most
  monthly refits put it at 0. The pooled fit's ratio, 1 L ≈ 1 mm over
  ~2.2 m², matches the plant spacing.
- **The band** is the 10–90 % quantile of held-out residuals per lead time.
  In-sample residuals gave a band that caught 56 % of outcomes, not 80 %.
- **The model is fitted locally and shipped as a committed parameters file.**
  Production never trains or refits it. No new data is expected to drive
  retraining, so ADR 0007's object store, nightly refit and fingerprint gate
  don't apply here. Refitting means running the CLI locally and committing
  the result.

## Considered Options

- **Chronos-2, a zero-shot foundation model with covariates.** It was the best
  at 7 days (+21 % skill vs +19 %) and had the best-calibrated band. Rejected:
  an extra watering lowered its forecast in 94 % of cases, because it learned
  "watering happens when dry". It is also not explainable to a farmer, and it
  needs torch.
- **A + Kalman filter (B).** No accuracy gain; it smoothed away real readings at
  the origin.
- **A two-bucket model with the unmeasured soil bed (A2).** It overfit with about
  400 training days, and the bed is unobservable.
- **A + a gradient-boosting residual (C), or gradient boosting alone (D).**
  Both broke the physics (a dry week wetting the soil), and D was worse than
  persistence.
- **Fitting the dose effect freely.** Rejected: advice would mostly say
  watering does nothing.

## Consequences

- The model is explainable in physical terms (mm of rain, mm of ET0, days to
  dry out) and transfers to a similar field by changing config.
- Advice rests on the dose prior. If a future log or a probe placed near a
  dripper shows the drip effect, `area_per_plant` is the knob to adjust.
- Forecast scores assume perfect weather. In operation, Open-Meteo's forecast
  error adds to them.
- The backtest harness stays in the repo, so the choice can be re-run when the
  2026 watering log arrives.
