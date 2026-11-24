"""Crop-cycle config for the red twin.

Red-only config read from the top-level ``crop_cycles`` key in red's
``metadata.yaml`` (the shared metadata registry ignores it, keeping ``shared/``
twin-agnostic). Mirrors ``risk/config.py``: the values are PROVISIONAL — the
cycle dates especially are horticultural assumptions awaiting confirmation from
Neurath/WP1 — so they live in config, never as code constants, and the models
declare **no defaults**: a missing or malformed block fails loudly at load
rather than silently falling back to guessed dates.

The conversion helpers turn red's vocabulary into the twin-agnostic
``shared.cycles`` model. A cohort is a plain span: it used to be subdivided into
developmental stages bound to growth sections, but the boundaries were fixed
offsets from the set date and so said nothing the start date did not, while
spending the chart's colour on a schedule instead of on observed conditions.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel

from wp6_data.shared.cycles import CohortSpec, CycleSpec
from wp6_data.shared.series import PeriodConfig, SeriesMetric


class CohortConfig(BaseModel):
    """How often a cohort sets, and how long it takes to complete."""

    interval_days: int
    duration_weeks: int


class ClimateMetric(SeriesMetric):
    """A shared daily series, optionally derived from raw readings.

    A climate metric *is* the wire-ready seam: because red models wire heights
    as devices (ADR 0001), pointing one at a growth section later means changing
    ``device`` to ``WS_01_01-h3``, with no code change and no new abstraction.

    Most metrics are a daily bucket the database aggregates (``agg``). A
    ``derive``d one cannot be: *day length* is time spent above a threshold,
    which no avg/min/max/sum of a bucket recovers, so it is computed from the
    raw readings. ``agg`` still says how its days combine over a cohort's
    window.
    """

    derive: Literal["day_length"] | None = None


# Red's name for the shared vocabulary.
CycleConfig = PeriodConfig


class ClimateConfig(BaseModel):
    metrics: list[ClimateMetric]


class CropCyclesConfig(BaseModel):
    cohort: CohortConfig
    climate: ClimateConfig
    cycles: list[CycleConfig]

    def metric(self, key: str) -> ClimateMetric:
        """The declared climate metric ``key``, or the first one as a fallback."""
        for m in self.climate.metrics:
            if m.key == key:
                return m
        return self.climate.metrics[0]


def load_crop_cycles(yaml_path: Path) -> CropCyclesConfig:
    """Load and validate the ``crop_cycles`` block from a twin metadata YAML.

    Raises if the file or the key is absent, or if any value is missing — the
    view must never run on implicit dates.
    """
    if not yaml_path.exists():
        raise ValueError(f"no metadata file at {yaml_path}")
    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    if "crop_cycles" not in raw:
        raise ValueError(f"no 'crop_cycles' block in {yaml_path}")
    # Each metric validates its own agg (SeriesMetric), so a bad one fails here
    # rather than reaching a provider call.
    return CropCyclesConfig(**raw["crop_cycles"])


def to_cohort_spec(config: CropCyclesConfig) -> CohortSpec:
    """Red's declared cohort rhythm as a twin-agnostic :class:`CohortSpec`."""
    return CohortSpec(
        interval=timedelta(days=config.cohort.interval_days),
        duration=timedelta(weeks=config.cohort.duration_weeks),
    )


def to_cycle_spec(cycle: CycleConfig) -> CycleSpec:
    """One declared cycle as a twin-agnostic :class:`CycleSpec`."""
    return CycleSpec(label=cycle.label, start=cycle.start, end=cycle.end)
