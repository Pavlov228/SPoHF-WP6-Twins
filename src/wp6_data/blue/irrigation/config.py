"""Load and validate the blue irrigation configuration (``config.yaml``)."""

from __future__ import annotations

from datetime import date, datetime, time
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml
from pydantic import BaseModel, ConfigDict, model_validator

CONFIG_PATH = Path(__file__).with_name("config.yaml")


class _Strict(BaseModel):
    # A typo in a knob must fail loudly, not fall back to nothing.
    model_config = ConfigDict(extra="forbid", frozen=True)


class FusionParams(_Strict):
    floor: float
    dropout_pad_hours: int
    spike_window_hours: int
    spike_k: float
    spike_min_spread: float
    reference_low_quantile: float
    reference_high_quantile: float
    consensus_window_days: int
    consensus_min_corr: float
    consensus_min_probes: int
    frozen_below_c: float

    @model_validator(mode="after")
    def _ordered_quantiles(self) -> FusionParams:
        if not 0 <= self.reference_low_quantile < self.reference_high_quantile <= 1:
            raise ValueError("reference quantiles must satisfy 0 <= low < high <= 1")
        return self


class RainParams(_Strict):
    station_device: str
    station_sensor: str
    max_mm_per_hour: float
    check_window_days: int
    min_reference_mm: float
    min_ratio: float


class WateringParams(_Strict):
    volume_sensor: str
    duration_sensor: str
    default_local_time: time
    timezone: str

    @model_validator(mode="after")
    def _known_timezone(self) -> WateringParams:
        ZoneInfo(self.timezone)  # raises for an unknown zone
        return self


class ModelParams(_Strict):
    area_per_plant_m2: float


class Exclusion(_Strict):
    probe: str
    start: datetime
    end: datetime
    reason: str


class IrrigationConfig(_Strict):
    probes: tuple[str, ...]
    sensor: str
    temperature_sensor: str
    history_start: date
    fusion: FusionParams
    rain: RainParams
    watering: WateringParams
    model: ModelParams
    exclude: tuple[Exclusion, ...] = ()

    @model_validator(mode="after")
    def _exclusions_name_probes(self) -> IrrigationConfig:
        unknown = {e.probe for e in self.exclude} - set(self.probes)
        if unknown:
            raise ValueError(f"exclude names unknown probes: {sorted(unknown)}")
        return self


def parse_config(raw: dict, known_devices: set[str]) -> IrrigationConfig:
    """Validate *raw*; every probe and the rain gauge must be devices ``metadata.yaml`` declares."""
    config = IrrigationConfig(**raw)
    missing = (set(config.probes) | {config.rain.station_device}) - known_devices
    if missing:
        raise ValueError(f"devices not declared in metadata.yaml: {sorted(missing)}")
    return config


@lru_cache(maxsize=1)
def load_config() -> IrrigationConfig:
    from wp6_data.blue.deps import metadata

    with CONFIG_PATH.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return parse_config(raw, set(metadata.devices))
