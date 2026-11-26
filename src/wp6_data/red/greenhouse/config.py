"""The ``greenhouse`` block of red's metadata: physics and advice.

No defaults here on purpose: a guessed U-value or vent curve would draw a
plausible-looking but invented greenhouse. Where each number comes from is
written next to it in ``metadata.yaml``, and as data in ``provenance``, which
the page shows its visitors next to the value.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, model_validator

#: Where a configured number comes from: fitted on red's own data, taken from
#: the literature, assumed for want of either, the grower's own figure or rule,
#: or a threshold the twin chose.
Provenance = Literal["calibrated", "literature", "assumed", "grower", "twin"]


class Devices(BaseModel):
    climate: str
    lamps: str
    water: str
    weather: str


class Cover(BaseModel):
    transmission: float
    u_open: float
    u_wind: float
    screen_saving: float
    screen_transmission: float
    reflected_share: float
    cover_temperature_share: float
    condensation_rate: float


class Ventilation(BaseModel):
    opening: list[float]
    rate: list[float]

    @model_validator(mode="after")
    def _monotone(self) -> Ventilation:
        if len(self.opening) != len(self.rate) or len(self.opening) < 2:
            raise ValueError("ventilation.opening and .rate need equal length ≥ 2")
        if any(b <= a for a, b in zip(self.opening, self.opening[1:], strict=False)):
            raise ValueError("ventilation.opening must increase")
        return self


class Lamps(BaseModel):
    par: float
    radiation_eq: float
    efficacy: float

    @property
    def power_w(self) -> float:
        """Electrical power at 100%, W per m² floor."""
        return self.par / self.efficacy


class Pipe(BaseModel):
    tag: str
    label: str
    diameter: float
    length: float


class Transpiration(BaseModel):
    intercept: float
    per_sun_w: float
    per_lamp_w: float
    per_hd: float
    per_pipe_w: float


class Advice(BaseModel):
    light_target_jcm2: float
    dark_min_h: int
    wake_lamp_h: float
    sun_max_w: float
    humid_rh: float
    humid_hours_target: float
    vent_open_pct: float
    dry_rh: float
    co2_lamp_floor_ppm: float


class GreenhouseConfig(BaseModel):
    devices: Devices
    cover: Cover
    ventilation: Ventilation
    lamps: Lamps
    pipes: list[Pipe]
    scale_area_m2: float
    transpiration: Transpiration
    advice: Advice
    #: dotted path (``cover.u_open``, or a whole block such as ``ventilation``)
    #: → where that number comes from
    provenance: dict[str, Provenance]

    @model_validator(mode="after")
    def _provenance_points_at_settings(self) -> GreenhouseConfig:
        for path in self.provenance:
            node: object = self
            for part in path.split("."):
                if not hasattr(node, part):
                    raise ValueError(f"provenance names an unknown setting: {path}")
                node = getattr(node, part)
        return self

    def source_of(self, path: str) -> Provenance:
        """The provenance of ``path``, or of the block it sits in."""
        while path:
            if path in self.provenance:
                return self.provenance[path]
            path = path.rpartition(".")[0]
        raise KeyError(f"no provenance for {path!r}")


def load_greenhouse(yaml_path: Path) -> GreenhouseConfig:
    """The ``greenhouse`` block of red's metadata. Raises when it is absent."""
    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    if "greenhouse" not in raw:
        raise ValueError(f"no 'greenhouse' block in {yaml_path}")
    return GreenhouseConfig(**raw["greenhouse"])
