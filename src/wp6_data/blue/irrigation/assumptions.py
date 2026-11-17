"""The irrigation twin's assumption register (``assumptions.yaml``).

Each assumption points at where its value lives, rather than copying the
number, so the register can't drift from the config or the shipped model.
:func:`resolve` looks the value up.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from wp6_data.blue.irrigation.config import IrrigationConfig
    from wp6_data.blue.irrigation.model import ShippedModel

ASSUMPTIONS_PATH = Path(__file__).with_name("assumptions.yaml")
#: Where a ``value`` path may point.
SOURCES = ("config", "model", "levels")


@dataclass(frozen=True)
class Assumption:
    id: str
    title: str
    statement: str
    basis: str
    impact: str
    test: str
    value: str | None = None
    unit: str = ""
    refs: tuple[str, ...] = ()


def read_assumptions(path: Path = ASSUMPTIONS_PATH) -> list[Assumption]:
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    out = []
    for entry in raw:
        entry = {k: (v.strip() if isinstance(v, str) else v) for k, v in entry.items()}
        entry["refs"] = tuple(entry.get("refs") or ())
        out.append(Assumption(**entry))
    return out


@lru_cache(maxsize=1)
def load_assumptions() -> list[Assumption]:
    return read_assumptions()


def resolve(path: str, config: IrrigationConfig, model: ShippedModel) -> object:
    """The current value behind ``config.x.y``, ``model.x.y`` or ``levels.x``."""
    from wp6_data.blue.irrigation.advice import levels

    source, _, rest = path.partition(".")
    if source not in SOURCES or not rest:
        raise ValueError(f"assumption value {path!r} must start with one of {SOURCES}")
    node: object = {"config": config, "model": model, "levels": levels(model, config)}[source]
    for part in rest.split("."):
        node = node[part] if isinstance(node, dict) else getattr(node, part)
    return node


def format_value(value: object) -> str:
    if isinstance(value, time):
        return f"{value:%H:%M}"
    if isinstance(value, float):
        return f"{value:.3g}"
    return str(value)
