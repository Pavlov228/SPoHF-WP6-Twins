"""How hard red's heating ran, as a relative index — the line gas cost follows.

LetsGrow reports no heating switch and no gas meter, only pipe and air
temperatures. A heating pipe gives off heat roughly in proportion to
``(pipe − air) ** exponent`` (1.25, the standard relation for greenhouse
heating pipes), so the **heat-output index** sums that over the circuits and
the hours:

    index per day = Σ hours Σ circuits max(pipe − air, 0) ** exponent

It grows with both how long and how hot the pipes ran. Hours alone do not: red
keeps its pipes warm ~22-24 hours a day for most of the year (minimum-pipe
operation), so a count of heating hours is flat while the gas bill is not.

Relative, not kWh: the circuits are weighted equally, and no pipe length,
diameter, boiler efficiency or gas price is in it. Those are constants a later
step can multiply in, without changing the line's shape.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]
import yaml
from pydantic import BaseModel


class HeatOutputConfig(BaseModel):
    """Where LetsGrow reports the heating, from ``metadata.yaml``'s ``heating``."""

    device: str
    air: str  # sensor tag: greenhouse air temperature, °C
    pipes: list[str]  # sensor tags: one measured pipe temperature per circuit, °C
    exponent: float


def load_heat_output(yaml_path: Path) -> HeatOutputConfig:
    """The ``heating`` block of red's metadata. Raises when it is absent.

    No default: a guessed tag would draw as a greenhouse that was never heated.
    """
    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    if "heating" not in raw:
        raise ValueError(f"no 'heating' block in {yaml_path}")
    return HeatOutputConfig(**raw["heating"])


def daily_heat_output(hourly: pd.Series) -> pd.DataFrame:
    """The hourly index summed per UTC day, **complete days only**.

    ``hourly`` is ``red.tsdb.fetch_heat_output``'s series. A day missing hours
    would sum short and draw as a mild day, so it is left out instead.

    Returns ``date, heat_index``.
    """
    columns = ["date", "heat_index"]
    if hourly.empty:
        return pd.DataFrame(columns=columns)
    by_day = hourly.groupby(hourly.index.date).agg(["sum", "count"])
    complete = by_day[by_day["count"] == 24]
    return pd.DataFrame({"date": complete.index, "heat_index": complete["sum"].to_numpy()})
