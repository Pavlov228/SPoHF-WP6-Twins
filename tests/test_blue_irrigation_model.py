"""The shipped forecast model (ADR 0009): the committed file must match the code."""

import pytest
import yaml

from wp6_data.blue.irrigation.config import CONFIG_PATH, IrrigationConfig
from wp6_data.blue.irrigation.model import (
    MODEL_PATH,
    ShippedModel,
    check_against,
    production_spec,
    read_model,
    write_model,
)


def _config() -> IrrigationConfig:
    with CONFIG_PATH.open(encoding="utf-8") as f:
        return IrrigationConfig(**yaml.safe_load(f))


def test_the_committed_model_matches_the_code_and_config():
    """Fails when the model's parameters or config change without re-running
    `wp6-blue-irrigation calibrate` and committing model.yaml."""
    model = read_model(MODEL_PATH)

    check_against(model, _config())
    assert len(model.band_low) == len(model.band_high)
    assert all(lo <= hi for lo, hi in zip(model.band_low, model.band_high, strict=True))


def test_a_model_with_other_parameters_is_refused():
    config = _config()
    model = read_model(MODEL_PATH)
    stale = ShippedModel(**(model.__dict__ | {"params": model.params | {"irr_eff": 0.01}}))

    with pytest.raises(ValueError, match="calibrate"):
        check_against(stale, config)


def test_write_then_read_round_trips(tmp_path):
    model = read_model(MODEL_PATH)
    path = tmp_path / "model.yaml"

    write_model(model, path)

    assert read_model(path) == model


def test_the_production_spec_ties_the_dose_to_rain():
    spec = production_spec(_config())

    assert "irr_eff" not in spec.params
    assert spec.area_per_plant_m2 == _config().model.area_per_plant_m2
