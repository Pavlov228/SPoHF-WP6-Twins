"""The assumption register and the model page (issue 070).

The register points at values instead of copying them, so these tests are what
keeps it honest: every pointer must resolve, and every parameter must have a
plain-language meaning on the page.
"""

import pytest
import yaml

from wp6_data.blue.irrigation.assumptions import (
    SOURCES,
    format_value,
    read_assumptions,
    resolve,
)
from wp6_data.blue.irrigation.config import CONFIG_PATH, IrrigationConfig
from wp6_data.blue.irrigation.model import MODEL_PATH, read_model
from wp6_data.blue.irrigation.water_balance import PARAM_DOCS, SINGLE_BUCKET, TWO_BUCKET


def _config() -> IrrigationConfig:
    with CONFIG_PATH.open(encoding="utf-8") as f:
        return IrrigationConfig(**yaml.safe_load(f))


def test_every_assumption_points_at_a_value_that_exists():
    config, model = _config(), read_model(MODEL_PATH)

    for a in read_assumptions():
        if a.value is None:
            continue
        assert a.value.split(".")[0] in SOURCES, a.id
        assert format_value(resolve(a.value, config, model)), a.id


def test_assumption_ids_are_unique_and_entries_are_complete():
    assumptions = read_assumptions()
    ids = [a.id for a in assumptions]

    assert len(ids) == len(set(ids))
    for a in assumptions:
        assert a.title and a.statement and a.basis and a.impact and a.test, a.id


def test_the_dose_area_assumption_is_the_one_the_model_uses():
    config, model = _config(), read_model(MODEL_PATH)
    dose = next(a for a in read_assumptions() if a.id == "dose-area")

    assert resolve(dose.value, config, model) == model.area_per_plant_m2


def test_a_pointer_outside_the_known_sources_is_refused():
    with pytest.raises(ValueError, match="must start with"):
        resolve("settings.anything", _config(), read_model(MODEL_PATH))


def test_every_model_parameter_has_a_plain_language_meaning():
    for name in (*SINGLE_BUCKET.params, *TWO_BUCKET.params):
        meaning, unit = PARAM_DOCS[name]
        assert meaning and unit


@pytest.mark.asyncio
async def test_model_page_shows_parameters_scores_and_assumptions():
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient

    from wp6_data.blue.routes import irrigation_model
    from wp6_data.shared.auth import verify_session_user
    from wp6_data.shared.templates.config import configure_dashboard

    configure_dashboard("blue")
    app = FastAPI()
    app.include_router(irrigation_model.router)
    app.dependency_overrides[verify_session_user] = lambda: "test-user"
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/irrigation/model")

    assert response.status_code == 200
    for a in read_assumptions():
        assert f'id="{a.id}"' in response.text
    for name in read_model(MODEL_PATH).params:
        assert f"<code>{name}</code>" in response.text
    assert "Outcomes inside the 80 % band" in response.text
