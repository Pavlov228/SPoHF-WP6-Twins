"""/version: the running build, polled by open pages to badge the favicon."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from wp6_data.shared.routes import health
from wp6_data.shared.templates import config as templates_config

BUILD = "v1.0.1-abc1234"


@pytest.fixture
def client(monkeypatch):
    app = FastAPI()
    app.include_router(health.router)
    monkeypatch.setattr(templates_config, "_app_version", BUILD)
    return TestClient(app)


def test_reports_the_configured_build(client):
    assert client.get("/version").json() == {"version": BUILD}


def test_is_never_cached(client):
    assert client.get("/version").headers["cache-control"] == "no-store"


def test_unbuilt_run_reports_no_version(client, monkeypatch):
    monkeypatch.setattr(templates_config, "_app_version", "")
    assert client.get("/version").json() == {"version": None}
