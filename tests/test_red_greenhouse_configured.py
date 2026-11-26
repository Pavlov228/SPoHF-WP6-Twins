"""What the /greenhouse page tells visitors about its numbers, without file names or ids.

A configured number reads "configured" and opens the values behind it, each
with where it comes from; a LetsGrow series or module reads by its plain name
with the id in a popup. These tests keep every such link resolvable, in every
language, and keep technical names out of the prose itself.
"""

import re
from pathlib import Path

import pytest
import yaml

from wp6_data.red.greenhouse import explain
from wp6_data.red.greenhouse.config import GreenhouseConfig, load_greenhouse
from wp6_data.red.greenhouse.state import COLUMNS
from wp6_data.shared import i18n

RED = Path(__file__).parents[1] / "src/wp6_data/red"
CFG = load_greenhouse(RED / "metadata.yaml")
FOLDERS = [i18n.SHARED_DIR, RED / "i18n"]
POPUP = re.compile(r"\[([^\]]+)\]\((cfg|tag|mod|term):([a-z0-9_]+)\)")
#: what a visitor should never have to read in the prose
TECHNICAL = re.compile(r"metadata\.yaml|\.py\b|`|\blg_[a-z_]+|letsgrow-[a-z0-9-]+")


@pytest.fixture(autouse=True)
def _catalogues():
    i18n.load(*FOLDERS)
    yield
    i18n.load(*FOLDERS)


def _in(lang: str):
    return i18n.use(lang)


def _greenhouse_prose(lang: str) -> dict[str, str]:
    cat = i18n.read_catalogue(RED / "i18n", lang)
    return {
        k: v
        for k, v in cat.items()
        if k.startswith("greenhouse.") and not k.startswith(("greenhouse.tag.", "greenhouse.pipe."))
    }


@pytest.mark.parametrize("lang", i18n.LANGS)
def test_every_explanation_fills_in_every_language(lang):
    token = _in(lang)
    try:
        notes = explain.catalogue(CFG)
    finally:
        i18n.reset(token)
    assert set(notes) == set(explain.ENTRIES)
    for key, note in notes.items():
        assert all(note[f].strip() for f in explain.FIELDS), (lang, key)
        assert "{" not in "".join(note[f] for f in explain.FIELDS), (lang, key)


@pytest.mark.parametrize("lang", i18n.LANGS)
def test_no_file_names_or_ids_in_what_visitors_read(lang):
    """Ids live behind popups; the prose itself names things plainly."""
    token = _in(lang)
    try:
        notes = explain.catalogue(CFG)
        rendered = {k: " ".join(n[f] for f in explain.FIELDS) for k, n in notes.items()}
    finally:
        i18n.reset(token)
    texts = {**_greenhouse_prose(lang), **{f"explain:{k}": v for k, v in rendered.items()}}
    for key, text in texts.items():
        visible = POPUP.sub(lambda m: m[1], text)
        assert not TECHNICAL.search(visible), f"{lang} {key}: {TECHNICAL.search(visible)[0]!r}"


@pytest.mark.parametrize("lang", i18n.LANGS)
def test_every_popup_link_resolves(lang):
    groups = set(explain.CONFIG_GROUPS) | {"all"}
    tags = {tag for (_, tag) in COLUMNS}
    roles = set(CFG.devices.model_dump())
    cat = i18n.read_catalogue(RED / "i18n", lang)
    for key, text in _greenhouse_prose(lang).items():
        for _, kind, ref in POPUP.findall(text):
            if kind == "cfg":
                assert ref in groups, (lang, key, ref)
            elif kind == "tag":
                assert ref in tags, (lang, key, ref)
                assert f"greenhouse.tag.{ref}" in cat, (lang, ref)
            elif kind == "mod":
                assert ref in roles, (lang, key, ref)
            else:
                assert f"greenhouse.ui.term.{ref}" in cat, (lang, key, ref)


@pytest.mark.parametrize("lang", i18n.LANGS)
def test_every_series_and_module_has_a_plain_name(lang):
    cat = i18n.read_catalogue(RED / "i18n", lang)
    for _, sources in explain.ENTRIES.values():
        for role, tags in sources:
            assert f"greenhouse.module.{role}.name" in cat, (lang, role)
            assert f"greenhouse.module.{role}.about" in cat, (lang, role)
            for tag in tags:
                assert f"greenhouse.tag.{tag}" in cat, (lang, tag)


@pytest.mark.parametrize("lang", i18n.LANGS)
def test_configured_values_carry_their_source(lang):
    token = _in(lang)
    try:
        view = explain.configured(CFG)
    finally:
        i18n.reset(token)
    assert set(view["groups"]) == set(explain.CONFIG_GROUPS)
    assert set(view["sources"]) == set(explain.PROVENANCE)
    for group, g in view["groups"].items():
        assert g["title"] and g["rows"], group
        for row in g["rows"]:
            assert row["label"] and row["value"], (group, row)
            assert row["source"] in explain.PROVENANCE, (group, row)
    assert set(view["modules"]) == set(CFG.devices.model_dump())


def test_configured_shows_the_numbers_the_balance_uses():
    token = _in("en")
    try:
        rows = explain.configured(CFG)["groups"]["cover"]["rows"]
    finally:
        i18n.reset(token)
    u = next(r for r in rows if "(U)" in r["label"])
    assert u["value"].startswith(f"{CFG.cover.u_open:g}")
    assert u["source"] == CFG.source_of("cover.u_open")


def test_every_listed_setting_has_a_provenance():
    for fields in explain.CONFIG_GROUPS.values():
        for path, _ in fields:
            assert CFG.source_of(path) in explain.PROVENANCE, path
    for path in ("ventilation", "pipes", "lamps.efficacy"):
        assert CFG.source_of(path) in explain.PROVENANCE, path


def test_provenance_must_name_a_real_setting():
    raw = yaml.safe_load((RED / "metadata.yaml").read_text(encoding="utf-8"))["greenhouse"]
    raw["provenance"] = {**raw["provenance"], "cover.u_typo": "assumed"}
    with pytest.raises(ValueError, match="unknown setting"):
        GreenhouseConfig(**raw)


def test_server_side_prose_renders_popup_buttons_and_escapes():
    html = explain.render_terms("all [configured](cfg:cover) & [a<b](term:research)")
    assert '<button type="button" class="gh-term" data-kind="cfg" data-id="cover">' in html
    assert "a&lt;b</button>" in html
    assert "[configured]" not in html
