"""Translations: language choice, number formatting, and catalogues kept in step.

Every catalogue folder (the shared shell's, each twin's) must define the same
keys in English, German and Dutch, with the same {placeholders} — a German
string that drops {target} would print a hole — and the same literature
markers, so no language loses a citation.
"""

import re
import string
from pathlib import Path

import pytest

from wp6_data.shared import i18n

SRC = Path(__file__).parents[1] / "src/wp6_data"
# The shared shell's catalogues, and every twin that brings its own.
FOLDERS = [
    i18n.SHARED_DIR,
    *sorted(p.parent for p in SRC.glob("*/i18n/en.yaml") if p.parent != i18n.SHARED_DIR),
]
MARKER = re.compile(r"\[\^([a-z0-9_]+)\]")
#: a word with a popup behind it: [text](cfg|tag|mod|term:id) — the kind and id must match
POPUP = re.compile(r"\]\(((?:cfg|tag|mod|term):[a-z0-9_]+)\)")


def _fields(text: str) -> set[str]:
    return {name for _, name, _, _ in string.Formatter().parse(text) if name}


@pytest.fixture
def loaded():
    i18n.load(*FOLDERS)
    yield
    i18n.load(*FOLDERS)


@pytest.mark.parametrize("folder", FOLDERS, ids=lambda f: f.parent.name)
@pytest.mark.parametrize("lang", [lang for lang in i18n.LANGS if lang != i18n.DEFAULT])
def test_every_language_has_the_same_keys(folder, lang):
    en = i18n.read_catalogue(folder, i18n.DEFAULT)
    other = i18n.read_catalogue(folder, lang)
    assert en, f"no English catalogue in {folder}"
    assert sorted(en.keys() - other.keys()) == [], f"missing in {lang}"
    assert sorted(other.keys() - en.keys()) == [], f"only in {lang}"


@pytest.mark.parametrize("folder", FOLDERS, ids=lambda f: f.parent.name)
@pytest.mark.parametrize("lang", [lang for lang in i18n.LANGS if lang != i18n.DEFAULT])
def test_translations_keep_placeholders_and_citations(folder, lang):
    en = i18n.read_catalogue(folder, i18n.DEFAULT)
    other = i18n.read_catalogue(folder, lang)
    for key, text in en.items():
        if key not in other:
            continue
        assert _fields(other[key]) == _fields(text), f"{lang} {key}: placeholders differ"
        assert sorted(MARKER.findall(other[key])) == sorted(MARKER.findall(text)), (
            f"{lang} {key}: citations differ"
        )
        assert sorted(POPUP.findall(other[key])) == sorted(POPUP.findall(text)), (
            f"{lang} {key}: popups differ"
        )


def test_html_markup_only_in_html_keys():
    for folder in FOLDERS:
        for lang in i18n.LANGS:
            for key, text in i18n.read_catalogue(folder, lang).items():
                if not key.endswith("_html"):
                    assert not re.search(r"<[a-z/]", text), f"{lang} {key} carries markup"


def test_a_key_defined_twice_is_refused(tmp_path):
    (tmp_path / "en.yaml").write_text("nav:\n  home: Mine\n", encoding="utf-8")
    with pytest.raises(ValueError, match="defined twice"):
        i18n.load(i18n.SHARED_DIR, tmp_path)
    i18n.load(*FOLDERS)


@pytest.mark.parametrize(
    ("query", "cookie", "header", "expected"),
    [
        ("de", "nl", "en", "de"),
        (None, "nl", "de", "nl"),
        (None, None, "nl-NL,nl;q=0.9,en;q=0.8", "nl"),
        (None, None, "fr-FR,fr;q=0.9,de;q=0.5", "de"),
        (None, None, "fr", "en"),
        ("xx", None, None, "en"),
        (None, None, None, "en"),
    ],
)
def test_negotiate_prefers_choice_then_cookie_then_browser(query, cookie, header, expected):
    assert i18n.negotiate(query, cookie, header) == expected


def test_numbers_follow_the_language(loaded):
    for lang, expected in [("en", "1,234.5"), ("de", "1.234,5"), ("nl", "1.234,5")]:
        token = i18n.use(lang)
        try:
            assert i18n.number(1234.5, ",.1f") == expected
        finally:
            i18n.reset(token)


def test_t_fills_placeholders_and_falls_back_to_english(loaded):
    token = i18n.use("de")
    try:
        assert i18n.t("nav.data_source", label="GTL") == "Datenquelle: GTL"
        assert i18n.t("greenhouse.ui.wi.vents_label", pct=80) == "Lüftungen bei 80%"
        assert i18n.t("no.such.key") == "no.such.key"
    finally:
        i18n.reset(token)


def test_th_escapes_plain_strings_but_not_html_keys(loaded):
    token = i18n.use("en")
    try:
        assert i18n.th("red_home.energy.button") == "Lamps &amp; Heating"
        assert "<strong>" in i18n.th("greenhouse.ui.tm.need_html", target=1000, h=8, dark=6)
    finally:
        i18n.reset(token)


def test_bundle_is_english_where_a_language_lacks_a_key(loaded):
    token = i18n.use("nl")
    try:
        ui = i18n.bundle("greenhouse.ui")
        assert ui["comp.vents"] == "Luchtramen"
        assert "comp.vents" in ui and not any(k.startswith("greenhouse.") for k in ui)
    finally:
        i18n.reset(token)


def test_the_switch_shows_only_where_the_twin_offers_languages():
    from wp6_data.shared.templates.pages import render_language_switch

    was = i18n.offered()
    try:
        i18n.offer(False)
        assert render_language_switch() == ""
        i18n.offer(True)
        switch = render_language_switch()
        assert all(f'data-lang="{lang}"' in switch for lang in i18n.LANGS)
    finally:
        i18n.offer(was)
