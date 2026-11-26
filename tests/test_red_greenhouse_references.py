"""The /greenhouse literature footnotes: every citation resolves, in every language.

references.yaml holds the bibliography and, per explanation and advice, the ids
it cites; the catalogues (red/i18n/<lang>.yaml) carry the same ids as ``[^id]``
markers in their prose. These tests keep the two in step, so a claim on the
page never points at a missing source and no language drops a citation.
"""

import re
from pathlib import Path

import pytest
import yaml

RED = Path(__file__).parents[1] / "src/wp6_data/red"
REFS = yaml.safe_load((RED / "greenhouse/references.yaml").read_text(encoding="utf-8"))
LANGS = ("en", "de", "nl")
MARKER = re.compile(r"\[\^([a-z0-9_]+)\]")
CHECKED = {"full", "abstract", "record"}


def _flatten(tree: dict, prefix: str = "") -> dict[str, str]:
    out: dict[str, str] = {}
    for key, value in tree.items():
        if isinstance(value, dict):
            out.update(_flatten(value, f"{prefix}{key}."))
        else:
            out[f"{prefix}{key}"] = str(value)
    return out


def _catalogue(lang: str) -> dict[str, str] | None:
    path = RED / "i18n" / f"{lang}.yaml"
    if not path.exists():
        return None
    return _flatten(yaml.safe_load(path.read_text(encoding="utf-8")) or {})


def _cited(catalogue: dict[str, str], prefix: str) -> set[str]:
    """Every marker id under ``prefix`` (all fields of one explanation or advice)."""
    return {
        ref
        for key, text in catalogue.items()
        if key.startswith(f"{prefix}.")
        for ref in MARKER.findall(text)
    }


def _footnotes() -> list[tuple[str, list[str]]]:
    fn = REFS["footnotes"]
    return [(f"greenhouse.explain.{k}", v) for k, v in fn["explain"].items()] + [
        (f"greenhouse.advice.{k}", v) for k, v in fn["advice"].items()
    ]


@pytest.mark.parametrize("ref_id", sorted(REFS["references"]))
def test_every_reference_is_complete_and_links_out(ref_id):
    ref = REFS["references"][ref_id]
    assert re.fullmatch(r"[a-z0-9_]+", ref_id), "ids are lowercase snake, as markers need"
    assert ref["label"].strip() and ref["cite"].strip()
    assert ref["url"].startswith("https://")
    assert ref["checked"] in CHECKED


def test_footnotes_cite_only_known_references_and_every_reference_is_used():
    known = set(REFS["references"])
    used = {ref for _, ids in _footnotes() for ref in ids}
    assert used <= known, f"unknown ids: {sorted(used - known)}"
    assert known <= used, f"never cited: {sorted(known - used)}"


@pytest.mark.parametrize("lang", LANGS)
def test_every_marker_in_the_catalogue_resolves(lang):
    catalogue = _catalogue(lang)
    if catalogue is None:
        pytest.skip(f"no {lang} catalogue yet")
    markers = {ref for text in catalogue.values() for ref in MARKER.findall(text)}
    assert markers <= set(REFS["references"]), sorted(markers - set(REFS["references"]))


@pytest.mark.parametrize("lang", LANGS)
def test_each_language_cites_what_the_footnotes_say(lang):
    """Same sources per explanation in en, de and nl — a translation keeps its citations."""
    catalogue = _catalogue(lang)
    if catalogue is None:
        pytest.skip(f"no {lang} catalogue yet")
    for prefix, ids in _footnotes():
        assert _cited(catalogue, prefix) == set(ids), f"{lang}: {prefix}"
