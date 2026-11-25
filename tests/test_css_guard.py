"""Guard against text that turns invisible because it read Pico's ``--pico-color``.

Pico redefines ``--pico-color`` per component: inside a ``<button>`` it is the
button's white label colour. A transparent-background button (a tab, a toggle)
that reads ``var(--pico-color)`` paints white text on a white page in light
mode. It happened to the table headers (b85cbc3) and again to the explore tabs'
hover. ``--dashboard-text`` is fixed at the root to the page's text colour; read
that instead.
"""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCANNED = [
    *REPO.glob("static/**/*.css"),
    *REPO.glob("static/**/*.js"),
    *REPO.glob("src/**/*.py"),
    *REPO.glob("src/**/*.css"),
    *REPO.glob("src/**/*.js"),
]
# The one place allowed to read it: where --dashboard-text is defined.
DEFINITION = re.compile(r"--dashboard-text:\s*var\(--pico-color\)")
CSS_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)


def _offenders(text: str) -> list[str]:
    code = DEFINITION.sub("", CSS_COMMENT.sub("", text))
    return [line.strip() for line in code.splitlines() if "var(--pico-color)" in line]


def test_no_style_reads_pico_color():
    offenders = {
        str(path.relative_to(REPO)): found
        for path in SCANNED
        if (found := _offenders(path.read_text(encoding="utf-8")))
    }

    assert SCANNED, "scanned no files: check the globs"
    assert not offenders, (
        "var(--pico-color) is white inside a button; use var(--dashboard-text): "
        f"{offenders}"
    )


def test_the_guard_catches_a_read_and_allows_the_definition():
    assert _offenders("button:hover { color: var(--pico-color); }")
    assert not _offenders(":root { --dashboard-text: var(--pico-color); }")
    assert not _offenders("/* never read var(--pico-color) */")


def test_dashboard_text_is_defined_at_the_root():
    css = (REPO / "static/shared/base.css").read_text(encoding="utf-8")

    assert DEFINITION.search(css)
