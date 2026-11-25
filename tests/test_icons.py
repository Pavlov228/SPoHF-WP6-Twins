"""The shared icon library: every icon renders as a themeable, decorative SVG."""

import pytest

from wp6_data.shared.templates.icons import ICON_NAMES, icon


@pytest.mark.parametrize("name", ICON_NAMES)
def test_every_icon_is_a_decorative_svg_in_current_color(name):
    svg = icon(name)

    assert svg.startswith("<svg") and svg.endswith("</svg>")
    assert 'aria-hidden="true"' in svg
    assert "currentColor" in svg


def test_size_sets_width_and_height():
    svg = icon("download", size=18)

    assert 'width="18" height="18"' in svg


def test_without_size_it_is_left_to_css():
    svg = icon("chart-line", cls="ct-icon")

    assert 'class="ct-icon"' in svg
    assert "width=" not in svg.split(">")[0]


def test_unknown_icon_fails_loudly():
    with pytest.raises(KeyError):
        icon("no-such-icon")
