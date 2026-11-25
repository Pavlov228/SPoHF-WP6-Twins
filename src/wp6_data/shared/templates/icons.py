"""Inline SVG icons, painted in ``currentColor`` so they follow the theme.

Inline rather than ``<img>`` so an icon takes its colour from the text around
it (dark mode, a button's active state) without a second file per colour.
Icons are decorative: the element they sit in carries the label.
"""

# Outline icons on a 24-unit grid, stroked like Feather icons.
_OUTLINE = (
    'viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"'
    ' stroke-linecap="round" stroke-linejoin="round"'
)
# Small glyphs on a 12-unit grid; each element sets its own paint.
_GLYPH = 'viewBox="0 0 12 12"'

_ICONS: dict[str, tuple[str, str]] = {
    "code": (
        _OUTLINE,
        '<polyline points="16 18 22 12 16 6"/>'
        '<polyline points="8 6 2 12 8 18"/>',
    ),
    "docs": (
        _OUTLINE,
        '<path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/>'
        '<path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/>',
    ),
    "download": (
        _OUTLINE,
        '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>'
        '<polyline points="7 10 12 15 17 10"/>'
        '<line x1="12" y1="15" x2="12" y2="3"/>',
    ),
    "chart-line": (
        _GLYPH,
        '<polyline points="1,9 4,5 7,7 11,2" fill="none" stroke="currentColor"'
        ' stroke-width="1.3" stroke-linejoin="round" stroke-linecap="round"/>',
    ),
    "chart-scatter": (
        _GLYPH,
        '<circle cx="2.5" cy="8.5" r="1.1" fill="currentColor"/>'
        '<circle cx="6" cy="5" r="1.1" fill="currentColor"/>'
        '<circle cx="9.5" cy="6.5" r="1.1" fill="currentColor"/>'
        '<circle cx="8" cy="2.5" r="1.1" fill="currentColor"/>',
    ),
    "chart-box": (
        _GLYPH,
        '<line x1="6" y1="1" x2="6" y2="11" stroke="currentColor" stroke-width="1.1"/>'
        '<rect x="3" y="4" width="6" height="4.5" fill="none" stroke="currentColor"'
        ' stroke-width="1.1"/>'
        '<line x1="3" y1="6.2" x2="9" y2="6.2" stroke="currentColor"'
        ' stroke-width="1.1"/>',
    ),
}

ICON_NAMES = tuple(_ICONS)


def icon(name: str, *, size: int | None = None, cls: str = "") -> str:
    """Render the named icon as an inline ``<svg>``.

    Args:
        name: One of :data:`ICON_NAMES`.
        size: Width and height in px. Omit to size it from CSS via ``cls``.
        cls: CSS class for the ``<svg>`` element.
    """
    attrs, body = _ICONS[name]
    size_attrs = f' width="{size}" height="{size}"' if size is not None else ""
    class_attr = f' class="{cls}"' if cls else ""
    return f'<svg{class_attr} {attrs}{size_attrs} aria-hidden="true">{body}</svg>'
