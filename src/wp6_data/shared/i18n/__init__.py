"""Translations: English, German and Dutch.

Strings live in YAML catalogues, one file per language (``en.yaml``,
``de.yaml``, ``nl.yaml``): this package's own for the shared shell, and one
folder per twin for its pages (``TwinConfig.i18n_dir``). Nested keys flatten to
dotted ids — ``nav.home`` — and a value may carry ``{name}`` placeholders.

The language is chosen per request (:func:`negotiate`: ``?lang=``, then the
``lang`` cookie, then the browser's ``Accept-Language``) and held in a context
variable, so :func:`t` needs no request passed around — the same way the shell
knows the current user. A key missing in German or Dutch falls back to English,
then to the key itself; a test keeps the three catalogues in step.

Catalogue strings are plain text: :func:`th` escapes one for HTML, and the
browser inserts its bundle as text. Only keys ending in ``_html`` may carry
markup, and are inserted as they are.

Numbers passed as placeholders are written the language's way: ``23.5`` in
English, ``23,5`` in German and Dutch, thousands the other way round.
"""

from __future__ import annotations

import html
import string
from contextvars import ContextVar
from pathlib import Path

import yaml

LANGS: tuple[str, ...] = ("en", "de", "nl")
DEFAULT = "en"
COOKIE = "lang"
NAMES = {"en": "English", "de": "Deutsch", "nl": "Nederlands"}
SHARED_DIR = Path(__file__).parent

_lang: ContextVar[str] = ContextVar("_lang", default=DEFAULT)
_catalogues: dict[str, dict[str, str]] = {lang: {} for lang in LANGS}
# Whether this process's twin offers a choice of language (it brings its own
# catalogues). One twin per process, as with the dashboard identity.
_offered = False


def offer(on: bool) -> None:
    """Offer the language choice (switch + negotiation) in this process, or not."""
    global _offered
    _offered = on


def offered() -> bool:
    """True when visitors can choose a language here; otherwise everything is English."""
    return _offered


def _flatten(tree: dict, prefix: str = "") -> dict[str, str]:
    out: dict[str, str] = {}
    for key, value in tree.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(_flatten(value, f"{path}."))
        else:
            out[path] = str(value)
    return out


def read_catalogue(folder: Path, lang: str) -> dict[str, str]:
    """One language's catalogue from ``folder``, flattened; empty if the file is absent."""
    path = folder / f"{lang}.yaml"
    if not path.exists():
        return {}
    return _flatten(yaml.safe_load(path.read_text(encoding="utf-8")) or {})


def load(*folders: Path | None) -> None:
    """Load the catalogues of ``folders`` (shared first, then a twin's).

    A key defined in two folders is a mistake — the twin would silently
    rename a shared string — so it raises.
    """
    for lang in LANGS:
        merged: dict[str, str] = {}
        for folder in folders:
            if folder is None:
                continue
            cat = read_catalogue(folder, lang)
            clash = merged.keys() & cat.keys()
            if clash:
                raise ValueError(f"i18n keys defined twice ({lang}): {sorted(clash)[:5]}")
            merged.update(cat)
        _catalogues[lang] = merged


def current() -> str:
    """The language of the request being served."""
    return _lang.get()


def use(lang: str):
    """Set the language for the rest of this context; returns the token to reset it."""
    return _lang.set(lang if lang in LANGS else DEFAULT)


def reset(token) -> None:
    _lang.reset(token)


def negotiate(query: str | None, cookie: str | None, accept_language: str | None) -> str:
    """The language to serve: an explicit choice, the remembered one, the browser's."""
    for choice in (query, cookie):
        if choice and choice.lower() in LANGS:
            return choice.lower()
    best, best_q = DEFAULT, -1.0
    for part in (accept_language or "").split(","):
        tag, _, params = part.strip().partition(";")
        lang = tag.split("-")[0].strip().lower()
        if lang not in LANGS:
            continue
        q = 1.0
        if params.strip().startswith("q="):
            try:
                q = float(params.strip()[2:])
            except ValueError:
                q = 0.0
        if q > best_q:
            best, best_q = lang, q
    return best


def number(value: float, spec: str = "", lang: str | None = None) -> str:
    """``value`` formatted with ``spec``, with the language's decimal and thousands marks."""
    text = format(value, spec)
    if (lang or current()) == "en":
        return text
    return text.translate(str.maketrans({".": ",", ",": "."}))


class _Formatter(string.Formatter):
    def format_field(self, value, format_spec: str) -> str:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return number(value, format_spec)
        return super().format_field(value, format_spec)


_formatter = _Formatter()


def lookup(key: str, lang: str | None = None) -> str | None:
    """The raw template for ``key`` in ``lang``, falling back to English; None if unknown."""
    lang = lang or current()
    return _catalogues.get(lang, {}).get(key) or _catalogues[DEFAULT].get(key)


def t(key: str, /, **params) -> str:
    """``key`` in the current language, its ``{placeholders}`` filled from ``params``."""
    template = lookup(key)
    if template is None:
        return key
    return _formatter.format(template, **params) if params else template


def th(key: str, /, **params) -> str:
    """:func:`t`, escaped for HTML text or an attribute (``_html`` keys are left as markup)."""
    text = t(key, **params)
    return text if key.endswith("_html") else html.escape(text, quote=True)


def bundle(prefix: str) -> dict[str, str]:
    """Every string under ``prefix`` in the current language (English where missing).

    For pages that build their text in the browser: the bundle is embedded in
    the page, keys shortened by the prefix.
    """
    head = f"{prefix}."
    out: dict[str, str] = {}
    for lang in (DEFAULT, current()):
        out.update({k[len(head) :]: v for k, v in _catalogues[lang].items() if k.startswith(head)})
    return out


def keys(lang: str) -> set[str]:
    """Every key the loaded catalogue for ``lang`` defines (for tests)."""
    return set(_catalogues[lang])
