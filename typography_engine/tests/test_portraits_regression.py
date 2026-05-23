"""Regression test over a diverse set of portraits.

Runs each fixture image through the full analysis + tonal-portrait pipeline and
asserts the core invariants that earlier batch testing established:

  * the render produces text (non-empty),
  * every emitted token is a whole approved word -- no fragments / floating
    letters,
  * the SVG is well-formed XML,
  * no error-level warnings are raised.

Fixture images are fetched on demand (see tests/fixtures); if none are
available (offline CI), the whole module is skipped.

Run with:  cd typography_engine && python -m pytest tests/test_portraits_regression.py -q
"""
from __future__ import annotations

from xml.etree import ElementTree as ET

import pytest

from app.config import RenderConfig
from app.pipeline.analyze import analyze_image
from app.pipeline.portrait import build_portrait
from app.pipeline.warnings import WarningCollector
from tests.fixtures import ensure_portraits

_WORDS = ["WOW", "AMAZING", "LOVING", "KIND", "BEAUTIFUL"]

_PORTRAITS = ensure_portraits()
if not _PORTRAITS:
    pytest.skip("No portrait fixtures available (offline?).", allow_module_level=True)


def _is_whole_words(token: str, words) -> bool:
    """True if `token` is exactly a concatenation of whole approved words."""
    n = len(token)
    ok = [False] * (n + 1)
    ok[0] = True
    for i in range(n):
        if not ok[i]:
            continue
        for w in words:
            if token.startswith(w, i):
                ok[i + len(w)] = True
    return ok[n]


@pytest.mark.parametrize("name", sorted(_PORTRAITS), ids=sorted(_PORTRAITS))
def test_portrait_renders_whole_words(name):
    warns = WarningCollector()
    an = analyze_image(_PORTRAITS[name].read_bytes(), RenderConfig(), warns)
    res = build_portrait(an, _WORDS, RenderConfig(), warns)

    assert res.runs, f"{name}: render produced no text"

    wordset = set(_WORDS)
    for run in res.runs:
        for token in run.text.split():
            assert _is_whole_words(token, wordset), f"{name}: non-word fragment {token!r}"

    ET.fromstring(res.svg)  # well-formed XML
    assert not warns.has_errors(), f"{name}: errors {[w['code'] for w in warns.as_list() if w['severity']=='error']}"
