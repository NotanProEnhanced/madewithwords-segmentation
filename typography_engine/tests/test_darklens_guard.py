"""Guard against the dark-lens (sunglasses) detector blacking out REAL eyes.

The Sculpt renderer suppresses the eyes of a face wearing tinted/mirrored lenses
and paints an opaque lens instead. That gate keys off eye-region darkness and
brightness, and a brightly-lit REAL face can look bright too -- so a too-loose gate
once painted opaque black discs over ordinary open eyes on single-subject memorial
portraits.

The critical invariant -- the one whose violation broke the memorial product -- is
that a brightly-lit REAL face is never flagged dark-lens (its eyes are never blacked
out). That is what these tests pin, across every known bright single-face fixture at
both the preview (SS=1) and full (SS=2) resolutions.

Note on the other direction: whether a genuine sunglasses lens is detected is
framing-dependent (it hinges on where MediaPipe lands the iris landmarks behind the
lens), so it is deliberately NOT asserted here -- a per-fixture "is flagged" check
would be brittle. A see-through lens is a minor cosmetic miss; a blacked-out real eye
is a product-breaking defect, so the guard is intentionally one-sided.

The renderer records its per-face eye classification into the optional `_diag`
dict, so these assertions exercise the REAL gate, not a copy that could drift.
"""
from __future__ import annotations

import pytest

from app.config import RenderConfig
from app.pipeline.analyze import analyze_image
from app.pipeline.displacement import render_displacement_portrait
from app.pipeline.warnings import WarningCollector
from tests.fixtures import ensure_portraits

_WORDS = ["KIND", "LOVE", "FAMILY"]
_PORTRAITS = ensure_portraits()

# Known single-subject portraits WITHOUT eyewear. Only the ones actually present in
# the cache are exercised (offline CI may have a subset).
_BRIGHT_SINGLE = ["alex", "lena_hat", "obama_blackmale", "biden_oldwhitemale",
                  "linmanuel", "obama2"]
_SINGLE_PRESENT = [n for n in _BRIGHT_SINGLE if n in _PORTRAITS]


def _classify(name: str, supersample: int, sunglasses: bool = False) -> dict:
    warns = WarningCollector()
    an = analyze_image(_PORTRAITS[name].read_bytes(), RenderConfig(), warns)
    diag: dict = {}
    # Render at the given supersample; SS=1 mirrors the on-screen preview, where the
    # regression was worst. Output is discarded -- we only read the classification.
    render_displacement_portrait(an, _WORDS, supersample=supersample,
                                 sunglasses=sunglasses, _diag=diag)
    return diag


@pytest.mark.skipif(not _SINGLE_PRESENT, reason="No single-face fixtures cached.")
@pytest.mark.parametrize("name", _SINGLE_PRESENT)
@pytest.mark.parametrize("supersample", [1, 2], ids=["preview_ss1", "full_ss2"])
def test_bright_single_face_keeps_real_eyes(name, supersample):
    # Default (no sunglasses flag): opacity is manual now, so NO face is ever dark-lensed.
    diag = _classify(name, supersample, sunglasses=False)
    assert diag.get("dark_lens_faces", 0) == 0, (
        f"{name} (SS={supersample}): a real, open-eyed face was flagged as a "
        f"sunglasses/dark lens and would render opaque black eye sockets."
    )
    # And the eyes must actually be rendered (not silently dropped as 'misfit'/closed).
    assert diag.get("eye_faces", 0) >= 1, (
        f"{name} (SS={supersample}): no eyes rendered ({diag})."
    )


@pytest.mark.skipif(not _SINGLE_PRESENT, reason="No single-face fixtures cached.")
@pytest.mark.parametrize("name", _SINGLE_PRESENT[:2])
def test_sunglasses_flag_forces_opaque_lens(name):
    # The manual toggle's contract: when the caller sets sunglasses=True, the eye region IS
    # rendered as an opaque lens (so a subject who really is wearing shades never gets
    # fabricated eyes). Guards against anyone silently neutering the flag.
    diag = _classify(name, supersample=1, sunglasses=True)
    assert diag.get("dark_lens_faces", 0) >= 1, (
        f"{name}: sunglasses=True did not opaque any lens ({diag})."
    )
