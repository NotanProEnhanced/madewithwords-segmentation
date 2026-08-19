"""Input-quality gate for the render endpoint.

Turns "garbage in -> silently bad portrait" into actionable feedback. The checks
are deliberately conservative: every real single portrait in our test batch
passes cleanly, and only inputs that genuinely lack a usable subject (a
non-portrait scene, an empty/tiny face, a subject filling the whole frame)
are flagged. Multi-face detection is intentionally omitted -- the available Haar
count is too noisy (it false-positives on single faces), so we would rather miss
a group photo than reject good single portraits.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .analyze import Analysis


@dataclass
class Issue:
    severity: str   # "error" (block render) | "warn" (render, but tell the user)
    code: str
    message: str


def assess_portrait_input(an: Analysis) -> List[Issue]:
    issues: List[Issue] = []
    W, H = an.img.w, an.img.h
    area = float(W * H) or 1.0
    cov = an.silhouette.coverage
    face_frac = 0.0
    if an.face_bbox:
        _, _, bw, bh = an.face_bbox
        face_frac = (bw * bh) / area

    if an.face_source == "none":
        issues.append(Issue(
            "error", "no_face",
            "We couldn't read this photo. Try one with better lighting, "
            "or a clearer, closer view of their face.",
        ))
        return issues

    if an.face_source != "mediapipe":
        # MediaPipe (the precise detector) found no face; only the crude Haar
        # fallback did. A tiny such box means this isn't really a portrait.
        if face_frac < 0.05:
            issues.append(Issue(
                "error", "no_clear_portrait",
                "This photo is too zoomed out — their face is too small. "
                "Try a closer shot where their face fills most of the frame.",
            ))
            return issues
        issues.append(Issue(
            "warn", "low_confidence",
            "This photo might be a bit unclear or at an angle. The portrait may "
            "turn out softer than ideal — but let's see how it looks.",
        ))

    if cov < 0.08:
        issues.append(Issue(
            "error", "subject_too_small",
            "The subject fills too little of the frame. Crop closer so the head "
            "and shoulders fill most of the photo.",
        ))
    elif cov > 0.96:
        issues.append(Issue(
            "warn", "no_clean_background",
            "We couldn't separate the subject from the background, so the outline "
            "may be imprecise. A plain, contrasting background works best.",
        ))

    # Resolution of the FACE, in TRUE source pixels. A small photo (or a close-up
    # screenshot) can have the face fill the frame -- passing the coverage check --
    # yet carry too few real pixels for the features, so the eyes render soft no
    # matter the engine. enhance_source upscales such a photo, which inflates the
    # working size, so judge from the pre-enhance file size (img.src_w). Calibrated
    # on the test set: a 490px screenshot -> ~165px face (soft); every clean portrait
    # is >= ~320px. Warn (never block) -- a grieving family may have only this photo.
    src_w = getattr(an.img, "src_w", 0) or an.img.w
    if an.face_bbox and an.img.w:
        face_w_src = an.face_bbox[2] * (float(src_w) / float(an.img.w))
        if face_w_src < 250.0:
            issues.append(Issue(
                "warn", "low_resolution",
                "This photo is low-resolution, so the portrait — especially the "
                "eyes — may look soft. If you have a larger or sharper close-up of "
                "them, it will render with more detail.",
            ))

    return issues
