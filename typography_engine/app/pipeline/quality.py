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

    # Human-face checks apply only to person subjects. Pet/other subjects have no
    # human face by definition, so only the coverage sanity checks below run.
    if getattr(an, "subject", "person") == "person":
        if an.face_source == "none":
            issues.append(Issue(
                "error", "no_face",
                "We couldn't find a face. Upload a clear, front-facing "
                "head-and-shoulders photo of one person.",
            ))
            return issues

        if an.face_source != "mediapipe":
            # MediaPipe (the precise detector) found no face; only the crude Haar
            # fallback did. A tiny such box means this isn't really a portrait.
            if face_frac < 0.05:
                issues.append(Issue(
                    "error", "no_clear_portrait",
                    "We couldn't find a clear face to work from. This looks like a "
                    "scene or a distant subject -- try a close head-and-shoulders photo.",
                ))
                return issues
            issues.append(Issue(
                "warn", "low_confidence",
                "We couldn't lock onto the face precisely, so the likeness may be "
                "rough. A clear, front-facing, well-lit photo works best.",
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

    return issues
