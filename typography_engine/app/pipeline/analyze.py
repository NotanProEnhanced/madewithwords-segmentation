"""Shared analysis orchestration used by debug and render endpoints."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from ..config import RenderConfig
from .edges import EdgeResult, detect_edges
from .landmarks import FaceLandmarks, detect_faces, haar_face_bbox
from .preprocess import LoadedImage, load_and_normalize
from .regions import RegionSet, build_regions
from .silhouette import Silhouette, extract_silhouette
from .warnings import WarningCollector


@dataclass
class Analysis:
    img: LoadedImage
    landmarks: Optional[FaceLandmarks]      # primary (largest) face
    face_bbox: Optional[tuple]
    face_source: str
    silhouette: Silhouette
    edges: EdgeResult
    regions: RegionSet
    faces: List[FaceLandmarks] = field(default_factory=list)  # all faces, largest first
    subject: str = "person"     # "person" | "pet" | "other"


def _is_person(subject: str) -> bool:
    return (subject or "person").strip().lower() in ("", "person", "people", "human")


def analyze_image(
    img_bytes: bytes,
    cfg: RenderConfig,
    warns: WarningCollector,
    subject: str = "person",
) -> Analysis:
    img = load_and_normalize(img_bytes, cfg.work_max_dim, warns)
    person = _is_person(subject)
    subj = "person" if person else (subject or "other").strip().lower()

    if person:
        faces = detect_faces(img, warns)
        landmarks = faces[0] if faces else None
        if landmarks is not None:
            face_bbox = landmarks.bbox
            face_source = "mediapipe"
        else:
            face_bbox = haar_face_bbox(img, warns)
            face_source = "haar" if face_bbox else "none"
    else:
        # Non-human subject (pet/other): no human face landmarks. The tonal
        # passes that anchor eyes/brows/lips all guard on an empty face list, so
        # the likeness comes from tone + edge-separation. Silhouette uses the
        # general matte instead of the human selfie segmenter.
        faces = []
        landmarks = None
        face_bbox = None
        face_source = subj

    sil = extract_silhouette(img, warns, face_bbox=face_bbox, subject=subj)
    edges = detect_edges(img, warns, cfg.canny_low, cfg.canny_high, mask=sil.mask)
    regions = build_regions(landmarks, sil, warns)

    return Analysis(
        img=img,
        landmarks=landmarks,
        face_bbox=face_bbox,
        face_source=face_source,
        silhouette=sil,
        edges=edges,
        regions=regions,
        faces=faces,
        subject=subj,
    )
