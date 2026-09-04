"""Shared analysis orchestration used by debug and render endpoints."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from ..config import RenderConfig
from .edges import EdgeResult, detect_edges
from .landmarks import FaceLandmarks, detect_faces, eyes_closed, haar_face_bbox
from .preprocess import LoadedImage, load_and_normalize
from .regions import RegionSet, build_regions
from .silhouette import Silhouette, extract_silhouette, silhouette_from_mask
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


def analyze_image(img_bytes: bytes, cfg: RenderConfig, warns: WarningCollector,
                  manual_mask=None) -> Analysis:
    img = load_and_normalize(img_bytes, cfg.work_max_dim, warns)

    faces = detect_faces(img, warns)
    # Tell the customer before they buy, not after: a closed eye has nothing bright
    # to paint over the naturally shadowed socket, so it renders as a flat dark disc
    # rather than an eye (see landmarks.eyes_closed). Non-fatal -- they can still
    # proceed if they choose to.
    if any(eyes_closed(f) for f in faces):
        warns.warn("input", "eyes_closed",
                   "One or more people in this photo have their eyes closed. "
                   "For the best result, choose a photo with eyes open.")
    landmarks = faces[0] if faces else None
    if landmarks is not None:
        face_bbox = landmarks.bbox
        face_source = "mediapipe"
    else:
        face_bbox = haar_face_bbox(img, warns)
        face_source = "haar" if face_bbox else "none"

    # A user-painted mask (manual background removal) overrides auto segmentation.
    if manual_mask is not None:
        sil = silhouette_from_mask(manual_mask, img.w, img.h)
    else:
        # Pass EVERY detected face, not just the first. The matte keeps any blob that
        # holds a face, and a group portrait has more than one.
        sil = extract_silhouette(img, warns, face_bbox=face_bbox,
                                 face_boxes=[f.bbox for f in faces] if faces else None)
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
    )
