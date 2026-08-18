"""Shared analysis orchestration used by debug and render endpoints."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from ..config import RenderConfig
from .edges import EdgeResult, detect_edges
from .landmarks import FaceLandmarks, detect_faces, haar_face_bbox
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
                  manual_mask=None, compute_layered: bool = True) -> Analysis:
    img = load_and_normalize(img_bytes, cfg.work_max_dim, warns)

    faces = detect_faces(img, warns)
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
        sil = extract_silhouette(img, warns, face_bbox=face_bbox)
    # `edges` (Canny) and `regions` are consumed ONLY by the layered Words/Message
    # renderer -- the displacement (Lifelike) sculpt never reads them. On the memorial
    # Lifelike preview that's pure wasted CPU, so callers that will render displacement
    # pass compute_layered=False to skip it. We return VALID EMPTY structures (not None)
    # so any incidental accessor degrades gracefully; the layered path always passes
    # compute_layered=True, so its output is byte-identical to before.
    if compute_layered:
        edges = detect_edges(img, warns, cfg.canny_low, cfg.canny_high, mask=sil.mask)
        regions = build_regions(landmarks, sil, warns)
    else:
        edges = EdgeResult(edges=np.zeros((img.h, img.w), np.uint8), contours=[])
        regions = RegionSet()

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
