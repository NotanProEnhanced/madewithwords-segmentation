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


def analyze_image(img_bytes: bytes, cfg: RenderConfig, warns: WarningCollector) -> Analysis:
    img = load_and_normalize(img_bytes, cfg.work_max_dim, warns)

    faces = detect_faces(img, warns)
    landmarks = faces[0] if faces else None
    if landmarks is not None:
        face_bbox = landmarks.bbox
        face_source = "mediapipe"
    else:
        face_bbox = haar_face_bbox(img, warns)
        face_source = "haar" if face_bbox else "none"

    sil = extract_silhouette(img, warns, face_bbox=face_bbox)
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
