"""Render debug overlay PNGs for the preprocessing stages."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from .edges import EdgeResult
from .landmarks import FaceLandmarks
from .preprocess import LoadedImage
from .silhouette import Silhouette


def _write(path: Path, img: np.ndarray) -> None:
    cv2.imwrite(str(path), img)


def render_debug_set(
    out_dir: Path,
    prefix: str,
    img: LoadedImage,
    sil: Silhouette,
    edges: EdgeResult,
    landmarks: Optional[FaceLandmarks],
    face_bbox: Optional[tuple],
) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    files: Dict[str, str] = {}

    # 1. Silhouette mask.
    sil_path = out_dir / f"{prefix}_silhouette.png"
    _write(sil_path, sil.mask)
    files["silhouette"] = sil_path.name

    # 2. Edges.
    edge_path = out_dir / f"{prefix}_edges.png"
    _write(edge_path, edges.edges)
    files["edges"] = edge_path.name

    # 3. Composite overlay on original working image.
    overlay = img.bgr.copy()
    # Tint silhouette interior.
    tint = overlay.copy()
    tint[sil.mask > 127] = (0, 180, 0)
    overlay = cv2.addWeighted(overlay, 0.7, tint, 0.3, 0)
    # Draw edges in red.
    overlay[edges.edges > 0] = (0, 0, 255)
    # Face bbox.
    if face_bbox is not None:
        x, y, w, h = [int(v) for v in face_bbox]
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 128, 0), 2)
    # Landmarks.
    if landmarks is not None:
        for px, py in landmarks.points:
            cv2.circle(overlay, (int(px), int(py)), 1, (255, 255, 0), -1)
    ov_path = out_dir / f"{prefix}_overlay.png"
    _write(ov_path, overlay)
    files["overlay"] = ov_path.name

    return files
