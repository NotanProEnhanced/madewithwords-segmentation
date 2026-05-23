"""Decompose a detected face + silhouette into named region paths.

Each region yields one or more ordered point loops/chains in working-image pixel
coordinates. Downstream stages smooth these into SVG paths and flow text along
them. When a feature cannot be derived we record a warning rather than inventing
geometry.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .edges import silhouette_contour
from .landmarks import FaceLandmarks
from .pathgen import order_edges_into_chains, resample_polyline
from .silhouette import Silhouette
from .warnings import WarningCollector


@dataclass
class RegionPath:
    name: str
    points: np.ndarray            # (N,2) float, working-image coords
    closed: bool
    kind: str                     # "primary" (major contour) | "secondary" (detail)


@dataclass
class RegionSet:
    paths: List[RegionPath] = field(default_factory=list)

    def by_name(self, name: str) -> List[RegionPath]:
        return [p for p in self.paths if p.name == name]

    def names(self) -> List[str]:
        return sorted({p.name for p in self.paths})


def _connection_edges(name: str):
    from mediapipe.tasks.python.vision import FaceLandmarksConnections as C

    obj = getattr(C, name, None)
    if obj is None:
        return []
    return [(e.start, e.end) for e in obj]


def _chain_points(lm: FaceLandmarks, edges, prefer_closed: bool) -> Optional[np.ndarray]:
    chains = order_edges_into_chains(edges)
    if not chains:
        return None
    if prefer_closed:
        closed = [c for c in chains if c[1]]
        if closed:
            best = max(closed, key=lambda c: len(c[0]))
            idx = best[0]
            return lm.points[idx]
    # else longest chain overall
    best = max(chains, key=lambda c: len(c[0]))
    return lm.points[best[0]]


# Region name -> (mediapipe connection attr, prefer_closed, kind)
_MP_REGIONS = {
    "face_oval": ("FACE_LANDMARKS_FACE_OVAL", True, "primary"),
    "left_eye": ("FACE_LANDMARKS_LEFT_EYE", True, "secondary"),
    "right_eye": ("FACE_LANDMARKS_RIGHT_EYE", True, "secondary"),
    "left_brow": ("FACE_LANDMARKS_LEFT_EYEBROW", False, "secondary"),
    "right_brow": ("FACE_LANDMARKS_RIGHT_EYEBROW", False, "secondary"),
    "lips": ("FACE_LANDMARKS_LIPS", True, "secondary"),
    "nose": ("FACE_LANDMARKS_NOSE", False, "secondary"),
}


def _split_silhouette_hair_neck(
    sil: Silhouette,
    face_oval: Optional[np.ndarray],
    warns: WarningCollector,
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    contour = silhouette_contour(sil.mask, warns)
    if contour is None:
        return out
    pts = contour.reshape(-1, 2).astype(np.float32)
    if face_oval is not None and len(face_oval) > 0:
        top_y = float(face_oval[:, 1].min())
        bot_y = float(face_oval[:, 1].max())
    else:
        ys = pts[:, 1]
        top_y = float(np.percentile(ys, 25))
        bot_y = float(np.percentile(ys, 70))

    hair = pts[pts[:, 1] <= top_y + 1.0]
    neck = pts[pts[:, 1] >= bot_y - 1.0]

    # Hair = upper envelope (crown line): one highest point per x-bin, smoothed.
    hair_env = _x_envelope(hair, take="min_y", bins=28)
    if hair_env is not None and len(hair_env) >= 4:
        out["hair"] = resample_polyline(hair_env, min(36, len(hair_env)))
    else:
        warns.warn("regions", "hair_sparse", "Too few silhouette points above the face to form a hair contour.")

    # Neck/shoulders = top envelope of the lower body (shoulder line).
    neck_env = _x_envelope(neck, take="min_y", bins=28)
    if neck_env is not None and len(neck_env) >= 4:
        out["neck"] = resample_polyline(neck_env, min(36, len(neck_env)))
    else:
        warns.warn("regions", "neck_sparse", "Too few silhouette points below the face to form a neck/shoulder contour.")

    return out


def _x_envelope(pts: np.ndarray, take: str, bins: int) -> Optional[np.ndarray]:
    """Reduce scattered contour points to one representative per x-bin.

    take="min_y" keeps the highest (smallest y) point per column, tracing the
    upper outline and rejecting the up/down spikes of a raw contour walk.
    """
    if pts is None or len(pts) < 4:
        return None
    x = pts[:, 0]
    x0, x1 = float(x.min()), float(x.max())
    if x1 - x0 < 1e-3:
        return None
    edges = np.linspace(x0, x1, bins + 1)
    idx = np.clip(np.digitize(x, edges) - 1, 0, bins - 1)
    out = []
    for b in range(bins):
        sel = pts[idx == b]
        if len(sel) == 0:
            continue
        if take == "min_y":
            row = sel[np.argmin(sel[:, 1])]
        else:
            row = sel[np.argmax(sel[:, 1])]
        out.append(row)
    if len(out) < 4:
        return None
    arr = np.array(out, dtype=np.float32)
    arr = arr[np.argsort(arr[:, 0])]
    # Rolling-median on y to suppress single-column spikes (e.g. a pole/medal
    # poking out of the silhouette) that would otherwise twist the text path.
    y = arr[:, 1].copy()
    k = 3
    half = k // 2
    ys = y.copy()
    for j in range(len(y)):
        lo, hi = max(0, j - half), min(len(y), j + half + 1)
        ys[j] = float(np.median(y[lo:hi]))
    arr[:, 1] = ys
    return arr


def build_regions(
    lm: Optional[FaceLandmarks],
    sil: Silhouette,
    warns: WarningCollector,
) -> RegionSet:
    rs = RegionSet()
    face_oval_pts: Optional[np.ndarray] = None

    if lm is not None:
        for name, (attr, prefer_closed, kind) in _MP_REGIONS.items():
            edges = _connection_edges(attr)
            if not edges:
                warns.warn("regions", "missing_connection", f"No MediaPipe connection data for {name}.")
                continue
            pts = _chain_points(lm, edges, prefer_closed)
            if pts is None or len(pts) < 3:
                warns.warn("regions", "region_failed", f"Could not order landmarks for {name}.")
                continue
            closed = prefer_closed
            rs.paths.append(RegionPath(name=name, points=pts, closed=closed, kind=kind))
            if name == "face_oval":
                face_oval_pts = pts
    else:
        warns.warn("regions", "no_landmarks", "No face landmarks; deriving face outline from silhouette only.")
        contour = silhouette_contour(sil.mask, warns)
        if contour is not None:
            pts = contour.reshape(-1, 2).astype(np.float32)
            rs.paths.append(RegionPath("face_oval", resample_polyline(pts, min(64, len(pts))), True, "primary"))
            face_oval_pts = rs.paths[-1].points

    # Hair + neck from silhouette (independent of landmark availability).
    sn = _split_silhouette_hair_neck(sil, face_oval_pts, warns)
    if "hair" in sn:
        rs.paths.append(RegionPath("hair", sn["hair"], False, "primary"))
    if "neck" in sn:
        rs.paths.append(RegionPath("neck", sn["neck"], False, "primary"))

    if not rs.paths:
        warns.error("regions", "no_regions", "No usable regions could be derived from the image.")

    return rs
