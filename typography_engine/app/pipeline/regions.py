"""Decompose a detected face + silhouette into named region paths.

Design for readability (the previous closed-loop approach produced upside-down
fragments where text wrapped around small loops):

  * ONE primary "silhouette" path: the outer head+shoulders boundary, opened at
    the bottom and oriented bottom-left -> over the crown -> bottom-right so that
    flowed text is at most rotated 90 degrees and never upside-down.
  * A few UPRIGHT horizontal feature arcs (brow line, jaw line, lip line) derived
    from landmarks to hint facial structure.
  * No text on tiny closed loops (eyes/nose) -- those are dropped.

Each region yields an ordered point chain in working-image pixel coords. When a
feature cannot be derived we record a warning rather than inventing geometry.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .edges import silhouette_contour
from .landmarks import FaceLandmarks
from .pathgen import resample_polyline
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


def _connection_indices(name: str) -> List[int]:
    from mediapipe.tasks.python.vision import FaceLandmarksConnections as C

    obj = getattr(C, name, None)
    if obj is None:
        return []
    idx = set()
    for e in obj:
        idx.add(e.start)
        idx.add(e.end)
    return sorted(idx)


def _resample_closed(pts: np.ndarray, n: int) -> np.ndarray:
    loop = np.vstack([pts, pts[:1]])
    rs = resample_polyline(loop, n + 1)
    return rs[:-1]


def _smooth(pts: np.ndarray, win: int, closed: bool) -> np.ndarray:
    if win <= 1 or len(pts) < 3:
        return pts
    win = min(win, len(pts) if closed else len(pts) // 2 * 2 + 1)
    if win < 3:
        return pts
    k = np.ones(win) / win
    if closed:
        x = np.concatenate([pts[-win:, 0], pts[:, 0], pts[:win, 0]])
        y = np.concatenate([pts[-win:, 1], pts[:, 1], pts[:win, 1]])
        sx = np.convolve(x, k, mode="same")[win:-win]
        sy = np.convolve(y, k, mode="same")[win:-win]
    else:
        pad = win // 2
        x = np.pad(pts[:, 0], pad, mode="edge")
        y = np.pad(pts[:, 1], pad, mode="edge")
        sx = np.convolve(x, k, mode="same")[pad:-pad] if pad else pts[:, 0]
        sy = np.convolve(y, k, mode="same")[pad:-pad] if pad else pts[:, 1]
    return np.stack([sx, sy], axis=1).astype(np.float32)


def _clean_outline(sil: Silhouette, warns: WarningCollector) -> Optional[np.ndarray]:
    """Outer silhouette boundary, opened at the bottom and oriented for upright text."""
    contour = silhouette_contour(sil.mask, warns)
    if contour is None:
        return None
    pts = contour.reshape(-1, 2).astype(np.float32)
    if len(pts) < 20:
        warns.warn("regions", "outline_sparse", "Silhouette boundary has too few points.")
        return None

    H = sil.mask.shape[0]
    pts = _resample_closed(pts, 260)
    # Heavy periodic smoothing rounds off thin protrusions (poles, stray medals).
    pts = _smooth(pts, win=17, closed=True)

    ys = pts[:, 1]
    yspan = float(ys.max() - ys.min()) or 1.0
    # Drop a generous bottom band so the open arc terminates on the smoother
    # neck/shoulder sides rather than winding through the neckline notch.
    band = 0.13 * yspan
    is_bottom = (ys >= ys.max() - band) | (ys >= H - 3)

    n = len(pts)
    if is_bottom.all() or (~is_bottom).sum() < 8:
        # No clear bottom edge: open the loop at the topmost point instead.
        start = int(np.argmin(ys))
        arc = np.roll(pts, -start, axis=0)
    else:
        # Start just after a bottom run ends; collect until bottom is hit again.
        start = None
        for i in range(n):
            if is_bottom[i] and not is_bottom[(i + 1) % n]:
                start = (i + 1) % n
                break
        if start is None:
            start = int(np.argmin(ys))
        arc_pts = []
        i = start
        for _ in range(n):
            if is_bottom[i]:
                break
            arc_pts.append(pts[i])
            i = (i + 1) % n
        arc = np.array(arc_pts, dtype=np.float32)

    if len(arc) < 12:
        warns.warn("regions", "outline_open_failed", "Could not open the silhouette boundary cleanly.")
        return None
    # Trim the slice endpoints, which sit in the steep cut region and otherwise
    # make the first/last glyphs curl.
    arc = arc[2:-2]
    # Orient so the crown reads left-to-right (start at the lower-left end).
    if arc[0, 0] > arc[-1, 0]:
        arc = arc[::-1].copy()
    arc = _smooth(arc, win=9, closed=False)
    return resample_polyline(arc, min(96, len(arc)))


def _horizontal_arc(
    lm: FaceLandmarks,
    indices: List[int],
    warns: WarningCollector,
    name: str,
    y_lo: float = 0.0,
    y_hi: float = 1.0,
) -> Optional[np.ndarray]:
    """Build an upright left-to-right arc from selected landmark points.

    y_lo/y_hi select a vertical sub-band (fractions of the group's own height) so
    we can take, e.g., only the upper lip line.
    """
    if not indices:
        return None
    pts = lm.points[indices]
    ymin, ymax = pts[:, 1].min(), pts[:, 1].max()
    span = (ymax - ymin) or 1.0
    lo = ymin + y_lo * span
    hi = ymin + y_hi * span
    sel = pts[(pts[:, 1] >= lo) & (pts[:, 1] <= hi)]
    if len(sel) < 3:
        warns.warn("regions", f"{name}_sparse", f"Too few points to form {name}.")
        return None
    sel = sel[np.argsort(sel[:, 0])]
    sel = _smooth(sel, win=3, closed=False)
    return resample_polyline(sel, min(40, len(sel)))


def build_regions(
    lm: Optional[FaceLandmarks],
    sil: Silhouette,
    warns: WarningCollector,
) -> RegionSet:
    rs = RegionSet()

    outline = _clean_outline(sil, warns)
    if outline is not None:
        rs.paths.append(RegionPath("silhouette", outline, closed=False, kind="primary"))

    if lm is not None:
        brow = _horizontal_arc(
            lm,
            _connection_indices("FACE_LANDMARKS_LEFT_EYEBROW")
            + _connection_indices("FACE_LANDMARKS_RIGHT_EYEBROW"),
            warns, "brow_line", y_lo=0.0, y_hi=0.6,
        )
        if brow is not None:
            rs.paths.append(RegionPath("brow_line", brow, closed=False, kind="secondary"))

        jaw = _horizontal_arc(
            lm, _connection_indices("FACE_LANDMARKS_FACE_OVAL"),
            warns, "jaw_line", y_lo=0.62, y_hi=1.0,
        )
        if jaw is not None:
            rs.paths.append(RegionPath("jaw_line", jaw, closed=False, kind="secondary"))
    else:
        warns.warn("regions", "no_landmarks", "No face landmarks; rendering silhouette outline only.")

    if not rs.paths:
        warns.error("regions", "no_regions", "No usable regions could be derived from the image.")

    return rs
