"""Geometry helpers: order edge sets into chains and emit smooth SVG path data."""
from __future__ import annotations

from collections import defaultdict
from typing import List, Sequence, Tuple

import numpy as np

Point = Tuple[float, float]


def order_edges_into_chains(edges: Sequence[Tuple[int, int]]) -> List[Tuple[List[int], bool]]:
    """Turn an unordered edge set into ordered vertex chains.

    Returns a list of (vertex_index_sequence, is_closed). Handles closed cycles,
    open chains, and branching graphs (branches are split at junctions).
    """
    adj = defaultdict(list)
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)

    # Deduplicate undirected edges for traversal bookkeeping.
    remaining = set()
    for a, b in edges:
        remaining.add((min(a, b), max(a, b)))

    def take_edge(u, v) -> bool:
        key = (min(u, v), max(u, v))
        if key in remaining:
            remaining.discard(key)
            return True
        return False

    chains: List[Tuple[List[int], bool]] = []

    # Prefer starting from degree-1 (endpoint) or degree>2 (junction) nodes so we
    # produce open chains where the graph isn't a clean cycle.
    def degree(n):
        return len(adj[n])

    def edge_left(u, c) -> bool:
        return (min(u, c), max(u, c)) in remaining

    def walk(start):
        path = [start]
        cur = start
        prev = None
        while True:
            nxt = None
            for cand in adj[cur]:
                if cand == prev:
                    continue
                if take_edge(cur, cand):
                    nxt = cand
                    break
            if nxt is None:
                break
            path.append(nxt)
            prev, cur = cur, nxt
            if cur == start:  # closed loop
                return path[:-1], True
            if degree(cur) != 2:  # reached a junction/endpoint -> stop this chain
                return path, False
        return path, False

    # First exhaust chains anchored at endpoints/junctions (open segments).
    for s in [n for n in adj if degree(n) != 2]:
        while any(edge_left(s, c) for c in adj[s]):
            path, closed = walk(s)
            if len(path) >= 2:
                chains.append((path, closed))

    # Remaining edges belong to pure cycles (all degree == 2).
    leftover_nodes = {n for e in remaining for n in e}
    for s in list(leftover_nodes):
        while any(edge_left(s, c) for c in adj[s]):
            path, closed = walk(s)
            if len(path) >= 2:
                chains.append((path, closed))

    return chains


def resample_polyline(points: np.ndarray, n: int) -> np.ndarray:
    """Resample an (N,2) polyline to n points evenly spaced by arc length."""
    if len(points) <= 2 or n <= 2:
        return points
    deltas = np.diff(points, axis=0)
    seg = np.hypot(deltas[:, 0], deltas[:, 1])
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    if total <= 0:
        return points
    targets = np.linspace(0, total, n)
    rx = np.interp(targets, cum, points[:, 0])
    ry = np.interp(targets, cum, points[:, 1])
    return np.stack([rx, ry], axis=1)


def catmull_rom_to_bezier_d(points: Sequence[Point], closed: bool, decimals: int = 2) -> str:
    """Build an SVG path 'd' string with cubic Beziers through the given points."""
    pts = [(float(x), float(y)) for x, y in points]
    n = len(pts)
    if n == 0:
        return ""
    if n == 1:
        x, y = pts[0]
        return f"M {round(x, decimals)} {round(y, decimals)}"
    if n == 2:
        (x0, y0), (x1, y1) = pts
        return f"M {round(x0,decimals)} {round(y0,decimals)} L {round(x1,decimals)} {round(y1,decimals)}"

    def P(i):
        if closed:
            return pts[i % n]
        return pts[min(max(i, 0), n - 1)]

    d = [f"M {round(pts[0][0], decimals)} {round(pts[0][1], decimals)}"]
    last = n if closed else n - 1
    for i in range(last):
        p0 = P(i - 1)
        p1 = P(i)
        p2 = P(i + 1)
        p3 = P(i + 2)
        c1x = p1[0] + (p2[0] - p0[0]) / 6.0
        c1y = p1[1] + (p2[1] - p0[1]) / 6.0
        c2x = p2[0] - (p3[0] - p1[0]) / 6.0
        c2y = p2[1] - (p3[1] - p1[1]) / 6.0
        d.append(
            "C "
            f"{round(c1x,decimals)} {round(c1y,decimals)} "
            f"{round(c2x,decimals)} {round(c2y,decimals)} "
            f"{round(p2[0],decimals)} {round(p2[1],decimals)}"
        )
    if closed:
        d.append("Z")
    return " ".join(d)
