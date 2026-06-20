#!/usr/bin/env python3
"""Generate Scribbler PWA icons with no third-party deps.

Draws a continuous pen-scribble "head" (denser toward the bottom, like the
reference portrait) onto a paper background, then writes PNGs via a tiny
hand-rolled encoder. Re-run to regenerate icons/*.png.
"""
import math
import os
import struct
import zlib

OUT = os.path.join(os.path.dirname(__file__), "icons")
os.makedirs(OUT, exist_ok=True)

PAPER = (246, 244, 238)
INK = (17, 20, 24)
ACCENT = (78, 139, 240)


def png_bytes(width, height, rgb):
    """rgb: flat bytearray length width*height*3 -> PNG bytes."""
    raw = bytearray()
    stride = width * 3
    for y in range(height):
        raw.append(0)  # filter type 0
        raw.extend(rgb[y * stride:(y + 1) * stride])
    comp = zlib.compress(bytes(raw), 9)

    def chunk(typ, data):
        return (struct.pack(">I", len(data)) + typ + data +
                struct.pack(">I", zlib.crc32(typ + data) & 0xffffffff))

    return (b"\x89PNG\r\n\x1a\n" +
            chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) +
            chunk(b"IDAT", comp) +
            chunk(b"IEND", b""))


class Canvas:
    def __init__(self, size, bg):
        self.n = size
        self.buf = bytearray(bg * size * size)

    def blend(self, x, y, color, a):
        if x < 0 or y < 0 or x >= self.n or y >= self.n or a <= 0:
            return
        i = (y * self.n + x) * 3
        a = min(1.0, a)
        for k in range(3):
            self.buf[i + k] = int(self.buf[i + k] * (1 - a) + color[k] * a)

    def disc(self, cx, cy, r, color, a):
        x0, x1 = int(cx - r - 1), int(cx + r + 1)
        y0, y1 = int(cy - r - 1), int(cy + r + 1)
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                d = math.hypot(x - cx, y - cy)
                if d <= r:
                    self.blend(x, y, color, a)
                elif d <= r + 1:
                    self.blend(x, y, color, a * (r + 1 - d))

    def line(self, x0, y0, x1, y1, color, width, a):
        steps = int(math.hypot(x1 - x0, y1 - y0)) + 1
        for s in range(steps + 1):
            t = s / steps
            self.disc(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, width, color, a)

    def to_png(self):
        return png_bytes(self.n, self.n, self.buf)


def rng(seed):
    state = {"s": seed & 0xffffffff or 1}

    def r():
        x = state["s"]
        x ^= (x << 13) & 0xffffffff
        x ^= x >> 17
        x ^= (x << 5) & 0xffffffff
        state["s"] = x & 0xffffffff
        return state["s"] / 0xffffffff
    return r


def draw_scribble_head(c, pad_frac=0.12):
    """Lay scribbles inside a head-shaped ellipse, denser near the bottom."""
    n = c.n
    r = rng(20240620)
    cx, cy = n * 0.5, n * 0.54
    rx, ry = n * (0.5 - pad_frac) * 0.82, n * (0.5 - pad_frac)

    def inside(x, y):
        return ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1.0

    width = max(0.55, n / 340)
    strokes = int(n * 0.62)
    for _ in range(strokes):
        # Pick a start inside the head; bias downward (more ink = chin/neck).
        for _try in range(12):
            x = cx + (r() - 0.5) * 2 * rx
            y = cy + (r() ** 0.7 - 0.4) * 2 * ry
            if inside(x, y):
                break
        else:
            continue
        ang = r() * math.tau
        segs = 6 + int(r() * 10)
        step = n / 70
        px, py = x, y
        for _ in range(segs):
            ang += (r() - 0.5) * 1.5
            nx = px + math.cos(ang) * step
            ny = py + math.sin(ang) * step
            if not inside(nx, ny):
                ang += math.pi  # bounce back inside
                nx = px + math.cos(ang) * step
                ny = py + math.sin(ang) * step
            c.line(px, py, nx, ny, INK, width, 0.3)
            px, py = nx, ny


def make(size, path, pad_frac, bg=PAPER, accent_ring=False):
    c = Canvas(size, bg)
    if accent_ring:
        # subtle accent baseline for maskable contrast
        for x in range(size):
            c.blend(x, size - 1, ACCENT, 0.0)
    draw_scribble_head(c, pad_frac)
    with open(path, "wb") as f:
        f.write(c.to_png())
    print("wrote", path, size)


if __name__ == "__main__":
    make(512, os.path.join(OUT, "icon-512.png"), 0.12)
    make(192, os.path.join(OUT, "icon-192.png"), 0.12)
    make(512, os.path.join(OUT, "maskable-512.png"), 0.22)  # extra safe padding
    make(180, os.path.join(OUT, "apple-touch-icon.png"), 0.12)
    make(64, os.path.join(OUT, "favicon.png"), 0.10)
