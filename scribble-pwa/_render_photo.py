#!/usr/bin/env python3
"""Run the app.js scribble engine offline against a real photo.

PIL is used ONLY to decode/resize the input and save the PNG; the scribble
logic mirrors scribble-pwa/app.js (tone map -> Sobel contour field ->
residual-balanced contour-following strokes). Lets us eyeball output without
a browser. Usage: python3 _render_photo.py <input> <output> [density] [contrast]
"""
import math
import sys
from PIL import Image

PAPER = (246, 244, 238)
INK = (17, 20, 24)


def rng(seed):
    st = [seed & 0xffffffff or 1]

    def r():
        x = st[0]
        x ^= (x << 13) & 0xffffffff
        x ^= x >> 17
        x ^= (x << 5) & 0xffffffff
        st[0] = x & 0xffffffff
        return st[0] / 4294967296
    return r


def ci(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def main():
    src = sys.argv[1]
    out = sys.argv[2]
    density = float(sys.argv[3]) if len(sys.argv) > 3 else 0.85
    gamma = float(sys.argv[4]) if len(sys.argv) > 4 else 1.55
    flow, opacity, weight, baseLen = 0.80, 0.20, 0.95, 24

    img = Image.open(src).convert("RGB")
    iw, ih = img.size
    MAP_MAX = 430
    fit = min(MAP_MAX / iw, MAP_MAX / ih, 1.0)
    mw, mh = max(8, round(iw * fit)), max(8, round(ih * fit))
    small = img.resize((mw, mh), Image.LANCZOS)
    px = small.load()

    tone = [0.0] * (mw * mh)
    for y in range(mh):
        for x in range(mw):
            r_, g_, b_ = px[x, y]
            tone[y * mw + x] = (0.2126 * r_ + 0.7152 * g_ + 0.0722 * b_) / 255

    def t(x, y):
        return tone[ci(y, 0, mh - 1) * mw + ci(x, 0, mw - 1)]

    grad = [0.0] * (mw * mh)
    for y in range(mh):
        for x in range(mw):
            gx = (-t(x-1,y-1) - 2*t(x-1,y) - t(x-1,y+1) + t(x+1,y-1) + 2*t(x+1,y) + t(x+1,y+1))
            gy = (-t(x-1,y-1) - 2*t(x,y-1) - t(x+1,y-1) + t(x-1,y+1) + 2*t(x,y+1) + t(x+1,y+1))
            grad[y * mw + x] = math.atan2(gx, -gy)

    residual = [0.0] * (mw * mh)
    total = 0.0
    for i in range(len(tone)):
        dd = (1 - tone[i]) ** gamma
        if dd < 0.015:
            dd = 0.0
        residual[i] = dd
        total += dd

    rs = 2
    ow, oh = mw * rs, mh * rs
    buf = bytearray(bytes(PAPER) * (ow * oh))

    def disc(cx, cy, rad, a):
        if a <= 0:
            return
        x0, x1 = int(cx - rad - 1), int(cx + rad + 1)
        y0, y1 = int(cy - rad - 1), int(cy + rad + 1)
        for yy in range(y0, y1 + 1):
            if yy < 0 or yy >= oh:
                continue
            row = yy * ow
            for xx in range(x0, x1 + 1):
                if xx < 0 or xx >= ow:
                    continue
                dd = math.hypot(xx - cx, yy - cy)
                aa = a if dd <= rad else (a * (rad + 1 - dd) if dd <= rad + 1 else 0)
                if aa <= 0:
                    continue
                idx = (row + xx) * 3
                buf[idx] = int(buf[idx] * (1 - aa) + INK[0] * aa)
                buf[idx+1] = int(buf[idx+1] * (1 - aa) + INK[1] * aa)
                buf[idx+2] = int(buf[idx+2] * (1 - aa) + INK[2] * aa)

    def seg(x0, y0, x1, y1, a):
        steps = int(math.hypot(x1 - x0, y1 - y0) * rs) + 1
        for s in range(steps + 1):
            tt = s / steps
            disc((x0 + (x1 - x0) * tt) * rs, (y0 + (y1 - y0) * tt) * rs, weight, a)

    def deposit(x, y, amt):
        ix, iy = int(x), int(y)
        k = amt * 1.35
        for dy in (-1, 0, 1):
            yy = iy + dy
            if yy < 0 or yy >= mh:
                continue
            for dx in (-1, 0, 1):
                xx = ix + dx
                if xx < 0 or xx >= mw:
                    continue
                w = 1 if dx == 0 and dy == 0 else 0.4
                idx = yy * mw + xx
                residual[idx] = max(0.0, residual[idx] - k * w)

    def alerp(a, b, tt):
        d = b - a
        while d > math.pi:
            d -= 2 * math.pi
        while d < -math.pi:
            d += 2 * math.pi
        return a + d * tt

    r = rng(0x9e3779b9)
    target = density
    strokes = 0
    maxs = 80000
    while strokes < maxs:
        if strokes % 800 == 0:
            cov = 1 - sum(residual) / max(1e-6, total)
            if cov >= target:
                break
        sx = sy = -1
        best = 0.0
        for _ in range(22):
            cx = int(r() * mw)
            cy = int(r() * mh)
            rr = residual[cy * mw + cx]
            if rr > best:
                best, sx, sy = rr, cx, cy
            if rr > 0.55 and r() < 0.6:
                break
        if sx < 0 or best < 0.03:
            strokes += 1
            continue
        strength = best
        segs = max(3, round(baseLen * (0.5 + 0.7 * strength)))
        step = 0.9 + r() * 0.5
        x = sx + (r() - 0.5)
        y = sy + (r() - 0.5)
        ang = grad[ci(int(y),0,mh-1)*mw+ci(int(x),0,mw-1)] + (r()-0.5)*(1-flow)*math.pi*1.4
        wob = 0.18 + (1 - flow) * 0.5
        a = min(0.85, max(0.02, opacity * (0.55 + 0.7 * strength)))
        pxn, pyn = x, y
        for i in range(segs):
            fa = grad[ci(int(pyn),0,mh-1)*mw+ci(int(pxn),0,mw-1)]
            ang = alerp(ang, fa, flow * 0.5) + (r() - 0.5) * wob
            nx = pxn + math.cos(ang) * step
            ny = pyn + math.sin(ang) * step
            if nx < 0 or ny < 0 or nx >= mw or ny >= mh:
                break
            if residual[int(ny) * mw + int(nx)] < 0.02 and i > 2:
                break
            seg(pxn, pyn, nx, ny, a)
            deposit(nx, ny, a)
            pxn, pyn = nx, ny
        strokes += 1

    Image.frombytes("RGB", (ow, oh), bytes(buf)).save(out)
    cov = 1 - sum(residual) / max(1e-6, total)
    print(f"{strokes} strokes, coverage {cov:.2f} -> {out} ({ow}x{oh})")


if __name__ == "__main__":
    main()
