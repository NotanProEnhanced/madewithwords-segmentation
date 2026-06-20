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

PAPER = (255, 255, 255)
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
    density = float(sys.argv[3]) if len(sys.argv) > 3 else 0.82
    gamma = float(sys.argv[4]) if len(sys.argv) > 4 else 1.30
    remove_bg = (sys.argv[5] != "0") if len(sys.argv) > 5 else True
    matte_path = sys.argv[6] if len(sys.argv) > 6 else None
    fill = float(sys.argv[7]) if len(sys.argv) > 7 else 0.18
    flow = float(sys.argv[8]) if len(sys.argv) > 8 else 0.72
    weight = float(sys.argv[9]) if len(sys.argv) > 9 else 1.0
    baseLen = float(sys.argv[10]) if len(sys.argv) > 10 else 58
    opacity = float(sys.argv[11]) if len(sys.argv) > 11 else 0.20

    img = Image.open(src).convert("RGB")
    iw, ih = img.size
    MAP_MAX = 430
    fit = min(MAP_MAX / iw, MAP_MAX / ih, 1.0)
    mw, mh = max(8, round(iw * fit)), max(8, round(ih * fit))
    small = img.resize((mw, mh), Image.LANCZOS)
    px = small.load()

    tone = [0.0] * (mw * mh)
    rgbm = [(0, 0, 0)] * (mw * mh)
    for y in range(mh):
        for x in range(mw):
            r_, g_, b_ = px[x, y]
            rgbm[y * mw + x] = (r_, g_, b_)
            tone[y * mw + x] = (0.2126 * r_ + 0.7152 * g_ + 0.0722 * b_) / 255

    # Foreground matte. If an external alpha matte is supplied (e.g. u2net),
    # use it; otherwise fall back to the border flood-fill (mirrors app.js).
    fg = [1] * (mw * mh)
    if matte_path:
        m = Image.open(matte_path).convert("L").resize((mw, mh), Image.LANCZOS)
        mp = m.load()
        fg = [1 if mp[x, y] > 90 else 0 for y in range(mh) for x in range(mw)]
    elif remove_bg:
        bg = [0] * (mw * mh)
        stack = []
        tol = 0.085 * 3 * 255

        def pushb(i):
            if not bg[i]:
                bg[i] = 1
                stack.append(i)
        for x in range(mw):
            pushb(x); pushb((mh - 1) * mw + x)
        for y in range(mh):
            pushb(y * mw); pushb(y * mw + mw - 1)
        while stack:
            i = stack.pop()
            xx, yy = i % mw, i // mw
            cr, cg, cb = rgbm[i]
            for nx2, ny2 in ((xx-1,yy),(xx+1,yy),(xx,yy-1),(xx,yy+1)):
                if nx2 < 0 or ny2 < 0 or nx2 >= mw or ny2 >= mh:
                    continue
                j = ny2 * mw + nx2
                if bg[j]:
                    continue
                dr, dg, db = rgbm[j]
                if abs(cr - dr) + abs(cg - dg) + abs(cb - db) < tol:
                    pushb(j)
        fg = [0 if bg[i] else 1 for i in range(mw * mh)]

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
        if not fg[i]:
            residual[i] = 0.0
            continue
        dd = (1 - tone[i]) ** gamma
        if dd < 0.015:
            dd = 0.0
        if fill > 0 and fg[i]:
            dd = max(dd, fill)
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
        k = amt * 0.5
        for dy in (-1, 0, 1):
            yy = iy + dy
            if yy < 0 or yy >= mh:
                continue
            for dx in (-1, 0, 1):
                xx = ix + dx
                if xx < 0 or xx >= mw:
                    continue
                w = 1 if dx == 0 and dy == 0 else 0.22
                idx = yy * mw + xx
                residual[idx] = max(0.0, residual[idx] - k * w)

    def deposit_line(x0, y0, x1, y1, amt):
        dist = math.hypot(x1 - x0, y1 - y0)
        n = max(1, round(dist))
        for s in range(1, n + 1):
            tt = s / n
            deposit(x0 + (x1 - x0) * tt, y0 + (y1 - y0) * tt, amt)

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
        len_scale = baseLen / 26
        stray = r() < 0.02
        segs = min(170 if stray else 110,
                   max(5, round(baseLen * (1.6 if stray else 0.5 + 0.7 * strength))))
        step = (1.0 + r() * 0.5) * len_scale
        a = min(0.85, max(0.02, opacity * (0.5 + 0.7 * strength) * (0.5 if stray else 1)))
        steer = 0.12 if stray else flow * 0.35
        drift = 0.05 + (1 - flow) * 0.10
        x = sx + (r() - 0.5)
        y = sy + (r() - 0.5)
        ang = grad[ci(int(y),0,mh-1)*mw+ci(int(x),0,mw-1)] + (r()-0.5)*(1-flow)*math.pi*1.5
        turn = (r() - 0.5) * 0.15
        pts = [(x, y)]
        pxn, pyn = x, y
        for i in range(segs):
            fa = grad[ci(int(pyn),0,mh-1)*mw+ci(int(pxn),0,mw-1)]
            ang = alerp(ang, fa, steer) + turn
            turn = max(-0.5, min(0.5, turn + (r() - 0.5) * drift))
            nx = pxn + math.cos(ang) * step
            ny = pyn + math.sin(ang) * step
            if nx < 0 or ny < 0 or nx >= mw or ny >= mh:
                break
            idx = int(ny) * mw + int(nx)
            if remove_bg and not fg[idx] and i > 0:
                break
            if (not stray) and residual[idx] < 0.02 and i > 3:
                break
            deposit_line(pxn, pyn, nx, ny, a)
            pxn, pyn = nx, ny
            pts.append((nx, ny))
        if len(pts) >= 2:
            for k in range(1, len(pts)):
                seg(pts[k-1][0], pts[k-1][1], pts[k][0], pts[k][1], a)
            if (not stray) and strength > 0.35 and r() < 0.3:
                a2 = a * 0.8
                for k in range(1, len(pts)):
                    seg(pts[k-1][0] + (r()-0.5)*0.8, pts[k-1][1] + (r()-0.5)*0.8,
                        pts[k][0] + (r()-0.5)*0.8, pts[k][1] + (r()-0.5)*0.8, a2)
        strokes += 1

    Image.frombytes("RGB", (ow, oh), bytes(buf)).save(out)
    cov = 1 - sum(residual) / max(1e-6, total)
    print(f"{strokes} strokes, coverage {cov:.2f} -> {out} ({ow}x{oh})")


if __name__ == "__main__":
    main()
