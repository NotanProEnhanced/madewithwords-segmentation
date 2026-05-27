#!/usr/bin/env python3
"""Turn a source photo + words into a matched 1:1 before/after pair via the engine.

render_pair(photo, words, ink, ...) -> (before_path, after_path)
  before = the photo, face-centred square crop
  after  = the rendered word-portrait, same crop, so the before/after slider aligns

Runs the same pipeline the app uses (analyze_image + render_layered_png), so it
needs the engine's deps (mediapipe, opencv, etc.) and models/ present.
"""
import os, sys, io
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import cv2
from PIL import Image
from app.config import RenderConfig
from app.pipeline.warnings import WarningCollector
from app.pipeline.analyze import analyze_image
from app.pipeline.tonal import render_layered_png

def render_pair(photo, words, ink="gold_noir", style="words", light=False,
                size=1100, out_dir=".", stem="item"):
    img_bytes = open(photo, "rb").read()
    cfg = RenderConfig(); warns = WarningCollector()
    an = analyze_image(img_bytes, cfg, warns)
    text = " ".join(words) if isinstance(words, (list, tuple)) else str(words)
    png, runs, ground, mask = render_layered_png(
        an, text, style, cfg, warns, ink=ink, remove_bg=True, light=light,
        out_width=1500, render_w=2400)
    if not png:
        raise RuntimeError("render failed for %s: %s" % (photo, warns.as_list()))
    after = Image.open(io.BytesIO(png)).convert("RGB")
    Wd, Hd = after.size
    before = Image.fromarray(cv2.cvtColor(an.img.bgr, cv2.COLOR_BGR2RGB)).resize((Wd, Hd), Image.LANCZOS)
    w0, h0 = an.img.bgr.shape[1], an.img.bgr.shape[0]
    sx, sy = Wd/float(w0), Hd/float(h0)
    if an.face_bbox:
        fx, fy, fw, fh = an.face_bbox; cx=(fx+fw/2)*sx; cy=(fy+fh/2)*sy
    else:
        cx, cy = Wd/2, Hd/2
    s = min(Wd, Hd)
    x0 = int(round(min(max(cx-s/2, 0), Wd-s)))
    y0 = int(round(min(max(cy-s*0.45, 0), Hd-s)))
    box = (x0, y0, x0+s, y0+s)
    os.makedirs(out_dir, exist_ok=True)
    bp = os.path.join(out_dir, stem+"-before.jpg")
    ap = os.path.join(out_dir, stem+"-after.png")
    before.crop(box).resize((size, size), Image.LANCZOS).save(bp, quality=90)
    after.crop(box).resize((size, size), Image.LANCZOS).save(ap)
    return bp, ap

if __name__ == "__main__":
    b, a = render_pair(sys.argv[1], sys.argv[2].split(","),
                       ink=(sys.argv[3] if len(sys.argv) > 3 else "gold_noir"),
                       out_dir="reels_out/_test", stem="test")
    print("before:", b, "\nafter: ", a)
