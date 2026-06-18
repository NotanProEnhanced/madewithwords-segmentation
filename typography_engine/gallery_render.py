"""Render gallery before/after sets for lovedinwords.com.

For each source photo it writes a matched trio into _gallery/:
  {name}-before.jpg  -- the source, cropped 4:5 and framed on the face
  {name}-after.png   -- the typographic portrait (Words / Message / Sculpt)
  {name}-zoom.png    -- a face-center crop of the portrait, so the words read

Runs the SAME code locally and inside the VPS container. Locally cairosvg is
missing, so the engine falls back to resvg and prints a DEGRADED warning -- fine
for the Words/Message noir styles, but Sculpt/photo-tinted output should be
regenerated in the container for full fidelity. The summary line reports which
rasterizer was used so you can confirm.

Usage:
    python gallery_render.py            # render every job in JOBS below
    python gallery_render.py grace ruth # render only the named jobs
"""
import io, os, sys, time
from PIL import Image

import app.pipeline.tonal as tonal
from app.config import RenderConfig
from app.pipeline.warnings import WarningCollector
from app.pipeline.analyze import analyze_image
from app.pipeline.tonal import render_layered_png
from app.pipeline.displacement import render_displacement_portrait
from app.pipeline import raster

HERE = os.path.dirname(__file__)
SRC = os.path.join(HERE, "marketing", "lovedinwords", "_src")
OUT = os.path.join(HERE, "marketing", "lovedinwords", "_gallery")
os.makedirs(OUT, exist_ok=True)

AR = 0.8  # 4:5 print aspect (width / height)

# --- The five remembrances ---------------------------------------------------
# file:    the photo in _src/ (set after the files are dropped)
# style:   "words" | "message" | "sculpt"
# ink:     palette key -- mono (noir) | sepia | navy   (sculpt uses it as ground)
# text:    space-separated WORDS for words/sculpt; a sentence for message
# caption: the gallery caption
# All five: Words style, Medium size (min_font 57), Original ink (photo) -- the
# full-colour lifelike look. NOTE: "photo" ink encodes the image as per-glyph
# tonal modulation in the mask, which the resvg fallback flattens -> muddy. Render
# this on a host WITH cairosvg (the VPS container or any cairo box) for the wow.
MIN_FONT = 57   # Medium

JOBS = {
    "grace": dict(file="1.jpg.jpeg", style="words", ink="photo",
                  text="MOTHER GRANDMOTHER GRACE KIND FAITHFUL GENTLE BELOVED ALWAYS",
                  caption="Mother, Grandmother, Faithful, Kind — the words that were her."),
    "ruth":  dict(file="2.jpg.jpeg", style="words", ink="photo",
                  text="MOM NANA JOYFUL DEVOTED WARM TENDER BELOVED ALWAYS LOVE",
                  caption="Mom, Nana, Devoted, Tender — her likeness, in her words."),
    "james": dict(file="3.jpg.jpeg", style="words", ink="photo",
                  text="FATHER GRANDFATHER FRIEND STEADY GENEROUS KIND BELOVED ALWAYS",
                  caption="Father, Grandfather, Steady, Generous — every word that was him."),
    "robert": dict(file="4.jpg.jpeg", style="words", ink="photo",
                   text="FATHER GRANDPA MENTOR STRONG WISE GENEROUS BELOVED ALWAYS",
                   caption="Father, Grandpa, Mentor, Wise — every word that was him."),
    "eleanor": dict(file="5.jpg.jpeg", style="words", ink="photo",
                    text="MOTHER GRANDMOTHER GRACE FAITH TENDER KIND BELOVED ALWAYS",
                    caption="Mother, Grace, Faith, Tender — woven into her likeness."),
}


def face_crop_45(orig: Image.Image, an) -> Image.Image:
    """Crop `orig` to 4:5, framed on the face (mirrors the studio's auto-crop)."""
    W, H = orig.size
    bb = an.face_bbox
    if bb:
        # face_bbox is in normalized (an.img) pixels -> scale to original.
        sx, sy = W / float(an.img.w), H / float(an.img.h)
        fx, fy, fw, fh = bb[0] * sx, bb[1] * sy, bb[2] * sx, bb[3] * sy
        ch = min(H, fh * 3.0); cw = ch * AR
        if cw > W:
            cw = W; ch = cw / AR
        cx = (fx + fw / 2.0) - cw / 2.0
        cy = (fy + fh / 2.0) - ch * 0.42      # face a touch above centre
    else:                                      # no face -> centre crop
        if W / H > AR:
            ch = H; cw = ch * AR
        else:
            cw = W; ch = cw / AR
        cx = (W - cw) / 2.0; cy = (H - ch) / 2.0
    cx = max(0, min(cx, W - cw)); cy = max(0, min(cy, H - ch))
    return orig.crop((round(cx), round(cy), round(cx + cw), round(cy + ch)))


def render_after(an, job) -> bytes:
    style, ink, text = job["style"], job["ink"], job["text"]
    cfg = RenderConfig(); cfg.min_font_px = job.get("min_font", MIN_FONT)   # Medium
    warns = WarningCollector()
    if style == "sculpt":
        # Sculpt has a separate ground (paper/navy/black) and ink (word colour).
        return render_displacement_portrait(an, text.split(), ground="black", ink=ink,
                                            out_width=1200, supersample=2, uppercase=True)
    tonal._SUPERSAMPLE = 2
    png, runs, ground, mask = render_layered_png(
        an, text, ("message" if style == "message" else "words"), cfg, warns,
        ink=ink, remove_bg=job.get("remove_bg", True), light=False,
        out_width=1200, render_w=2000, tone_density=job.get("density", 0.6))
    return png


def zoom(after_path, out_path, side_frac=0.46):
    im = Image.open(after_path).convert("RGB")
    W, H = im.size
    side = int(min(W, H) * side_frac)
    cx, cy = W // 2, int(H * 0.40)
    box = (max(0, cx - side // 2), max(0, cy - side // 2),
           min(W, cx + side // 2), min(H, cy + side // 2))
    im.crop(box).save(out_path)


def do(name, job):
    src = os.path.join(SRC, job["file"])
    if not os.path.exists(src):
        print(f"  ! {name}: missing {job['file']} in _src/"); return
    cfg = RenderConfig(); warns = WarningCollector()
    raw = open(src, "rb").read()
    an = analyze_image(raw, cfg, warns)

    # before: source cropped 4:5 on the face
    orig = Image.open(io.BytesIO(raw)).convert("RGB")
    before = face_crop_45(orig, an)
    if before.width > 1080:
        before = before.resize((1080, round(1080 / AR)), Image.LANCZOS)
    bpath = os.path.join(OUT, f"{name}-before.jpg")
    before.save(bpath, quality=90)

    # after: render from the SAME crop so framing matches
    buf = io.BytesIO(); before.save(buf, format="JPEG", quality=95)
    an2 = analyze_image(buf.getvalue(), cfg, warns)
    t0 = time.time()
    png = render_after(an2, job)
    apath = os.path.join(OUT, f"{name}-after.png")
    open(apath, "wb").write(png)
    zoom(apath, os.path.join(OUT, f"{name}-zoom.png"))
    print(f"  {name}: {job['style']}/{job['ink']}  {len(png)//1024}KB  {time.time()-t0:.1f}s")


if __name__ == "__main__":
    backend = "cairosvg (full fidelity)" if raster.cairosvg_available() else "resvg (DEGRADED -- regenerate Sculpt in the container)"
    print(f"rasterizer: {backend}\n")
    want = sys.argv[1:] or list(JOBS)
    for name in want:
        if name in JOBS:
            do(name, JOBS[name])
        else:
            print(f"  ? unknown job '{name}'")
    print(f"\nwrote -> {OUT}")
