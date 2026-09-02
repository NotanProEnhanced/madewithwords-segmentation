#!/usr/bin/env python3
"""
Landmark-free typographic-portrait prototype (pets / any subject).

WHY THIS EXISTS
---------------
The production engine (displacement.py) needs a human MediaPipe FACE MESH and the
"selfie" person-segmenter -- neither fires on a pet, so it raises
`displacement_needs_face`. This prototype proves the CORE idea works WITHOUT any
landmarks:

  1) segment the subject (GrabCut -- no model download; or pass a mask),
  2) build a DETAIL map from the photo (Laplacian energy -> where the fur/eyes/nose are),
  3) fill the silhouette with words, in two SIZE TIERS driven by that detail
     (fine words on high-detail features, coarse on flat body),
  4) color every glyph by the photo itself, so the pet emerges FROM the words.

This is species-agnostic: it never looks for eyes or a face, only for photographic
detail, so it works on a dog, cat, parrot or rabbit alike.

USAGE
-----
  python pet_prototype.py PHOTO.jpg "NAME, GOODBOY, LOYAL, RESCUE" out.png [--ground paper|dark|slate]
  python pet_prototype.py --demo out_dir            # synthetic subject (no photo needed)

Real pet photos are where quality is judged -- run the first form in your container
(where cv2/PIL are installed) on a few real pets.
"""
import sys, os, math
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

_FONT = next((p for p in [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
] if os.path.exists(p)), None)

GROUNDS = {                     # BGR
    "paper": (232, 240, 244),   # warm ivory -> ink-on-paper look
    "dark":  (40, 26, 20),      # deep navy  -> colors pop (most portrait-like)
    "slate": (216, 221, 226),   # cool gallery gray
}


def foreground_mask(bgr):
    """GrabCut with a border-init rectangle -- no model download. Good enough for a
    centered subject on a distinct background (the common pet-photo case). In the app
    this is replaced by a real matte; here it just has to be serviceable."""
    h, w = bgr.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    rect = (int(w * 0.06), int(h * 0.06), int(w * 0.88), int(h * 0.88))
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(bgr, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
        m = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype(np.float32)
    except cv2.error:
        m = np.zeros((h, w), np.float32); m[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]] = 1.0
    m = cv2.morphologyEx((m * 255).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.GaussianBlur(m, (0, 0), sigmaX=max(1.0, w * 0.004)) / 255.0
    return np.clip(m, 0, 1)


def detail_map(gray):
    """Local detail = smoothed Laplacian energy, normalized 0..1. High on eyes,
    nose, whiskers and fur edges; low on flat body. This is the landmark-free
    stand-in for the face-mesh feature anchoring."""
    lap = np.abs(cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F, ksize=3))
    lap = cv2.GaussianBlur(lap, (0, 0), sigmaX=max(1.0, gray.shape[1] * 0.012))
    lap /= (np.percentile(lap, 99) + 1e-6)
    return np.clip(lap, 0, 1)


def _char_stream(words):
    """Endless stream of uppercase letters with single spaces between words."""
    toks = [w for w in "".join(c if (c.isalnum() or c in " ,") else " " for c in words.upper()).split() if w]
    if not toks:
        toks = ["LOVE"]
    i = 0
    while True:
        for ch in toks[i % len(toks)]:
            yield ch
        yield " "
        i += 1


def _render_tier(bgr, mask, size, ground_is_light):
    """Draw one constant-size layer of photo-colored text across the whole mask.
    Returns (rgb float HxWx3, alpha float HxW). On a light ground the glyph alpha is
    the ink DENSITY (dark photo -> opaque, light -> faint); on dark it's flat."""
    h, w = bgr.shape[:2]
    font = ImageFont.truetype(_FONT, size) if _FONT else ImageFont.load_default()
    layer = Image.new("RGB", (w, h), (0, 0, 0))
    alpha = Image.new("L", (w, h), 0)
    dl, da = ImageDraw.Draw(layer), ImageDraw.Draw(alpha)
    cw = max(3, int(font.getlength("M")))
    lh = int(size * 1.02)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    stream = _char_stream(_render_tier.words)
    y = 0
    while y < h:
        x = 0
        while x < w:
            ch = next(stream)
            cx, cy = min(w - 1, x + cw // 2), min(h - 1, y + size // 2)
            if ch != " " and mask[cy, cx] > 0.5:
                b, g, r = bgr[cy, cx]
                col = (int(r), int(g), int(b))
                if ground_is_light:                       # ink density: dark photo -> strong ink
                    a = int(255 * (0.35 + 0.65 * (1.0 - gray[cy, cx])))
                    col = tuple(int(c * 0.72) for c in col)   # deepen so light hues still read on paper
                else:
                    a = 255
                dl.text((x, y), ch, font=font, fill=col)
                da.text((x, y), ch, font=font, fill=a)
            x += cw
        y += lh
    return np.asarray(layer, np.float32), np.asarray(alpha, np.float32) / 255.0


def render_word_portrait(bgr, mask, words, ground="paper"):
    """Two detail-driven tiers composited over the ground, clipped to the subject."""
    h, w = bgr.shape[:2]
    _render_tier.words = words
    gbgr = GROUNDS.get(ground, GROUNDS["paper"])
    light = (ground != "dark")
    det = detail_map(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
    base = max(9, int(round(w / 46)))                 # coarse tier size
    fine = max(6, int(round(base * 0.55)))            # fine tier (features)
    coarse_rgb, coarse_a = _render_tier(bgr, mask, base, light)
    fine_rgb,   fine_a   = _render_tier(bgr, mask, fine, light)
    # Where detail is high, prefer the FINE tier; elsewhere the coarse tier.
    sel = np.clip((det - 0.32) / 0.33, 0, 1)[..., None]      # 0 flat -> 1 detailed
    rgb = coarse_rgb * (1 - sel) + fine_rgb * sel
    a = (coarse_a[..., None] * (1 - sel) + fine_a[..., None] * sel) * mask[..., None]
    ground_rgb = np.full((h, w, 3), gbgr[::-1], np.float32)  # BGR const -> RGB canvas
    out = ground_rgb * (1 - a) + rgb * a
    return np.clip(out, 0, 255).astype(np.uint8)            # RGB


def synthetic_pet(w=680, h=850):
    """A stylised 'pet' (head + ears + two dark eyes + nose + fur noise) with real
    brightness + detail structure, so the smoke-test exercises the true code path
    without a photo. NOT a real animal -- just enough to show the technique."""
    img = np.full((h, w, 3), (60, 70, 80), np.uint8)        # BGR background
    cx, cy = w // 2, int(h * 0.5)
    # warm fur body/head
    cv2.ellipse(img, (cx, cy), (int(w * 0.30), int(h * 0.32)), 0, 0, 360, (70, 120, 175), -1)
    cv2.ellipse(img, (int(w * 0.30), int(h * 0.24)), (70, 110), 20, 0, 360, (60, 105, 160), -1)  # ears
    cv2.ellipse(img, (int(w * 0.70), int(h * 0.24)), (70, 110), -20, 0, 360, (60, 105, 160), -1)
    cv2.ellipse(img, (cx, int(h * 0.60)), (int(w * 0.20), int(h * 0.16)), 0, 0, 360, (150, 200, 235), -1)  # light muzzle
    # fur texture (detail)
    rng = np.random.default_rng(7)
    noise = rng.normal(0, 16, (h, w, 1)).astype(np.float32)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    # eyes + nose (dark, high-detail features)
    for ex in (int(w * 0.40), int(w * 0.60)):
        cv2.circle(img, (ex, int(h * 0.44)), 34, (35, 40, 55), -1)
        cv2.circle(img, (ex, int(h * 0.44)), 15, (20, 20, 25), -1)
        cv2.circle(img, (ex - 6, int(h * 0.44) - 8), 6, (240, 240, 245), -1)   # catchlight
    cv2.ellipse(img, (cx, int(h * 0.56)), (30, 22), 0, 0, 360, (30, 30, 40), -1)  # nose
    # mask = the head+ears+muzzle silhouette
    m = np.zeros((h, w), np.uint8)
    cv2.ellipse(m, (cx, cy), (int(w * 0.30), int(h * 0.32)), 0, 0, 360, 255, -1)
    cv2.ellipse(m, (int(w * 0.30), int(h * 0.24)), (70, 110), 20, 0, 360, 255, -1)
    cv2.ellipse(m, (int(w * 0.70), int(h * 0.24)), (70, 110), -20, 0, 360, 255, -1)
    cv2.ellipse(m, (cx, int(h * 0.60)), (int(w * 0.20), int(h * 0.16)), 0, 0, 360, 255, -1)
    mask = cv2.GaussianBlur(m, (0, 0), sigmaX=3) / 255.0
    return img, np.clip(mask, 0, 1)


def main():
    args = sys.argv[1:]
    if args and args[0] == "--demo":
        outdir = args[1] if len(args) > 1 else "."
        os.makedirs(outdir, exist_ok=True)
        bgr, mask = synthetic_pet()
        words = "BUDDY, GOOD BOY, LOYAL, RESCUE, FETCH, ZOOMIES, CUDDLES, BEST FRIEND"
        for g in ("paper", "dark", "slate"):
            out = render_word_portrait(bgr, mask, words, ground=g)
            Image.fromarray(out).save(os.path.join(outdir, f"pet_demo_{g}.png"))
            print("wrote", os.path.join(outdir, f"pet_demo_{g}.png"))
        Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).save(os.path.join(outdir, "pet_demo_source.png"))
        return
    if len(args) < 3:
        print(__doc__); sys.exit(1)
    photo, words, out = args[0], args[1], args[2]
    ground = "paper"
    if "--ground" in args:
        ground = args[args.index("--ground") + 1]
    bgr = cv2.imread(photo, cv2.IMREAD_COLOR)
    if bgr is None:
        print("could not read", photo); sys.exit(1)
    H = 1000
    if bgr.shape[0] != H:
        bgr = cv2.resize(bgr, (int(bgr.shape[1] * H / bgr.shape[0]), H), interpolation=cv2.INTER_AREA)
    mask = foreground_mask(bgr)
    Image.fromarray(render_word_portrait(bgr, mask, words, ground=ground)).save(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
