from flask import Flask, request, jsonify, send_file
from rembg import remove
from PIL import Image
import io, os, time, uuid

app = Flask(__name__)

# Short-lived store. PHP should fetch mask immediately and cache on its side.
STORE = {}
TTL_SECONDS = 1200  # 20 minutes

def _cleanup():
    now = time.time()
    dead = [k for k,v in STORE.items() if now - v["ts"] > TTL_SECONDS]
    for k in dead:
        STORE.pop(k, None)

def _mask_from_image_bytes(img_bytes: bytes) -> Image.Image:
    inp = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
    out = remove(inp)  # RGBA with transparent background
    alpha = out.split()[-1]  # alpha channel
    return alpha

def _bbox_from_mask(mask: Image.Image):
    w, h = mask.size
    pix = mask.load()
    thr = 16  # tolerant to soft edges
    minx, miny = w, h
    maxx, maxy = -1, -1
    fg = 0
    for y in range(h):
        for x in range(w):
            if pix[x, y] > thr:
                fg += 1
                if x < minx: minx = x
                if y < miny: miny = y
                if x > maxx: maxx = x
                if y > maxy: maxy = y
    if maxx < 0:
        return {"x":0,"y":0,"w":0,"h":0}, 0.0
    bbox = {"x": int(minx), "y": int(miny), "w": int(maxx-minx+1), "h": int(maxy-miny+1)}
    coverage = fg / float(w*h)
    return bbox, float(coverage)

def _confidence(mask: Image.Image, bbox: dict, coverage: float) -> float:
    w, h = mask.size
    if coverage <= 0.0001:
        return 0.0

    size_score = min(1.0, coverage / 0.35)  # ~0.35 coverage is healthy
    extreme_pen = 1.0
    if coverage < 0.06:
        extreme_pen *= 0.35
    if coverage > 0.75:
        extreme_pen *= 0.55

    touch = 0
    if bbox["x"] <= 2: touch += 1
    if bbox["y"] <= 2: touch += 1
    if bbox["x"] + bbox["w"] >= w-3: touch += 1
    if bbox["y"] + bbox["h"] >= h-3: touch += 1
    border_pen = [1.0, 0.90, 0.78, 0.62, 0.48][touch]

    conf = 0.25 + 0.55*size_score
    conf *= extreme_pen
    conf *= border_pen
    return float(max(0.0, min(1.0, conf)))

@app.get("/health")
def health():
    return jsonify(ok=True, service="segmentation", model="rembg-u2net", ttl_seconds=TTL_SECONDS)

@app.post("/segment")
def segment():
    _cleanup()
    t0 = time.time()

    if "image" not in request.files:
        return jsonify(ok=False, error="missing_image"), 400

    f = request.files["image"]
    img_bytes = f.read()
    if not img_bytes:
        return jsonify(ok=False, error="empty_upload"), 400

    try:
        mask = _mask_from_image_bytes(img_bytes)
        bbox, coverage = _bbox_from_mask(mask)
        conf = _confidence(mask, bbox, coverage)

        buf = io.BytesIO()
        mask.save(buf, format="PNG")
        png_bytes = buf.getvalue()

        mask_id = uuid.uuid4().hex[:16]
        STORE[mask_id] = {"ts": time.time(), "png": png_bytes}

        ms = int((time.time() - t0) * 1000)
        base_url = request.host_url.rstrip("/")
        return jsonify(
            ok=True,
            mask_id=mask_id,
            mask_url=f"{base_url}/mask/{mask_id}.png",
            bbox=bbox,
            coverage=coverage,
            confidence=conf,
            model="rembg-u2net",
            ms=ms
        )
    except Exception as e:
        return jsonify(ok=False, error="processing_failed", message=str(e)), 500

@app.get("/mask/<mask_id>.png")
def mask_png(mask_id):
    _cleanup()
    item = STORE.get(mask_id)
    if not item:
        return jsonify(ok=False, error="mask_not_found_or_expired"), 404
    return send_file(io.BytesIO(item["png"]), mimetype="image/png")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")))
