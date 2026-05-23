# Typography Portrait Engine

An **isolated, deterministic** vector-rendering service that turns an uploaded
portrait photo plus a user-supplied word list into a clean black-and-white
typographic portrait. Words flow along the contours of the face, hair, eyes,
brows, nose, lips, and neck/shoulders.

This service is fully self-contained under `typography_engine/` and does **not**
touch the existing live segmentation app (`/app.py` at the repo root).

- No diffusion / generative image models.
- Only the user's approved words are ever drawn (cycled to fill each contour).
- Output SVG is well-formed and uses **hex colors only** (no `rgb()/rgba()/hsl()`).
- Text is never rendered below a configurable minimum font size; paths that
  cannot host a readable word are skipped with an explicit warning.

## Architecture

```
app/main.py                 FastAPI: / , /health, /debug/preprocess, /debug/regions, /render
app/config.py               RenderConfig (hex colors, min font, canvas, tuning)
app/pipeline/
  preprocess.py             OpenCV load / resize / CLAHE
  silhouette.py             GrabCut silhouette (head+shoulders seed) + confidence
  edges.py                  Canny edges + contour extraction
  landmarks.py              MediaPipe FaceLandmarker (optional) + Haar fallback
  regions.py                Decompose into named contour paths
  pathgen.py                Edge-set ordering + Catmull-Rom -> cubic Bezier 'd'
  textmeasure.py            Exact width via the font CairoSVG renders
  textlayout.py             Adaptive sizing, path fitting, orientation
  svgbuild.py               Strict hex-only SVG assembly + validation
  portrait.py               Full portrait assembler
  raster.py                 CairoSVG SVG -> PNG
  warnings.py               Debug-warning collector
static/index.html           Test frontend
tests/test_smoke.py         End-to-end TestClient suite
```

## Setup

```bash
cd typography_engine
pip install -r requirements.txt

# MediaPipe (optional, for precise facial landmarks) needs system GL libs once:
sudo apt-get install -y libgles2 libegl1 libglib2.0-0
# The face_landmarker.task model auto-downloads to models/ on first use.
# If MediaPipe or the model is unavailable, the engine degrades gracefully to an
# OpenCV Haar face box + silhouette outline and reports a warning.
```

## Run

```bash
./run.sh                 # serves on http://127.0.0.1:8077  (TYPO_PORT to override)
# open http://127.0.0.1:8077/  for the test frontend
```

## Endpoints

| Method | Path                 | Purpose                                         |
|--------|----------------------|-------------------------------------------------|
| GET    | `/health`            | Capabilities + version                          |
| POST   | `/debug/preprocess`  | Silhouette / edges / overlay debug PNGs         |
| POST   | `/debug/regions`     | Stroked region paths (SVG + PNG)                |
| POST   | `/render`            | Final typographic portrait (SVG + PNG)          |

`/render` form fields: `image` (file), `words` (comma/newline list) or
`words_json` (JSON array), `min_font_px`, `uppercase`, `foreground_hex`,
`background_hex`.

## Test (curl)

```bash
# Health
curl -s http://127.0.0.1:8077/health | python3 -m json.tool

# Render
curl -s -X POST \
  -F "image=@portrait.jpg" \
  -F "words=DREAM,COURAGE,CODE,NAVY,PIONEER,LEGEND" \
  -F "min_font_px=14" \
  http://127.0.0.1:8077/render | python3 -m json.tool
# -> JSON with svg/png URLs under /outputs/...
```

## Test (suite)

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

The suite uses FastAPI's `TestClient`, which exercises the full HTTP request /
response path (useful in sandboxes where loopback sockets are restricted).
