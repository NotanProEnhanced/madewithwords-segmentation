# Scribbler — Photo → Scribble Art (PWA)

A small, installable **Progressive Web App** that turns an uploaded photo into a
continuous pen‑scribble portrait, drawn **in realtime** right on the canvas.
Everything runs on the device — the photo is never uploaded anywhere.

![scribble portrait style](icons/icon-512.png)

## Features

- **Upload by drag‑and‑drop, file picker, or paste** (Ctrl/⌘+V).
- **Live pen drawing** — watch it draw as if by hand: each stroke is revealed
  progressively with a moving pen‑tip cursor, with a **Drawing speed** control
  and a **Show the pen** toggle. (Turn off *Animate the drawing* to render
  instantly.)
- **Hand‑scribbled strokes** — long, sweeping curves flow along the photo's
  tonal contours (hair, jawline, cheekbones) with a persistent "curl", a light
  second pen‑retrace over darks, and occasional stray overshoot lines, so the
  result reads like a real hand‑drawn scribble portrait rather than machine fuzz.
- **Single continuous line** — an optional mode that renders the whole portrait
  as one unbroken stroke (the pen never lifts): the detailed contour‑following
  scribbles are generated, then chained end‑to‑end by nearest endpoint so the
  pen draws them as a single line, preserving the facial detail.
- **One‑tap style presets** — *Sketch* (clean line‑art), *Bold* (heavier,
  dramatic strokes), and *Ink‑storm* (dense) set every slider at once. Your
  choice (and any manual tweaks) is **remembered** between visits and
  auto‑applied to the next photo.
- **Face fill** — a baseline ink demand across the subject so lit skin still
  gets light scribbles and the face reads fuller.
- **Remove background (subject only)** — on‑device subject segmentation isolates
  the person so ink stays on them and the background drops to clean paper. Uses
  **MediaPipe Selfie Segmentation** (runs in‑browser via WebAssembly,
  self‑hosted for full offline use); if it can't load, a colour‑keyed border
  flood‑fill takes over.
- **Trace facial features (likeness)** — **MediaPipe Face Landmarker** detects
  the eyes, brows, nose, lips, and jawline and draws them as guaranteed lines so
  the portrait actually resembles the subject (especially in single‑line mode).
  Skips gracefully when no face is detected.
- **Live controls**: density, contrast, scribble size, flow (chaotic ↔ contour),
  line weight, ink opacity, plus ink/paper colours.
- **Print-quality export** — **PNG** at ~3600 px long edge (≈ 300 dpi for a 12″
  print) and **SVG** vector (infinitely scalable, ideal for large prints and
  pen-plotters). Exports are recomputed deterministically so they match the
  on-screen sketch exactly, at any resolution.
- **Installable & offline** — PWA manifest + service worker cache the app shell.

## How it works

1. The photo is sampled into a small **tone map** (luminance grid).
2. A **contour field** is derived from the image gradient (Sobel) so strokes can
   flow along iso‑tone lines.
3. A **residual ink field** tracks how much ink each region still "wants"
   (darker = more). The engine repeatedly:
   - picks a dark spot (rejection sampling on the residual field),
   - lays down a short scribble stroke that follows the local contour with
     jitter, and
   - **subtracts** the ink it deposited so coverage self‑balances and tones come
     out even instead of clumping.

Strokes are drawn in batches via `requestAnimationFrame`, which is what makes the
drawing animate smoothly in realtime.

When **Remove background** is on, MediaPipe Selfie Segmentation produces a
subject matte first; the residual ink field is zeroed outside the subject so no
strokes land on the background. When **Trace facial features** is on, MediaPipe
Face Landmarker contributes feature lines that are drawn last (scribble mode) or
woven into the path (single‑line mode).

All logic lives in [`app.js`](app.js); no build step. The MediaPipe runtime
(JS + WASM) and both models are **self‑hosted** under
[`vendor/mediapipe/`](vendor/mediapipe) and precached by the service worker, so
the entire app — including segmentation and face tracing — works **fully
offline** with no first‑run network call. If those assets somehow fail to load,
it falls back to flood‑fill with no face traces.

### Updating the MediaPipe assets
```bash
# JS + WASM (SIMD build) from npm:
npm pack @mediapipe/tasks-vision@0.10.20
#   -> copy package/vision_bundle.mjs and package/wasm/vision_wasm_internal.{js,wasm}
#      into vendor/mediapipe/ and vendor/mediapipe/wasm/
# Models:
curl -o vendor/mediapipe/models/selfie_segmenter.tflite \
  https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite
curl -o vendor/mediapipe/models/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
```
Only the SIMD WASM build is bundled (supported by all current browsers); add the
`nosimd` variant if you must support very old browsers offline.

`_render_photo.py` is a dev‑only harness that mirrors the engine for eyeballing
output without a browser; it is not part of the shipped app.

## Run it locally

It's a static site — serve the folder over HTTP (a service worker needs
`http://`/`https://`, not `file://`):

```bash
cd scribble-pwa
python3 -m http.server 8000
# open http://localhost:8000
```

To install as an app, open it in Chrome/Edge/Safari and use **Install app** /
**Add to Home Screen**.

## Deploy

Drop the `scribble-pwa/` folder onto any static host (GitHub Pages, Netlify,
Vercel, Cloudflare Pages, S3, etc.). No server code required.

## Regenerating icons

App icons are generated by a dependency‑free script:

```bash
python3 make_icons.py
```
