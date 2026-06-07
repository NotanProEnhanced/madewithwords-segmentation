# Pet (non-human) subject support — design + spike

Status: **v1 spike** on branch `feat/pet-support`. Backend path implemented behind
a `subject` flag (defaults to `"person"`, so the human path is byte-identical).
Frontend toggle and tuning are deferred until the spike validates render quality.

## Why it's mostly already there

The tonal renderer builds the likeness from **tone + edges**
(`tonal.build_tonal_portrait` → `_tone_field`, `_edge_separate`) and only
*enhances* it with human-specific passes (`_emphasize_features`, `_sharpen_eyes`,
`_face_ovals`, `_balance_faces`). Two facts make pets feasible with minimal change:

1. **The face passes already self-skip.** `_faces_of()` returns empty for a pet,
   so those passes `return dark` unchanged. No crash — just no facial anchoring.
2. **The only hard blocker was the silhouette.** `silhouette.extract_silhouette`
   used MediaPipe's human **selfie segmenter**; on a pet it yields a poor/empty
   mask → nothing to fill. Give it a clean pet silhouette and the existing tonal
   renderer produces a recognizable pet word-portrait (fur, dark eyes/nose emerge
   from tone + edge separation).

## What the spike changes (all gated on `subject`)

| File | Change |
|---|---|
| `app/config.py` | `SUBJECT_DEFAULT`, `REMBG_MODEL` (env-tunable). |
| `app/pipeline/analyze.py` | `analyze_image(..., subject="person")` + `Analysis.subject`. For non-person: skip human face detection (`faces=[]`), route silhouette to the matte. |
| `app/pipeline/silhouette.py` | `_general_matte()` (rembg, lazy import) + `extract_silhouette(..., subject=)`. Person → selfie segmenter; pet/other → matte. GrabCut remains the fallback for both. |
| `app/pipeline/quality.py` | The input gate's human-face checks run only for `subject=="person"`; pets keep just the coverage checks (else a pet is rejected as "no face"). |
| `app/main.py` | `/render` accepts `subject` (Form), threads it to `analyze_image`, and persists `an.subject` in the job recipe. `_ensure_clean_png` (paid high-res recompose) reads `subject` from the recipe so the download matches the preview. |
| `requirements.txt` | `rembg` + `onnxruntime` (the matte + CPU inference). |
| `Dockerfile` | `libgomp1` (onnxruntime OpenMP runtime). |

Default `subject="person"` everywhere ⇒ **zero behavior change** for people, and
rembg/onnxruntime are imported lazily so the person path works even if they're
absent.

## How to test the spike on STAGING (not prod)

Build it in the staging environment by pointing the staging worktree at this
branch:
```bash
cd ~/typortrait-staging/typography_engine
git fetch origin
git checkout feat/pet-support
./staging.sh up -d --build         # installs rembg/onnxruntime; first build is slow
```
First pet render lazily downloads the rembg model (~170 MB) to the container —
the first one is slow, then cached. Test via the API (the UI toggle is v1, not in
the spike), authenticating with your staging Basic Auth:
```bash
curl -s -u 'youruser:pass' -X POST https://staging.typortrait.com/render \
  -F image=@/path/to/dog.jpg \
  -F subject=pet \
  -F 'words=loyal brave good boy rascal companion' \
  -F ink=gold_noir -F style=mosaic \
  | python3 -c "import sys,json;d=json.load(sys.stdin);print(d.get('ok'),d.get('job'),d.get('preview_url'))"
```
Open the returned preview URL (watermarked) and judge likeness across a few pets
(short-haired dog, cat, fluffy dog) and inks. **Judgement call:** is tone-only
likeness good enough to sell, or do we need v2 feature anchoring?

## Risks / things to validate

- **Dependency conflict (most likely):** `rembg` pulls `opencv-python-headless`,
  but this repo uses `opencv-contrib-python`. Two OpenCV distributions can clash.
  If the build or `import cv2` breaks, reconcile to one OpenCV (e.g. install rembg
  with `--no-deps` plus its other deps, or switch the repo to headless+contrib).
- **Image size / RAM:** onnxruntime + model add ~150–250 MB and meaningful RAM.
  Confirm the VPS has headroom before promoting.
- **First-render latency:** model download on first pet render. Optional: prefetch
  in the Dockerfile once the model choice is settled.
- **Quality variance:** flat/low-contrast or very fluffy pets read weaker without
  feature anchoring. The free watermarked preview is the safety net.

## Roadmap

- **v1 (after spike validates):** frontend `Person · Pet` toggle in `static/index.html`
  (sends `subject`, adds a `tp_track` event); a small "pet profile" in `tonal`
  (slightly higher `_edge_separate` for fur, tuned `auto_tone`); optional Dockerfile
  model prefetch.
- **v2:** "tap the eyes" → synthesize a face/eye region so the finer word tier +
  eye emphasis kick in (big likeness boost). Cat faces: OpenCV ships
  `haarcascade_frontalcatface.xml` (zero new deps); dogs need a small YOLO model.
- **v3:** auto subject detection (try human face; fall back to matte) with a UI
  override; later, multi-subject (person + pet together).

## Marketing gate

Do **not** re-add "pet portraits" to the marketing FAQ / Pinterest (or create the
Pet board + pins) until v1 ships and likeness is validated on real pets. It was
removed from the live FAQ in commit `3131ef9`.
