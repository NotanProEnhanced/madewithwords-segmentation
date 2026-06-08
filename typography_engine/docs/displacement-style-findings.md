# Displacement Typographic Portrait — Findings & Productionization Spec

**Status:** Prototype validated (scratch, isolated). **Not** in production. Pending a build‑vs‑shelve decision.
**Date:** 2026‑06‑08
**Scope:** A new human‑portrait *Style* for the studio ("Displacement"), alongside the live **Words** and **Passage** styles.

---

## 1. What it is

A premium typographic‑portrait style where **horizontal rows of the user's meaningful words are warped by the photo's luminance** so the text drapes over the facial form (the classic Photoshop "type‑follows‑the‑form" / displacement look), combined with a **multi‑tier feature‑detail system** so eyes/nose/mouth render with finer text than the broad form.

It is distinct from the current renderers:
- **Words** = words scattered by tone (mosaic).
- **Passage** = prose laid in tonal rows.
- **Displacement** = rows *warped over the 3‑D form* + tiered feature detail. The "iconic designer" look.

**Key advantage over the manual Photoshop technique:** Photoshop displaces by luminance only. We have the **MediaPipe 478‑point face mesh**, enabling *feature‑aware* displacement and anchoring — so we can match and exceed the manual technique automatically, in seconds.

---

## 2. The working recipe (reproducible)

All of this ran in an **isolated scratch script** that imports the production pipeline **read‑only** (`analyze_image` for face mesh + silhouette) and renders with its own logic. Nothing in `app/` or `static/` was modified. The recipe below is the validated "round‑3" version.

Pipeline (per image):
1. **Analyze** → `analyze_image` gives `img.gray`, `silhouette.mask`, `landmarks.points` (478×2 px), `face_bbox`.
2. **Supersample** working buffers ×2 (crisp glyphs), downscale final for output.
3. **Subject‑relative scale** `s = clip(face_frac / 0.47, 0.5, 1.3)` where `face_frac = face_bbox_w / image_w`. **Critical** — without it the recipe overfits one framing and the face over‑warps on differently‑framed photos.
4. **Render 3 text tiers** (random word order + random horizontal phase per row → no "rivers"):
   - Coarse rows `64·s` px (broad form / drape)
   - Fine rows `22·s` px (features)
   - Micro rows `13·s` px (eyes)
5. **Feature masks** from landmark groups (eyes, brows, nose, lips):
   - `feat` (tight) → where fine text replaces coarse
   - `eye` → where micro text replaces fine
   - `feat_damp` (wide, feathered) → where displacement is **dampened**
6. **Displacement field — clean vertical drape only:** `my = y + 64·s·dn·(1 − 0.85·feat_damp)`, `mx = x`, where `dn = (blur(gray) − 0.5)·2`. Dampening in the feature band keeps eyes/nose/mouth flat and crisp while cheeks/forehead/body drape. (Earlier high‑frequency gradient terms caused feature "mush" and were removed.)
7. **Composite tiers:** `warped = coarse·(1−feat) + fine·feat`, then `·(1−eye) + micro·eye`. Remap all by the same field.
8. **Tonal mapping (full range):** percentile‑stretch the subject's luminance (4th–96th). For **dark ground** (the hero look) the white text follows the **highlights** (`light = stretched lum`); for **light ground** it follows the shadows. **Local‑contrast boost** in the face (`+0.40·highpass·face_weight`) so flat‑lit features separate.
9. **Progressive shadow density:** dilate/thicken text where the source is darkest (dense blacks → faint highlights).
10. **Feature anchoring:** thin dark lines on eye rings + lip seam, dark pupils + nostrils, from landmarks. **Essential** — without it, features under‑read on flat‑lit faces.
11. **Edge feathering** of the silhouette so hair/shoulders dissolve into the ground.
12. **Compose** onto the chosen ground.

### Color grounds (slot into the existing ink‑swatch UI)
- **Black‑on‑white** (paper) — classic.
- **White‑on‑navy** — premium, gallery. **Recommended hero look.**
- **White‑on‑black** — maximum drama / cinematic.

The dark grounds are the standout (more premium than paper). They are a *correct* dark mode (white text in highlights, dissolving to ground in shadow) — not the washed‑out "light mode" that was previously removed.

---

## 3. What is validated

- ✅ **Bare faces:** robust and beautiful across **5 diverse AI‑generated faces** (varied age, gender, skin tone) **+ hero + Margot**, after adding subject‑relative scaling.
- ✅ **Three color grounds** all work; navy/black are the hero looks.
- ✅ **Bold / acetate eyeglass frames** render **faithfully from the source** — a thick frame is a substantial dark region in the photo, so the tonal/text rendering reproduces it. No simulation needed. (Confirmed on a generated portrait with heavy black square frames.)

### Hard problems solved during prototyping (all already solved in the production tonal renderer)
1. **Subject‑relative scaling** — overfitting to one framing → scale displacement + font tiers by face size.
2. **Feature over‑warp ("mush")** — strong displacement destroyed features → drop gradient terms + dampen displacement in the feature band.
3. **Feature anchoring** — flat‑lit faces lost definition → explicit eye/lip/nostril anchors + local‑contrast boost.

---

## 4. Known limitations & untested edge cases

- ⚠️ **Thin metal / wire eyeglass frames:** render **faint or lost** — they're barely present in the tonal signal. Arguably *true to source* (thin wire is subtle in reality), but a glasses‑wearer in wire frames won't see prominent glasses. Improvable **only** via real frame **detection + source‑true rendering** (hard CV/ML R&D) — **never** by simulating/drawing fake frames (explicitly rejected: not true to the source image).
- ❓ **Untested:** hats, heavy facial hair, strong profile / ¾ angles, hair/hand occlusion, multiple faces in frame, extreme skin‑tone + lighting combinations, very tight or very loose framing extremes.
- The hardest real faces are still **acceptable** but less premium than the most forgiving ones.

### Rejected approaches (do not revisit)
- **Simulated/drawn glasses** (stylized frames from landmarks): produced a readable result but is **not true to the source** — rejected on principle. A portrait must reflect the actual photo.
- **CV frame detection for thin wire** (blackhat, connected components, adaptive threshold): all failed — frames too low‑contrast to isolate reliably, and a "has‑glasses" gate from these signals false‑positives on brows/lashes.

---

## 5. Productionization plan

This is a **real renderer build (multi‑day), not a scratch tweak.** The prototype proved the look and mapped the cost; the efficient path is to **reuse the production renderer's solved foundations** rather than re‑deriving them.

**Build outline:**
1. New module `app/pipeline/displacement.py` → `render_displacement_portrait(...)`, mirroring `tonal.py`'s interface (SVG/PNG out, same `analyze_image` input).
2. **Reuse** the production analysis, **subject‑relative scaling**, and especially the existing **eye/feature anchoring** code (don't hand‑roll the scratch versions).
3. Add **"Displacement"** as a third `style` in `/render` and a third option in the `#styleSeg` control (Words / Passage / Displacement).
4. Add **ground** selection (paper / navy / black) — fits the existing ink‑swatch mechanism; navy default.
5. **Test harness:** run across a large, diverse face set (≥30 incl. glasses, hats, beards, angles, skin tones) before any deploy. Gate on "reads as a defined face."
6. **Edge‑case backlog (each its own task):** thin‑wire glasses (detection R&D), hats/occlusion, profiles, multiple faces.
7. Deploy via the existing `promote.sh` flow once verified on staging.

**Effort estimate:** ~1–2 weeks for a polished, well‑tested Style (excludes the thin‑wire‑glasses R&D, which is open‑ended and optional).

---

## 6. Strategic note

Adding styles to the human‑portrait studio is **lower‑risk, higher‑ROI** than new markets (e.g. Homes), because every style reuses the existing analysis → render → watermark → checkout → print pipeline with **no new dependencies**, on a proven, live funnel. Displacement is the strongest premium candidate and is uniquely enabled by the face mesh you already compute.

**Recommended sequence:** deepen the human studio (styles) first; revisit Homes (new market + new ML, still under research) second.
