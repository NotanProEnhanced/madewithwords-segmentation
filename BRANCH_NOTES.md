# `claude/calligram-classic` — Branch Notes

**Branched from `2e082a1` (2026-05-30 noon) to restore the renderer lineage Jeff remembers as "the finest."**

The current `claude/printful-pod` ship line (`calligram-shipline-700bf09`) was deemed to have drifted from that aesthetic target over the 2026-05-31 → 2026-06-01 iteration window. Rather than revert destructively, this branch resumes work from the pre-drift state, in parallel.

## Why 2e082a1

That commit is the first complete state of the `build_calligram` builder after the introduction of **per-glyph photo modulation** (commit `a70eea3`) — photo tone runs THROUGH each letter shape rather than each letter being a flat colour. It also added a stronger in-face curve. This is the technique that defines the engine's character.

## Engine API at branch point

- Entry: `app.pipeline.tonal.build_calligram(an, passage, cfg, warns, ink_hex, bg_hex)`
- Palettes (`_CALLIGRAM`): `navy, sepia, burgundy, forest, gold_noir, mono, photo`
- Returns `(svg, runs, modulation_png)` — `modulation_png` is the per-pixel photo-modulated raster
- Light_bg and dark_bg paths both produce modulated PNG
- Subject-only handled internally
- Source photo: `typography_engine/Margot-Source.JPG`

## Aesthetic target

Jeff's reference for "finest" is the **canonical Margot rendering** — typography_engine/Margot-Source.JPG run through gold_noir with the per-glyph modulation. Characteristics that constitute "good":

- **Pupils read as dark voids** — letters at the pupil location fade to bg colour
- **Sclera reads as bright** — letters at sclera pop with full ink
- **Per-glyph photo tone** — each letter carries the photo's brightness from underneath, not a uniform fill
- **Density follows tone** — denser typography where the photo has more information
- **Word-level legibility** — keywords stay readable as words, not letter mush
- **No rivers or banding** — vertical/diagonal column alignment must be broken

## Recovery tags (every prior milestone is preserved)

```
calligram-current-784a10c     (5/30 evening, original tag)
calligram-shipline-700bf09    (6/01 ship line, full pipeline)
calligram-restored-3fa24f7    (6/01 first "restore" attempt)
calligram-photoreal-ab901d3   (6/01 photorealism pass)
calligram-jitter-4903587      (6/01 anti-river jitter work)
calligram-maxfx-1795d28       (6/01 max micro-features)
calligram-anatomy-e01ab96     (6/01 anatomical gradient features)
+ 9 more — see `git tag -l 'calligram-*'`
```

## Test photo set (use for every iteration)

```
typography_engine/Margot-Source.JPG                                    (toddler, light skin, soft light)
typography_engine/source.png                                           (young adult, medium skin)
typography_engine/tests/fixtures/cache/obama_blackmale.jpg             (dark skin, studio light)
typography_engine/tests/fixtures/cache/biden_oldwhitemale.jpg          (light skin, elderly, news light)
```

Render all 4 every change. Side-by-side comparison is the only honest test.

## Known-FAILED experiments (avoid repeating)

These were tried in the ship-line timeline and rejected by Jeff's eye. Skip them:

- **Bebas Neue font** — looked "stamped on" against the typewriter feel Margot has
- **Silhouette-edge DOF blur** — read as Photoshop blur, not photographic
- **Lacrimal caruncle brightness lift** — produced visible crescent artifacts
- **Stochastic density skip in highlights** — created Swiss-cheese face
- **Hard 2.10× contrast stretch on light_bg** — clipped all features to identical 1.0 ink
- **Painted solid black pupil disks on light_bg** — looked like stickers
- **Forced catchlight via cv2.circle** — looked like a sticker
- **Inverting a dark_bg render to fake light_bg** — produced photographic negatives

## Known-WIN infrastructure available to port forward

These changes from the ship line (`claude/printful-pod`) are aesthetic-neutral or pure wins. **Do NOT port them blindly** — but they're available to cherry-pick when you have a stable classic build:

- `resvg-py` SVG rasterizer in `raster.py` (~2.5× faster, identical output)
- Per-row word shuffle + per-letter sub-pixel jitter (kills banding/rivers, no aesthetic change)
- Adaptive iris/xxs tier sizes (scales to detected eye width)
- Keepsake `compose_poster_png` for PIL-based poster output
- Root URL serving `/static/index.html` directly (no redirect)
- 5-palette ship line UI: black_ink, gold_noir, navy_marigold, white_black, white_spectrum

Cherry-pick command pattern:
```bash
git cherry-pick <commit-from-ship-line>
```

## Quick start for the next chat

```bash
# Verify branch
git branch --show-current   # should show claude/calligram-classic

# Render Margot through current engine state
cd typography_engine
python -c "
from app.config import RenderConfig
from app.pipeline.analyze import analyze_image
from app.pipeline.tonal import build_calligram, _CALLIGRAM
from app.pipeline.warnings import WarningCollector
img = open('Margot-Source.JPG', 'rb').read()
cfg = RenderConfig(); warns = WarningCollector()
an = analyze_image(img, cfg, warns)
passage = ' '.join(['MARGOT', 'BEAUTIFUL', 'SISTER', 'DAUGHTER', 'ADORABLE'] * 80)
ink, bg = _CALLIGRAM['gold_noir']
svg, runs, mod = build_calligram(an, passage, cfg, warns, ink_hex=ink, bg_hex=bg)
if mod: open('/tmp/test.png', 'wb').write(mod)
print('runs:', len(runs), 'modulation:', bool(mod))
"
```

## Open questions for next session

1. **Spec the targets quantitatively.** Currently "finest" is Jeff's eye. Before tuning, render the 4 test photos and have Jeff annotate what specifically constitutes the win — eye depth, lip definition, hair texture, etc.
2. **Compare against ship line on same photos.** Render `claude/calligram-classic` vs `claude/printful-pod` on the same 4 photos. Identify what each does better. The answer may be "ship one, ship both as a style menu, or merge."
3. **Decide on light_bg.** This branch's `_CALLIGRAM` includes `mono` / `photo` (light bg). The ship line dropped them then re-added `white_black` / `white_spectrum`. Decide whether the classic branch wants light_bg at all.

## Ground rules

- One change per commit. Descriptive message with the WHY.
- Render the 4 test photos after each change. Compare visually.
- Tag every milestone Jeff approves as "good." Lossless rollback.
- Don't tune numbers; tune *what we measure with*. Add a feature only when there's a clear visual deficiency in the test set.
- Never delete or destructively rewrite history. Branch instead.
