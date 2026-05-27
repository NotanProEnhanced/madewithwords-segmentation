# Typortrait promo reel

`make_reel.py` generates the ~18s vertical (9:16) promo reel as an animated GIF
from the marketing hero assets.

**Beats:** hook (first frame = thumbnail) → "Start with a photo." → "Then, add
the words that matter." (words dissolve in) → portrait resolves → before/after
**drag** slider → "Made from your words." + CTA (held ~3s).

## Generate it

Requires Python 3 + Pillow. Fonts are auto-located (Windows Georgia/Arial, or
Linux Liberation/DejaVu/FreeSerif).

### Windows
```
pip install Pillow
cd typography_engine
python tools\make_reel.py
```

### Linux / VPS
```
sudo apt-get install -y python3-pip fonts-liberation && pip3 install Pillow
cd typography_engine
python3 tools/make_reel.py
```

Output: `marketing/reel.gif` (or pass a path: `python3 tools/make_reel.py out.gif`).

## Export a posting-ready MP4 (recommended for social)

A GIF is heavy and lower quality; platforms prefer MP4 (H.264). With `ffmpeg`
installed (`winget install Gyan.FFmpeg` on Windows, `sudo apt-get install -y
ffmpeg` on Linux):

```
ffmpeg -i marketing/reel.gif -movflags +faststart -pix_fmt yuv420p \
  -vf "scale=1080:-2:flags=lanczos,pad=1080:1920:0:(1920-ih)/2:color=0xFAF9F7" reel.mp4
```

## Batch — hundreds of reels (the content factory)

`batch_reels.py` reads a JSON manifest and produces one folder per item under
`reels_out/<id>/` with the rendered **before/after pair**, the **reel GIF** (and
**MP4** if `ffmpeg` is installed), and a paste-ready **caption + hashtags**.

```
python3 tools/batch_reels.py                      # uses tools/reels_manifest.example.json
python3 tools/batch_reels.py my_manifest.json     # your own list
```

Manifest item keys: `id`, `words` (required); `photo` (rendered via the engine)
**or** a prebuilt `before`+`after` pair; plus `ink`, `style`, `light`, `caption`.
See `reels_manifest.example.json`. (Engine rendering needs the app deps + `models/`,
so run it on the server or inside the container; the prebuilt-pair path only needs
Pillow.) Output under `reels_out/` is git-ignored.

The pieces are shared: `reel_template.build_reel(cfg)` is the single drawing
routine used by both `make_reel.py` and `batch_reels.py`, and it's the same code
you'd call server-side for a per-user **"Share as a reel"** button.

## Flywheel

```
Create a portrait  ->  auto-package as a reel (this template)  ->  post to
Reels/Stories/TikTok/Pinterest  ->  reach  ->  new users create  ->  (repeat)
```

- **Seed fuel:** batch-generate demo reels (faces × occasions × colours) to keep
  the channels full — that's what the manifest + `batch_reels.py` are for.
- **Compounding fuel:** wire `build_reel()` into the app so every render offers a
  one-tap **"Share as a reel."** User-made reels are authentic, free distribution
  that grows with usage.
- **Attribution:** give each reel a UTM link (e.g. `typortrait.com/?utm_source=ig&utm_campaign=mothersday`),
  watch which occasions/styles convert, and weight the next batch toward winners.
- **Cadence:** batch builds a backlog; schedule a steady drip rather than dumping
  all at once.

## Customising

All near the top of `make_reel.py`, clearly labelled:
- `WORDS` — the words shown.
- captions — in `frame()`.
- timeline — `HOOK_END`, `WORDS_IN/DUR`, `REVEAL0/1`, `SLIDER_END`, `DUR`, `HOLD`.
- look — `BG`, `INK`, `MUTED`; size `W, H, FPS`.
- The slider drag speed is the `SLIDER_END` window and the `0.38*sin(...)` sweep.

> Note: the committed `hero-before.jpg` / `hero-after.png` are a **synthetic demo
> face**. Drop in a real, owned portrait pair (same square crop) to feature a real
> person before publishing.

---

# Posting kit

### On-screen text (already baked into the reel)
1. **A portrait, written in your words.**  ·  *here's how ↓*
2. **Start with a photo.**
3. **Then, add the words that matter.**
4. **Drag to compare ↔**
5. **Made from your words.**  ·  *Create yours free · typortrait.com*

### Suggested caption
> Their portrait — written in the words that describe them. 🤍
>
> Upload a photo, add the words that matter (a name, their traits, a date, even a
> whole letter), and watch it become a portrait made entirely of text. A keepsake
> as personal as the message inside it.
>
> Free to create & preview — pay only if you love it.
> ✨ Make one → typortrait.com

### Hashtags (mix broad + niche; trim to ~10–15 per post)
```
#typortrait #wordart #wordportrait #typographyart #portraitart
#personalizedgift #customgift #meaningfulgifts #sentimentalgifts #keepsake
#giftideas #anniversarygift #memorialgift #giftsforher #giftsforhim
```

### Audio
GIF/MP4 carry no sound — add a track in each platform's editor. Pick a trending,
emotional/uplifting song and line a beat-hit up with the **reveal (~8s)** and the
**drag (~9–12s)**. Reels with native trending audio get more reach than uploaded
audio.

### Specs & where it posts
- Master: **1080×1920**, MP4 (H.264 + AAC), < 90s. Set the **cover/thumbnail to
  the hook frame**.
- Works as-is on **Instagram Reels/Stories, Facebook Reels/Stories, TikTok,
  YouTube Shorts, Pinterest**.
- **X/Twitter**: not vertical-first — also export a **1:1 or 16:9** crop.
- Keep all text in the **centre safe zone** (the reel already does) so platform
  UI (caption, buttons, profile) doesn't cover it.
- **Don't** burn in another platform's watermark (e.g. the TikTok logo) when
  cross-posting — Meta down-ranks it; re-export clean from the master.
