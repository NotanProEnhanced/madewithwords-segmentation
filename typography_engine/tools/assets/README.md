# Reel music bed

The reel pipeline (`tools/batch_reels.py` and the live `/reel` endpoint)
will mix an audio file into the generated MP4 if one is present at:

    tools/assets/reel_audio.mp3

Or wherever you point the `TYPO_REEL_AUDIO` env var (absolute path).

If the file is missing, the pipeline silently falls back to producing a
**silent** MP4 — which works, but TikTok and Instagram Reels both
algorithmically downrank content without sound, so you'll lose reach.

## How the audio is used

ffmpeg loops the track if it's shorter than the reel, fades in over
0.6s, fades out over 1.2s before the end of the video, and mixes at
volume 0.45 so it sits under the visuals rather than dominating them.
Output codec is AAC 96kbps — small (~200KB extra per reel) and
universally supported.

## License — important

The track you drop in here will be published on Instagram, TikTok, etc.
as part of every consented reel. **Only use tracks that are explicitly
licensed for commercial use with no attribution required.** A few
sources that meet that bar:

### Pixabay Music — recommended
- https://pixabay.com/music/
- License: Pixabay License — free for commercial use, no attribution
  required. Some tracks marked "Use without attribution required".
- Search: `ambient cinematic`, `soft piano`, `emotional cinematic`,
  `inspiring background`. Aim for **18+ seconds**, calm/warm vibe.

### Mixkit
- https://mixkit.co/free-stock-music/
- License: Mixkit License — free for commercial use, no attribution
  required for music in this section.

### YouTube Audio Library
- https://studio.youtube.com/ → Audio Library
- License varies per track. Filter to **"No attribution required"**.
- Caveat: tracks are licensed for use on YouTube; for cross-posting to
  IG / TikTok you'll want a Pixabay or Mixkit track instead to avoid
  ambiguity.

### Avoid
- Spotify / Apple Music tracks (no commercial license)
- "Free" tracks that require crediting the artist (you can't put a
  credit in a 9:16 reel cleanly, and platforms strip text overlays
  the longer they live)
- Anything you found via Google search without an explicit license link

## Recommended workflow

1. Pick one track that matches the brand (calm, cinematic, warm — not
   energetic dance/EDM, not melancholy minor-key).
2. Download as MP3.
3. Rename to `reel_audio.mp3` and drop here, **or** put it somewhere on
   the server and set `TYPO_REEL_AUDIO=/abs/path/to/file.mp3` in `.env`.
4. Commit the file to git (so it ships with the deploy) OR keep it
   out-of-tree if you don't want the binary in your repo history.
5. Rebuild: `sudo docker compose up -d --build`.
6. Generate a new reel via `/admin/reels/<job>/resend` or by going
   through a fresh consented checkout — verify the MP4 plays with audio.

## Switching tracks later

Replacing `reel_audio.mp3` and rebuilding affects only **new** reels.
Already-generated MP4s in `data/outputs/` keep whatever audio they
shipped with. If you need to regenerate an older reel with the new
track, delete its `_reel.mp4` file from `data/outputs/` and use the
dashboard's **Re-send notification email** action on a queued job
(which doesn't rebuild — you'd need to re-run the user-facing flow).
A future enhancement could add an admin-side "rebuild reel" button if
this becomes a frequent need.
