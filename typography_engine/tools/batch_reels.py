#!/usr/bin/env python3
"""Batch-generate Typortrait reels from a manifest — the content factory.

Reads a JSON manifest (list of items) and, for each one, produces a folder under
reels_out/<id>/ containing:
  <id>-before.jpg, <id>-after.png   (rendered from the photo, unless supplied)
  <id>.gif                          (the reel)
  <id>.mp4                          (if ffmpeg is installed)
  <id>-caption.txt                  (caption + hashtags, ready to paste)

Manifest item keys:
  id      (str, required)            output folder / file stem
  words   (list[str], required)      words shown in the reel
  photo   (path)                     source photo -> rendered via the engine
    ...or supply a prebuilt pair instead of `photo`:
  before  (path)                     1:1 photo crop
  after   (path)                     1:1 rendered word-portrait
  ink     (str, default gold_noir)   mono|navy|sepia|burgundy|forest|gold_noir|spectrum|aurora|photo
  style   (str, default words)       words|message
  light   (bool, default false)
  caption (str)                      extra line appended to the caption

Usage:
  python3 tools/batch_reels.py [manifest.json]   # default: tools/reels_manifest.example.json
"""
import os, sys, json, shutil, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from reel_template import build_reel

CAPTION = (
    "Their portrait — written in the words that describe them. \U0001f90d\n\n"
    "Upload a photo, add the words that matter ({words}…), and watch it become "
    "a portrait made entirely of text.\n\n"
    "Free to create & preview — pay only if you love it.\n"
    "✨ Make one → typortrait.com"
)
HASHTAGS = ("#typortrait #wordart #wordportrait #typographyart #portraitart "
            "#personalizedgift #customgift #meaningfulgifts #keepsake #giftideas "
            "#anniversarygift #memorialgift #giftsforher #giftsforhim")

def _default_audio_path():
    """Optional music bed. Override with TYPO_REEL_AUDIO=/abs/path/to/track.mp3.
    See tools/assets/README.md for sourcing license-clear tracks."""
    env = os.environ.get("TYPO_REEL_AUDIO", "").strip()
    if env:
        return env
    default = os.path.join(HERE, "assets", "reel_audio.mp3")
    return default if os.path.exists(default) else None


def to_mp4(gif, mp4):
    """Backwards-compat wrapper: routes to reel_template.convert_gif_to_mp4
    so this script and the live /reel endpoint share one encoder + audio path."""
    from reel_template import convert_gif_to_mp4
    return convert_gif_to_mp4(gif, mp4, audio_path=_default_audio_path())

def main():
    manifest = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "reels_manifest.example.json")
    items = json.load(open(manifest))
    out_root = os.path.join(HERE, "..", "reels_out")
    ok = 0
    for it in items:
        try:
            stem = it["id"]; words = it["words"]
            d = os.path.join(out_root, stem); os.makedirs(d, exist_ok=True)
            if it.get("photo"):
                from render_pair import render_pair        # imported lazily (heavy engine deps)
                before, after = render_pair(it["photo"], words, ink=it.get("ink","gold_noir"),
                                            style=it.get("style","words"), light=it.get("light", False),
                                            out_dir=d, stem=stem)
            else:
                before, after = it["before"], it["after"]
            gif = os.path.join(d, stem+".gif")
            build_reel({"before":before, "after":after, "words":words, "out":gif,
                        "cta":it.get("cta","Create yours free  ·  typortrait.com")})
            cap = CAPTION.format(words=", ".join(words[:5]))
            if it.get("caption"): cap += "\n\n" + it["caption"]
            open(os.path.join(d, stem+"-caption.txt"), "w", encoding="utf-8").write(cap + "\n\n" + HASHTAGS + "\n")
            mp4 = os.path.join(d, stem+".mp4"); made_mp4 = to_mp4(gif, mp4)
            ok += 1
            print("[ok] %-16s -> %s%s" % (stem, gif, "  (+mp4)" if made_mp4 else ""))
        except Exception as e:
            print("[FAIL] %s: %s" % (it.get("id","?"), e))
    print("done: %d/%d  (output in reels_out/)" % (ok, len(items)))

if __name__ == "__main__":
    main()
