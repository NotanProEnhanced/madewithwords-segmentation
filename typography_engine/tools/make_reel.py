#!/usr/bin/env python3
"""Generate the single demo Typortrait promo reel (animated GIF).

Thin wrapper around reel_template.build_reel using the marketing hero assets.
For many reels from a manifest, use batch_reels.py. See tools/README.md.

Usage:
    python3 tools/make_reel.py                 # writes marketing/reel.gif
    python3 tools/make_reel.py out.gif         # custom output path
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from reel_template import build_reel

BASE = os.path.join(HERE, "..", "marketing")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE, "reel.gif")

out = build_reel({
    "before": os.path.join(BASE, "hero-before.jpg"),
    "after":  os.path.join(BASE, "hero-after.png"),
    "words": ["cherished","kind","radiant","timeless","beloved","gentle","brave",
              "1962","forever","laughter","grace","always our light"],
    "out": OUT,
})
print("wrote", out, "(%d KB)" % (os.path.getsize(out)//1024))

# Posting-ready MP4 (needs ffmpeg):
#   ffmpeg -i marketing/reel.gif -movflags +faststart -pix_fmt yuv420p \
#          -vf "scale=1080:-2:flags=lanczos,pad=1080:1920:0:(1920-ih)/2:color=0xFAF9F7" reel.mp4
