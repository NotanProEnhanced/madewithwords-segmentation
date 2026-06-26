"""Print-ready memorial / prayer card from a word-portrait.

Produces a 2-page PDF (front, back) at the standard prayer-card finished size
2.5 x 4.25 in, with 0.125 in bleed and corner crop marks — the exact file a funeral
printer or a print-on-demand API (Prodigi/Gelato) needs. Parameterized so it can be
wired into the studio product later: make_card(portrait, name, dates, words, verse).

Deps: reportlab, pillow.  CLI: python memorial_card.py  -> a sample on the Desktop.
"""
import os, textwrap
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from PIL import Image

PT = 72.0                                   # points per inch
TRIM_W, TRIM_H = 2.5 * PT, 4.25 * PT        # finished card
BLEED = 0.125 * PT                          # extend art past the trim
SLUG  = 0.20 * PT                           # margin around the bleed for crop marks
SAFE  = 0.16 * PT                           # keep text this far inside the trim
PAGE_W = TRIM_W + 2 * (BLEED + SLUG)
PAGE_H = TRIM_H + 2 * (BLEED + SLUG)
# trim box origin (from page corner)
TX0, TY0 = SLUG + BLEED, SLUG + BLEED
TX1, TY1 = TX0 + TRIM_W, TY0 + TRIM_H

CREAM = (244/255, 239/255, 230/255); INK = (45/255, 38/255, 31/255)
MUTED = (120/255, 112/255, 100/255); ROSE = (176/255, 90/255, 80/255)
LINE  = (214/255, 205/255, 191/255); PANEL = (13/255, 13/255, 15/255)
GOLD  = (150/255, 120/255, 70/255)

SERIF, SERIF_I, SERIF_B = "Times-Roman", "Times-Italic", "Times-Bold"

VERSES = {
    "words":   "Her portrait was made from the words of everyone who loved her — "
               "so her story stays close, always.",
    "psalm23": "The Lord is my shepherd; I shall not want. "
               "He maketh me to lie down in green pastures.",   # Psalm 23 (KJV, public domain)
    "light":   "What we have once enjoyed we can never lose; "
               "all that we love deeply becomes a part of us.", # Helen Keller (public domain)
    "none":    "",
}

def _wrap(c, text, font, size, max_w):
    c.setFont(font, size)
    out, line = [], ""
    for w in text.split():
        t = (line + " " + w).strip()
        if c.stringWidth(t, font, size) <= max_w: line = t
        else: out.append(line); line = w
    if line: out.append(line)
    return out

def _crop_marks(c):
    c.setStrokeColorRGB(0, 0, 0); c.setLineWidth(0.4)
    g, L = 3, 11                                  # gap from trim corner, mark length
    for (x, sx) in ((TX0, -1), (TX1, 1)):
        for (y, sy) in ((TY0, -1), (TY1, 1)):
            c.line(x + sx*g, y, x + sx*(g+L), y)  # horizontal tick
            c.line(x, y + sy*g, x, y + sy*(g+L))  # vertical tick

def _bg(c):
    c.setFillColorRGB(*CREAM)
    c.rect(SLUG, SLUG, TRIM_W + 2*BLEED, TRIM_H + 2*BLEED, fill=1, stroke=0)  # art to bleed
    # double inner keyline just inside the trim
    c.setStrokeColorRGB(*LINE); c.setLineWidth(1)
    c.rect(TX0+5, TY0+5, TRIM_W-10, TRIM_H-10, fill=0, stroke=1)
    c.setLineWidth(0.6); c.rect(TX0+9, TY0+9, TRIM_W-18, TRIM_H-18, fill=0, stroke=1)

def _ctr(c, y, text, font, size, rgb):
    c.setFillColorRGB(*rgb); c.setFont(font, size)
    c.drawCentredString((TX0+TX1)/2, y, text)

def _front(c, portrait, name, dates):
    _bg(c)
    cx = (TX0 + TX1) / 2
    _ctr(c, TY1 - 26, "In Loving Memory", SERIF_I, 13, MUTED)
    # portrait, inset with a thin dark panel
    im = Image.open(portrait).convert("RGB"); ar = im.height / im.width
    pw = TRIM_W - 2*22; ph = pw * ar
    px, py = cx - pw/2, TY1 - 40 - ph
    c.setFillColorRGB(*PANEL); c.rect(px-3, py-3, pw+6, ph+6, fill=1, stroke=0)
    c.drawImage(ImageReader(im), px, py, pw, ph)
    y = py - 20
    _ctr(c, y, name, SERIF_B, 12.5, INK); y -= 17
    c.setStrokeColorRGB(*GOLD); c.setLineWidth(0.6)
    c.line(cx-34, y+4, cx-9, y+4); c.line(cx+9, y+4, cx+34, y+4)
    _ctr(c, y, "+", SERIF, 9, ROSE); y -= 15
    _ctr(c, y, dates, SERIF, 9.5, MUTED)

def _back(c, name, dates, words, verse):
    _bg(c); cx = (TX0 + TX1) / 2
    y = TY1 - 34
    _ctr(c, y, name, SERIF_B, 11.5, INK); y -= 14
    _ctr(c, y, dates, SERIF_I, 8.5, MUTED); y -= 20
    _ctr(c, y, "+", SERIF, 10, ROSE); y -= 16
    c.setStrokeColorRGB(*LINE); c.setLineWidth(0.6); c.line(TX0+24, y, TX1-24, y); y -= 16
    for ln in _wrap(c, words, SERIF, 9.5, TRIM_W - 2*20):
        _ctr(c, y, ln, SERIF, 9.5, INK); y -= 13
    y -= 6; c.line(TX0+24, y, TX1-24, y); y -= 18
    if verse:
        for ln in _wrap(c, verse, SERIF_I, 8.5, TRIM_W - 2*22):
            _ctr(c, y, ln, SERIF_I, 8.5, MUTED); y -= 11
        y -= 12
    _ctr(c, y, "Forever in our hearts", SERIF_B, 9.5, INK)
    _ctr(c, TY0 + 18, "Loved in Words  -  lovedinwords.com", SERIF, 6.5, MUTED)

def make_card(portrait, name, dates, words, verse="words", out_pdf="memorial-card.pdf"):
    """Write a print-ready 2-page (front/back) prayer card PDF. `verse` is a key in
    VERSES or any custom string. Returns out_pdf."""
    vtext = VERSES.get(verse, verse)
    c = canvas.Canvas(out_pdf, pagesize=(PAGE_W, PAGE_H))
    _front(c, portrait, name, dates); _crop_marks(c); c.showPage()
    _back(c, name, dates, words, vtext); _crop_marks(c); c.showPage()
    c.save(); return out_pdf

if __name__ == "__main__":
    desk = os.path.join(os.path.expanduser("~"), "Desktop")
    out = make_card(
        portrait="marketing/lovedinwords/grace-after.png",
        name="Grace Eleanor Hartwell", dates="1942  -  2026",
        words="Mother  -  Grandmother  -  Faithful  -  Kind  -  Gentle  -  Beloved",
        verse="words",
        out_pdf=os.path.join(desk, "Memorial-Card-PRINT-READY.pdf"))
    print("OK ->", out)
