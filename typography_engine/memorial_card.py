"""Print-ready memorial / prayer card from a word-portrait.

Produces a 2-page PDF (front, back) at the standard prayer-card finished size
2.5 x 4.25 in, with 0.125 in bleed and corner crop marks -- the exact file a funeral
printer or a print-on-demand API (Prodigi/Gelato) needs. Parameterized so it can be
wired into the studio product: make_card(portrait, name, dates, words, verse, layout, divider).

Layouts:  "classic" (framed portrait on cream)  |  "bleed" (full-bleed portrait, white
type over a scrim)  |  "elegant" (single gold frame, airier, fleuron).
Dividers: "cross" | "heart" | "fleuron" | "rule".

Deps: reportlab, pillow.  CLI: python memorial_card.py [variants] -> samples on the Desktop.
"""
import os, sys
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from PIL import Image

PT = 72.0
TRIM_W, TRIM_H = 2.5 * PT, 4.25 * PT
BLEED = 0.125 * PT
SLUG  = 0.20 * PT
PAGE_W = TRIM_W + 2 * (BLEED + SLUG)
PAGE_H = TRIM_H + 2 * (BLEED + SLUG)
TX0, TY0 = SLUG + BLEED, SLUG + BLEED
TX1, TY1 = TX0 + TRIM_W, TY0 + TRIM_H
CX = (TX0 + TX1) / 2

CREAM=(244/255,239/255,230/255); INK=(45/255,38/255,31/255); MUTED=(120/255,112/255,100/255)
ROSE=(176/255,90/255,80/255); LINE=(214/255,205/255,191/255); PANEL=(13/255,13/255,15/255)
GOLD=(150/255,120/255,70/255); WHITE=(248/255,245/255,240/255)
SERIF,SERIF_I,SERIF_B="Times-Roman","Times-Italic","Times-Bold"

VERSES={
 "words":"Her portrait was made from the words of everyone who loved her - so her story stays close, always.",
 "psalm23":"The Lord is my shepherd; I shall not want. He maketh me to lie down in green pastures.",
 "light":"What we have once enjoyed we can never lose; all that we love deeply becomes a part of us.",
 "none":"",
}

def _wrap(c,text,font,size,max_w):
    c.setFont(font,size); out,line=[],""
    for w in text.split():
        t=(line+" "+w).strip()
        if c.stringWidth(t,font,size)<=max_w: line=t
        else: out.append(line); line=w
    if line: out.append(line)
    return out

def _crop_marks(c):
    c.setStrokeColorRGB(0,0,0); c.setLineWidth(0.4); g,L=3,11
    for x,sx in ((TX0,-1),(TX1,1)):
        for y,sy in ((TY0,-1),(TY1,1)):
            c.line(x+sx*g,y,x+sx*(g+L),y); c.line(x,y+sy*g,x,y+sy*(g+L))

def _ctr(c,y,text,font,size,rgb):
    c.setFillColorRGB(*rgb); c.setFont(font,size); c.drawCentredString(CX,y,text)

def _heart(c,cx,cy,s,rgb):
    c.setFillColorRGB(*rgb); r=s*0.5
    c.circle(cx-r*0.52,cy+r*0.28,r*0.6,fill=1,stroke=0); c.circle(cx+r*0.52,cy+r*0.28,r*0.6,fill=1,stroke=0)
    p=c.beginPath(); p.moveTo(cx-r*1.06,cy+r*0.42); p.lineTo(cx,cy-r*0.98); p.lineTo(cx+r*1.06,cy+r*0.42); p.close()
    c.drawPath(p,fill=1,stroke=0)

def _divider(c,y,kind):
    if kind=="rule":
        c.setStrokeColorRGB(*GOLD); c.setLineWidth(0.6); c.line(CX-40,y+3,CX+40,y+3); return
    c.setStrokeColorRGB(*GOLD); c.setLineWidth(0.6); c.line(CX-34,y+3,CX-10,y+3); c.line(CX+10,y+3,CX+34,y+3)
    if kind=="heart": _heart(c,CX,y+4,9,ROSE)
    elif kind=="fleuron": _ctr(c,y,"*",SERIF,11,GOLD)
    else: _ctr(c,y,"+",SERIF,9,ROSE)            # cross

def _bg_cream(c,double=True):
    c.setFillColorRGB(*CREAM); c.rect(SLUG,SLUG,TRIM_W+2*BLEED,TRIM_H+2*BLEED,fill=1,stroke=0)
    c.setStrokeColorRGB(*LINE); c.setLineWidth(1); c.rect(TX0+5,TY0+5,TRIM_W-10,TRIM_H-10,fill=0,stroke=1)
    if double:
        c.setLineWidth(0.6); c.rect(TX0+9,TY0+9,TRIM_W-18,TRIM_H-18,fill=0,stroke=1)

def _portrait_cover(portrait,aspect):
    im=Image.open(portrait).convert("RGB"); w,h=im.size; tar=aspect
    if w/h>tar: nw=int(h*tar); im=im.crop(((w-nw)//2,0,(w-nw)//2+nw,h))
    else: nh=int(w/tar); im=im.crop((0,(h-nh)//2,w,(h-nh)//2+nh))
    return im

# ---------------- FRONT LAYOUTS ----------------
def _front_classic(c,portrait,name,dates,divider):
    _bg_cream(c,double=True)
    _ctr(c,TY1-26,"In Loving Memory",SERIF_I,13,MUTED)
    im=Image.open(portrait).convert("RGB"); ar=im.height/im.width
    pw=TRIM_W-2*22; ph=pw*ar; px,py=CX-pw/2,TY1-40-ph
    c.setFillColorRGB(*PANEL); c.rect(px-3,py-3,pw+6,ph+6,fill=1,stroke=0); c.drawImage(ImageReader(im),px,py,pw,ph)
    y=py-20; _ctr(c,y,name,SERIF_B,12.5,INK); y-=16; _divider(c,y,divider); y-=15; _ctr(c,y,dates,SERIF,9.5,MUTED)

def _front_bleed(c,portrait,name,dates,divider):
    im=_portrait_cover(portrait,(TRIM_W+2*BLEED)/(TRIM_H+2*BLEED))
    c.drawImage(ImageReader(im),SLUG,SLUG,TRIM_W+2*BLEED,TRIM_H+2*BLEED)
    # bottom scrim for legibility
    for i,a in enumerate([0.0,0.18,0.42,0.66,0.82]):
        c.setFillColorRGB(*PANEL); c.setFillAlpha(a); c.rect(TX0,TY0+ (4-i)*16 +6,TRIM_W,16,fill=1,stroke=0)
    c.setFillAlpha(0.5); c.setFillColorRGB(*PANEL); c.rect(TX0,TY1-30,TRIM_W,24,fill=1,stroke=0); c.setFillAlpha(1)
    _ctr(c,TY1-22,"In Loving Memory",SERIF_I,11.5,WHITE)
    y=TY0+40; _ctr(c,y,name,SERIF_B,13,WHITE); y-=15
    c.setStrokeColorRGB(*WHITE); c.setLineWidth(0.5); c.line(CX-30,y+3,CX-9,y+3); c.line(CX+9,y+3,CX+30,y+3)
    if divider=="heart": _heart(c,CX,y+4,9,(0.93,0.86,0.86))
    else: _ctr(c,y,"+" if divider!="fleuron" else "*",SERIF,9,WHITE)
    y-=14; _ctr(c,y,dates,SERIF,9.5,(0.92,0.90,0.86))

def _front_elegant(c,portrait,name,dates,divider):
    _bg_cream(c,double=False)
    c.setStrokeColorRGB(*GOLD); c.setLineWidth(0.8); c.rect(TX0+7,TY0+7,TRIM_W-14,TRIM_H-14,fill=0,stroke=1)
    _divider(c,TY1-30,"fleuron"); _ctr(c,TY1-46,"In Loving Memory",SERIF_I,12,MUTED)
    im=Image.open(portrait).convert("RGB"); ar=im.height/im.width
    pw=TRIM_W-2*30; ph=pw*ar; px,py=CX-pw/2,TY1-58-ph
    c.setStrokeColorRGB(*GOLD); c.setLineWidth(0.6); c.rect(px-4,py-4,pw+8,ph+8,fill=0,stroke=1); c.drawImage(ImageReader(im),px,py,pw,ph)
    y=py-22; _ctr(c,y,name,SERIF_B,13,INK); y-=16; _ctr(c,y,dates,SERIF_I,9.5,MUTED); y-=14; _divider(c,y,divider)

_FRONTS={"classic":_front_classic,"bleed":_front_bleed,"elegant":_front_elegant}

# ---------------- BACK (shared) ----------------
def _back(c,name,dates,words,verse,divider):
    _bg_cream(c,double=True); y=TY1-34
    _ctr(c,y,name,SERIF_B,11.5,INK); y-=14; _ctr(c,y,dates,SERIF_I,8.5,MUTED); y-=18; _divider(c,y,divider); y-=16
    c.setStrokeColorRGB(*LINE); c.setLineWidth(0.6); c.line(TX0+24,y,TX1-24,y); y-=16
    for ln in _wrap(c,words,SERIF,9.5,TRIM_W-2*20): _ctr(c,y,ln,SERIF,9.5,INK); y-=13
    y-=6; c.line(TX0+24,y,TX1-24,y); y-=18
    if verse:
        for ln in _wrap(c,verse,SERIF_I,8.5,TRIM_W-2*22): _ctr(c,y,ln,SERIF_I,8.5,MUTED); y-=11
        y-=12
    _ctr(c,y,"Forever in our hearts",SERIF_B,9.5,INK)
    _ctr(c,TY0+18,"Loved in Words  -  lovedinwords.com",SERIF,6.5,MUTED)

def make_card(portrait,name,dates,words,verse="words",layout="classic",divider="cross",out_pdf="memorial-card.pdf"):
    vtext=VERSES.get(verse,verse); c=canvas.Canvas(out_pdf,pagesize=(PAGE_W,PAGE_H))
    _FRONTS.get(layout,_front_classic)(c,portrait,name,dates,divider); _crop_marks(c); c.showPage()
    _back(c,name,dates,words,vtext,divider); _crop_marks(c); c.showPage()
    c.save(); return out_pdf

if __name__=="__main__":
    desk=os.path.join(os.path.expanduser("~"),"Desktop")
    P="marketing/lovedinwords/grace-after.png"
    NAME,DATES="Grace Eleanor Hartwell","1942  -  2026"
    WORDS="Mother  -  Grandmother  -  Faithful  -  Kind  -  Gentle  -  Beloved"
    if "variants" in sys.argv:
        for lay in ("classic","bleed","elegant"):
            make_card(P,NAME,DATES,WORDS,"words",lay,"cross",f"_card_{lay}.pdf"); print("OK",lay)
    else:
        make_card(P,NAME,DATES,WORDS,"words","classic","cross",os.path.join(desk,"Memorial-Card-PRINT-READY.pdf"))
        print("OK -> Memorial-Card-PRINT-READY.pdf")
