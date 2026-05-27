#!/usr/bin/env python3
"""Generate the Typortrait promo reel (animated GIF) from the marketing assets.

~16s vertical (9:16) reel, built for Reels/Stories best practice:
  HOOK (first frame = thumbnail)  ->  "Start with a photo."  ->
  "Then, add the words that matter." (words dissolve in)  ->  portrait resolves  ->
  before/after slider sweep  ->  "Made from your words." + CTA (held ~3s).

Usage:
    python3 tools/make_reel.py                 # writes marketing/reel.gif
    python3 tools/make_reel.py out.gif         # custom output path

Needs: Python 3 + Pillow  (pip install Pillow). Serif + sans fonts are located
automatically (Windows Georgia/Arial, or Linux Liberation/DejaVu/FreeSerif).
To post it as a real video, convert the GIF to MP4 with ffmpeg (see footer note).
"""
import os, sys, math, random
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "..", "marketing")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE, "reel.gif")

def find_font(names):
    dirs = ["C:/Windows/Fonts", "/usr/share/fonts/truetype/liberation",
            "/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/truetype/freefont",
            "/Library/Fonts", "/System/Library/Fonts/Supplemental"]
    for n in names:
        for d in dirs:
            p = os.path.join(d, n)
            if os.path.exists(p):
                return p
    return None

SERIF_B = find_font(["georgiab.ttf","timesbd.ttf","LiberationSerif-Bold.ttf","DejaVuSerif-Bold.ttf","FreeSerifBold.ttf"])
SERIF   = find_font(["georgia.ttf","times.ttf","LiberationSerif-Regular.ttf","DejaVuSerif.ttf","FreeSerif.ttf"])
SANS    = find_font(["arial.ttf","LiberationSans-Regular.ttf","DejaVuSans.ttf","FreeSans.ttf"])
if not (SERIF_B and SERIF and SANS):
    sys.exit("Could not find fonts. Install 'fonts-liberation' (Linux) or run on a machine with Georgia/Arial.")

random.seed(7)                                  # deterministic -> identical output
W, H, FPS = 450, 800, 10
BG, INK, MUTED, LINE = (250,249,247), (13,27,58), (120,130,150), (40,30,20)
SQ, IMG_X, IMG_Y = 340, 55, 200

photo = Image.open(os.path.join(BASE,"hero-before.jpg")).convert("RGB").resize((SQ,SQ), Image.LANCZOS)
portrait = Image.open(os.path.join(BASE,"hero-after.png")).convert("RGB").resize((SQ,SQ), Image.LANCZOS)

WORDS = ["cherished","kind","radiant","timeless","beloved","gentle","brave",
         "1962","forever","laughter","grace","always our light"]
def font(p,s): return ImageFont.truetype(p,s)
f_brand=font(SERIF_B,38); f_hook=font(SERIF_B,29); f_tag=font(SERIF,22)
f_cap=font(SERIF_B,28); f_cta=font(SANS,21); f_small=font(SANS,15); f_tagpill=font(SANS,13)
def ctext(d,cx,y,txt,fnt,fill):
    b=d.textbbox((0,0),txt,font=fnt); d.text((cx-(b[2]-b[0])/2,y),txt,font=fnt,fill=fill)

placements=[]
for i in range(34):
    gx=random.random()+random.random()+random.random()-1.5
    gy=random.random()+random.random()+random.random()-1.5
    placements.append((WORDS[i%len(WORDS)].upper(), SQ*0.5+gx*SQ*0.30, SQ*0.5+gy*SQ*0.32,
                       random.uniform(0.85,1.3), random.random()))
def words_layer(scale):
    layer=Image.new("RGBA",(SQ,SQ),(0,0,0,0)); d=ImageDraw.Draw(layer)
    for word,x,y,sc,ph in placements:
        a=int(max(0,min(1,scale-ph*0.25))*235)
        if a<=0: continue
        fw=font(SERIF_B,max(12,int(19*sc))); b=d.textbbox((0,0),word,font=fw)
        d.text((x-(b[2]-b[0])/2,y-11),word,font=fw,fill=(13,27,58,a))
    return layer

# ---- timeline (seconds) ----
HOOK_END=2.4; P1_END=4.4; WORDS_IN=4.9; WORDS_DUR=1.7
REVEAL0=7.3; REVEAL1=8.1; SLIDER_END=11.0; DUR=13.8; HOLD=3.0

def tagpill(d,txt,x,y,anchor_left):
    b=d.textbbox((0,0),txt,font=f_tagpill); w=b[2]-b[0]; pad=6
    x0 = x if anchor_left else x-(w+2*pad)
    d.rounded_rectangle([x0,y,x0+w+2*pad,y+22], radius=8, fill=(13,27,58))
    d.text((x0+pad,y+3), txt, font=f_tagpill, fill=(250,249,247))

def slider_img(frac):
    x=int(max(0.06,min(0.94,frac))*SQ)
    base=portrait.copy()
    if x>0: base.paste(photo.crop((0,0,x,SQ)),(0,0))
    d=ImageDraw.Draw(base)
    d.line([(x,0),(x,SQ)], fill=(255,255,255), width=3)
    d.ellipse([x-15,SQ//2-15,x+15,SQ//2+15], fill=(255,255,255))
    d.line([(x-6,SQ//2),(x+6,SQ//2)], fill=(13,27,58), width=2)
    tagpill(d,"PHOTO",10,10,True); tagpill(d,"TYPORTRAIT",SQ-10,10,False)
    return base

def img_for(t):
    if t < P1_END:
        return photo
    if t < REVEAL0:
        base=photo.convert("RGBA")
        wa=min(1.0,(t-WORDS_IN)/WORDS_DUR) if t>WORDS_IN else 0.0
        if wa>0:
            base=Image.alpha_composite(base,Image.new("RGBA",(SQ,SQ),(250,249,247,int(150*wa))))
            base=Image.alpha_composite(base,words_layer(wa))
        return base.convert("RGB")
    if t < REVEAL1:
        k=(t-REVEAL0)/(REVEAL1-REVEAL0)
        base=Image.alpha_composite(Image.alpha_composite(photo.convert("RGBA"),
              Image.new("RGBA",(SQ,SQ),(250,249,247,150))), words_layer(1.0)).convert("RGB")
        return Image.blend(base,portrait,k)
    if t < SLIDER_END:
        u=(t-REVEAL1)/(SLIDER_END-REVEAL1)
        return slider_img(0.6+0.4*math.cos(2*math.pi*u))   # there-and-back sweep
    return portrait

def frame(t):
    im=Image.new("RGB",(W,H),BG); d=ImageDraw.Draw(im)
    ctext(d,W/2,60,"Typortrait",f_brand,INK)
    im.paste(img_for(t),(IMG_X,IMG_Y))
    d.rectangle([IMG_X-3,IMG_Y-3,IMG_X+SQ+2,IMG_Y+SQ+2], outline=LINE, width=2)
    cy=IMG_Y+SQ+26
    if t < HOOK_END:
        ctext(d,W/2,cy,"A portrait, written in your words.",f_hook,INK)
        ctext(d,W/2,cy+46,"here's how  ↓",f_small,MUTED)
    elif t < P1_END:
        ctext(d,W/2,cy,"Start with a photo.",f_tag,MUTED)
    elif t < REVEAL0:
        ctext(d,W/2,cy,"Then, add the words that matter.",f_tag,MUTED)
    elif t < REVEAL1:
        pass
    elif t < SLIDER_END:
        ctext(d,W/2,cy,"Your photo, made of those words.",f_tag,MUTED)
    else:
        ctext(d,W/2,cy,"Made from your words.",f_cap,MUTED)
        ctext(d,W/2,cy+52,"cherished · kind · radiant · timeless · beloved",f_small,MUTED)
        cta="Create yours free  ·  typortrait.com"
        b=d.textbbox((0,0),cta,font=f_cta); cw=b[2]-b[0]; px0=(W-cw)//2-22; py0=cy+96
        d.rounded_rectangle([px0,py0,px0+cw+44,py0+50], radius=25, fill=INK)
        d.text(((W-cw)//2,py0+13), cta, font=f_cta, fill=(250,249,247))
    return im

n=int(DUR*FPS)
frames=[frame(i/FPS) for i in range(n)]
frames += [frames[-1]]*int(HOLD*FPS)        # hold the CTA so viewers can process the offer
frames[0].save(OUT, save_all=True, append_images=frames[1:], duration=int(1000/FPS), loop=0, optimize=True)
print("wrote", OUT, "(%.1fs, %d frames, %d KB)" % (len(frames)/FPS, len(frames), os.path.getsize(OUT)//1024))

# Export a posting-ready MP4 (needs ffmpeg):
#   ffmpeg -i marketing/reel.gif -movflags +faststart -pix_fmt yuv420p \
#          -vf "scale=1080:-2:flags=lanczos,pad=1080:1920:0:(1920-ih)/2:color=0xFAF9F7" reel.mp4
