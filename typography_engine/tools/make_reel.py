#!/usr/bin/env python3
"""Generate the Typortrait promo reel (animated GIF) from the marketing assets.

Reproduces the ~12s vertical (9:16) reel:
  photo  ->  "add the words that matter" (words dissolve in)  ->  portrait  ->  CTA.

Usage:
    python3 tools/make_reel.py                 # writes marketing/reel.gif
    python3 tools/make_reel.py out.gif         # custom output path

Needs: Python 3 + Pillow  (pip install Pillow). A serif + sans font are located
automatically (Windows Georgia/Times, or Linux Liberation/DejaVu/FreeSerif).
To post it as a real video, convert the GIF to MP4 with ffmpeg (see README note
at the bottom of this file).
"""
import os, sys, random
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

SERIF_B = find_font(["georgiab.ttf","timesbd.ttf","LiberationSerif-Bold.ttf","DejaVuSerif-Bold.ttf","FreeSerifBold.ttf","Georgia Bold.ttf"])
SERIF   = find_font(["georgia.ttf","times.ttf","LiberationSerif-Regular.ttf","DejaVuSerif.ttf","FreeSerif.ttf","Georgia.ttf"])
SANS    = find_font(["arial.ttf","LiberationSans-Regular.ttf","DejaVuSans.ttf","FreeSans.ttf","Arial.ttf"])
if not (SERIF_B and SERIF and SANS):
    sys.exit("Could not find fonts. Install 'fonts-liberation' (Linux) or run on a machine with Georgia/Arial.")

random.seed(7)                         # deterministic -> identical output every run
W, H, FPS = 450, 800, 11
BG, INK, MUTED = (250,249,247), (13,27,58), (120,130,150)
SQ, IMG_Y = 360, 250
IMG_X = (W - SQ)//2

photo = Image.open(os.path.join(BASE,"hero-before.jpg")).convert("RGB").resize((SQ,SQ), Image.LANCZOS)
portrait = Image.open(os.path.join(BASE,"hero-after.png")).convert("RGB").resize((SQ,SQ), Image.LANCZOS)

WORDS = ["cherished","kind","radiant","timeless","beloved","gentle","brave",
         "1962","forever","laughter","grace","always our light"]
def font(p,s): return ImageFont.truetype(p,s)
f_brand=font(SERIF_B,40); f_tag=font(SERIF,22); f_cap=font(SERIF_B,28); f_cta=font(SANS,21); f_small=font(SANS,15)
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
        fw=font(SERIF_B,max(13,int(20*sc))); b=d.textbbox((0,0),word,font=fw)
        d.text((x-(b[2]-b[0])/2,y-12),word,font=fw,fill=(13,27,58,a))
    return layer

WORDS_IN, WORDS_DUR, REVEAL0, REVEAL1 = 2.7, 1.9, 5.7, 6.5
def img_for(t):
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
    return portrait

def caption(t):
    if t<2.2:     return "Start with a photo."
    if t<REVEAL0: return "Then, add the words that matter."
    if t<REVEAL1: return None
    return "Made from your words."

def frame(t):
    im=Image.new("RGB",(W,H),BG); d=ImageDraw.Draw(im)
    ctext(d,W/2,70,"Typortrait",f_brand,INK)
    d.rectangle([IMG_X-3,IMG_Y-3,IMG_X+SQ+2,IMG_Y+SQ+2],outline=(40,30,20),width=2)
    im.paste(img_for(t),(IMG_X,IMG_Y))
    cap=caption(t)
    if cap:
        big=t>=REVEAL1
        ctext(d,W/2,IMG_Y+SQ+34,cap,(f_cap if big else f_tag),MUTED)
    if t>=REVEAL1:
        ctext(d,W/2,IMG_Y+SQ+92,"cherished · kind · radiant · timeless · beloved",f_small,MUTED)
        cta="Create yours free  ·  typortrait.com"
        b=d.textbbox((0,0),cta,font=f_cta); cw=b[2]-b[0]; px0=(W-cw)//2-22; py0=H-120
        d.rounded_rectangle([px0,py0,px0+cw+44,py0+50],radius=25,fill=INK)
        d.text(((W-cw)//2,py0+13),cta,font=f_cta,fill=(250,249,247))
    return im

DUR=9.0; n=int(DUR*FPS)
frames=[frame(i/FPS) for i in range(n)]
frames += [frames[-1]]*int(3.0*FPS)      # hold the CTA ~3s so viewers can process the offer
frames[0].save(OUT, save_all=True, append_images=frames[1:], duration=int(1000/FPS), loop=0, optimize=True)
print("wrote", OUT, "(%.1fs, %d frames, %d KB)" % (len(frames)/FPS, len(frames), os.path.getsize(OUT)//1024))

# To export a posting-ready MP4 (needs ffmpeg installed):
#   ffmpeg -i marketing/reel.gif -movflags +faststart -pix_fmt yuv420p \
#          -vf "scale=1080:-2:flags=lanczos,pad=1080:1920:0:(1920-ih)/2:color=0xFAF9F7" reel.mp4
