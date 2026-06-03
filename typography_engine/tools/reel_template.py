#!/usr/bin/env python3
"""Reusable Typortrait reel builder — one source of truth for the promo reel.

build_reel(cfg) renders the ~18s 9:16 reel GIF from a before/after image pair and
a word list. Used by make_reel.py (single demo) and batch_reels.py (hundreds).

cfg keys:
  before (path), after (path)  -- the 1:1 photo and the rendered word-portrait
  words  (list[str])           -- words shown in the dissolve + the word-list caption
  out    (path)                -- output .gif
  brand  (str, default "Typortrait")
  cta    (str, default "Create yours free  ·  typortrait.com")
  fps    (int, default 10)
"""
import os, math, random
from PIL import Image, ImageDraw, ImageFont

def _find_font(names):
    dirs = ["C:/Windows/Fonts", "/usr/share/fonts/truetype/liberation",
            "/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/truetype/freefont",
            "/Library/Fonts", "/System/Library/Fonts/Supplemental"]
    for n in names:
        for d in dirs:
            p = os.path.join(d, n)
            if os.path.exists(p): return p
    return None

SERIF_B = _find_font(["georgiab.ttf","timesbd.ttf","LiberationSerif-Bold.ttf","DejaVuSerif-Bold.ttf","FreeSerifBold.ttf"])
SERIF   = _find_font(["georgia.ttf","times.ttf","LiberationSerif-Regular.ttf","DejaVuSerif.ttf","FreeSerif.ttf"])
SANS    = _find_font(["arial.ttf","LiberationSans-Regular.ttf","DejaVuSans.ttf","FreeSans.ttf"])

W, H = 450, 800
BG, INK, MUTED, LINE = (250,249,247), (13,27,58), (120,130,150), (40,30,20)
SQ, IMG_X, IMG_Y = 340, 55, 200
# Pacing — every content beat is scaled by SLOW so the reel breathes. Bump SLOW
# in one place to lengthen/slow the whole thing (1.52 adds ~8s over the original).
SLOW = 1.52
HOOK_END=2.4*SLOW; P1_END=4.4*SLOW; WORDS_IN=4.9*SLOW; WORDS_DUR=1.7*SLOW
REVEAL0=7.3*SLOW; REVEAL1=8.1*SLOW; SLIDER_END=12.9*SLOW; DUR=15.4*SLOW; HOLD=3.0

def build_reel(cfg):
    if not (SERIF_B and SERIF and SANS):
        raise RuntimeError("Fonts not found. Install 'fonts-liberation' or run where Georgia/Arial exist.")
    before = Image.open(cfg["before"]).convert("RGB").resize((SQ,SQ), Image.LANCZOS)
    after  = Image.open(cfg["after"]).convert("RGB").resize((SQ,SQ), Image.LANCZOS)
    words  = [w for w in (cfg.get("words") or ["love"]) if w]
    brand  = cfg.get("brand", "Typortrait")
    cta    = cfg.get("cta", "Create yours free")
    fps    = cfg.get("fps", 10)
    out    = cfg["out"]
    wordlist = " · ".join(words[:5])

    def font(p,s): return ImageFont.truetype(p,s)
    f_brand=font(SERIF_B,38); f_hook=font(SERIF_B,29); f_tag=font(SERIF,22)
    f_cap=font(SERIF_B,28); f_cta=font(SANS,21); f_small=font(SANS,15); f_pill=font(SANS,13)
    f_foot=font(SERIF_B,23)
    def ctext(d,cx,y,txt,fnt,fill):
        b=d.textbbox((0,0),txt,font=fnt); d.text((cx-(b[2]-b[0])/2,y),txt,font=fnt,fill=fill)

    # The buyer's words, shown as a tidy centered block — uniform font, size and
    # colour, no overlap, every word fully inside the frame. Greedy-wrapped into
    # centred lines that fade in gently, one after another.
    display_words = [w.upper() for w in dict.fromkeys(words) if w][:12]
    def words_layer(scale):
        layer=Image.new("RGBA",(SQ,SQ),(0,0,0,0)); d=ImageDraw.Draw(layer)
        fw=font(SERIF_B,22); sep="   ·   "; pad=20; maxw=SQ-2*pad
        def measure(s):
            b=d.textbbox((0,0),s,font=fw); return b[2]-b[0]
        lines=[]; cur=""
        for w in display_words:
            cand = w if not cur else cur+sep+w
            if measure(cand)>maxw and cur:
                lines.append(cur); cur=w
            else:
                cur=cand
        if cur: lines.append(cur)
        lh=34; y0=(SQ-lh*len(lines))//2
        for i,s in enumerate(lines):
            a=int(max(0.0,min(1.0,scale-i*0.10))*235)
            if a<=0: continue
            d.text((SQ/2-measure(s)/2, y0+i*lh), s, font=fw, fill=(13,27,58,a))
        return layer

    def pill(d,txt,x,y,left):
        b=d.textbbox((0,0),txt,font=f_pill); w=b[2]-b[0]; pad=6
        x0 = x if left else x-(w+2*pad)
        d.rounded_rectangle([x0,y,x0+w+2*pad,y+22], radius=8, fill=(13,27,58,235))
        d.text((x0+pad,y+3), txt, font=f_pill, fill=(250,249,247,255))
    def slider(frac, pulse):
        x=int(max(0.07,min(0.93,frac))*SQ); cy=SQ//2
        base=after.convert("RGBA")
        if x>0: base.paste(before.crop((0,0,x,SQ)).convert("RGBA"),(0,0))
        d=ImageDraw.Draw(base,"RGBA")
        d.line([(x,0),(x,SQ)], fill=(255,255,255,255), width=3)
        rr=20+int(9*pulse); d.ellipse([x-rr,cy-rr,x+rr,cy+rr], outline=(13,27,58,110), width=3)
        d.ellipse([x-17,cy-17,x+17,cy+17], fill=(255,255,255,255))
        d.line([(x-4,cy-7),(x-10,cy),(x-4,cy+7)], fill=(13,27,58,255), width=2)
        d.line([(x+4,cy-7),(x+10,cy),(x+4,cy+7)], fill=(13,27,58,255), width=2)
        pill(d,"PHOTO",10,10,True); pill(d,"TYPORTRAIT",SQ-10,10,False)
        return base.convert("RGB")

    def img_for(t):
        if t < P1_END: return before
        if t < REVEAL0:
            b=before.convert("RGBA"); wa=min(1.0,(t-WORDS_IN)/WORDS_DUR) if t>WORDS_IN else 0.0
            if wa>0:
                b=Image.alpha_composite(b,Image.new("RGBA",(SQ,SQ),(250,249,247,int(150*wa))))
                b=Image.alpha_composite(b,words_layer(wa))
            return b.convert("RGB")
        if t < REVEAL1:
            k=(t-REVEAL0)/(REVEAL1-REVEAL0)
            b=Image.alpha_composite(Image.alpha_composite(before.convert("RGBA"),
              Image.new("RGBA",(SQ,SQ),(250,249,247,150))), words_layer(1.0)).convert("RGB")
            return Image.blend(b,after,k)
        if t < SLIDER_END:
            u=(t-REVEAL1)/(SLIDER_END-REVEAL1)
            return slider(0.5+0.38*math.sin(2*math.pi*u), 0.5+0.5*math.sin(u*2*math.pi*3))
        return after

    def frame(t):
        im=Image.new("RGB",(W,H),BG); d=ImageDraw.Draw(im)
        ctext(d,W/2,60,brand,f_brand,INK)
        # Present the rendering like a hung print: a navy frame with a wide
        # white matt around it, to punctuate the reveal. Same outer footprint,
        # so the rest of the layout is unchanged; the art insets to fit the matt.
        FR, MT = 9, 16
        inner = SQ - 2*(FR+MT)
        ox, oy = IMG_X+FR+MT, IMG_Y+FR+MT
        d.rectangle([IMG_X, IMG_Y, IMG_X+SQ-1, IMG_Y+SQ-1], fill=INK)                     # frame
        d.rectangle([IMG_X+FR, IMG_Y+FR, IMG_X+SQ-1-FR, IMG_Y+SQ-1-FR], fill=(255,255,255))  # white matt
        im.paste(img_for(t).resize((inner,inner), Image.LANCZOS), (ox,oy))                # the rendering
        d.rectangle([ox-1, oy-1, ox+inner, oy+inner], outline=(206,201,192), width=1)     # matt opening edge
        cy=IMG_Y+SQ+26
        if t < P1_END:    ctext(d,W/2,cy,"Start with a photo.",f_tag,MUTED)
        elif t < REVEAL0: ctext(d,W/2,cy,"Then, add the words that matter.",f_tag,MUTED)
        elif t < REVEAL1: pass
        elif t < SLIDER_END: ctext(d,W/2,cy,"Your Typortrait is revealed!",f_tag,MUTED)
        else:
            ctext(d,W/2,cy,"Made from your words.",f_cap,MUTED)
            ctext(d,W/2,cy+52,wordlist,f_small,MUTED)
            b=d.textbbox((0,0),cta,font=f_cta); cw=b[2]-b[0]; px0=(W-cw)//2-22; py0=cy+96
            d.rounded_rectangle([px0,py0,px0+cw+44,py0+50], radius=25, fill=INK)
            d.text(((W-cw)//2,py0+13), cta, font=f_cta, fill=(250,249,247))
        # Persistent brand watermark — centered along the bottom of every frame.
        ctext(d,W/2,H-34,"Typortrait.com",f_foot,INK)
        return im

    n=int(DUR*fps)
    frames=[frame(i/fps) for i in range(n)]
    frames += [frames[-1]]*int(HOLD*fps)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=int(1000/fps), loop=0, optimize=True)
    return out


def convert_gif_to_mp4(gif, mp4, audio_path=None):
    """Convert the reel GIF to a 1080x1920 H.264 MP4 with optional music bed.

    If audio_path is provided and the file exists, the audio is looped so it
    always covers the video, mixed at a tasteful volume (0.45), and fades in
    at the start and out before the end so silent-video downranking on
    TikTok / IG Reels / YT Shorts isn't triggered.

    Falls back to silent MP4 if ffmpeg isn't installed or audio_path is missing.
    Returns True on success, False otherwise.
    """
    import shutil, subprocess, os
    if not shutil.which("ffmpeg"):
        return False
    vf = "scale=1080:-2:flags=lanczos,pad=1080:1920:0:(1920-ih)/2:color=0xFAF9F7"
    use_audio = bool(audio_path) and os.path.exists(audio_path)
    if use_audio:
        # Probe the GIF duration so the audio fade-out lines up with the
        # actual end of the reel, rather than relying on the template's
        # constants (DUR+HOLD) drifting later.
        dur = float(DUR + HOLD)
        if shutil.which("ffprobe"):
            try:
                out = subprocess.check_output(
                    ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                     "-of", "default=noprint_wrappers=1:nokey=1", gif],
                    stderr=subprocess.DEVNULL,
                )
                dur = float(out.strip())
            except Exception:  # noqa: BLE001
                pass
        fade_out_start = max(0.0, dur - 1.2)
        afilter = (
            f"[0:a]volume=0.45,"
            f"afade=t=in:st=0:d=0.6,"
            f"afade=t=out:st={fade_out_start:.2f}:d=1.2[aud]"
        )
        cmd = [
            "ffmpeg", "-y",
            "-stream_loop", "-1", "-i", str(audio_path),
            "-i", str(gif),
            "-filter_complex", afilter,
            "-map", "1:v", "-map", "[aud]",
            "-shortest",
            "-movflags", "+faststart",
            "-pix_fmt", "yuv420p",
            "-vf", vf,
            "-c:a", "aac", "-b:a", "96k",
            str(mp4),
        ]
    else:
        cmd = [
            "ffmpeg", "-y", "-i", str(gif),
            "-movflags", "+faststart",
            "-pix_fmt", "yuv420p",
            "-vf", vf,
            str(mp4),
        ]
    try:
        subprocess.run(cmd, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:  # noqa: BLE001
        return False
