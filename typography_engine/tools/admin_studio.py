"""Admin Studio — a LOCAL batch generator for retail typography portraits.

Runs ONLY on your machine (http://127.0.0.1:8090). It reuses the exact Typortrait
render engine, but is a separate process that is NEVER deployed and never touches
Prod/Staging, their database, or app/ code. There is no consent gate and no Stripe
here -- it is an admin tool for producing art you will sell.

    1. Copy tools/admin_studio.env.example -> tools/admin_studio.env and set a
       username + a long password.
    2. From the engine root:  python tools/admin_studio.py
    3. Open http://127.0.0.1:8090  (log in with those credentials).

Single mode: upload a photo, type words, pick style/ground/ink/size -> Generate.
Batch mode:  point at a CSV manifest + a folder of source images -> renders them
all into tools/_gallery_out/ (masters + previews + catalog.json), ready to publish
to the storefront in a separate, deliberate step.

Manifest columns:  item_id,image,title,words,style,ground,ink,price
  style: lifelike | letter | passage   (lifelike/letter need a detectable face)
See tools/manifest.example.csv.
"""
from __future__ import annotations

import csv
import io
import json
import os
import secrets
import sys
import threading
from pathlib import Path

# Make `app` importable when run from the engine root.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "tools" / "_gallery_out"
ENV_FILE = ROOT / "tools" / "admin_studio.env"


def _load_env() -> tuple[str, str]:
    if not ENV_FILE.exists():
        sys.exit(
            f"\n  Missing {ENV_FILE.name}.\n"
            f"  Copy tools/admin_studio.env.example -> tools/admin_studio.env and set\n"
            f"  ADMIN_USER + ADMIN_PASS, then run again.\n")
    user = pw = ""
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k.strip() == "ADMIN_USER":
            user = v.strip()
        elif k.strip() == "ADMIN_PASS":
            pw = v.strip()
    if not user or not pw or pw == "change-me-to-something-long":
        sys.exit(f"\n  Set a real ADMIN_USER and ADMIN_PASS in {ENV_FILE.name}.\n")
    return user, pw


ADMIN_USER, ADMIN_PASS = _load_env()

# --- render engine (same code the studio uses) --------------------------------
_STYLE_MAP = {"lifelike": "displacement", "letter": "displacement",
              "passage": "message", "mosaic": "words"}


def _norm_style(s: str) -> tuple[str, bool]:
    s = (s or "lifelike").strip().lower()
    return _STYLE_MAP.get(s, s), (s == "letter")


def render_portrait(img_bytes: bytes, words: str, style: str, ground: str,
                    ink: str | None, out_width: int, supersample: int) -> bytes:
    from app.config import RenderConfig
    from app.pipeline.analyze import analyze_image
    from app.pipeline.warnings import WarningCollector

    eng_style, flow = _norm_style(style)
    warns = WarningCollector()
    an = analyze_image(img_bytes, RenderConfig(), warns)
    wlist = [w for w in str(words).replace("\n", " ").split() if w]
    if not wlist:
        raise ValueError("no words supplied")
    if eng_style == "displacement":
        from app.pipeline.displacement import render_displacement_portrait
        return render_displacement_portrait(
            an, wlist, ground=ground or "navy", out_width=out_width,
            supersample=supersample, ink=(ink or None), flow=flow)
    from app.pipeline.tonal import render_layered_png
    png, *_ = render_layered_png(
        an, " ".join(wlist), eng_style, RenderConfig(), warns,
        ink=(ink or "navy"), out_width=out_width)
    if not png:
        raise ValueError("render produced no image: " + str(warns.as_list()))
    return png


def _save_master_preview(item_id: str, png: bytes, catalog_row: dict) -> None:
    from PIL import Image
    (OUT / "masters").mkdir(parents=True, exist_ok=True)
    (OUT / "previews").mkdir(parents=True, exist_ok=True)
    img = Image.open(io.BytesIO(png)).convert("RGB")
    img.save(OUT / "masters" / f"{item_id}.png", "PNG")
    pw = 900
    prev = img.resize((pw, round(img.height * pw / img.width)), Image.LANCZOS) if img.width > pw else img
    prev.save(OUT / "previews" / f"{item_id}.png", "PNG")


# --- approvals ----------------------------------------------------------------
# Per-item review state {id: "approved"|"rejected"} in _gallery_out/approvals.json.
# Kept separate from catalog.json so re-running a batch never wipes your decisions.
def _approvals_path() -> Path:
    return OUT / "approvals.json"


def _load_approvals() -> dict:
    p = _approvals_path()
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return {}
    return {}


def _save_approvals(data: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _approvals_path().write_text(json.dumps(data, indent=2), encoding="utf-8")


# --- batch state --------------------------------------------------------------
BATCH: dict = {"running": False, "total": 0, "done": 0, "current": "",
               "errors": [], "cancel": False, "log": []}


def run_batch(manifest_path: str, images_dir: str, out_width: int, supersample: int) -> None:
    global BATCH
    try:
        rows = list(csv.DictReader(open(manifest_path, encoding="utf-8-sig")))
    except Exception as e:  # noqa: BLE001
        BATCH.update(running=False, errors=[f"manifest read failed: {e}"])
        return
    BATCH.update(running=True, total=len(rows), done=0, current="",
                 errors=[], cancel=False, log=[])
    catalog = []
    idir = Path(images_dir)
    for row in rows:
        if BATCH.get("cancel"):
            BATCH["log"].append("canceled by user")
            break
        item = (row.get("item_id") or "").strip()
        BATCH["current"] = item
        try:
            if not item:
                raise ValueError("empty item_id")
            img_path = idir / (row.get("image") or "").strip()
            if not img_path.exists():
                raise FileNotFoundError(f"image not found: {img_path.name}")
            png = render_portrait(
                img_path.read_bytes(), row.get("words", ""),
                row.get("style", "lifelike"), row.get("ground", "navy"),
                (row.get("ink") or "").strip() or None, out_width, supersample)
            _save_master_preview(item, png, row)
            catalog.append({
                "id": item, "title": (row.get("title") or item).strip(),
                "subject": (row.get("subject") or row.get("title") or item).strip(),
                "category": (row.get("category") or "").strip(),
                "price": (row.get("price") or "").strip(),
                "style": (row.get("style") or "lifelike").strip(),
                "ground": (row.get("ground") or "navy").strip(),
                "words": (row.get("words") or "").strip(),
            })
            BATCH["log"].append(f"OK  {item}")
        except Exception as e:  # noqa: BLE001
            BATCH["errors"].append(f"{item or '(no id)'}: {type(e).__name__}: {e}")
            BATCH["log"].append(f"ERR {item}: {e}")
        BATCH["done"] += 1
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "catalog.json").write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    BATCH.update(running=False, current="")


# --- web app ------------------------------------------------------------------
from fastapi import Depends, FastAPI, Form, HTTPException, UploadFile, File  # noqa: E402
from fastapi.responses import HTMLResponse, JSONResponse, Response, FileResponse  # noqa: E402
from fastapi.security import HTTPBasic, HTTPBasicCredentials  # noqa: E402

app = FastAPI(title="Typortrait Admin Studio")
_security = HTTPBasic()


def auth(cred: HTTPBasicCredentials = Depends(_security)) -> str:
    ok = (secrets.compare_digest(cred.username, ADMIN_USER)
          and secrets.compare_digest(cred.password, ADMIN_PASS))
    if not ok:
        raise HTTPException(401, "bad credentials", headers={"WWW-Authenticate": "Basic"})
    return cred.username


@app.get("/", response_class=HTMLResponse)
def home(user: str = Depends(auth)) -> HTMLResponse:
    return HTMLResponse(PAGE)


@app.post("/api/render")
async def api_render(user: str = Depends(auth), image: UploadFile = File(...),
                     words: str = Form(...), style: str = Form("lifelike"),
                     ground: str = Form("navy"), ink: str = Form("photo"),
                     out_width: int = Form(1400), supersample: int = Form(2)):
    try:
        png = render_portrait(await image.read(), words, style, ground,
                              ink or None, int(out_width), int(supersample))
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=400)
    return Response(png, media_type="image/png")


@app.post("/api/batch")
def api_batch(user: str = Depends(auth), manifest: str = Form(...),
              images_dir: str = Form(...), out_width: int = Form(3000),
              supersample: int = Form(2)):
    if BATCH.get("running"):
        return JSONResponse({"error": "a batch is already running"}, status_code=409)
    t = threading.Thread(target=run_batch,
                         args=(manifest, images_dir, int(out_width), int(supersample)),
                         daemon=True)
    t.start()
    return {"ok": True}


@app.get("/api/batch/status")
def api_batch_status(user: str = Depends(auth)):
    return {k: BATCH[k] for k in ("running", "total", "done", "current", "errors", "log")}


@app.post("/api/batch/cancel")
def api_batch_cancel(user: str = Depends(auth)):
    BATCH["cancel"] = True
    return {"ok": True}


@app.get("/api/outputs")
def api_outputs(user: str = Depends(auth)):
    prev = OUT / "previews"
    appr = _load_approvals()
    ids = sorted(p.stem for p in prev.glob("*.png")) if prev.exists() else []
    items = [{"id": i, "status": appr.get(i, "pending")} for i in ids]
    return {"count": len(items), "items": items}


@app.post("/api/review")
def api_review(user: str = Depends(auth), item: str = Form(...), status: str = Form(...)):
    item = (item or "").strip()
    appr = _load_approvals()
    if status in ("approved", "rejected"):
        appr[item] = status
    else:                                   # anything else clears back to pending
        appr.pop(item, None)
    _save_approvals(appr)
    return {"ok": True, "status": appr.get(item, "pending")}


@app.get("/out/preview/{item}.png")
def out_preview(item: str, user: str = Depends(auth)):
    p = OUT / "previews" / f"{item}.png"
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(str(p), media_type="image/png")


@app.get("/out/master/{item}.png")
def out_master(item: str, user: str = Depends(auth)):
    p = OUT / "masters" / f"{item}.png"
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(str(p), media_type="image/png", filename=f"{item}.png")


PAGE = """<!doctype html><meta charset=utf-8><title>Typortrait Admin Studio</title>
<style>
 :root{--bg:#0f1420;--card:#1a2130;--line:#2b3448;--ink:#e7ecf6;--mut:#94a0b8;--acc:#4f7dff}
 *{box-sizing:border-box}body{margin:0;font:14px/1.5 system-ui,Segoe UI,sans-serif;background:var(--bg);color:var(--ink)}
 header{padding:12px 20px;border-bottom:1px solid var(--line);display:flex;gap:16px;align-items:center}
 header h1{font-size:15px;margin:0;font-weight:650}.mut{color:var(--mut)}
 .tabs{display:flex;gap:4px;padding:10px 20px 0}.tabs button{background:none;border:none;color:var(--mut);padding:8px 14px;border-radius:8px 8px 0 0;cursor:pointer;font-size:14px}
 .tabs button.on{background:var(--card);color:var(--ink)}
 .wrap{padding:20px;max-width:1000px}.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px;margin-bottom:16px}
 label{display:block;font-size:12.5px;color:var(--mut);margin:10px 0 4px}
 input,select,textarea{width:100%;background:#10161f;border:1px solid var(--line);border-radius:8px;color:var(--ink);padding:9px 11px;font:inherit}
 textarea{min-height:70px;resize:vertical}.row{display:flex;gap:12px}.row>div{flex:1}
 button.go{background:var(--acc);border:none;color:#fff;border-radius:9px;padding:11px 18px;font-weight:600;cursor:pointer;margin-top:14px}
 button.sec{background:#26304a;border:1px solid var(--line);color:var(--ink);border-radius:9px;padding:9px 14px;cursor:pointer}
 .hide{display:none}img.result{max-width:100%;border-radius:10px;margin-top:14px;border:1px solid var(--line)}
 .bar{height:10px;background:#10161f;border-radius:6px;overflow:hidden;margin:10px 0}.bar>i{display:block;height:100%;background:var(--acc);width:0}
 .log{font:12px/1.5 ui-monospace,monospace;background:#0b0f17;border:1px solid var(--line);border-radius:8px;padding:10px;max-height:220px;overflow:auto;white-space:pre-wrap}
 .err{color:#ff8b8b}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:12px}
 .grid a{display:block}.grid img{width:100%;border-radius:8px;border:1px solid var(--line)}.grid .cap{font-size:11px;color:var(--mut);margin-top:3px;word-break:break-all}
 .rcard{background:#141b28;border:1px solid var(--line);border-radius:9px;padding:8px}
 .rcard img{width:100%;border-radius:6px;border:1px solid var(--line);cursor:pointer}
 .rcard.approved{outline:2px solid #2fa35f}.rcard.rejected{outline:2px solid #c05262;opacity:.6}
 .rcard .cap{font-size:11px;color:var(--mut);margin:5px 0 6px;word-break:break-all}
 .badge{font-size:10px;padding:1px 6px;border-radius:5px;text-transform:uppercase;letter-spacing:.03em}
 .badge.approved{background:#1f7a44;color:#fff}.badge.rejected{background:#7a2431;color:#fff}.badge.pending{background:#2b3448;color:var(--mut)}
 .rv{display:flex;gap:6px}.rv button{flex:1;border:1px solid var(--line);border-radius:6px;padding:6px;cursor:pointer;font-size:12px;background:#10161f;color:var(--ink)}
 .rv .ap.on{background:#1f7a44;border-color:#2fa35f;color:#fff}.rv .rj.on{background:#7a2431;border-color:#c05262;color:#fff}
 .filt button.on{background:var(--acc);border-color:var(--acc);color:#fff}
</style>
<header><h1>🎨 Typortrait Admin Studio</h1><span class=mut>local · no consent · no Stripe · not connected to prod</span></header>
<div class=tabs><button class=on data-t=single>Single</button><button data-t=batch>Batch (CSV)</button><button data-t=out>Output</button></div>
<div class=wrap>
 <div id=single><div class=card>
  <label>Source image</label><input type=file id=s_img accept=image/*>
  <label>Words</label><textarea id=s_words placeholder="JESUS CHRIST LIGHT LOVE FAITH GRACE ..."></textarea>
  <div class=row>
   <div><label>Style</label><select id=s_style><option>lifelike</option><option>letter</option><option>passage</option></select></div>
   <div><label>Ground</label><select id=s_ground><option>navy</option><option>paper</option><option>black</option></select></div>
   <div><label>Ink</label><input id=s_ink value=photo></div>
  </div>
  <div class=row>
   <div><label>Width px</label><input id=s_w type=number value=1400></div>
   <div><label>Supersample</label><select id=s_ss><option>1</option><option selected>2</option></select></div>
  </div>
  <label>Download name (item id)</label><input id=s_name placeholder="good-shepherd">
  <button class=go id=s_go>Generate</button>
  <div id=s_status class=mut style=margin-top:10px></div>
  <img id=s_out class="result hide">
  <div id=s_dl class=hide style=margin-top:12px>
   <button class=sec id=s_dl_img>⬇ Download image (PNG)</button>
   <button class=sec id=s_dl_txt>⬇ Download words (TXT)</button>
  </div>
 </div></div>

 <div id=batch class=hide><div class=card>
  <p class=mut>Point at a CSV manifest and the folder of source images. Renders every row into <code>tools/_gallery_out/</code> (masters + previews + catalog.json). Runs locally; ~2–20s per portrait.</p>
  <label>Manifest CSV (full path)</label><input id=b_manifest placeholder="C:\\Users\\...\\manifest.csv">
  <label>Images folder (full path)</label><input id=b_dir placeholder="C:\\Users\\...\\images">
  <div class=row>
   <div><label>Master width px (print)</label><input id=b_w type=number value=3000></div>
   <div><label>Supersample</label><select id=b_ss><option>1</option><option selected>2</option></select></div>
  </div>
  <button class=go id=b_go>Start batch</button> <button class=sec id=b_cancel style=margin-top:14px>Cancel</button>
  <div class=bar><i id=b_fill></i></div>
  <div id=b_stat class=mut></div>
  <div class=log id=b_log style=margin-top:10px></div>
 </div></div>

 <div id=out class=hide><div class=card>
  <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">
   <button class=sec id=o_refresh>Refresh</button>
   <span class=mut style=margin-left:auto>Filter:</span>
   <span class=filt>
    <button class="sec on" data-f=all>All</button>
    <button class=sec data-f=pending>Pending</button>
    <button class=sec data-f=approved>Approved</button>
    <button class=sec data-f=rejected>Rejected</button>
   </span>
  </div>
  <div id=o_count class=mut style=margin-top:8px></div>
  <div class=grid id=o_grid style=margin-top:14px></div>
 </div></div>
</div>
<script>
const $=s=>document.querySelector(s), tabs=document.querySelectorAll('.tabs button');
let lastBlob=null;
function _dl(b,n){const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download=n;document.body.appendChild(a);a.click();a.remove();}
tabs.forEach(b=>b.onclick=()=>{tabs.forEach(x=>x.classList.remove('on'));b.classList.add('on');
 for(const id of['single','batch','out'])$('#'+id).classList.toggle('hide',id!==b.dataset.t);
 if(b.dataset.t==='out')loadOut();});

$('#s_go').onclick=async()=>{
 const f=$('#s_img').files[0]; if(!f){$('#s_status').textContent='Pick an image first.';return;}
 $('#s_status').textContent='Rendering…'; $('#s_out').classList.add('hide');
 const fd=new FormData();
 fd.append('image',f);fd.append('words',$('#s_words').value);fd.append('style',$('#s_style').value);
 fd.append('ground',$('#s_ground').value);fd.append('ink',$('#s_ink').value);
 fd.append('out_width',$('#s_w').value);fd.append('supersample',$('#s_ss').value);
 const r=await fetch('/api/render',{method:'POST',body:fd});
 if(!r.ok){const j=await r.json().catch(()=>({error:r.status}));$('#s_status').innerHTML='<span class=err>'+(j.error||'failed')+'</span>';return;}
 const b=await r.blob();lastBlob=b;$('#s_out').src=URL.createObjectURL(b);$('#s_out').classList.remove('hide');$('#s_dl').classList.remove('hide');$('#s_status').textContent='Done.';};
$('#s_dl_img').onclick=()=>{ if(lastBlob) _dl(lastBlob,(($('#s_name').value.trim())||'portrait')+'.png'); };
$('#s_dl_txt').onclick=()=>{ const n=($('#s_name').value.trim())||'portrait';
 const t='item_id: '+n+'\\nwords: '+$('#s_words').value.trim()+'\\nstyle: '+$('#s_style').value+'\\nground: '+$('#s_ground').value+'\\nink: '+$('#s_ink').value+'\\nwidth: '+$('#s_w').value;
 _dl(new Blob([t],{type:'text/plain'}), n+'.txt'); };

let poll=null;
$('#b_go').onclick=async()=>{
 const fd=new FormData();
 fd.append('manifest',$('#b_manifest').value);fd.append('images_dir',$('#b_dir').value);
 fd.append('out_width',$('#b_w').value);fd.append('supersample',$('#b_ss').value);
 const r=await fetch('/api/batch',{method:'POST',body:fd});
 if(!r.ok){const j=await r.json().catch(()=>({}));$('#b_stat').innerHTML='<span class=err>'+(j.error||'failed')+'</span>';return;}
 if(poll)clearInterval(poll); poll=setInterval(status,1200); status();};
$('#b_cancel').onclick=()=>fetch('/api/batch/cancel',{method:'POST'});
async function status(){
 const s=await (await fetch('/api/batch/status')).json();
 const pct=s.total?Math.round(100*s.done/s.total):0;
 $('#b_fill').style.width=pct+'%';
 $('#b_stat').textContent=`${s.done}/${s.total} · ${s.running?('rendering '+s.current):'idle'} · ${s.errors.length} errors`;
 $('#b_log').innerHTML=(s.log||[]).slice(-40).map(l=>l.startsWith('ERR')?'<span class=err>'+l+'</span>':l).join('\\n')
   +(s.errors.length?'\\n\\n'+s.errors.map(e=>'<span class=err>'+e+'</span>').join('\\n'):'');
 if(!s.running&&poll){clearInterval(poll);poll=null;}}
let OUTFILTER='all';
async function loadOut(){
 const s=await (await fetch('/api/outputs')).json();
 const items=s.items||[];
 const c={all:items.length,pending:0,approved:0,rejected:0};
 items.forEach(i=>c[i.status||'pending']++);
 $('#o_count').textContent=`${c.all} generated · ${c.approved} approved · ${c.pending} pending · ${c.rejected} rejected`;
 const show=items.filter(i=>OUTFILTER==='all'||(i.status||'pending')===OUTFILTER);
 $('#o_grid').innerHTML=show.map(i=>{const st=i.status||'pending';return `
  <div class="rcard ${st}">
   <a href="/out/master/${i.id}.png" target=_blank><img src="/out/preview/${i.id}.png"></a>
   <div class=cap>${i.id} <span class="badge ${st}">${st}</span></div>
   <div class=rv>
    <button class="ap ${st==='approved'?'on':''}" data-id="${i.id}" data-s=approved>✓ Approve</button>
    <button class="rj ${st==='rejected'?'on':''}" data-id="${i.id}" data-s=rejected>✗ Reject</button>
   </div>
  </div>`}).join('')||'<p class=mut style="grid-column:1/-1">No items in this filter.</p>';
 $('#o_grid').querySelectorAll('.rv button').forEach(b=>b.onclick=async()=>{
  const newS=b.classList.contains('on')?'pending':b.dataset.s;
  await fetch('/api/review',{method:'POST',body:new URLSearchParams({item:b.dataset.id,status:newS})});
  loadOut();});
}
$('#o_refresh').onclick=loadOut;
document.querySelectorAll('.filt button').forEach(b=>b.onclick=()=>{
 document.querySelectorAll('.filt button').forEach(x=>x.classList.remove('on'));
 b.classList.add('on'); OUTFILTER=b.dataset.f; loadOut();});
</script>"""


if __name__ == "__main__":
    import uvicorn
    print("\n  Typortrait Admin Studio  ->  http://127.0.0.1:8090")
    print(f"  Output folder: {OUT}")
    print("  (local only; not connected to prod/staging)\n")
    uvicorn.run(app, host="127.0.0.1", port=8090, log_level="warning")
