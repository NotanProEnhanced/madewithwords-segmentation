"""Automate the Sacred Collection source material.

GPT plans the figures (title, category, unique words, an image prompt); the OpenAI
Images API draws each source portrait; a FACE-QC gate drops any that don't give the
Lifelike engine a clean front-facing face; and the result is an images folder + a
manifest.csv ready for the Admin Studio batch -> review -> publish.

Local only. Uses your OPENAI_API_KEY (env or .env). Never touches prod. Uses plain
HTTP (httpx) exactly like app/suggest.py -- no extra package to install.

    # 1) put OPENAI_API_KEY=... in your local .env (this stays on your machine)
    # 2) a cheap test batch first:
    python tools/sacred_generate.py --count 25
    # 3) the full run (resumable; re-run to continue / retry failures):
    python tools/sacred_generate.py --count 400 --resume

Output (default tools/_sacred_src/):
    images/<item_id>.png   the AI source portraits that PASSED the face check
    manifest.csv           item_id,image,title,category,words,style,ground,ink,price
    plan.json              the full plan + per-item QC (for resume / auditing)

Then in the Admin Studio: Batch tab -> this manifest + the images folder -> Output
tab -> Approve/Reject -> python tools/gallery_publish.py (publishes approved only).
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import re
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_DEFAULT = ROOT / "tools" / "_sacred_src"
_TIMEOUT = 120.0


def _load_key() -> str:
    import os
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if key:
        return key
    env = ROOT / ".env"
    if env.exists():
        for ln in env.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if ln.startswith("OPENAI_API_KEY") and "=" in ln:
                return ln.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def _slug(s: str) -> str:
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", s.lower())).strip("-") or "item"


def _post(url: str, key: str, body: dict, tries: int = 4) -> dict:
    """POST with retry on 429/5xx. Raises on final failure."""
    last = None
    for i in range(tries):
        try:
            r = httpx.post(url, headers={"Authorization": f"Bearer {key}"},
                           json=body, timeout=_TIMEOUT)
            if r.status_code == 200:
                return r.json()
            last = f"HTTP {r.status_code}: {r.text[:200]}"
            if r.status_code in (429, 500, 502, 503, 529):
                time.sleep(2 * (i + 1))
                continue
            raise RuntimeError(last)          # 4xx (e.g. content policy) -> don't retry
        except httpx.HTTPError as e:
            last = str(e)
            time.sleep(2 * (i + 1))
    raise RuntimeError(last or "request failed")


_CATEGORIES = ("Life of Christ", "Marian Devotions", "Saints & Figures",
               "Old Testament", "Psalms & Prayers")
_PLACEHOLDER = ("unnamed", "not yet", "not named", "placeholder", "unknown", "tbd", "various")
_BANNED = {"PROSTITUTE", "SINNER", "DENIER", "TRAITOR", "HARLOT", "ADULTERESS"}

_PLAN_SYSTEM = (
    "You curate a tasteful, reverent retail collection of typographic word-portraits of "
    'Christian figures. Respond ONLY as JSON {"items":[...]}.\n'
    "Each item: item_id (unique kebab-case slug); title (the figure's common name); "
    "subject (the natural noun phrase that completes 'the words that belong to ___' -- for a "
    "person, their name; for a devotion or scene, the person depicted, e.g. Divine Mercy -> "
    "Christ, Our Lady of Sorrows -> the Blessed Mother, Ecce Homo -> Christ); "
    "category (EXACTLY ONE of: " + "; ".join(_CATEGORIES) + "); "
    "words (EXACTLY 12-16 UPPERCASE words, space-separated, no punctuation, DISTINCTIVE to "
    "THIS figure -- prioritize their specific name, unique roles, deeds, place, and "
    "associated scripture; do NOT fill every figure with the same generic virtues like "
    "GRACE LOVE HOPE FAITH -- at most two or three such general words); "
    "image_prompt (a photorealistic FRONT-FACING single-subject portrait: the face fills "
    "the frame, soft reverent studio light, plain dark background, dignified; the "
    "appearance HISTORICALLY and CULTURALLY appropriate to THIS specific figure in AGE, "
    "ethnicity, and bearing (e.g. the Virgin Mary a serene young woman of about 16-20; "
    "youthful figures young, elder apostles and patriarchs older); Middle-Eastern for "
    "biblical / Holy Land figures, region-appropriate for later saints; "
    "it is a bare photograph with absolutely NO caption, title, name, label, watermark, "
    "border, or letters of any kind anywhere in the frame).\n"
    "STRICT RULES:\n"
    "- Every subject must be a REAL, distinct, NAMED figure. NEVER invent placeholders or "
    "unnamed/'various' entries. If you cannot produce the requested number of NEW distinct "
    "figures, return FEWER items -- never pad to hit a count.\n"
    "- Treat alternate names as the SAME subject and never repeat one (e.g. Peter = Simon "
    "Peter = Cephas; Saul = Paul; the Virgin Mary = Madonna). Never repeat a subject from "
    "the provided 'already used' list.\n"
    "- Words must be REVERENT and affirming. NEVER use contested, negative, sinful, or "
    "potentially offensive labels (never words like 'prostitute', 'sinner', 'denier', "
    "'traitor', 'harlot').\n"
    "- Use the category labels spelled EXACTLY as listed; do not invent new categories."
)


def plan_items(key: str, model: str, theme: str, want: int, seen_titles: list[str],
               per_call: int) -> list[dict]:
    items: list[dict] = []
    seen = set(seen_titles)
    while len(items) < want:
        n = min(per_call, want - len(items))
        used = ", ".join(sorted(seen))[:3000]
        user = (f"Theme: {theme}\nProduce {n} NEW distinct subjects.\n"
                f"Already used (do not repeat): {used or '(none)'}")
        data = _post("https://api.openai.com/v1/chat/completions", key, {
            "model": model,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": _PLAN_SYSTEM},
                         {"role": "user", "content": user}],
            "temperature": 0.8,
        })
        content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "") or "{}"
        try:
            batch = json.loads(content).get("items", [])
        except json.JSONDecodeError:
            batch = []
        added = 0
        for it in batch:
            title = str(it.get("title") or "").strip()
            low = title.lower()
            if not title or low in {t.lower() for t in seen}:
                continue
            if any(b in low for b in _PLACEHOLDER):          # backstop: reject placeholders
                continue
            words = [w for w in str(it.get("words") or "").upper().split() if w not in _BANNED]
            if len(words) < 8:                               # backstop: substantive sets only
                continue
            cat = str(it.get("category") or "").strip()
            cat = next((c for c in _CATEGORIES if c.lower() == cat.lower()), "Saints & Figures")
            iid = _slug(str(it.get("item_id") or title))
            items.append({"item_id": iid, "title": title, "category": cat,
                          "subject": (str(it.get("subject") or "").strip() or title),
                          "words": " ".join(words),
                          "image_prompt": str(it.get("image_prompt") or "").strip()})
            seen.add(title)
            added += 1
        print(f"  planned {len(items)}/{want} (+{added})")
        if added == 0:                        # model stopped producing new ones
            break
    return items[:want]


_WORDS_SYSTEM = (
    "You are given specific Christian figures, each with a FIXED title and category. "
    'Respond ONLY as JSON {"items":[...]} with ONE item per given figure, in the same '
    "order. Do NOT add, drop, merge, or rename figures.\n"
    "Each item: item_id (kebab-case slug of the title); title (EXACTLY the given title); "
    "subject (the natural noun phrase that completes 'the words that belong to ___' -- for a "
    "person, their name; for a devotion or scene, the person depicted, e.g. Divine Mercy -> "
    "Christ, Our Lady of Sorrows -> the Blessed Mother); "
    "category (EXACTLY the given category); "
    "words (EXACTLY 12-16 UPPERCASE words, space-separated, no punctuation, DISTINCTIVE to "
    "THIS figure -- their specific name, unique roles, deeds, place, and associated "
    "scripture; use at most two or three generic virtues like GRACE LOVE HOPE FAITH; never "
    "contested, negative, or offensive labels); "
    "image_prompt (a photorealistic FRONT-FACING single-subject portrait: the face fills the "
    "frame, soft reverent studio light, plain dark background, dignified; appearance "
    "appropriate to THIS figure in AGE, ethnicity, and bearing (e.g. the Virgin Mary a serene "
    "young woman about 16-20; elders older); Middle-Eastern for biblical / Holy Land figures, "
    "region-appropriate for later saints; a bare photograph with absolutely NO caption, title, "
    "name, label, watermark, border, or letters)."
)


def _mkey(t: str) -> str:
    return "".join(c for c in t.lower() if c.isalnum())


def load_figures(path: str) -> list[tuple[str, str]]:
    """Read a curated 'Title | Category' list (one per line; # comments ignored)."""
    out = []
    for ln in Path(path).read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        if "|" in ln:
            t, c = ln.split("|", 1)
            out.append((t.strip(), c.strip()))
        else:
            out.append((ln, "Saints & Figures"))
    return out


def plan_from_figures(key: str, model: str, figures: list[tuple[str, str]],
                      per_call: int) -> list[dict]:
    """Write words + image_prompt for a FIXED list of figures (titles/categories kept)."""
    items: list[dict] = []
    for i in range(0, len(figures), per_call):
        chunk = figures[i:i + per_call]
        listing = "\n".join(f"- {t} | {c}" for t, c in chunk)
        data = _post("https://api.openai.com/v1/chat/completions", key, {
            "model": model,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": _WORDS_SYSTEM},
                         {"role": "user", "content": f"Figures (title | category):\n{listing}"}],
            "temperature": 0.7,
        })
        content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "") or "{}"
        try:
            batch = json.loads(content).get("items", [])
        except json.JSONDecodeError:
            batch = []
        by_key = {_mkey(str(it.get("title") or "")): it for it in batch}
        for t, c in chunk:
            it = by_key.get(_mkey(t)) or {}
            words = [w for w in str(it.get("words") or "").upper().split() if w not in _BANNED]
            if not words:
                print(f"  (!) no words returned for '{t}' -- skipped"); continue
            items.append({"item_id": _slug(t), "title": t, "category": c,
                          "subject": (str(it.get("subject") or "").strip() or t),
                          "words": " ".join(words),
                          "image_prompt": str(it.get("image_prompt") or "").strip()})
        print(f"  words written {len(items)}/{len(figures)}")
    return items


def gen_image(key: str, model: str, prompt: str, size: str) -> bytes:
    data = _post("https://api.openai.com/v1/images/generations", key, {
        "model": model, "prompt": prompt, "size": size, "n": 1})
    b64 = (data.get("data") or [{}])[0].get("b64_json")
    if not b64:
        raise RuntimeError("no image data returned")
    return base64.b64decode(b64)


def has_face(img_bytes: bytes) -> bool:
    from app.config import RenderConfig
    from app.pipeline.analyze import analyze_image
    from app.pipeline.warnings import WarningCollector
    an = analyze_image(img_bytes, RenderConfig(), WarningCollector())
    return getattr(an, "landmarks", None) is not None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--count", type=int, default=25, help="how many figures to plan/generate (auto mode)")
    ap.add_argument("--figures", default=None, help="path to a curated 'Title | Category' list; "
                    "uses EXACTLY those figures (GPT only writes words + prompts). Ignores --count/--theme.")
    ap.add_argument("--theme", default="Christian sacred figures: Jesus, the Virgin Mary, "
                    "the apostles, saints, prophets, and biblical scenes")
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--price", default="29")
    ap.add_argument("--style", default="lifelike")
    ap.add_argument("--ground", default="navy")
    ap.add_argument("--ink", default="photo")
    ap.add_argument("--size", default="1024x1536", help="OpenAI image size (portrait)")
    ap.add_argument("--model-text", default="gpt-4o", help="planner model (gpt-4o gives far more "
                    "distinctive, accurate words than gpt-4o-mini; cost is still pennies)")
    ap.add_argument("--model-image", default="gpt-image-1")
    ap.add_argument("--per-call", type=int, default=8, help="figures per planning call "
                    "(smaller = more distinctive words per figure)")
    ap.add_argument("--resume", action="store_true", help="skip figures already generated")
    ap.add_argument("--dry-run", action="store_true", help="plan only; generate no images")
    a = ap.parse_args()

    key = _load_key()
    if not key:
        sys.exit("No OPENAI_API_KEY found (env or .env). Add it to your local .env and retry.")

    out = Path(a.out)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    plan_path = out / "plan.json"

    # Resume: reuse an existing plan so item_ids/words stay stable across runs.
    plan = []
    if a.resume and plan_path.exists():
        try:
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            plan = []
    if a.figures:
        figs = load_figures(a.figures)
        have = {p["title"].strip().lower() for p in plan}
        todo = [(t, c) for t, c in figs if t.strip().lower() not in have]
        if todo:
            print(f"Writing words for {len(todo)} curated figure(s) via {a.model_text}…")
            plan.extend(plan_from_figures(key, a.model_text, todo, a.per_call))
        keep = {t.strip().lower() for t, _ in figs}
        plan = [p for p in plan if p["title"].strip().lower() in keep]   # exactly the curated set
    else:
        if len(plan) < a.count:
            print(f"Planning {a.count - len(plan)} more figure(s) via {a.model_text}…")
            more = plan_items(key, a.model_text, a.theme, a.count - len(plan),
                              [p["title"] for p in plan], a.per_call)
            plan.extend(more)
        plan = plan[:a.count]
    plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Plan ready: {len(plan)} figures -> {plan_path}")

    if a.dry_run:
        print("DRY-RUN: no images generated. Review plan.json.")
        return

    # Rough cost heads-up (gpt-image-1 is billed per image by size/quality).
    print(f"Generating up to {len(plan)} images ({a.size}) via {a.model_image}. "
          f"Image generation is billed per image — watch your OpenAI usage.\n")

    rows, passed, failed, skipped, errored = [], 0, 0, 0, []
    for i, it in enumerate(plan, 1):
        iid = it["item_id"]
        dest = img_dir / f"{iid}.png"
        tag = f"[{i}/{len(plan)}] {iid}"
        if a.resume and dest.exists() and it.get("qc") == "face":
            rows.append(it); skipped += 1
            print(f"{tag}: skip (already generated)")
            continue
        try:
            png = gen_image(key, a.model_image, it["image_prompt"], a.size)
            dest.write_bytes(png)
        except Exception as e:  # noqa: BLE001 -- content policy, rate, etc.
            it["qc"] = "error"; errored.append(f"{iid}: {e}")
            print(f"{tag}: IMAGE ERROR {e}")
            continue
        try:
            ok = has_face(png)
        except Exception:  # noqa: BLE001
            ok = False
        it["qc"] = "face" if ok else "no-face"
        if ok:
            rows.append(it); passed += 1
            print(f"{tag}: OK (face)")
        else:
            failed += 1
            print(f"{tag}: no clean face -> excluded from manifest (image kept)")
        plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")

    # Manifest = only the face-passing rows, in the studio's exact format.
    man = out / "manifest.csv"
    with open(man, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "image", "title", "subject", "category", "words", "style", "ground", "ink", "price"])
        for it in rows:
            w.writerow([it["item_id"], f"{it['item_id']}.png", it["title"],
                        it.get("subject") or it["title"], it["category"],
                        it["words"], a.style, a.ground, a.ink, a.price])

    print(f"\nDone. face-pass {passed} · no-face {failed} · resumed {skipped} · errors {len(errored)}")
    if errored:
        print("  errors:", errored[:8])
    print(f"  images   -> {img_dir}")
    print(f"  manifest -> {man}  ({len(rows)} ready to render)")
    print("\n  Next: Admin Studio -> Batch (this manifest + images folder) -> Output -> "
          "Approve/Reject -> python tools/gallery_publish.py")


if __name__ == "__main__":
    main()
