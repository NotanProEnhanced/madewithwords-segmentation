#!/usr/bin/env python3
"""Blind judging of two test-set runs: the standing way to decide "is B better than A".

WHY
  Every judgement so far was made knowing which side was the change. That is how a
  change that merely looks different gets read as better, and how a real gain on one
  face is generalised from one glance. The engine is deterministic and the test set is
  fixed, so the comparison can be made blind: the same ten portraits (or the pets)
  rendered by two builds, shown as pairs in a random order with the sides swapped at
  random, judged without labels, and only then decoded.

USE
  ./blind-judge.py build <run A> <run B> [--label NAME] [--pets]
        Builds a page under the staging tree's static/blind/<label>/ and prints its URL.
        The page holds the images by pair number and side only; which side is which run
        is written to the KEY, kept outside the served tree:
            /root/typortrait-testset/blind/<label>-key.json
        Judge on the page (click, or keys 1 / 2 / 0 for left / right / no preference),
        then press "copy results" and paste the line into:
  ./blind-judge.py tally <label> '<pasted line>' ['<another judge's line>' ...]
        Decodes with the key and prints, per image, which run was preferred, the totals,
        and a two-sided sign test on the non-tied pairs. Several judges' lines tally
        together; each judge is named on the page.
  ./blind-judge.py rm <label>
        Removes the served page (the key stays).

RULE (see TESTSET.md): a change is adopted on a blind win, not on a look. Wins on at
least two thirds of the non-tied pairs with no loss on the hard cases is a clear win;
anything closer is "no difference", which is also an answer.

Runs are directories under the set's out/ (human) or pets/out/ (--pets): the names
render-testset.sh and render-petset.sh file them under. Only stdlib; runs on the host.
"""
from __future__ import annotations

import html
import json
import os
import random
import shutil
import sys
from math import comb

SET = os.environ.get("SET", "/root/typortrait-testset")
TREE = os.environ.get("TREE", "/root/typortrait-stg")
BASE = os.environ.get("BASE", "https://staging.typortrait.com")
DEST_ROOT = os.path.join(TREE, "typography_engine", "static", "blind")
KEY_DIR = os.path.join(SET, "blind")

HARD = ("05", "06", "07", "08")   # couple, sidelight, dark-on-dark, white-hair


def _resolve(outs: str, name: str) -> str:
    if os.path.isdir(name):
        return name
    p = os.path.join(outs, name)
    if os.path.isdir(p):
        return p
    hits = sorted(d for d in os.listdir(outs) if d.startswith(name)) if os.path.isdir(outs) else []
    if hits:
        return os.path.join(outs, hits[0])
    sys.exit(f"no run named {name!r} under {outs}")


def build(a_name: str, b_name: str, label: str, pets: bool) -> None:
    outs = os.path.join(SET, "pets", "out") if pets else os.path.join(SET, "out")
    A, B = _resolve(outs, a_name), _resolve(outs, b_name)
    names = sorted(f for f in os.listdir(A) if f.lower().endswith(".png") and os.path.isfile(os.path.join(B, f)))
    if not names:
        sys.exit(f"no matching .png files in {A} and {B}")
    dest = os.path.join(DEST_ROOT, label)
    if os.path.exists(dest):
        shutil.rmtree(dest)
    os.makedirs(os.path.join(dest, "L"))
    os.makedirs(os.path.join(dest, "R"))
    os.makedirs(KEY_DIR, exist_ok=True)
    rng = random.Random(f"{label}|{os.path.basename(A)}|{os.path.basename(B)}")
    order = list(names)
    rng.shuffle(order)
    key = {"label": label, "A": os.path.basename(A), "B": os.path.basename(B), "pairs": []}
    for i, f in enumerate(order, 1):
        left_is_a = rng.random() < 0.5
        src_l, src_r = (A, B) if left_is_a else (B, A)
        pid = "%02d" % i
        shutil.copyfile(os.path.join(src_l, f), os.path.join(dest, "L", pid + ".png"))
        shutil.copyfile(os.path.join(src_r, f), os.path.join(dest, "R", pid + ".png"))
        key["pairs"].append({"id": pid, "file": f, "left": "A" if left_is_a else "B"})
    with open(os.path.join(KEY_DIR, label + "-key.json"), "w") as fh:
        json.dump(key, fh, indent=1)
    with open(os.path.join(dest, "index.html"), "w") as fh:
        fh.write(_page(label, [p["id"] for p in key["pairs"]]))
    print(f"{len(names)} pairs.  page: {BASE}/static/blind/{label}/index.html   key: {KEY_DIR}/{label}-key.json")
    print("Judge every pair without opening the key. Then: ./blind-judge.py tally", label, "'<results line>'")


def _page(label: str, ids: list) -> str:
    items = "\n".join(
        f'<section class="pair" data-id="{i}"><div class="hd"><span class="n">pair {i}</span>'
        f'<span class="v" id="v{i}">unjudged</span></div>'
        f'<div class="imgs"><figure><img src="L/{i}.png" alt="" loading="lazy"><figcaption>left</figcaption></figure>'
        f'<figure><img src="R/{i}.png" alt="" loading="lazy"><figcaption>right</figcaption></figure></div>'
        f'<div class="btns"><button data-c="L">left is better</button><button data-c="T">no preference</button>'
        f'<button data-c="R">right is better</button></div></section>'
        for i in ids)
    lab = html.escape(label)
    return f"""<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Blind judging: {lab}</title>
<style>
  :root{{color-scheme:dark}}
  body{{margin:0;padding:20px;font:14px/1.5 system-ui,sans-serif;background:#111;color:#eee}}
  h1{{font-size:16px;font-weight:600;margin:0 0 4px}}
  p.note{{margin:0 0 16px;color:#9aa;max-width:70ch}}
  .who{{margin:0 0 20px}} .who input{{font:inherit;padding:4px 8px;background:#1b1f27;color:#eee;border:1px solid #2a2f3a;border-radius:4px}}
  .pair{{margin:0 0 30px;padding:0 0 12px;border-bottom:1px solid #2a2f3a}}
  .pair.cur{{outline:2px solid #6ea8ff;outline-offset:8px}}
  .hd{{display:flex;justify-content:space-between;font:600 13px/1.4 ui-monospace,monospace;color:#cfd3da;margin:0 0 6px}}
  .v{{color:#8b93a1}} .v.done{{color:#9ec5ff}}
  .imgs{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}
  figure{{margin:0}} figcaption{{font:11px/1.4 ui-monospace,monospace;color:#8b93a1;padding:4px 0}}
  img{{width:100%;height:auto;display:block;background:#000;border:1px solid #2a2f3a}}
  .btns{{display:flex;gap:8px;margin:8px 0 0}}
  button{{font:inherit;padding:6px 12px;background:#1b1f27;color:#eee;border:1px solid #3a4150;border-radius:4px;cursor:pointer}}
  button.on{{background:#2a4a7a;border-color:#6ea8ff}}
  .foot{{position:sticky;bottom:0;background:#111;padding:12px 0;border-top:1px solid #2a2f3a;display:flex;gap:12px;align-items:center;flex-wrap:wrap}}
  textarea{{width:100%;font:12px ui-monospace,monospace;background:#1b1f27;color:#eee;border:1px solid #2a2f3a;border-radius:4px;padding:6px}}
</style>
<h1>Blind judging: {lab}</h1>
<p class="note">Two builds rendered the same portraits. Which side is which is not on this page and
differs from pair to pair. Judge each pair as a customer would judge a print: likeness, the words,
the eyes, the hair, the edge. Keys: <b>1</b> left, <b>2</b> right, <b>0</b> no preference; the page
scrolls to the next pair. Choices are kept in this browser until you copy them out.</p>
<p class="who">Judge: <input id="who" placeholder="your name" size="18"></p>
{items}
<div class="foot"><button id="copy">copy results</button><span id="prog"></span>
<textarea id="out" rows="2" readonly placeholder="results appear here; paste the line into blind-judge.py tally"></textarea></div>
<script>
(function(){{
  const label={json.dumps(label)}, ids={json.dumps(ids)};
  const K="blind-"+label;
  let st={{}}; try{{st=JSON.parse(localStorage.getItem(K)||"{{}}")}}catch(e){{st={{}}}}
  const who=document.getElementById("who"); who.value=st._who||"";
  who.addEventListener("input",()=>{{st._who=who.value;save()}});
  function save(){{try{{localStorage.setItem(K,JSON.stringify(st))}}catch(e){{}}render()}}
  function render(){{
    let n=0;
    ids.forEach(i=>{{const s=document.querySelector('.pair[data-id="'+i+'"]');const c=st[i];
      s.querySelectorAll("button").forEach(b=>b.classList.toggle("on",b.dataset.c===c));
      const v=document.getElementById("v"+i); v.textContent=c?({{L:"left",R:"right",T:"no preference"}})[c]:"unjudged"; v.classList.toggle("done",!!c);
      if(c)n++;}});
    document.getElementById("prog").textContent=n+" of "+ids.length+" judged";
    const line=(st._who||"anon").replace(/[^\\w.-]/g,"_")+" "+label+" "+ids.map(i=>i+":"+(st[i]||"-")).join(",");
    document.getElementById("out").value=line;
  }}
  let cur=0;
  function setCur(k){{cur=Math.max(0,Math.min(ids.length-1,k));document.querySelectorAll(".pair").forEach((s,j)=>s.classList.toggle("cur",j===cur));
    document.querySelectorAll(".pair")[cur].scrollIntoView({{behavior:"smooth",block:"start"}});}}
  document.querySelectorAll(".pair").forEach((s,j)=>{{
    s.querySelectorAll("button").forEach(b=>b.addEventListener("click",()=>{{st[s.dataset.id]=b.dataset.c;save();setCur(j+1)}}));
    s.addEventListener("click",()=>{{cur=j;document.querySelectorAll(".pair").forEach((x,k)=>x.classList.toggle("cur",k===j))}});
  }});
  document.addEventListener("keydown",e=>{{const m={{"1":"L","2":"R","0":"T"}}[e.key];if(!m||e.target.tagName==="INPUT")return;
    st[ids[cur]]=m;save();setCur(cur+1)}});
  document.getElementById("copy").addEventListener("click",()=>{{const t=document.getElementById("out");t.select();
    try{{navigator.clipboard.writeText(t.value)}}catch(e){{document.execCommand("copy")}}}});
  render();setCur(0);
}})();
</script>
"""


def _sign_test(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    k = max(wins, losses)
    p = sum(comb(n, j) for j in range(k, n + 1)) / 2.0 ** n
    return min(1.0, 2.0 * p)


def tally(label: str, lines: list) -> None:
    with open(os.path.join(KEY_DIR, label + "-key.json")) as fh:
        key = json.load(fh)
    side = {p["id"]: p for p in key["pairs"]}
    votes = {}   # file -> list of (judge, "A"|"B"|"T")
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 3 or parts[1] != label:
            sys.exit(f"not a results line for {label}: {line!r}")
        judge = parts[0]
        for item in parts[2].split(","):
            pid, c = item.split(":")
            if c == "-":
                continue
            p = side[pid]
            r = "T" if c == "T" else (p["left"] if c == "L" else ("B" if p["left"] == "A" else "A"))
            votes.setdefault(p["file"], []).append((judge, r))
    A, B = key["A"], key["B"]
    print(f"A = {A}\nB = {B}\n")
    wins = losses = ties = 0
    hard_loss = []
    for f in sorted(votes):
        vs = votes[f]
        a = sum(1 for _, r in vs if r == "A"); b = sum(1 for _, r in vs if r == "B"); t = len(vs) - a - b
        verdict = "B" if b > a else ("A" if a > b else "tie")
        wins += verdict == "B"; losses += verdict == "A"; ties += verdict == "tie"
        if verdict == "A" and f[:2] in HARD:
            hard_loss.append(f)
        print(f"  {f:34s} A {a}  B {b}  none {t}   -> {'B better' if verdict == 'B' else 'A better' if verdict == 'A' else 'no preference'}")
    n = wins + losses
    p = _sign_test(wins, losses)
    print(f"\nB preferred on {wins}, A on {losses}, no preference on {ties}  (judges: {len(lines)})")
    print(f"sign test on the {n} decided pairs: p = {p:.3f}")
    if n and wins >= 2 * n / 3 and not hard_loss:
        print("CLEAR WIN for B: adopt.")
    elif n and losses >= 2 * n / 3:
        print("CLEAR LOSS for B: do not adopt.")
    else:
        print("NO CLEAR DIFFERENCE" + (f" (B lost a hard case: {', '.join(hard_loss)})" if hard_loss else "") + ": do not adopt on this evidence.")


def main(argv: list) -> None:
    if len(argv) < 2 or argv[1] not in ("build", "tally", "rm"):
        sys.exit(__doc__)
    cmd = argv[1]
    if cmd == "build":
        args = [a for a in argv[2:] if not a.startswith("--")]
        pets = "--pets" in argv
        label = None
        if "--label" in argv:
            label = argv[argv.index("--label") + 1]
            args = [a for a in args if a != label]
        if len(args) != 2:
            sys.exit("usage: blind-judge.py build <run A> <run B> [--label NAME] [--pets]")
        label = label or f"{os.path.basename(args[0])}-vs-{os.path.basename(args[1])}"
        build(args[0], args[1], label, pets)
    elif cmd == "tally":
        if len(argv) < 4:
            sys.exit("usage: blind-judge.py tally <label> '<results line>' [...]")
        tally(argv[2], argv[3:])
    else:
        dest = os.path.join(DEST_ROOT, argv[2])
        shutil.rmtree(dest, ignore_errors=True)
        print("removed", dest)


if __name__ == "__main__":
    main(sys.argv)
