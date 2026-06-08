# Typortrait

Premium typographic-portrait studio. Upload a portrait + a list of words, get back a typographic illustration where the words form the portrait.

Two render modes:
1. **Premium AI** (`/render-production`) — OpenAI `gpt-image-1` image-edit with a hand-tuned prompt. ~30–90 s, ~$0.04–0.17 per render.
2. **Notan Preview** (`/render`) — fast procedural OpenCV pipeline that thresholds the photo into notan regions and packs the user's words into each region as SVG. No API key needed, instant.

Within Premium mode there are two prompt styles:
- **Pure Typographic (seanings-style)** — every visible mark is a letter; letters BUILD the portrait. Default.
- **Editorial Lettering** — illustrated portrait with selective hand-lettering accents.

## Architecture

```
typortrait/
├── server/
│   ├── main.py             FastAPI app: /render, /render-production, /health
│   ├── requirements.txt    Python deps (FastAPI, OpenCV, OpenAI SDK, etc.)
│   └── .env.example        Template — copy to .env and add your OpenAI key
├── frontend/
│   └── index.html          Single-file UI (HTML/CSS/JS, no build step)
└── typo-portrait.html      Earlier standalone client-only renderer (legacy, kept for reference)
```

Frontend is served by FastAPI as static files at `/`, so everything is same-origin (no CORS issues for local dev).

## Setup

Requires Python 3.10+ (tested on Python 3.13 / Windows 11).

### Windows (PowerShell)

```powershell
cd path\to\typortrait\server
py -m pip install -r requirements.txt
Copy-Item .env.example .env
notepad .env          # paste OPENAI_API_KEY=sk-..., save, close
py -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

### macOS / Linux

```bash
cd path/to/typortrait/server
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
nano .env             # paste OPENAI_API_KEY=sk-..., Ctrl+O Enter Ctrl+X
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Then visit http://localhost:8000/.

The top-right badge will read `server online · OpenAI configured` once the key is loaded. If you only want the Notan Preview, you can run without an `.env` — Premium AI will return 503 until a key is configured.

## API

### `GET /health`
Returns `{ ok, renderer, version, openai_configured }`.

### `POST /render` — Notan procedural preview
Multipart form:
- `image` (file, required)
- `words` (string)
- `threshold` (int, default 128)
- `density` (float, default 1.0)
- `detail` (int, default 9)

Returns `{ ok, regions, wordsPlaced, svg }`.

### `POST /render-production` — Premium AI render
Multipart form:
- `image` (file, required)
- `words` (string, required)
- `detail` (int 15..75, default 55)
- `drama` (int 20..75, default 65)
- `style` (`typographic` or `editorial`, default `typographic`)

Returns `{ success, image }` where `image` is a `data:image/png;base64,...` URL.

## Prompts

Two prompt templates live at the top of `main.py`:

- `PURE_TYPOGRAPHIC_PROMPT` — aggressive "every mark is a letter" prompt. Currently the best-performing version is what's checked in (referred to as "v2" in the development history). Per-feature recipes for hair / brows / eyes / nose / lips / jaw. Complete-word rule. Identity-preservation section.
- `PRODUCTION_PROMPT` — the longer editorial-lettering prompt ported from the earlier `typortrait-stage1` codebase. ~500 lines, heavily iterated.

Both use `{words}`, `{detail_instruction}`, `{drama_instruction}` placeholders.

## Known limits & what we tried

`gpt-image-1` edit mode preserves photographic facial detail, which is why the face often stays drawn even when hair and clothing become typographic. Things attempted that **didn't** help:

- Server-side preprocessing (CLAHE + bilateral + posterize): triggered OpenAI moderation reliably.
- "Reconstruct the face from scratch" prompt language: triggered moderation.
- Shorter output-first prompt: face stayed drawn AND less hair typography came through; v2 is better.

This appears to be a model-architecture ceiling, not a prompt-engineering ceiling. To break past it, the realistic next steps are:

1. **FLUX.1 via Replicate** — img2img has stronger transformation strength and looser moderation for stylistic face transforms. ~$0.03–0.05 / render. Best chance at "letters BUILD the face."
2. **Stability AI SD 3.5 Ultra** — similar profile to FLUX.
3. **DALL·E 3 text-to-image (generate, not edit)** — same OpenAI key, no identity match but more typographic freedom; identity is described in the prompt ("a woman with hair pulled up, brown eyes…").

A FLUX backend would add a `/render-flux` endpoint parallel to `/render-production` and a third option to the frontend Style dropdown. ~1–2 hours of work.

## File-by-file notes

- `server/main.py`
  - Imports: FastAPI, OpenCV, PIL, NumPy, OpenAI SDK, dotenv.
  - `load_dotenv(dotenv_path=...)` uses an absolute path next to `main.py` so the key loads regardless of where uvicorn is launched from.
  - `print('[env] ...')` diagnostics on startup show whether the key is present.
  - `preprocess_for_typographic()` exists but is currently NOT called — kept for reference; calling it reliably trips OpenAI moderation.
  - Static frontend is mounted at `/` so the SPA and API are same-origin.

- `frontend/index.html` — single file, no bundler. Inline CSS + JS. Talks to same-origin `/render` and `/render-production`.

- `typo-portrait.html` — early standalone client-side procedural renderer using canvas. Kept for reference. Not loaded by the FastAPI app.

## What's NOT in this zip

- `server/.env` — contains the OpenAI API key. Each developer needs their own.
- `__pycache__/`, `.venv/` — regenerated by pip.
- `.git/` — fresh handoff; reinit if needed.
