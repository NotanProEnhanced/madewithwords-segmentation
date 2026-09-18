# Runbook for an outside agent: photos in, portraits out, on staging

Written for an agent that can generate images and run code (ChatGPT, or any other),
handed to it by the owner. It renders typographic portraits on the staging mirror at
`https://staging.typortrait.com` through the site's own render endpoint, the same call
the page makes and the same call the gate scripts (`ops/render-testset.sh`,
`ops/render-petset.sh`) make. Nothing here touches a live site or a paid path.

## 0. What to ask the owner for before starting

1. **The staging login.** The whole of staging.typortrait.com sits behind HTTP Basic
   Auth (nginx). You need a username and a password. Keep them in environment
   variables (`STG_USER`, `STG_PASS`) for the session; never write them into a file you
   keep, a notebook, a prompt or a log.
2. **Which brands to render for.** `typortrait`, `lovedinwords`, `faithinwords` for
   people; `pawsinwords` for cats and dogs. The brand changes the watermark, the words'
   voice and which backdrops make sense; it does not change the engine.
3. **Where the results go.** A folder of PNGs plus a CSV of what was rendered. The
   owner pulls them down or publishes them; do not post them anywhere yourself.

## 1. The source images you generate

These portraits are made from a real photograph's hair, skin and light, so the source
has to look like a real photograph, not an illustration.

- One subject per image. A person facing the camera or three-quarter, head and
  shoulders or waist-up; the face at least a quarter of the frame's height. For pets,
  one cat or dog, face visible, eyes open.
- Portrait orientation, 4:5 is ideal (e.g. 1600 x 2000). Minimum 1200 px on the short
  side; maximum file 20 MB (the proxy refuses above 25 MB).
- Natural light, in focus, no motion blur. Some hair against the background is good:
  that is what the matte is judged on.
- Plain or simple background. Avoid text, watermarks, frames, filters, heavy grain.
- Variety across the batch: ages, skin tones, hair (short, long, curly, white, none),
  glasses, hats, beards, side light, dark clothing on dark, light clothing on light.
  The hard cases are the useful ones.
- Save as JPEG quality 90+ or PNG. File names: `NN-short-description.jpg`.

State plainly in your notes that the people are synthetic. The site's consent field
(below) is for a real person's biometrics; synthetic faces have no such person.

## 2. The words you write

The format the page sends is a comma-separated list, the **name first**, then six to
nine traits, in capitals:

    ELEANOR, GRACE, KIND, BRAVE, WISE, WARM, LOYAL, HOME

Voice by brand: `typortrait` is a gift, warm and specific ("GOOFY, BRILLIANT, MOM");
`lovedinwords` is a memorial, gentle, past tense in spirit ("GRACE, ALWAYS, LOVED");
`faithinwords` is devotional ("GRACE, FAITH, HOPE, PSALM 23"); `pawsinwords` is a pet
("MILO, LOYAL, GENTLE, GOOFY, SOUL"). Short words fill a face better than long ones;
one or two long words are fine.

## 3. The render call

`POST https://staging.typortrait.com/render`, multipart form, Basic Auth. Fields:

| field | value |
|---|---|
| `image` | the photo file |
| `words` | the list from section 2 |
| `message` | the same string (the page sends both) |
| `style` | `displacement` for the **Words** style, `woven` for the **Photo** style (people) |
| `ink` | `photo` (the default look); also `mono`, `sepia`, `navy`, `burgundy`, `forest`, `gold_noir`, `spectrum`, `aurora` |
| `ground` | `navy` |
| `backdrop` | optional: `studio`, `sand`, `slate`, `transparent`; on lovedinwords/faithinwords also `wildflowers`, `roses`, `eucalyptus`, `line` |
| `png_width` | `1400` (a preview; `1500` is the most the site serves unpaid) |
| `aspect` | `0.8` |
| `remove_bg` | `true` |
| `uppercase` | `true` |
| `brand` | one of the four |
| `ref` | the same as `brand` |
| `biometric_consent` | `on` (required; the request is refused without it) |

For a pet (brand `pawsinwords`): omit `style`, and send `pet=1`, `pet_type=0.30`
(the typography size: 0.30 Small, 0.42 Medium, 0.56 Large), `ground=dark` (or `mid`,
`photo`).

The response is JSON. On success `ok` is true and `preview` is a path such as
`/outputs/<job>_preview.png`. Fetch `https://staging.typortrait.com` + that path with
the same Basic Auth to get the PNG. On failure `ok` is false and `error` names the
reason (`biometric_consent_required`, `no_face`, `image_too_small`, ...): record it and
move on; do not retry a refusal.

Example in Python:

```python
import os, requests, time
BASE = "https://staging.typortrait.com"
AUTH = (os.environ["STG_USER"], os.environ["STG_PASS"])

def render(path, words, brand, style="displacement", ink="photo", backdrop=None, pet=False, pet_type=0.30):
    data = {"words": words, "message": words, "ink": ink, "ground": "dark" if pet else "navy",
            "png_width": "1400", "aspect": "0.8", "remove_bg": "true", "uppercase": "true",
            "brand": brand, "ref": brand, "biometric_consent": "on"}
    if pet:
        data.update({"pet": "1", "pet_type": str(pet_type)})
    else:
        data["style"] = style
    if backdrop:
        data["backdrop"] = backdrop
    t0 = time.time()
    with open(path, "rb") as f:
        r = requests.post(f"{BASE}/render", data=data, files={"image": f}, auth=AUTH, timeout=180)
    j = r.json()
    if not j.get("ok"):
        return None, j.get("error", f"http {r.status_code}"), time.time() - t0
    png = requests.get(BASE + j["preview"], auth=AUTH, timeout=60).content
    return png, None, time.time() - t0
```

## 4. Timing and the one rule that matters

The heavy engine (the Photo style, and every pet) renders **one at a time across the
whole server** by design. The proxy cuts any request at 120 seconds. So:

- Send renders **sequentially**. Never in parallel, never from two scripts at once.
- Expect about 10 s for Words, 30 to 45 s for Photo, 25 to 40 s for a pet.
- If a request returns 502 or 504, or times out, wait 30 s and try that one once more.
  If it fails twice, record it and continue.
- A batch of 20 photos, both styles, is roughly 20 minutes. Do not run more than
  about 60 renders an hour; the box is shared with the live sites.

The same photo with the same words renders once and is then served from a cache, so a
second style or a backdrop change on it is faster than the first render.

## 5. The batch

Write a manifest first, one row per portrait, and run from it:

    file,brand,style,ink,backdrop,words
    01-woman-side-light.jpg,typortrait,displacement,photo,,ELEANOR, GRACE, KIND, BRAVE, WISE
    01-woman-side-light.jpg,typortrait,woven,photo,,ELEANOR, GRACE, KIND, BRAVE, WISE
    07-tabby.jpg,pawsinwords,pet,photo,,MILO, LOYAL, GENTLE, GOOFY, SOUL

Save each result as `<file-stem>_<brand>_<style>_<ink>[_<backdrop>].png`, and write
`results.csv` with: file, brand, style, ink, backdrop, ok, error, seconds, bytes,
width, height. Open every PNG once and confirm it decodes and its height is about
1.25 times its width; anything else is a failed render even if `ok` was true.

Report to the owner: how many rendered, how many refused and why, the slowest, and the
folder. Then stop. Do not judge the portraits; the owner does that by eye.

## 6. Do not

- Do not send anything to typortrait.com, lovedinwords.com, faithinwords.com or
  pawsinwords.com. Staging only.
- Do not call `/download`, `/checkout` or any Stripe or Printful path; those are the
  paid flow.
- Do not run renders in parallel.
- Do not put the login anywhere but the two environment variables.
- Do not upload photographs of real people you did not generate.
