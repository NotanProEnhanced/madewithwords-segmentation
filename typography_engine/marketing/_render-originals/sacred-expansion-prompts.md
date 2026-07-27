# FaithInWords — Sacred Expansion: source + typography prompts

Production record for the nine-collection expansion (Phase 3). For each new portrait:
1. **Source prompt** → paste into ChatGPT (image gen) to produce the base face.
2. **Word list** → the words the engine weaves into that face.
3. **Render** → I run it through the engine (Lifelike, navy ground) and wire it in.

Pipeline per piece: you generate the source → hand me `{id}-source.png` → I run
`gallery_render` + build the thumbnail/product images + add to the catalog → it goes live.
(Unlike the marketing demos, the framed mockups are auto-generated from the master — no
separate mockup prompt needed.)

This batch = **Portraits of Christ + Easter** (the approved "produce first" priority).

---

## House style — keep every source image consistent

Match the existing 46 renders (see `st-francis-of-assisi.png`, `the-blessed-virgin-mary.png`):

- **Photorealistic**, reverent devotional portrait — a real, lifelike human face (NOT a
  painting, drawing, icon, or 3D render).
- **Head-and-shoulders, centered, facing the camera straight on.**
- **Clear, luminous eyes in sharp focus** — the eyes are the focal point (the renderer is
  driven by them; looking-away or closed eyes break the render).
- Soft, gentle **directional light** (Rembrandt-style), natural skin tones, a dignified,
  serene expression.
- **Plain, dark, unadorned background** (it's replaced by the navy ground at render time).
- As if shot on an **85mm lens at f/4**, **4:5 vertical**, high resolution.
- **No text, no watermark, no border, no lettering, no printed halo.**

**Christ consistency (critical):** every "Portraits of Christ" and "Easter" piece must read
as the *same Christ*. Use this fixed likeness in every prompt: *a man in his early thirties,
warm olive Middle-Eastern skin, long dark brown wavy hair parted in the middle falling past
the shoulders, a full soft beard, deep compassionate brown eyes, a noble serene brow.* For
the tightest match, generate the whole set **in one ChatGPT session**, and/or upload one
existing Christ render as a face reference before each prompt.

**Render command (same for all — Lifelike):**
```
python tools/gallery_render.py <id> path/to/<id>-source.png --ground navy --ink photo
```

---

## Portraits of Christ (9 new)

### sacred-heart-of-jesus — Sacred Heart
**Source prompt:**
> A photorealistic, reverent devotional portrait of Jesus Christ — a man in his early thirties, warm olive Middle-Eastern skin, long dark brown wavy hair parted in the middle, a full soft beard, deep compassionate brown eyes, a tender serene expression. Head-and-shoulders, centered, facing the camera straight on, one hand resting gently over the center of his chest. Soft directional Rembrandt lighting, natural skin tones, plain dark unadorned background. 85mm at f/4, 4:5 vertical, high resolution, sharp focus on the eyes. No text, no printed halo, no watermark.

**Words:** `SACRED HEART LOVE MERCY DEVOTION FLAME THORNS COMPASSION BURNING PIERCED REFUGE GRACE`
**Ground:** navy

### christ-the-teacher — Christ the Teacher
**Source prompt:**
> A photorealistic, reverent devotional portrait of Jesus Christ — [same likeness: early thirties, olive Middle-Eastern skin, long dark wavy center-parted hair, full soft beard, deep brown eyes] — with a calm, authoritative, kindly expression. Head-and-shoulders, centered, facing the camera straight on, one hand raised in a gentle open teaching gesture near the chest. Soft directional light, natural skin tones, plain dark background. 85mm at f/4, 4:5 vertical, high resolution, eyes clear and in focus. No text, no halo, no watermark.

**Words:** `CHRIST TEACHER RABBI WISDOM PARABLES BEATITUDES SERMON TRUTH WORD DISCIPLES LIGHT`
**Ground:** navy

### christ-in-prayer — Christ in Prayer
**Source prompt:**
> A photorealistic, reverent devotional portrait of Jesus Christ — [same likeness] — a prayerful, serene expression of quiet surrender, eyes softly open and reverent, hands joined together in prayer just beneath the chin. Head-and-shoulders, centered, facing the camera straight on. Gentle light from above, natural skin tones, plain dark background. 85mm at f/4, 4:5 vertical, high resolution, eyes clear and in focus. No text, no halo, no watermark.

**Words:** `CHRIST PRAYER GETHSEMANE FATHER SURRENDER COMMUNION VIGIL SILENCE WATCH WILL TRUST`
**Ground:** navy

### christ-the-healer — Christ the Healer
**Source prompt:**
> A photorealistic, reverent devotional portrait of Jesus Christ — [same likeness] — a compassionate, gentle expression, one hand extended forward in a healing gesture of blessing. Head-and-shoulders, centered, facing the camera straight on. Soft directional light, natural skin tones, plain dark background. 85mm at f/4, 4:5 vertical, high resolution, sharp focus on the eyes. No text, no halo, no watermark.

**Words:** `CHRIST HEALER TOUCH MERCY FAITH RESTORE WHOLENESS BLIND LAME LEPER COMPASSION`
**Ground:** navy

### christ-the-carpenter — Christ the Carpenter
**Source prompt:**
> A photorealistic, reverent devotional portrait of Jesus Christ as the young carpenter of Nazareth — [same likeness, perhaps a touch younger and more rugged] — a humble, steady expression, wearing a simple rough working tunic, strong weathered hands visible near the chest. Head-and-shoulders, centered, facing the camera straight on. Warm workshop light, natural skin tones, plain dark background. 85mm at f/4, 4:5 vertical, high resolution, eyes clear and in focus. No text, no halo, no watermark.

**Words:** `CHRIST CARPENTER NAZARETH LABOR HANDS HUMILITY HIDDEN WORKMAN WOOD YOKE TRADE`
**Ground:** navy

### christ-the-bridegroom — The Bridegroom
**Source prompt:**
> A photorealistic, reverent devotional portrait of Jesus Christ as the Bridegroom — [same likeness] — a warm, joyful, loving expression, a serene tenderness in the eyes. Head-and-shoulders, centered, facing the camera straight on, wearing a fine simple robe. Soft warm directional light, natural skin tones, plain dark background. 85mm at f/4, 4:5 vertical, high resolution, sharp focus on the eyes. No text, no halo, no watermark.

**Words:** `CHRIST BRIDEGROOM COVENANT LOVE CHURCH FEAST JOY UNION FAITHFUL PROMISE AWAITING`
**Ground:** navy

### bread-of-life — Bread of Life
**Source prompt:**
> A photorealistic, reverent devotional portrait of Jesus Christ — [same likeness] — a gentle, giving expression, both hands holding and offering a small round loaf of bread near the chest. Head-and-shoulders, centered, facing the camera straight on. Soft directional light, natural skin tones, plain dark background. 85mm at f/4, 4:5 vertical, high resolution, eyes clear and in focus. No text, no halo, no watermark.

**Words:** `CHRIST BREAD LIFE MANNA EUCHARIST NOURISH HUNGER GIVEN BROKEN BODY SUSTAIN`
**Ground:** navy

### living-water — Living Water
**Source prompt:**
> A photorealistic, reverent devotional portrait of Jesus Christ — [same likeness] — a serene, inviting expression, cupped hands holding clear water near the chest, a few bright droplets catching the light. Head-and-shoulders, centered, facing the camera straight on. Soft directional light, natural skin tones, plain dark background. 85mm at f/4, 4:5 vertical, high resolution, sharp focus on the eyes. No text, no halo, no watermark.

**Words:** `CHRIST LIVING WATER WELL THIRST SPRING SPIRIT CLEANSE FLOW ETERNAL SAMARITAN`
**Ground:** navy

### alpha-and-omega — Alpha and Omega
**Source prompt:**
> A photorealistic, reverent devotional portrait of Jesus Christ, eternal and majestic — [same likeness] — a timeless, noble, serene expression, a faint radiance behind him. Head-and-shoulders, centered, facing the camera straight on, wearing a regal simple robe. Soft directional light with a subtle glow, natural skin tones, plain dark background. 85mm at f/4, 4:5 vertical, high resolution, sharp focus on the eyes. No text, no letters, no halo, no watermark.

**Words:** `CHRIST ALPHA OMEGA BEGINNING END ETERNAL FIRST LAST ALMIGHTY WAS IS`
**Ground:** navy

---

## Easter Collection (5 new)

### risen-christ — Risen Christ
**Source prompt:**
> A photorealistic, reverent devotional portrait of the risen Jesus Christ — [same likeness] — a serene, triumphant, luminous expression, calm eyes full of life, one hand gently raised showing a wound in the palm, a faint radiance around him. Head-and-shoulders, centered, facing the camera straight on, wearing a white robe. Soft glowing directional light, natural skin tones, plain dark background. 85mm at f/4, 4:5 vertical, high resolution, sharp focus on the eyes. No text, no halo, no watermark.

**Words:** `CHRIST RISEN RESURRECTION VICTORY LIFE TOMB EMPTY ALLELUIA LIGHT DEATH CONQUERED`
**Ground:** navy

### christ-crowned-with-thorns — Christ Crowned with Thorns
**Source prompt:**
> A photorealistic, reverent devotional portrait of Jesus Christ crowned with thorns — [same likeness] — a sorrowful, dignified, silent expression, eyes open and gentle, a woven crown of thorns on his brow with a few small traces of blood, bare shoulders. Head-and-shoulders, centered, facing the camera straight on. Low, soft chiaroscuro light, natural skin tones, plain dark background. 85mm at f/4, 4:5 vertical, high resolution, sharp focus on the eyes. Restrained and reverent, not gory. No text, no halo, no watermark.

**Words:** `CHRIST THORNS CROWN PASSION SUFFERING KING MOCKED SORROW SILENT SACRIFICE LOVE`
**Ground:** navy *(optional: black for a darker Passion tone)*

### christ-before-the-cross — Christ Before the Cross
**Source prompt:**
> A photorealistic, reverent devotional portrait of Jesus Christ — [same likeness] — a resolute, sorrowful expression of loving surrender, the rough dark beam of a wooden cross resting against one shoulder. Head-and-shoulders, centered, facing the camera straight on. Low, soft directional light, natural skin tones, plain dark background. 85mm at f/4, 4:5 vertical, high resolution, sharp focus on the eyes. No text, no halo, no watermark.

**Words:** `CHRIST CROSS SURRENDER OBEDIENCE SACRIFICE LOVE WEIGHT REDEMPTION SILENT OFFERING FATHER`
**Ground:** navy *(optional: black)*

### resurrection-morning — Resurrection Morning
**Source prompt:**
> A photorealistic, reverent devotional portrait of the risen Jesus Christ at dawn — [same likeness] — a gentle, hopeful, radiant expression, calm luminous eyes, soft golden morning light on his face as if in a garden at sunrise, wearing a white robe. Head-and-shoulders, centered, facing the camera straight on. Warm soft directional dawn light, natural skin tones, plain dark background. 85mm at f/4, 4:5 vertical, high resolution, sharp focus on the eyes. No text, no halo, no watermark.

**Words:** `CHRIST RESURRECTION MORNING DAWN GARDEN RISEN GLORY HOPE NEW LIFE FIRSTBORN`
**Ground:** navy

### christ-triumphant — Christ Triumphant
**Source prompt:**
> A photorealistic, reverent devotional portrait of Jesus Christ triumphant and reigning — [same likeness] — a majestic, serene, victorious expression, a noble regal bearing, wearing a fine robe, a faint radiance of glory behind him. Head-and-shoulders, centered, facing the camera straight on. Soft directional light with a subtle glow, natural skin tones, plain dark background. 85mm at f/4, 4:5 vertical, high resolution, sharp focus on the eyes. No text, no crown lettering, no watermark.

**Words:** `CHRIST TRIUMPHANT KING GLORY VICTORY REIGN MAJESTY CONQUEROR THRONE EXALTED LORD`
**Ground:** navy

---

## Still to come (on your go-ahead)

Same format, next batches: **Blessed Virgin Mary (5)** · **Angels (4)** · **Apostles (9)** ·
**Great Saints (13, incl. Aquinas / Teresa of Ávila / Monica)** · **Prophets & Old Testament
(8)** · **Women of Faith (7)** · **Christmas (3)**. ~49 more, produced in whatever order you
want to render them.
