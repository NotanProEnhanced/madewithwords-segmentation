# FaithInWords — Sacred Expansion: source + typography prompts

Production record for the nine-collection expansion (Phase 3). For each new portrait:
**Source prompt** (paste into ChatGPT image gen), **Words** (what the engine weaves in), and
the **Render** command. Every prompt below is self-contained — copy any single code block
and paste it straight in, no editing.

Pipeline per piece: you generate the source → hand me `{id}-source.png` → I run the render +
build the thumbnail/product images + add it to the catalog → it goes live.

This batch = **Portraits of Christ + Easter** (the approved "produce first" priority).

---

## House style (already baked into every prompt below)

Matches the existing 46 renders: photorealistic reverent portrait, real lifelike face,
head-and-shoulders, straight-on, luminous eyes in sharp focus, soft directional light, plain
dark background, 85mm/f4, 4:5, no text/halo/watermark. Every "Christ" prompt uses one fixed
likeness so the whole collection reads as the *same* Jesus.

**Tip for facial consistency:** generate the Christ + Easter set in **one ChatGPT session**,
and/or upload one finished Christ render as a reference image before each prompt.

---

## Portraits of Christ (9 new)

### sacred-heart-of-jesus — Sacred Heart
```
A photorealistic, reverent devotional portrait of Jesus Christ — a man in his early thirties with warm olive Middle-Eastern skin, long dark brown wavy hair parted in the middle falling past his shoulders, a full soft beard, and deep compassionate brown eyes. A tender, serene expression. Head-and-shoulders, centered, facing the camera straight on, one hand resting gently over the center of his chest. Soft directional Rembrandt lighting, natural skin tones, plain dark unadorned background. Shot as if on an 85mm lens at f/4, 4:5 vertical, high resolution, sharp focus on the eyes. No text, no printed halo, no watermark, no border.
```
```
SACRED HEART LOVE MERCY DEVOTION FLAME THORNS COMPASSION BURNING PIERCED REFUGE GRACE
```
**Render:** `python tools/gallery_render.py sacred-heart-of-jesus sacred-heart-of-jesus-source.png --ground navy --ink photo`

### christ-the-teacher — Christ the Teacher
```
A photorealistic, reverent devotional portrait of Jesus Christ — a man in his early thirties with warm olive Middle-Eastern skin, long dark brown wavy hair parted in the middle falling past his shoulders, a full soft beard, and deep compassionate brown eyes. A calm, authoritative, kindly expression. Head-and-shoulders, centered, facing the camera straight on, one hand raised in a gentle open teaching gesture near the chest. Soft directional light, natural skin tones, plain dark unadorned background. Shot as if on an 85mm lens at f/4, 4:5 vertical, high resolution, eyes clear and in sharp focus. No text, no halo, no watermark, no border.
```
```
CHRIST TEACHER RABBI WISDOM PARABLES BEATITUDES SERMON TRUTH WORD DISCIPLES LIGHT
```
**Render:** `python tools/gallery_render.py christ-the-teacher christ-the-teacher-source.png --ground navy --ink photo`

### christ-in-prayer — Christ in Prayer
```
A photorealistic, reverent devotional portrait of Jesus Christ — a man in his early thirties with warm olive Middle-Eastern skin, long dark brown wavy hair parted in the middle falling past his shoulders, a full soft beard, and deep compassionate brown eyes. A prayerful, serene expression of quiet surrender, eyes softly open and reverent, hands joined together in prayer just beneath the chin. Head-and-shoulders, centered, facing the camera straight on. Gentle light from above, natural skin tones, plain dark unadorned background. Shot as if on an 85mm lens at f/4, 4:5 vertical, high resolution, eyes clear and in sharp focus. No text, no halo, no watermark, no border.
```
```
CHRIST PRAYER GETHSEMANE FATHER SURRENDER COMMUNION VIGIL SILENCE WATCH WILL TRUST
```
**Render:** `python tools/gallery_render.py christ-in-prayer christ-in-prayer-source.png --ground navy --ink photo`

### christ-the-healer — Christ the Healer
```
A photorealistic, reverent devotional portrait of Jesus Christ — a man in his early thirties with warm olive Middle-Eastern skin, long dark brown wavy hair parted in the middle falling past his shoulders, a full soft beard, and deep compassionate brown eyes. A compassionate, gentle expression, one hand extended forward in a healing gesture of blessing. Head-and-shoulders, centered, facing the camera straight on. Soft directional light, natural skin tones, plain dark unadorned background. Shot as if on an 85mm lens at f/4, 4:5 vertical, high resolution, sharp focus on the eyes. No text, no halo, no watermark, no border.
```
```
CHRIST HEALER TOUCH MERCY FAITH RESTORE WHOLENESS BLIND LAME LEPER COMPASSION
```
**Render:** `python tools/gallery_render.py christ-the-healer christ-the-healer-source.png --ground navy --ink photo`

### christ-the-carpenter — Christ the Carpenter
```
A photorealistic, reverent devotional portrait of Jesus Christ as the young carpenter of Nazareth — a man in his early thirties with warm olive Middle-Eastern skin, long dark brown wavy hair parted in the middle falling past his shoulders, a full soft beard, deep compassionate brown eyes, and a slightly rugged, sun-weathered look. A humble, steady expression, wearing a simple rough working tunic, strong weathered hands visible near the chest. Head-and-shoulders, centered, facing the camera straight on. Warm workshop light, natural skin tones, plain dark unadorned background. Shot as if on an 85mm lens at f/4, 4:5 vertical, high resolution, eyes clear and in sharp focus. No text, no halo, no watermark, no border.
```
```
CHRIST CARPENTER NAZARETH LABOR HANDS HUMILITY HIDDEN WORKMAN WOOD YOKE TRADE
```
**Render:** `python tools/gallery_render.py christ-the-carpenter christ-the-carpenter-source.png --ground navy --ink photo`

### christ-the-bridegroom — The Bridegroom
```
A photorealistic, reverent devotional portrait of Jesus Christ as the Bridegroom — a man in his early thirties with warm olive Middle-Eastern skin, long dark brown wavy hair parted in the middle falling past his shoulders, a full soft beard, and deep compassionate brown eyes. A warm, joyful, loving expression with a serene tenderness in the eyes, wearing a fine simple robe. Head-and-shoulders, centered, facing the camera straight on. Soft warm directional light, natural skin tones, plain dark unadorned background. Shot as if on an 85mm lens at f/4, 4:5 vertical, high resolution, sharp focus on the eyes. No text, no halo, no watermark, no border.
```
```
CHRIST BRIDEGROOM COVENANT LOVE CHURCH FEAST JOY UNION FAITHFUL PROMISE AWAITING
```
**Render:** `python tools/gallery_render.py christ-the-bridegroom christ-the-bridegroom-source.png --ground navy --ink photo`

### bread-of-life — Bread of Life
```
A photorealistic, reverent devotional portrait of Jesus Christ — a man in his early thirties with warm olive Middle-Eastern skin, long dark brown wavy hair parted in the middle falling past his shoulders, a full soft beard, and deep compassionate brown eyes. A gentle, giving expression, both hands holding and offering a small round loaf of bread near the chest. Head-and-shoulders, centered, facing the camera straight on. Soft directional light, natural skin tones, plain dark unadorned background. Shot as if on an 85mm lens at f/4, 4:5 vertical, high resolution, eyes clear and in sharp focus. No text, no halo, no watermark, no border.
```
```
CHRIST BREAD LIFE MANNA EUCHARIST NOURISH HUNGER GIVEN BROKEN BODY SUSTAIN
```
**Render:** `python tools/gallery_render.py bread-of-life bread-of-life-source.png --ground navy --ink photo`

### living-water — Living Water
```
A photorealistic, reverent devotional portrait of Jesus Christ — a man in his early thirties with warm olive Middle-Eastern skin, long dark brown wavy hair parted in the middle falling past his shoulders, a full soft beard, and deep compassionate brown eyes. A serene, inviting expression, cupped hands holding clear water near the chest with a few bright droplets catching the light. Head-and-shoulders, centered, facing the camera straight on. Soft directional light, natural skin tones, plain dark unadorned background. Shot as if on an 85mm lens at f/4, 4:5 vertical, high resolution, sharp focus on the eyes. No text, no halo, no watermark, no border.
```
```
CHRIST LIVING WATER WELL THIRST SPRING SPIRIT CLEANSE FLOW ETERNAL SAMARITAN
```
**Render:** `python tools/gallery_render.py living-water living-water-source.png --ground navy --ink photo`

### alpha-and-omega — Alpha and Omega
```
A photorealistic, reverent devotional portrait of Jesus Christ, eternal and majestic — a man in his early thirties with warm olive Middle-Eastern skin, long dark brown wavy hair parted in the middle falling past his shoulders, a full soft beard, and deep compassionate brown eyes. A timeless, noble, serene expression, wearing a regal simple robe, a faint radiance behind him. Head-and-shoulders, centered, facing the camera straight on. Soft directional light with a subtle glow, natural skin tones, plain dark unadorned background. Shot as if on an 85mm lens at f/4, 4:5 vertical, high resolution, sharp focus on the eyes. No text, no letters, no halo, no watermark, no border.
```
```
CHRIST ALPHA OMEGA BEGINNING END ETERNAL FIRST LAST ALMIGHTY WAS IS
```
**Render:** `python tools/gallery_render.py alpha-and-omega alpha-and-omega-source.png --ground navy --ink photo`

---

## Easter Collection (5 new)

### risen-christ — Risen Christ
```
A photorealistic, reverent devotional portrait of the risen Jesus Christ — a man in his early thirties with warm olive Middle-Eastern skin, long dark brown wavy hair parted in the middle falling past his shoulders, a full soft beard, and deep compassionate brown eyes. A serene, triumphant, luminous expression, calm eyes full of life, one hand gently raised showing a wound in the palm, wearing a white robe, a faint radiance around him. Head-and-shoulders, centered, facing the camera straight on. Soft glowing directional light, natural skin tones, plain dark unadorned background. Shot as if on an 85mm lens at f/4, 4:5 vertical, high resolution, sharp focus on the eyes. No text, no halo, no watermark, no border.
```
```
CHRIST RISEN RESURRECTION VICTORY LIFE TOMB EMPTY ALLELUIA LIGHT DEATH CONQUERED
```
**Render:** `python tools/gallery_render.py risen-christ risen-christ-source.png --ground navy --ink photo`

### christ-crowned-with-thorns — Christ Crowned with Thorns
```
A photorealistic, reverent devotional portrait of Jesus Christ crowned with thorns — a man in his early thirties with warm olive Middle-Eastern skin, long dark brown wavy hair parted in the middle falling past his shoulders, a full soft beard, and deep compassionate brown eyes. A sorrowful, dignified, silent expression, eyes open and gentle, a woven crown of thorns on his brow with only a few small traces of blood, bare shoulders. Head-and-shoulders, centered, facing the camera straight on. Low, soft chiaroscuro light, natural skin tones, plain dark unadorned background. Restrained and reverent, not gory. Shot as if on an 85mm lens at f/4, 4:5 vertical, high resolution, sharp focus on the eyes. No text, no halo, no watermark, no border.
```
```
CHRIST THORNS CROWN PASSION SUFFERING KING MOCKED SORROW SILENT SACRIFICE LOVE
```
**Render:** `python tools/gallery_render.py christ-crowned-with-thorns christ-crowned-with-thorns-source.png --ground navy --ink photo`

### christ-before-the-cross — Christ Before the Cross
```
A photorealistic, reverent devotional portrait of Jesus Christ — a man in his early thirties with warm olive Middle-Eastern skin, long dark brown wavy hair parted in the middle falling past his shoulders, a full soft beard, and deep compassionate brown eyes. A resolute, sorrowful expression of loving surrender, the rough dark beam of a wooden cross resting against one shoulder. Head-and-shoulders, centered, facing the camera straight on. Low, soft directional light, natural skin tones, plain dark unadorned background. Shot as if on an 85mm lens at f/4, 4:5 vertical, high resolution, sharp focus on the eyes. No text, no halo, no watermark, no border.
```
```
CHRIST CROSS SURRENDER OBEDIENCE SACRIFICE LOVE WEIGHT REDEMPTION SILENT OFFERING FATHER
```
**Render:** `python tools/gallery_render.py christ-before-the-cross christ-before-the-cross-source.png --ground navy --ink photo`

### resurrection-morning — Resurrection Morning
```
A photorealistic, reverent devotional portrait of the risen Jesus Christ at dawn — a man in his early thirties with warm olive Middle-Eastern skin, long dark brown wavy hair parted in the middle falling past his shoulders, a full soft beard, and deep compassionate brown eyes. A gentle, hopeful, radiant expression, calm luminous eyes, soft golden morning light on his face as if in a garden at sunrise, wearing a white robe. Head-and-shoulders, centered, facing the camera straight on. Warm soft directional dawn light, natural skin tones, plain dark unadorned background. Shot as if on an 85mm lens at f/4, 4:5 vertical, high resolution, sharp focus on the eyes. No text, no halo, no watermark, no border.
```
```
CHRIST RESURRECTION MORNING DAWN GARDEN RISEN GLORY HOPE NEW LIFE FIRSTBORN
```
**Render:** `python tools/gallery_render.py resurrection-morning resurrection-morning-source.png --ground navy --ink photo`

### christ-triumphant — Christ Triumphant
```
A photorealistic, reverent devotional portrait of Jesus Christ triumphant and reigning — a man in his early thirties with warm olive Middle-Eastern skin, long dark brown wavy hair parted in the middle falling past his shoulders, a full soft beard, and deep compassionate brown eyes. A majestic, serene, victorious expression with a noble regal bearing, wearing a fine robe, a faint radiance of glory behind him. Head-and-shoulders, centered, facing the camera straight on. Soft directional light with a subtle glow, natural skin tones, plain dark unadorned background. Shot as if on an 85mm lens at f/4, 4:5 vertical, high resolution, sharp focus on the eyes. No text, no crown lettering, no halo, no watermark, no border.
```
```
CHRIST TRIUMPHANT KING GLORY VICTORY REIGN MAJESTY CONQUEROR THRONE EXALTED LORD
```
**Render:** `python tools/gallery_render.py christ-triumphant christ-triumphant-source.png --ground navy --ink photo`

---

## Still to come (on your go-ahead)

Same self-contained, copy-paste format: **Blessed Virgin Mary (5)** · **Angels (4)** ·
**Apostles (9)** · **Great Saints (13, incl. Aquinas / Teresa of Ávila / Monica)** ·
**Prophets & Old Testament (8)** · **Women of Faith (7)** · **Christmas (3)**. ~49 more.
