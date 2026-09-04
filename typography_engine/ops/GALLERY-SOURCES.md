# Gallery source images: ChatGPT prompts + renderer text

32 synthetic source-photo prompts (18 Typortrait/human, 14 PawsInWords/pet) for
building a showcase gallery, plus the word list to paste into each render. Every
prompt is self-contained — copy the whole fenced block into ChatGPT image
generation, nothing else needed.

## Why synthetic sources, not real customer photos

Typortrait.com deletes source photos after `TYPO_RETENTION_DAYS` (see
COMPLIANCE.md) and never promised any customer their photo would appear in
marketing. Generating stock-style sources with ChatGPT sidesteps that entirely —
no real, identifiable person, no consent question, no retention conflict. Keep
these images and their renders filed separately from `data/private/*.src`
(real customer uploads) so the two never get confused later.

## Standing instructions for every ChatGPT prompt

Paste this once at the start of the session, or prepend it to each prompt if
generating one at a time:

```
Generate a single photorealistic photograph (NOT an illustration, painting,
3D render, CGI, or stylized image) matching the description below. Requirements
for every image:
- Real-photography style: natural skin/fur texture, realistic lighting falloff,
  a genuine camera-like depth of field. No airbrushed/plastic "AI skin."
- Portrait orientation, roughly 4:5 aspect ratio.
- The subject's eyes must be OPEN and clearly visible, with a visible iris and
  a natural catchlight -- even in a candid or laughing pose, the eyes should
  not read as shut or squeezed fully closed.
- Correct, undistorted anatomy: natural hands, ears, and facial proportions.
  No extra/missing fingers, no warped features.
- Simple, uncluttered background appropriate to the lighting described --
  nothing that competes with the subject.
- No text, logos, watermarks, or borders anywhere in the image.
- Highest resolution/quality setting available.
```

Then the per-image description below is the only thing that changes between
prompts.

## After generating

1. Save each source as `h01-...jpg` .. `h18-...jpg` (human) or `p01-...jpg` ..
   `p14-...jpg` (pet), named for the category (matches the numbering below).
2. Render each through the normal customer flow (or the admin studio at
   `/admin/studio` if seeding the pre-made storefront gallery) using the word
   list given for that image.
3. Confirm eyes render cleanly (no disc) and the renderNote warning does NOT
   fire on any of these -- if it does, that source didn't meet the "eyes open"
   requirement above and should be regenerated, not used.

---

## Category → landing-page assignments

These are the `/examples/<slug>` pages this set is actually meant to build
(see the 2026-09-04 conversation: generating renders with no destination is
a waste of a render). Each page needs 4-5 solid images -- the ones below are
picked for that, not for engine-diversity coverage. Everything NOT listed
here (H01-H03, H07-H09, H12-H14, H17) is general variety, useful as
supporting/rotating images inside a page but not the reason any page exists.

| Page | Slug | Images |
|---|---|---|
| Gifts for Parents & Grandparents | `portraits-for-parents-grandparents` | H04, H05, H15, H16 |
| Couple & Anniversary Portraits | `couple-anniversary-portraits` | H10, H19, H20 |
| Family Portraits | `family-portraits` | H11, H18, H21, H22 |
| Kids & Baby Portraits | `kids-baby-portraits` | H06, H23, H24, H25 |
| Dog Portraits | `dog-portraits` | P01, P02, P03, P04, P08, P11, P13 |
| Cat Portraits | `cat-portraits` | P05, P06, P07, P12 |
| Pet Memorial Portraits | `pet-memorial-portraits` | P09, P15, P16, P17 |

## Typortrait.com — humans (25)

### H01 — light skin, dark hair, studio softbox, solo
```
A woman in her late 20s with light skin and dark shoulder-length hair, softly
lit by a large studio softbox from the front-left, plain warm-gray backdrop,
gentle closed-mouth smile, facing the camera directly.
```
Words: `EMMA, GRACE, RADIANT, WARMTH, GENTLE, KIND, SHINE, TOGETHER`

### H02 — medium skin, blonde wavy hair, natural window light, 3/4 view
```
A woman in her mid-30s with medium olive skin and long wavy blonde hair,
lit by soft natural window light from the side, plain cream wall behind her,
three-quarter angle toward the camera, relaxed genuine smile.
```
Words: `SOPHIA, LIGHT, GRACE, WARM, JOY, GENTLE, FREE, HOME`

### H03 — deep skin, short natural hair, dramatic sidelight, solo
```
A man in his early 30s with deep brown skin and short natural textured hair,
dramatic hard sidelight from the right creating strong shadow on one half of
his face, plain dark charcoal background, calm confident direct gaze.
```
Words: `MARCUS, STRENGTH, BOLD, STEADY, RISE, FOCUS, PROUD, POWER`

### H04 — senior woman, white hair, glasses, studio soft light
```
A woman in her early 70s with white hair in a soft bob, wearing thin
wire-frame glasses, light skin, softly lit by an even studio light, plain
light-gray backdrop, warm gentle smile with visible eyes behind the glasses.
```
Words: `ELEANOR, WISDOM, GRACE, GENTLE, TIMELESS, LOVE, PEACE, HOME`

### H05 — senior man, white hair, golden-hour outdoor
```
A man in his mid-70s with white hair and light skin, outdoors at golden hour,
warm low sunlight from behind and to the side creating a soft rim light on
his hair, blurred green foliage background, thoughtful, calm expression,
looking just off-camera with eyes clearly open.
```
Words: `WILLIAM, LEGACY, STRENGTH, WISDOM, STEADY, HONOR, TIME, GRACE`

### H06 — young child, bright natural light, laughing
```
A 5-year-old child with light skin and light brown curly hair, outdoors in
bright even natural daylight, mid-laugh with a big open smile, eyes open and
bright (not squeezed shut), plain soft green blurred background.
```
Words: `LUCAS, JOY, WONDER, BRIGHT, GIGGLE, FREE, LIGHT, PLAY`

### H07 — middle-aged man, dark skin, glasses, professional headshot
```
A man in his mid-40s with dark skin, short black hair, and rectangular
black-framed glasses, professional corporate headshot lighting (even, soft,
front-on), plain neutral blue-gray backdrop, confident closed-mouth smile,
direct eye contact with the camera.
```
Words: `DAVID, FOCUS, STRENGTH, STEADY, RISE, TRUST, BUILD, LEAD`

### H08 — woman, curly textured hair, medium skin, outdoor natural light
```
A woman in her late 20s with medium-brown skin and voluminous curly natural
hair, outdoors in soft overcast natural light, three-quarter profile angle,
plain blurred neutral background, soft genuine smile, eyes clearly open.
```
Words: `MAYA, VIBRANT, FREE, RADIANT, BOLD, JOY, ROOTS, SHINE`

### H09 — bald/short hair man, light skin, low-key dark lighting
```
A man in his late 30s with a shaved head and light skin, low-key dramatic
lighting -- a single soft light from above-front, the rest of the frame
falling to near-black, plain very dark background, serious calm expression,
eyes clearly open and lit.
```
Words: `JAMES, STEADY, DEPTH, STRENGTH, QUIET, RESOLVE, STILL, FOCUS`

### H10 — couple, varied skin tones, studio soft light
```
A couple in their 30s -- one with light skin and short brown hair, one with
deep brown skin and short black hair -- cheek to cheek, both facing the
camera, softly lit by an even studio light, plain warm-white backdrop, both
smiling genuinely with eyes open.
```
Words: `TOGETHER, LOVE, US, FOREVER, HOME, JOURNEY, TRUST, GROW`

### H11 — small family, outdoor natural light
```
A parent in their mid-30s with medium skin tone and two young children (ages
about 4 and 7), all with light-to-medium skin, outdoors in soft natural
afternoon light, plain blurred park-green background, all three facing the
camera with genuine smiles and eyes open.
```
Words: `FAMILY, TOGETHER, HOME, LOVE, JOY, GROW, ALWAYS, US`

### H12 — woman with hat, medium skin, outdoor soft light
```
A woman in her mid-20s with medium skin tone and a wide-brim straw sunhat,
outdoors in soft diffused daylight (overcast), plain blurred neutral outdoor
background, relaxed smile, eyes clearly visible under the hat brim.
```
Words: `OLIVIA, FREE, SUNLIT, EASY, WANDER, WARM, BRIGHT, CALM`

### H13 — man with full beard, light skin, sidelight
```
A man in his early 40s with light skin and a full well-groomed dark beard,
soft directional sidelight from the left, plain neutral gray background,
neutral/serious expression, eyes clearly open and lit.
```
Words: `HENRY, STEADY, STRONG, ROOTED, FIRM, QUIET, RESOLVE, TRUE`

### H14 — young woman, deep skin, glasses, bright studio light
```
A woman in her early 20s with deep brown skin and round tortoiseshell
glasses, bright even studio lighting, plain white backdrop, big genuine
open smile, eyes clearly visible behind the glasses.
```
Words: `ZOE, BRIGHT, BOLD, RADIANT, JOY, RISE, SHINE, FREE`

### H15 — senior woman, deep skin, warm indoor window light
```
A woman in her late 60s with deep brown skin and short gray natural hair,
warm soft indoor window light from the side, plain soft-beige background,
warm gentle smile, eyes clearly open.
```
Words: `RUTH, GRACE, WISDOM, WARM, GENTLE, HONOR, TIMELESS, LOVE`

### H16 — man, gray receding hair, warm indoor lamp light
```
A man in his mid-50s with light skin and short graying hair receding at the
temples, warm low indoor lamp lighting from one side, plain dark-wood
background suggestion (softly blurred), three-quarter angle, thoughtful calm
expression, eyes clearly open.
```
Words: `ROBERT, STEADY, DEPTH, WISE, CALM, ANCHOR, STRONG, TRUE`

### H17 — woman, auburn hair, backlit golden hour outdoor
```
A woman in her late 20s with light skin and long auburn/red hair, outdoors
at golden hour with warm backlight creating a soft rim glow through her
hair, front of her face still evenly lit and clearly visible, plain blurred
warm outdoor background, soft natural smile, eyes clearly open.
```
Words: `IVY, GOLDEN, FREE, RADIANT, WARM, GLOW, GRACE, WILD`

### H18 — group of 3 friends, mixed ages/skin tones, outdoor
```
Three adult friends in their 30s, mixed skin tones (one light, one medium,
one deep) and mixed hair colors, standing close together outdoors in soft
natural daylight, plain blurred neutral outdoor background, all facing the
camera, genuine smiles, all eyes clearly open.
```
Words: `FRIENDS, TOGETHER, JOY, US, ALWAYS, HOME, LAUGH, TRUE`

### H19 — young couple, mixed skin tones, outdoor golden hour
```
A couple in their late 20s -- one with light skin and dark hair, one with
medium-brown skin and curly black hair -- standing close together outdoors
at golden hour, foreheads touching, soft warm backlight, plain blurred
golden-field background, both smiling gently with eyes clearly open.
```
Words: `TOGETHER, FOREVER, US, JOURNEY, LOVE, HOME, GROW, ALWAYS`

### H20 — senior couple, anniversary, warm indoor light
```
A couple in their late 60s, both with light skin and gray/white hair,
cheek to cheek, warm soft indoor lamp lighting, plain warm-beige
background, both smiling warmly with eyes clearly open.
```
Words: `ANNIVERSARY, TIMELESS, US, FOREVER, GRACE, JOURNEY, LOVE, HOME`

### H21 — family of 4, outdoor natural light
```
Two parents (one light skin, one medium skin tone) with two children
(ages about 6 and 9), all facing the camera, outdoors in soft natural
daylight, plain blurred park-green background, genuine smiles, all eyes
clearly open.
```
Words: `FAMILY, TOGETHER, HOME, GROW, LOVE, ALWAYS, US, ROOTS`

### H22 — multi-generational family, indoor warm light
```
Three generations -- a grandparent (70s, gray hair), a parent (40s), and
a child (8) -- light-to-medium skin tones, standing together indoors,
warm soft window light, plain neutral indoor background, all facing the
camera with genuine smiles, eyes clearly open.
```
Words: `LEGACY, FAMILY, ROOTS, LOVE, GENERATIONS, HOME, ALWAYS, GRACE`

### H23 — baby, soft studio light, close-up
```
A close-up of a 1-year-old baby's face, light skin, soft wispy hair,
softly lit by an even studio light, plain warm-cream background, calm
curious expression, eyes wide open and bright.
```
Words: `NOAH, WONDER, PRECIOUS, NEW, JOY, PURE, BEGIN, LOVE`

### H24 — two young siblings, outdoor bright light
```
Two siblings (ages about 4 and 7), light-to-medium skin tone, standing
close together outdoors in bright natural daylight, plain blurred green
background, both laughing genuinely, eyes clearly open and bright.
```
Words: `SIBLINGS, JOY, TOGETHER, GIGGLE, PLAY, FOREVER, BOND, LIGHT`

### H25 — toddler, big grin, outdoors
```
A 3-year-old child with medium skin tone and short curly hair, outdoors
in soft natural daylight, big genuine open-mouthed grin, plain blurred
neutral outdoor background, eyes clearly open and bright.
```
Words: `SOPHIE, JOY, WONDER, BRIGHT, GIGGLE, PURE, LIGHT, PLAY`

---

## PawsInWords.com — pets (17)

### P01 — Golden Retriever, outdoor natural light, close-up
```
A close-up of a Golden Retriever's face, light golden short-to-medium coat,
outdoors in bright natural daylight, plain blurred green background, mouth
slightly open in a happy pant, eyes wide open with a clear catchlight.
```
Words: `BUDDY, LOYAL, JOY, GOLDEN, FAITHFUL, FRIEND, HAPPY, HOME`

### P02 — Black Labrador, studio soft light, close-up
```
A close-up of a Black Labrador's face, solid black short coat, softly lit by
an even studio light with a clear catchlight visible in both eyes, plain
warm-gray backdrop, alert attentive expression, mouth closed.
```
Words: `SHADOW, LOYAL, STRONG, STEADY, TRUE, GUARD, FAITHFUL, BOLD`

### P03 — Curly cream Poodle, indoor soft light, close-up
```
A close-up of a cream-colored curly-coated Poodle's face, soft indoor
window light, plain light-beige background, bright alert eyes with a clear
catchlight, gentle content expression.
```
Words: `COCO, SWEET, CURLY, GENTLE, JOY, PRECIOUS, CHARM, LOVE`

### P04 — Border Collie, outdoor natural light, full body
```
A Border Collie standing full-body in a grassy field, black-and-white
patterned coat, bright natural outdoor daylight, alert ears up, facing the
camera, eyes clearly open with visible catchlight, plain blurred green
background.
```
Words: `SCOUT, ALERT, SWIFT, LOYAL, SMART, ENERGY, FOCUS, RUN`

### P05 — Orange tabby cat, indoor window light, close-up
```
A close-up of an orange tabby cat's face, soft natural window light from the
side, plain neutral cream background, eyes wide open with a bright clear
catchlight, calm curious expression.
```
Words: `TIGER, CURIOUS, WARM, PLAYFUL, SUNNY, GENTLE, CHARM, HOME`

### P06 — Black cat, dramatic sidelight, close-up
```
A close-up of a solid black cat's face, dramatic soft sidelight with a
strong, clearly visible catchlight in each eye so the eyes read bright
against the dark fur, plain dark-charcoal background, calm direct gaze.
```
Words: `MIDNIGHT, MYSTERY, SLEEK, CALM, BOLD, QUIET, GRACE, WISE`

### P07 — Gray/silver cat, soft studio light, close-up
```
A close-up of a silver-gray shorthair cat's face (Russian Blue type),
evenly lit by a soft studio light, plain light-gray backdrop, bright green
eyes wide open with a clear catchlight, serene expression.
```
Words: `SMOKEY, SERENE, GENTLE, ELEGANT, CALM, GRACE, SOFT, TRUE`

### P08 — Fluffy white dog, bright outdoor light, close-up
```
A close-up of a fluffy white-coated dog's face (Samoyed type), bright
natural outdoor daylight, plain soft-blue sky background, mouth open in a
happy "smile," eyes clearly open with visible catchlight.
```
Words: `SNOWY, JOY, BRIGHT, FLUFFY, HAPPY, LIGHT, PLAYFUL, PURE`

### P09 — Senior gray-muzzle dog, warm indoor lamp light
```
A close-up of an older dog's face with a graying muzzle, medium-brown
short coat, warm soft indoor lamp lighting, plain warm-brown blurred
background, gentle calm expression, eyes clearly open and soft.
```
Words: `DUKE, GENTLE, FAITHFUL, STEADY, FOREVER, LOYAL, WARM, TRUE`

### P10 — Rabbit, soft studio light, close-up
```
A close-up of a light-brown rabbit's face, ears upright, softly lit by an
even studio light, plain soft-cream background, eyes clearly open and
bright, calm alert expression.
```
Words: `CLOVER, GENTLE, SWEET, SOFT, CURIOUS, CHARM, QUIET, JOY`

### P11 — Small dog (Corgi), outdoor golden-hour light, full body
```
A Corgi standing full-body outdoors at golden hour, tan-and-white coat,
warm low sunlight, plain blurred golden-field background, ears up, facing
the camera, eyes clearly open with visible catchlight, mouth open happily.
```
Words: `BISCUIT, CHEERFUL, GOLDEN, PLAYFUL, JOY, SPUNKY, WARM, FUN`

### P12 — Calico cat, natural window light, close-up
```
A close-up of a calico cat's face (patches of orange, black, and white),
soft natural window light, plain neutral-cream background, sitting pose,
eyes clearly open and bright with a visible catchlight.
```
Words: `PATCHES, UNIQUE, SWEET, GENTLE, CHARM, BRIGHT, PLAYFUL, JOY`

### P13 — German Shepherd, dramatic outdoor sidelight, full body
```
A German Shepherd standing full-body outdoors, black-and-tan coat,
dramatic natural sidelight from the low sun, plain blurred outdoor
background, alert upright ears, direct gaze at the camera, eyes clearly
open with visible catchlight.
```
Words: `TITAN, LOYAL, STRONG, GUARD, NOBLE, STEADY, BRAVE, TRUE`

### P14 — Parrot/cockatiel, studio soft light, close-up
```
A close-up of a cockatiel perched on a plain wooden dowel, soft even studio
lighting, plain light-gray background, head turned toward the camera, eye
clearly open and bright with a visible catchlight.
```
Words: `SKY, BRIGHT, CHEERFUL, FREE, VIBRANT, JOY, LIGHT, SING`

### P15 — senior gray-muzzle cat, warm indoor light
```
A close-up of an older cat's face with visible gray/white fur around the
muzzle, warm soft indoor lamp lighting, plain warm-brown blurred
background, calm gentle expression, eyes clearly open and soft with a
visible catchlight.
```
Words: `WHISKERS, GENTLE, FAITHFUL, FOREVER, WARM, LOYAL, PEACE, TRUE`

### P16 — dog with graying face, resting pose, golden light
```
A close-up of a dog's face with a visibly graying muzzle and around the
eyes, medium-brown coat, resting calmly, warm soft golden-hour light,
plain blurred warm background, gentle peaceful expression, eyes clearly
open and soft with a visible catchlight.
```
Words: `BEAR, GENTLE, FAITHFUL, PEACE, FOREVER, LOYAL, WARM, REST`

### P17 — older cat resting in a sunbeam, warm gentle mood
```
A cat with a light tabby coat and some visible gray around the face,
resting in a warm beam of soft window light, plain neutral background,
calm serene expression, eyes clearly open and soft with a visible
catchlight.
```
Words: `SUNNY, PEACE, GENTLE, WARM, FOREVER, SOFT, LOYAL, TRUE`
