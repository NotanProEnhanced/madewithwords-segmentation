# The test set

    sources   /root/typortrait-testset/src/*.jpg
    words     /root/typortrait-testset/words.txt
    results   /root/typortrait-testset/out/<commit>/

Deliberately outside every tree, so a revert cannot take it.

The engine is deterministic — `rng` is seeded with a fixed value — so the same photograph
and the same words produce a byte-identical render. That is what makes `compare-testset.sh`
exact: two runs either are the same render or they are not, and there is no judgment in it.
It only holds if the photographs and the words never change. Add images; never edit or
replace one.

    ./ops/render-testset.sh              render every image at the current commit
    ./ops/render-testset.sh 05 07        just those
    PET=1 ./ops/render-testset.sh        through the pet engine instead
    ./ops/compare-testset.sh A B         which images moved between two runs

## What each image is for

The first ten were generated to vary the SUBJECT. The original prompts were not kept, only
what each was chosen to exercise:

| # | image | exercises |
|---|-------|-----------|
| 01 | hat | headwear the mesh has no landmarks for |
| 02 | glasses | thin frames over the eye landmarks |
| 03 | sunglasses | eyes the model cannot see at all |
| 04 | beard | texture where the chin landmarks say skin |
| 05 | couple | **the only multi-subject image**, and the only landscape source |
| 06 | sidelight | half the face in shadow |
| 07 | dark-on-dark | dark clothing against a dark ground — where the matte gives up |
| 08 | white-hair | light hair against a light ground |
| 09 | dark-skin | tone mapping across a wide skin range |
| 10 | smile | teeth and open-mouth shadow |

## Known gaps

**Aspect.** Nine of the ten sources are 1122x1402 — exactly the 4:5 the harness renders at.
So a framing change produces no visible difference on nine images out of ten, and the set is
nearly blind to that whole class of bug. Real uploads are phone photographs: 3:4, 9:16,
occasionally square. Almost none are 4:5.

**Animals.** The pet engine (`pet_proto.py`) is landmark-free and serves PawsInWords, and
until images 11-15 there was not one animal in the set. Every claim made about that engine
was inferred from photographs of people. `PET_HOLE_MAX` in particular is calibrated on a
single two-person photograph; a dog with a real gap between its legs is exactly the case
that could show the threshold to be wrong.

**Multi-subject.** One image. A fix validated on `05-couple` is validated on a sample of one.

**Backgrounds.** Every one of the first ten is a clean studio backdrop. Not one has another
person behind the subject, and the matte model on the memorial brands
(`TYPO_MATTE_MODEL=1` -> RVM) answers exactly one question: *which pixels are a person?* A
crowd is people, so it says yes, and `_clean_mask` keeps every blob down to 15% of the
largest. A LovedInWords portrait shipped with rows of spectators cut out above the
subject's head, and the set could not have caught it: there is nothing behind anyone in it.
Sources 16 and 17 exist to fix that, and they are deliberately a PAIR -- see below.


## Prompts for the pet sources (11-15)

Paste each into an image generator. Ask for **1122x1402 (4:5 portrait)** so they match the
rest of the set, and save as the filename given — the leading number is how
`render-testset.sh` selects an image.

**11-black-dog.jpg**
> A photorealistic studio portrait of a black Labrador, head and chest, facing the camera.
> Charcoal gray seamless backdrop, soft key light from the upper left. The dog's dark fur
> and the dark background are close in tone, with only rim light separating them. Sharp
> focus on the eyes. Natural fur texture, no collar, no props. Vertical 4:5 portrait.

*Why: the matte's hardest case — subject and ground nearly the same value. The human
equivalent, 07-dark-on-dark, is where the model already loses the torso.*

**12-white-cat.jpg**
> A photorealistic studio portrait of a long-haired white Persian cat, head and shoulders,
> facing the camera. Bright white seamless backdrop, soft even lighting. The cat's white
> fur is a similar value to the background, with fine wispy hairs at the edges of the
> silhouette. Sharp focus on the eyes. Vertical 4:5 portrait.

*Why: light fur on white is the exact case `_solidify_matte` was written for — the model
loses low-confidence edges and the body drops out, leaving a floating head.*

**13-dog-and-cat.jpg**
> A photorealistic studio portrait of a golden retriever and a tabby cat sitting side by
> side, close together, both facing the camera, heads nearly touching. Mid-gray seamless
> backdrop, soft frontal lighting. A small gap of visible background between their bodies
> below where their heads meet. Vertical 4:5 portrait.

*Why: the animal equivalent of 05-couple. Two subjects walling off a pocket of background
is what `PET_HOLE_MAX` was introduced for, and it has never been tested on animals.*

**14-dog-sitting.jpg**
> A photorealistic full-body studio portrait of a beagle sitting upright on a plain floor,
> facing the camera, front legs straight and clearly apart with visible background between
> them and beneath its chest. Light gray seamless backdrop, soft even lighting, the whole
> animal within the frame with space below it. Vertical 4:5 portrait.

*Why: **the important one.** A large, genuine, fully enclosed pocket of background that is
NOT between two subjects. If `PET_HOLE_MAX=0.012` is wrong, this is the image that shows it
— the gap between the legs will exceed the threshold and correctly stay background, or the
threshold will prove to be measuring the wrong thing.*

**15-fluffy-dog.jpg**
> A photorealistic studio portrait of a long-haired Border Collie, head and chest, facing
> the camera. Mid-tone blue-gray seamless backdrop, soft key light. Long wispy fur at the
> edges of the ears and chest breaking up the silhouette against the background. Sharp
> focus on the eyes. Vertical 4:5 portrait.

*Why: wispy edges are where the soft matte matters and where a solid threshold destroys
detail. The union of the two is the part of `_solidify_matte` nothing currently tests.*

## After adding them

    cp ~/Downloads/11-black-dog.jpg /root/typortrait-testset/src/     # and 12..15
    PET=1 ./ops/render-testset.sh

Existing runs will show the new images as "only in B", which the comparison reports rather
than hides. Re-render whichever baseline you want to compare against so both runs cover the
same set.

## Prompts for the background sources (16-17)

These two are a **pair**, and the pair is the point. A crowd that is separate from the
subject can be dropped by keeping only the blobs that contain a detected face
(`TYPO_FACE_ANCHORED_MATTE`). A crowd that TOUCHES the subject cannot -- it is one
connected component holding one face, so it is kept whole. The fix works on 16 and not on
17, and a set containing only 16 would report a solved problem.

**16-crowd-behind.jpg**
> A photorealistic candid photograph of two people, head and shoulders, smiling at the
> camera at an outdoor sports stadium. Behind them and clearly separated by several feet,
> a blurred crowd of spectators in blue and yellow. Daylight. The two subjects are sharply
> focused and their outlines do not overlap anyone behind them. Vertical 4:5 portrait.

*Why: the separable case. The crowd is its own connected region with no detected face in
it, so face-anchored filtering should remove it and leave both subjects whole.*

**17-crowd-touching.jpg**
> A photorealistic candid photograph of one person, head and shoulders, at a crowded
> outdoor event, taken from close range. Spectators stand directly behind them so that the
> subject's hair overlaps the people behind with no gap of background between them.
> Daylight, everyone in similar tones. Vertical 4:5 portrait.

*Why: the case component filtering CANNOT fix, and the reason `TYPO_FACE_ANCHORED_MATTE`
ships off. `TYPO_MASK_DEBUG` should report "nothing removed ... CONTIGUOUS" here. Whatever
is written next -- a face-scale or distance rule -- has to be judged on this one, not on 16.*

Render these with `TYPO_MATTE_MODEL=1` set, or the matte path under test does not run.
