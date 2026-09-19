"""Every engine switch, in one place.

Each key that the pet engine, the pet base module, the pet landmarks and the Lifelike
renderer read from the environment is registered here with its default and one line on
what it does. The code reads through raw(name), which returns exactly what
os.environ.get(name, default) returned before this registry existed, so moving a key
here changed no pixel (gated byte-identical on 2026-09-18). The parsing (float, int,
on/off) stays at the point of use, where its meaning is.

Conventions the readers use, for anyone setting a value:
  float / int   ""  means the default (the reader does `float(v or default)`).
  on/off        most readers treat "0", "false", "off" and "" as off and anything else
                as on; a few (marked "on = 1/true/on/yes") accept only those four.
  path / text   used as given.

ops/engine.env is the box-wide copy of the values production runs; a key listed there
must be registered here (tests/test_settings.py), and no engine module may read the
environment directly.
"""
from __future__ import annotations

import os
import tempfile
from typing import Callable, Dict, List, Optional, Union

_Default = Union[None, str, Callable[[], str]]


class Key:
    __slots__ = ("name", "kind", "default", "doc")

    def __init__(self, name: str, kind: str, default: _Default, doc: str):
        self.name, self.kind, self.default, self.doc = name, kind, default, doc


REGISTRY: Dict[str, Key] = {}


def _k(name: str, kind: str, default: _Default, doc: str) -> None:
    REGISTRY[name] = Key(name, kind, default, doc)


# ---- pet engine (app/pet_v2/engine.py) ----------------------------------------------
_k("PET_ENGINE", "text", "",
   "Which pet engine serves PawsInWords: 'v2' (streamlines, collision-aware) or anything else for the row-based original.")
_k("PET_V2_ITERS", "int", "4",
   "Rounds of regrow-rasterise-compare-correct. 4 measured best; the loop also stops early when the score turns down.")
_k("PET_V2_MAX_RENDER_PX", "int", "2400",
   "Cap on the render height; a taller print is rendered at this height and upscaled at the end.")
_k("PET_V2_SHARP_PRINT", "on/off", "1",
   "Paid files: draw the type again at print scale and finish there, instead of enlarging the working render. 0 = the old enlargement.")
_k("PET_V2_CACHE_ENTRIES", "int", "4",
   "Finished renders kept in memory so previews, loupes and backdrop swaps reuse one render.")
_k("PET_V2_HAIR_REVEAL", "float", "0.7",
   "People only: how much of the photograph shows through outside the face ellipse (hair, clothing). 0 = none.")
_k("PET_V2_LANDMARKS", "on/off", "1",
   "Run the pet pose model for eye and nose coordinates; off falls back to the brightness heuristic.")
_k("PET_V2_VERBOSE", "on/off", "",
   "Print the per-render metrics and stage notes to the container log.")
_k("PET_V2_KEEP_FIELDS", "on/off", "",
   "Keep the intermediate fields on the render state for tools that inspect them (memory cost).")
_k("PET_V2_RESEED_DEBUG", "text", None,
   "Debug: print the streamline reseed decisions.")
_k("GOP_MAX_OVERLAP", "float", "0.08",
   "Collision policy: the most a placed word may overlap existing ink, as a fraction of its area.")
_k("GOP_SCALE", "float", "1",
   "Debug/QA: multiply the render resolution; 1 in production.")
_k("GOP_LANDMARKS", "text", "",
   "Debug/QA: supply eye and nose coordinates directly ('x,y;x,y;x,y'), bypassing detection.")
_k("GOP_DEBUG_PT", "text", None,
   "Debug: 'x,y' prints every compositing stage's value at that pixel.")

# ---- pet base module (app/pet_proto.py) --------------------------------------------
_k("PET_MATTE_MODEL", "text", "isnet",
   "Foreground matte model for pets: 'isnet' (default, best on fur) or 'u2net'.")
_k("PET_MATTE_URL", "path", "",
   "Override the matte model's download URL.")
_k("PET_MATTE_DIR", "path", lambda: tempfile.gettempdir(),
   "Where the matte model file is kept; default the system temp dir.")
_k("PET_MATTE_FILL", "float", "0.35",
   "Alpha threshold below which the matte is treated as background when solidifying the silhouette.")
_k("PET_HOLE_MAX", "float", "0.012",
   "Largest enclosed hole (fraction of the mask) that is filled as matte error rather than kept as a real gap.")
_k("PET_HOLE_DEBUG", "on/off", "",
   "Debug: report the holes the matte cleanup considered.")
_k("PET_TORSO_FILL", "float", "0",
   "Fraction of the height below which a mask column counts as reaching the bottom, for torso fill. 0 = off.")
_k("PET_TRACK", "float", "1.0",
   "Original engine: inter-phrase spacing multiplier on a row.")
_k("PET_ROW_GAP", "float", "1.12",
   "Original engine: multiplier on the row step; 1.0 = rows touching.")
_k("PET_LANDMARKS", "on/off", "0",
   "Original engine: run the pose model for feature coordinates.")
_k("PET_FEATURE_PROTECT", "float", "0.7",
   "Strength of the feature field (eyes, nose) that protects them from the de-whisker and confines the photo blend.")
_k("PET_FEATURE_SCOPE", "float", "0.06",
   "Blur radius, as a fraction of width, of the neighbourhood the feature field compares against.")
_k("PET_FEATURE_FILL", "float", "0",
   "Fill the interior of a feature rim so eye and nose interiors do not fall to the coarse tier. 0 = off.")
_k("PET_DEWHISKER", "float", "0.85",
   "How strongly thin bright lines (whiskers) are dissolved into the coat before rendering.")
_k("PET_LOCAL_CONTRAST", "float", "0.4",
   "Blend of local-contrast enhancement so a dark muzzle keeps structure. 0 = off.")
_k("PET_TYPE_SCALE", "float", "0.42",
   "Fallback typography size when the request carries none: Small 0.30, Medium 0.42, Large 0.56.")
_k("PET_TIER_COARSE", "float", "64",
   "Size of the coarsest word tier; lower shrinks only the largest words.")
_k("PET_WORD_TIERS", "on = 1/true/on/yes", "",
   "Give each size tier its own word list (name large, adjectives as texture) instead of the full stream.")
_k("PET_TIER_GAMMA", "float", "1.0",
   "Reshapes the tier distribution: <1 toward fine, >1 toward coarse.")
_k("PET_DRAPE", "float", "68",
   "Original engine: how far rows warp vertically to ride the form, in pixels at reference scale.")
_k("PET_DRAPE_SMOOTH", "float", "0.045",
   "Smoothing of the drape field as a fraction of width, so rows follow broad form not local edges.")
_k("PET_DRAPE_DAMP", "float", "0.92",
   "How much the drape is damped on features so eyes and nose stay crisp.")
_k("PET_DRAPE_DETAIL_DAMP", "float", "1.9",
   "Gain on the detail edges that extend drape protection into feature interiors.")
_k("PET_HERO_CENTRE", "float", "0",
   "Keep the largest words within this radius of the head (normalised to the mask box). 0 = off.")
_k("PET_SHADOW_LIFT", "float", "1.0",
   "Floor the word colour to a dim warm ink in shadowed fur so a dark chest is not an empty void.")
_k("PET_NEGATIVE_SPACE", "float", "0",
   "Thin the type toward the ground in the deepest shadows so they read as depth. 0 = off.")
_k("PET_SUBJECT_BASE", "float", "0",
   "Show the dimmed photograph behind the words inside the silhouette (0..1); the flat ground stays behind the subject.")
_k("PET_SUBJECT_DIM", "float", "0.45",
   "How much the photograph behind the words is dimmed when PET_SUBJECT_BASE is on.")
_k("PET_DUMP_FIELDS", "path", "",
   "Debug: directory to write the alpha, ink and base fields of a render.")
_k("PET_EDGE_INK", "float", "0.62",
   "Darkness of the edge ink drawn along the silhouette.")
_k("PET_PHOTO", "float", "0.45",
   "Strength of the photographic blend on the features (eyes, nose), which anchors the piece.")
_k("PET_PHOTO_FUR", "float", "0.10",
   "Residual photographic blend on the coat; keep low so the coat stays words.")
_k("PET_TONAL", "float", "1.0",
   "Tonal depth inside the subject: deepen shadows, lift highlights. 0 = off.")
_k("PET_SHARPEN", "float", "0.5",
   "Gentle unsharp mask inside the subject so eyes and nose snap.")
_k("PET_EYE_POP", "float", "0.6",
   "Restore the catchlight and iris rim on the feature field. 0 = off.")
_k("PET_VIBRANCE", "float", "0.35",
   "Vibrance: boost less-saturated colours more so warm fur glows without going garish.")
_k("PET_VIGNETTE", "float", "0.32",
   "Studio vignette toward the canvas corners, ground included.")
_k("PET_EDGE_TIGHTEN", "float", "0.18",
   "Fade from ground to full render over a short band inside the silhouette, removing the matte halo.")
_k("PET_MAX_RENDER_PX", "int", "2400",
   "Original engine: cap on the render height before upscaling to the print size.")

# ---- pet landmarks (app/pet_landmarks.py) ------------------------------------------
_k("PET_LM_DET_MODEL", "path", "models/yolox_m.onnx",
   "Path of the animal detector model.")
_k("PET_LM_POSE_MODEL", "path", "models/rtmpose_ap10k_m.onnx",
   "Path of the animal pose model (prefetched at image build; never downloaded at run time).")
_k("PET_LM_DET_URL", "path", "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_m.onnx",
   "Download URL of the animal detector model when the file is missing.")
_k("PET_LM_MIN_SCORE", "float", "0.7",
   "Minimum detector score for a subject; real animals scored 1.37 to 1.84 on the test set.")
_k("PET_LM_SECOND_MIN", "float", "0.25",
   "A second animal counts as a subject when its box is at least this fraction of the largest one's area.")
_k("PET_LM_USE_NECK", "on/off", "1",
   "Use the neck keypoint when placing the head region.")
_k("PET_LM_RETRY_AFTER", "float", "300",
   "Seconds to wait before retrying the pose models after a failed load.")
_k("PET_LM_DEBUG", "on/off", "",
   "Debug: log why a detection was or was not used.")

# ---- Lifelike renderer (app/pipeline/displacement.py) ------------------------------
_k("TYPO_FONT", "path", "",
   "The typeface both engines draw with: a file under typography_engine/fonts/ or a path. Unset = the box font (DejaVu Sans Bold).")
_k("TYPO_FONT_WEIGHT", "float", "",
   "Weight axis for a variable typeface (e.g. 700); a static face ignores it.")
_k("TYPO_SHARP_PRINT", "on/off", "1",
   "Paid files: draw the type again at print scale and rebuild the composite over it, instead of enlarging the working render. 0 = the old enlargement.")
_k("TYPO_FLORAL_DIR", "path", "",
   "Directory of the watercolour floral frames; empty uses the bundled art.")
_k("TYPO_MATTE_FLOOR", "float", "0.12",
   "Matte alpha below which the transition band goes to 0, so gaps between hair strands carry no words.")
_k("TYPO_MATTE_GAMMA", "float", "1.5",
   "Gamma on the matte's transition band after the floor.")
_k("TYPO_DARKLENS", "on/off", "1",
   "Kill switch for the sunglasses treatment the customer can request; off forces it off.")
_k("TYPO_EYE_DEBUG", "on/off", "",
   "Debug: report which gate each face took in the eye synthesis.")
_k("TYPO_MISFIT_GAP", "float", "0.02",
   "Suppress a face whose iris sits above its brow by more than this fraction of face height (a misfitted mesh).")
_k("TYPO_IRIS_MIN_PX", "float", "8.0",
   "Minimum iris radius in pixels for a face to get real eyes; smaller faces get the fallback.")
_k("TYPO_EYE_OPEN_EAR", "float", "",
   "Eye-aspect-ratio below which an eye counts as shut; empty uses the code constant.")
_k("TYPO_LENS_DARK_MAX", "float", "115",
   "Sunglasses: an eye region darker than this mean counts as a tinted lens.")
_k("TYPO_LENS_DARK_MED", "float", "105",
   "Sunglasses: same test on the median.")
_k("TYPO_LENS_REALEYE", "float", "1.6",
   "Sunglasses: a sclera ring this many times brighter than the pupil proves a bare eye, which is never blacked out.")
_k("TYPO_LENS_DARKRATIO", "float", "0.62",
   "Sunglasses: fraction of the eye box that must be dark.")
_k("TYPO_DRAPE", "float", "64",
   "How far rows warp vertically to ride the form, in pixels at reference scale.")
_k("TYPO_FLOW_JITTER", "float", "3.0",
   "Per-row horizontal offset in units of the row font size, breaking the wallpaper lattice. 0 = aligned.")
_k("TYPO_PER_FACE", "on/off", "1",
   "Size the type per face in a group photo rather than one global scale.")
_k("TYPO_FACE_DETAIL", "float", "0.95",
   "How strongly the face is forced to the fine tier.")
_k("TYPO_GRADUATE_BODY", "on/off", "1",
   "Step the type down continuously below the chin and above the face so body and hair are not giant words.")
_k("TYPO_GROUP_UNIFORM", "on/off", "1",
   "In a group, use one type scale for all faces of similar size.")
_k("TYPO_NECK_FINE", "float", "1.0",
   "Scale of the fine-type boost applied to the neck.")
_k("TYPO_HILIGHT_FINE", "float", "0.30",
   "Push the brightest 30% of the face to finer type so highlights breathe.")
_k("TYPO_CREASE", "float", "0.22",
   "Strength of the fine high-pass that renders creases (smile lines, crow's feet) as delicate darker type.")
_k("TYPO_DUMP_STAGES", "path", "",
   "Debug: directory to write a frame after each finishing stage.")
_k("TYPO_EDGE_FALLOFF", "float", "0.45",
   "Fade the shadow-side edge band into space, gated by the face's own lighting asymmetry.")
_k("TYPO_HILIGHT_WASH", "float", "0.5",
   "Wash of light into the brightest skin. 0 = off.")
_k("TYPO_SHADOW_LIFT", "float", "0.18",
   "Lift of the deepest shadows on the subject. 0 = off.")
_k("TYPO_EYE_BLOB", "on = 1/true/on/yes", "0",
   "Restore the legacy dark disc drawn where an eye could not be synthesised.")
_k("TYPO_LENS_SIZE", "float", "1.0",
   "Scale of the lens ellipses drawn for sunglasses.")
_k("TYPO_TEETH_DARK", "float", "60.0",
   "Open-mouth gate: the mouth interior must be darker than this.")
_k("TYPO_TEETH_BRIGHT", "float", "205.0",
   "Open-mouth gate: teeth must be brighter than this.")
_k("TYPO_TEETH_DEBUG", "on = 1/true/on/yes", "",
   "Debug: report the open-mouth decision per face.")
_k("TYPO_PAPER_FACE", "float", "0.30",
   "Paper grounds: ink-density floor on the face.")
_k("TYPO_PAPER_HAIR", "float", "0.35",
   "Paper grounds: ink-density floor on the hair so light hair renders as grey words instead of vanishing.")
_k("TYPO_DUMP_FIELDS", "path", "",
   "Debug: directory to write the mask, alpha, base, ink and size fields of a render.")
_k("TYPO_INK_SAT", "float", "1.02",
   "Saturation multiplier on the photo colours the words take.")
_k("TYPO_INK_LIFT", "float", "1.14",
   "Multiplier on the words' brightness, compensating for the dark ground showing between glyphs.")
_k("TYPO_INK_LIFT_ADD", "float", "14",
   "Offset added to the words' brightness after the multiplier.")
_k("TYPO_SUBJECT_BASE", "float", "0",
   "Show the dimmed photograph behind the words inside the silhouette (0..1).")
_k("TYPO_SUBJECT_DIM", "float", "0.45",
   "How much the photograph behind the words is dimmed when TYPO_SUBJECT_BASE is on.")
_k("TYPO_POLARITY", "on = 1/true/on/yes", "1",
   "Shadow model: carry tone in the letter colour across the full range, dense near-black type in shadow.")
_k("TYPO_POLARITY_GAMMA", "float", "1.35",
   "Shadow model: >1 drives deep shadow harder to black.")
_k("TYPO_POLARITY_FLOOR", "float", "0.18",
   "Shadow model: how black the gaps between glyphs get in shadow; 0 = true black.")
_k("TYPO_POLARITY_INK_FLOOR", "float", "40",
   "Shadow model: the letters' own black point, so eyes in shadow do not render as holes.")
_k("TYPO_POLARITY_PEDESTAL", "float", "8",
   "Shadow model: pedestal on the hue division so a pixel with no light comes out neutral, not blue. 0 = old math.")
_k("TYPO_EYE_PLAIN", "on = 1/true/on/yes", "",
   "Skip the whole eye synthesis (ring, sclera, photo paste); the eyes are words like the rest.")
_k("TYPO_IRIS_ALPHA", "float", "0",
   "Opacity of the tinted-iris paint in the eye synthesis.")
_k("TYPO_IRIS_LIFT", "float", "1.35",
   "Brightness lift on the iris paint.")
_k("TYPO_IRIS_PER_FACE", "on = 1/true/on/yes", "",
   "Sample the iris colour per face rather than once per photo.")
_k("TYPO_WORD_EYES", "on = 1/true/on/yes", "0",
   "Photo look: skip the photographic eye paste and build the eye from the synthetic layers.")
_k("TYPO_EYE_PHOTO", "float", "1.0",
   "Opacity of the photographic eye paste; 1 = the photo's own pixels.")
_k("TYPO_EYE_SHARPEN", "float", "0.6",
   "Sharpen inside the eye opening, scaled up for smaller eyes. 0 = off.")
_k("TYPO_EYE_POP", "float", "0.8",
   "Paint back the catchlight and iris rim on a soft source eye. 0 = off.")
_k("TYPO_NOIR_CONTRAST", "float", "1.08",
   "Contrast applied when the Noir ink desaturates the Lifelike render.")
_k("TYPO_DEPOSTERIZE", "float", "0.6",
   "Bring the photo's low-frequency shading back as a gentle multiply on the subject, light grounds only. 0 = off.")
_k("TYPO_BG_LIGHTEN", "float", "0",
   "Lighten the ground outside the subject toward white by this fraction.")
_k("TYPO_VIBRANCE", "float", "0.22",
   "Vibrance on the finished frame.")
_k("TYPO_SAT_CAP", "float", "150",
   "Soft cap on saturation; only saturation above it is compressed. 0 = off.")


def raw(name: str) -> Optional[str]:
    """os.environ.get(name, registered default). Unset with no default returns None."""
    k = REGISTRY[name]
    d = k.default() if callable(k.default) else k.default
    return os.environ.get(name, d)


def names() -> List[str]:
    return list(REGISTRY)


def describe() -> str:
    """One line per key: name, kind, default, description. For the ops journal and README."""
    out = []
    for k in REGISTRY.values():
        d = k.default() if callable(k.default) else k.default
        out.append(f"{k.name}  [{k.kind}, default {d!r}]  {k.doc}")
    return "\n".join(out)
