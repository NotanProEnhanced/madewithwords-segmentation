# Floral background frames (lovedinwords)

Curated watercolor frames composited OUTSIDE the subject silhouette, behind the
Paper sculpt (dark ink on warm ivory). Selecting a floral in the studio's
Background axis forces the Paper ground automatically.

Drop the four PNGs here with these exact names (the pipeline loads by key):

| Key           | File               | Concept              | Placement            |
|---------------|--------------------|----------------------|----------------------|
| wildflowers   | `wildflowers.png`  | Soft Wildflowers     | side columns         |
| roses         | `roses.png`        | Soft Roses           | bottom border        |
| eucalyptus    | `eucalyptus.png`   | Eucalyptus & Lily    | top arch             |
| line          | `line.png`         | Quiet Line           | line-art corners     |

## Specs
- Aspect: EXACTLY 4:5 portrait (width:height = 4:5) — must be exact (no stretch/crop).
- Size: 4800x6000 recommended (300 PPI on 16x20); 3600x4500 minimum.
- Format: PNG; cream ground ~#f4efe8 to match the Paper ivory; center kept clear.

## Notes
- This directory is bind-mounted, so swapping art needs NO rebuild — just replace
  the file and the next render (cache is per-process; restart the container to force
  a reload, or the cache refreshes on the next cold worker).
- A missing/broken file degrades gracefully to a plain cream mat (never crashes).
- Override the directory with env TYPO_FLORAL_DIR if the art lives elsewhere.
