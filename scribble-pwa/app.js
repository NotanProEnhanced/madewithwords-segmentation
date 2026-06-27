/* Scribbler — turn a photo into a continuous pen-scribble portrait, in realtime.
 * Everything runs on-device. No uploads, no network calls during rendering.
 *
 * Pipeline:
 *   1. Fit the image into a working "tone map" (luminance grid).
 *   2. Build a residual darkness field = how much ink each cell still wants.
 *   3. Repeatedly lay down scribble strokes: pick dark spots, flow strokes
 *      along tonal contours, then subtract the ink we deposited so coverage
 *      self-balances. Animate batches with requestAnimationFrame.
 */
(() => {
  'use strict';

  // ---- DOM ----
  const $ = (id) => document.getElementById(id);
  const stage = $('stage');
  const canvasWrap = $('canvasWrap');
  const canvas = $('canvas');
  const ctx = canvas.getContext('2d');
  const penCanvas = $('pen');
  const penCtx = penCanvas.getContext('2d');
  const dropzone = $('dropzone');
  const fileInput = $('fileInput');
  const statusEl = $('status');
  const statusText = $('statusText');
  const barFill = $('barFill');

  const controls = {
    density: $('density'),
    contrast: $('contrast'),
    length: $('length'),
    flow: $('flow'),
    weight: $('weight'),
    opacity: $('opacity'),
    fill: $('fill'),
    speed: $('speed'),
    ink: $('inkColor'),
    paper: $('paperColor'),
    removeBg: $('removeBg'),
    animate: $('animateChk'),
    showPen: $('showPen'),
    singleLine: $('singleLine'),
  };
  const labels = {
    density: $('densityVal'),
    contrast: $('contrastVal'),
    length: $('lengthVal'),
    flow: $('flowVal'),
    weight: $('weightVal'),
    opacity: $('opacityVal'),
    fill: $('fillVal'),
    speed: $('speedVal'),
  };
  const buttons = {
    pick: $('pickBtn'),
    redraw: $('redrawBtn'),
    pause: $('pauseBtn'),
    download: $('downloadBtn'),
    install: $('installBtn'),
  };

  // ---- State ----
  let sourceBitmap = null;     // ImageBitmap or HTMLImageElement of the upload
  let mapW = 0, mapH = 0;      // tone map dimensions
  let renderScale = 1;         // map px -> canvas px
  let tone = null;             // Float32Array luminance 0..1 (mapW*mapH)
  let residual = null;         // Float32Array remaining ink demand
  let gradAngle = null;        // Float32Array contour direction (radians)
  let fgMask = null;           // Uint8Array 1=subject, 0=background
  let initialInk = 0;          // sum of residual at start
  let strokeCount = 0;
  let rafId = 0;
  let paused = false;
  let running = false;
  let penStroke = null;        // stroke currently being revealed by the pen
  let head = { x: 0, y: 0 };   // pen-tip position in tone-map coords
  const MAX_STROKES = 75000;
  const params = {};           // engine settings captured at sketch start
  let lastFile = null;         // most recent upload, for re-matting on toggle
  let matteImg = null;         // ML subject cutout (transparent bg), or null
  let activePreset = null;     // currently selected style preset, if any
  const STORE_KEY = 'scribbler.settings.v1';
  const PRESET_KEYS = ['density', 'contrast', 'length', 'flow', 'weight', 'opacity', 'fill'];

  // On-device subject segmentation (u2net-class model, runs in WebAssembly and
  // caches itself after first load). Loaded lazily from CDN the first time the
  // user removes a background; falls back to the flood-fill matte on failure.
  const BG_REMOVAL_CDN = 'https://cdn.jsdelivr.net/npm/@imgly/background-removal@1.5.8/+esm';
  const RAND_STATE = { s: 0x2545f491 };

  // Deterministic PRNG so a re-sketch with identical settings is reproducible.
  function rnd() {
    let x = RAND_STATE.s;
    x ^= x << 13; x ^= x >>> 17; x ^= x << 5;
    RAND_STATE.s = x >>> 0;
    return RAND_STATE.s / 4294967296;
  }
  function seed(n) { RAND_STATE.s = (n >>> 0) || 1; }

  // ---- Image intake ----
  async function loadFile(file) {
    if (!file || !file.type.startsWith('image/')) return;
    lastFile = file;
    matteImg = null;
    const url = URL.createObjectURL(file);
    try {
      sourceBitmap = await loadImage(url);
    } catch (err) {
      console.error(err);
      alert('Could not read that image. Try a JPG or PNG.');
      URL.revokeObjectURL(url);
      return;
    }
    URL.revokeObjectURL(url);

    canvasWrap.hidden = false;
    dropzone.style.display = 'none';
    await prepareMatte();        // compute the ML cutout if "Remove background" is on
    buildToneMap();
    enableControls(true);
    startSketch();
  }

  // Run on-device background removal for the current file. Best-effort: any
  // failure (offline, model blocked) leaves matteImg null so we fall back.
  async function prepareMatte() {
    matteImg = null;
    if (!controls.removeBg.checked || !lastFile) return;
    busy('Isolating subject…');
    try {
      const mod = await import(BG_REMOVAL_CDN);
      const blob = await mod.removeBackground(lastFile);
      const u = URL.createObjectURL(blob);
      matteImg = await loadImage(u);
      URL.revokeObjectURL(u);
    } catch (err) {
      console.warn('On-device matte unavailable, using flood-fill fallback.', err);
      matteImg = null;
    }
  }

  function loadImage(src) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = reject;
      img.decoding = 'async';
      img.src = src;
    });
  }

  // Build luminance grid, contour directions, and the residual ink field.
  function buildToneMap() {
    const MAP_MAX = 540;                 // longest side of the tone grid
    const iw = sourceBitmap.width, ih = sourceBitmap.height;
    const fit = Math.min(MAP_MAX / iw, MAP_MAX / ih, 1.6);
    mapW = Math.max(8, Math.round(iw * fit));
    mapH = Math.max(8, Math.round(ih * fit));

    // Sample the photo into a small offscreen canvas.
    const off = document.createElement('canvas');
    off.width = mapW; off.height = mapH;
    const octx = off.getContext('2d', { willReadFrequently: true });
    octx.drawImage(sourceBitmap, 0, 0, mapW, mapH);
    const data = octx.getImageData(0, 0, mapW, mapH).data;

    tone = new Float32Array(mapW * mapH);
    for (let i = 0, p = 0; i < tone.length; i++, p += 4) {
      // Rec. 709 luma, normalized 0..1.
      tone[i] = (0.2126 * data[p] + 0.7152 * data[p + 1] + 0.0722 * data[p + 2]) / 255;
    }

    // Foreground matte: flood-fill inward from the border across smoothly
    // connected colour (the studio background), leaving the subject as 1.
    buildForegroundMask(data);

    // Lightly blur the tone map so strokes read regions, not pixel noise.
    tone = boxBlur(tone, mapW, mapH, 1);

    // Contour direction = perpendicular to the luminance gradient (Sobel).
    gradAngle = new Float32Array(mapW * mapH);
    for (let y = 0; y < mapH; y++) {
      for (let x = 0; x < mapW; x++) {
        const gx =
          -t(x - 1, y - 1) - 2 * t(x - 1, y) - t(x - 1, y + 1) +
           t(x + 1, y - 1) + 2 * t(x + 1, y) + t(x + 1, y + 1);
        const gy =
          -t(x - 1, y - 1) - 2 * t(x, y - 1) - t(x + 1, y - 1) +
           t(x - 1, y + 1) + 2 * t(x, y + 1) + t(x + 1, y + 1);
        // Flow along the iso-tone line (rotate gradient by 90deg).
        gradAngle[y * mapW + x] = Math.atan2(gx, -gy);
      }
    }

    // Compute the render scale: keep output crisp but bounded.
    const outMax = 1500;
    renderScale = clamp(outMax / Math.max(mapW, mapH), 1.4, 3);
    canvas.width = Math.round(mapW * renderScale);
    canvas.height = Math.round(mapH * renderScale);
    penCanvas.width = canvas.width;
    penCanvas.height = canvas.height;

    function t(x, y) {
      x = x < 0 ? 0 : x >= mapW ? mapW - 1 : x;
      y = y < 0 ? 0 : y >= mapH ? mapH - 1 : y;
      return tone[y * mapW + x];
    }
  }

  // Build the foreground mask. Prefer the ML cutout's alpha channel; otherwise
  // fall back to flood-fill: background = the region connected to the border
  // whose colour matches the estimated backdrop colour. Keying on backdrop
  // colour (not just local smoothness) stops the fill from leaking through a
  // soft edge into the subject, and keeps differently-coloured clothing/hair
  // that touches the frame.
  function buildForegroundMask(data) {
    const N = mapW * mapH;

    if (matteImg) {
      const tc = document.createElement('canvas');
      tc.width = mapW; tc.height = mapH;
      const tx = tc.getContext('2d');
      tx.drawImage(matteImg, 0, 0, mapW, mapH);
      const md = tx.getImageData(0, 0, mapW, mapH).data;
      fgMask = new Uint8Array(N);
      for (let i = 0; i < N; i++) fgMask[i] = md[i * 4 + 3] > 100 ? 1 : 0;
      return;
    }

    // Estimate backdrop colour from the four corners (median per channel).
    const corners = [[0, 0], [mapW - 1, 0], [0, mapH - 1], [mapW - 1, mapH - 1]]
      .map(([x, y]) => { const p = (y * mapW + x) * 4; return [data[p], data[p + 1], data[p + 2]]; });
    const med = (k) => { const v = corners.map((c) => c[k]).sort((a, b) => a - b); return (v[1] + v[2]) / 2; };
    const bgR = med(0), bgG = med(1), bgB = med(2);
    const GTOL = 150;                  // max channel-sum distance from backdrop
    const isBg = (i) => {
      const p = i * 4;
      return Math.abs(data[p] - bgR) + Math.abs(data[p + 1] - bgG) + Math.abs(data[p + 2] - bgB) < GTOL;
    };

    const bg = new Uint8Array(N);     // 1 = background
    const stack = new Int32Array(N);
    let sp = 0;
    const push = (i) => { if (!bg[i] && isBg(i)) { bg[i] = 1; stack[sp++] = i; } };

    for (let x = 0; x < mapW; x++) { push(x); push((mapH - 1) * mapW + x); }
    for (let y = 0; y < mapH; y++) { push(y * mapW); push(y * mapW + mapW - 1); }

    while (sp > 0) {
      const i = stack[--sp];
      const x = i % mapW, y = (i / mapW) | 0;
      if (x > 0) push(i - 1);
      if (x < mapW - 1) push(i + 1);
      if (y > 0) push(i - mapW);
      if (y < mapH - 1) push(i + mapW);
    }

    fgMask = new Uint8Array(N);
    for (let i = 0; i < N; i++) fgMask[i] = bg[i] ? 0 : 1;
  }

  function buildResidual() {
    const gamma = parseFloat(controls.contrast.value);
    const removeBg = controls.removeBg.checked && fgMask;
    const fill = parseFloat(controls.fill.value);   // baseline ink for the subject
    residual = new Float32Array(mapW * mapH);
    initialInk = 0;
    for (let i = 0; i < tone.length; i++) {
      if (removeBg && !fgMask[i]) { residual[i] = 0; continue; }
      // Darkness demand, contrast-shaped. Pure white asks for almost no ink.
      let d = Math.pow(1 - tone[i], gamma);
      if (d < 0.015) d = 0;
      // Fill boost: give the subject a baseline so lit skin still gets light
      // contour scribbles instead of dropping to blank paper. Gated to the
      // foreground (fgMask) so it never inks the background.
      if (fill > 0 && fgMask && fgMask[i]) d = Math.max(d, fill);
      residual[i] = d;
      initialInk += d;
    }
  }

  // ---- The scribble engine ----
  function startSketch() {
    cancelAnimationFrame(rafId);
    seed(0x9e3779b9);                    // fixed seed -> reproducible sketch
    buildResidual();
    strokeCount = 0;
    penStroke = null;
    paused = false;
    running = true;
    buttons.pause.textContent = 'Pause';

    // Capture engine settings for this sketch.
    params.opacity = parseFloat(controls.opacity.value);
    params.weight = parseFloat(controls.weight.value);
    params.flow = parseFloat(controls.flow.value);
    params.baseLen = parseInt(controls.length.value, 10);
    params.target = parseFloat(controls.density.value);

    // Paper.
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = controls.paper.value;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.scale(renderScale, renderScale);
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.lineWidth = params.weight;
    ctx.strokeStyle = controls.ink.value;
    clearPen();

    canvasWrap.hidden = false;
    dropzone.style.display = 'none';
    showStatus(true);

    // Single continuous line: build one TSP-style path and draw it unbroken.
    if (controls.singleLine.checked) {
      ctx.lineWidth = Math.max(0.9, params.weight);
      const path = buildSingleLinePath();
      if (controls.animate.checked && path.length > 1) {
        penStroke = { pts: path, i: 0 };
        rafId = requestAnimationFrame(tickLine);
      } else {
        ctx.globalAlpha = 1;
        if (path.length > 1) strokePath(path);
        finishSketch();
      }
      return;
    }

    if (controls.animate.checked) {
      rafId = requestAnimationFrame(tick);   // progressive pen reveal
    } else {
      drawInstant();                          // all at once
    }
  }

  // Reveal a single unbroken path with the pen — never lifts.
  function tickLine() {
    if (paused) return;
    if (!penStroke || penStroke.pts.length < 2) { finishSketch(); return; }
    const s = penStroke;
    let budget = penSpeed();
    ctx.globalAlpha = 1;
    while (s.i < s.pts.length - 1 && budget > 0) {
      const a = s.pts[s.i], b = s.pts[s.i + 1];
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.stroke();
      head.x = b.x; head.y = b.y;
      s.i++; budget--;
    }
    drawPen();
    setProgress(s.i / (s.pts.length - 1));
    if (s.i >= s.pts.length - 1) { finishSketch(); return; }
    rafId = requestAnimationFrame(tickLine);
  }

  // Slider 1..10 -> tone-map points drawn per animation frame.
  function penSpeed() {
    const v = parseInt(controls.speed.value, 10);
    return Math.max(2, Math.round(v * v * 4));
  }

  function coverage() {
    return 1 - sumResidual() / Math.max(1e-6, initialInk);
  }

  // Animated frame: reveal the current stroke point-by-point with the pen tip,
  // pulling the next stroke when one finishes — so it looks hand-drawn.
  function tick() {
    if (paused) return;
    let budget = penSpeed();

    while (budget > 0) {
      if (!penStroke) {
        if (coverage() >= params.target) { finishSketch(); return; }
        penStroke = nextStroke();
        if (!penStroke) { finishSketch(); return; }
        penStroke.i = 0;
      }
      const s = penStroke;
      ctx.globalAlpha = s.alpha;
      // Draw revealed segments as a flowing curve up to the new head.
      while (s.i < s.pts.length - 1 && budget > 0) {
        const a = s.pts[s.i], b = s.pts[s.i + 1];
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.stroke();
        head.x = b.x; head.y = b.y;
        s.i++; budget--;
      }
      if (s.i >= s.pts.length - 1) {
        // Finalize: hand-redraw pass thickens darks like a real pen.
        if (s.redrawPts) { ctx.globalAlpha = s.alpha * 0.8; strokePath(s.redrawPts); }
        penStroke = null;
      }
    }

    drawPen();
    setProgress(coverage() / params.target);
    rafId = requestAnimationFrame(tick);
  }

  // Non-animated: paint everything synchronously.
  function drawInstant() {
    while (strokeCount < MAX_STROKES) {
      if ((strokeCount & 1023) === 0 && coverage() >= params.target) break;
      const s = nextStroke();
      if (!s) break;
      paintStroke(s);
    }
    finishSketch();
  }

  // Pull the next drawable stroke (skips empty attempts). Null when done.
  function nextStroke() {
    for (let a = 0; a < 120; a++) {
      if (strokeCount >= MAX_STROKES) return null;
      const s = buildStroke();
      strokeCount++;
      if (s) return s;
      // A run of misses means the dark regions are spent.
      if (a > 40 && coverage() >= params.target * 0.96) return null;
    }
    return null;
  }

  // Paint a fully-built stroke at once (instant mode / preset preview).
  function paintStroke(s) {
    ctx.globalAlpha = s.alpha;
    strokePath(s.pts);
    if (s.redrawPts) { ctx.globalAlpha = s.alpha * 0.8; strokePath(s.redrawPts); }
  }

  // Build one scribble stroke (pick a dark start, walk a curling path along the
  // contours, deposit ink). Returns {pts, alpha, redrawPts} or null.
  function buildStroke() {
    const { opacity, flow, baseLen } = params;
    // 1) Pick a starting cell, biased toward remaining darkness (rejection).
    let sx = -1, sy = -1, best = 0;
    for (let a = 0; a < 22; a++) {
      const cx = (rnd() * mapW) | 0;
      const cy = (rnd() * mapH) | 0;
      const r = residual[cy * mapW + cx];
      if (r > best) { best = r; sx = cx; sy = cy; }
      if (r > 0.55 && rnd() < 0.6) break; // good enough, stop early
    }
    if (sx < 0 || best < 0.03) return null;

    // 2) Stroke geometry. Strength scales with how dark the start is.
    const matte = controls.removeBg.checked && fgMask;
    const strength = clamp(best, 0, 1);
    const lenScale = baseLen / 26;                 // ~1 at the old default

    // Occasionally throw a long, light "stray" sweep across the subject — the
    // kind of overshooting line a hand makes — for organic chaos.
    const stray = rnd() < 0.02;
    const segs = Math.min(stray ? 170 : 110,
      Math.max(5, Math.round(baseLen * (stray ? 1.6 : 0.5 + 0.7 * strength))));
    const step = (1.0 + rnd() * 0.5) * lenScale;   // tone-map units per segment
    const alpha = clamp(opacity * (0.5 + 0.7 * strength) * (stray ? 0.5 : 1), 0.02, 0.85);
    const steer = stray ? 0.12 : flow * 0.35;      // how hard it hugs contours
    const drift = 0.05 + (1 - flow) * 0.10;        // how fast the curl wanders

    let x = sx + (rnd() - 0.5);
    let y = sy + (rnd() - 0.5);
    let ang = sampleAngle(x, y) + (rnd() - 0.5) * (1 - flow) * Math.PI * 1.5;
    let turn = (rnd() - 0.5) * 0.15;               // persistent turn rate (curl)

    const pts = [{ x, y }];
    for (let i = 0; i < segs; i++) {
      const flowAng = sampleAngle(x, y);
      ang = angleLerp(ang, flowAng, steer) + turn; // contour pull + smooth curl
      turn = clamp(turn + (rnd() - 0.5) * drift, -0.5, 0.5);

      const nx = x + Math.cos(ang) * step;
      const ny = y + Math.sin(ang) * step;
      if (nx < 0 || ny < 0 || nx >= mapW || ny >= mapH) break;

      const idx = (ny | 0) * mapW + (nx | 0);
      if (matte && !fgMask[idx] && i > 0) break;   // never spill onto background
      // Stop once we wander into a light region (stray sweeps carry on).
      if (!stray && residual[idx] < 0.02 && i > 3) break;

      depositLine(x, y, nx, ny, alpha);            // deduct ink along the run
      x = nx; y = ny;
      pts.push({ x, y });
    }

    if (pts.length < 2) return null;
    // Decide the hand-redraw pass now (keeps RNG stream stable across modes).
    let redrawPts = null;
    if (!stray && strength > 0.35 && rnd() < 0.3) {
      redrawPts = pts.map((p) => ({ x: p.x + (rnd() - 0.5) * 0.8, y: p.y + (rnd() - 0.5) * 0.8 }));
    }
    return { pts, alpha, redrawPts };
  }

  // Draw the pen/nib at the current head position on the overlay canvas.
  function clearPen() {
    penCtx.setTransform(1, 0, 0, 1, 0, 0);
    penCtx.clearRect(0, 0, penCanvas.width, penCanvas.height);
  }
  function drawPen() {
    clearPen();
    if (!controls.showPen.checked || !running) return;
    const px = head.x * renderScale, py = head.y * renderScale;
    const L = Math.max(26, canvas.width * 0.05);   // barrel length
    const ang = -Math.PI * 0.72;                    // held up and to the right
    const cx = Math.cos(ang), cy = Math.sin(ang);
    const ex = px + cx * L, ey = py + cy * L;
    const ox = -cy, oy = cx;                        // perpendicular
    const w = Math.max(3, L * 0.18);

    // barrel
    penCtx.lineCap = 'round';
    penCtx.strokeStyle = '#11151c';
    penCtx.lineWidth = w;
    penCtx.beginPath(); penCtx.moveTo(px + cx * L * 0.28, py + cy * L * 0.28); penCtx.lineTo(ex, ey); penCtx.stroke();
    // accent grip
    penCtx.strokeStyle = '#6ea8fe';
    penCtx.lineWidth = w * 0.7;
    penCtx.beginPath();
    penCtx.moveTo(px + cx * L * 0.30, py + cy * L * 0.30);
    penCtx.lineTo(px + cx * L * 0.55, py + cy * L * 0.55);
    penCtx.stroke();
    // nib triangle at the tip
    penCtx.fillStyle = '#11151c';
    penCtx.beginPath();
    penCtx.moveTo(px, py);
    penCtx.lineTo(px + cx * L * 0.30 + ox * w * 0.6, py + cy * L * 0.30 + oy * w * 0.6);
    penCtx.lineTo(px + cx * L * 0.30 - ox * w * 0.6, py + cy * L * 0.30 - oy * w * 0.6);
    penCtx.closePath(); penCtx.fill();
  }

  // Draw a point list as a smooth curve (quadratic through segment midpoints).
  function strokePath(pts) {
    ctx.beginPath();
    ctx.moveTo(pts[0].x, pts[0].y);
    if (pts.length === 2) {
      ctx.lineTo(pts[1].x, pts[1].y);
    } else {
      for (let i = 1; i < pts.length - 1; i++) {
        const mx = (pts[i].x + pts[i + 1].x) * 0.5;
        const my = (pts[i].y + pts[i + 1].y) * 0.5;
        ctx.quadraticCurveTo(pts[i].x, pts[i].y, mx, my);
      }
      const n = pts.length;
      ctx.quadraticCurveTo(pts[n - 2].x, pts[n - 2].y, pts[n - 1].x, pts[n - 1].y);
    }
    ctx.stroke();
  }

  // ---- Single continuous line (one unbroken stroke) ----
  // The scribble engine already yields a detailed, contour-following portrait.
  // For single-line we generate those strokes, then chain them end-to-end by
  // nearest endpoint so the pen never lifts — keeping the facial detail while
  // producing one continuous line. (TSP stipple gave only a flat silhouette.)
  function buildSingleLinePath() {
    // Single-line needs dense coverage so the face resolves into detail (it
    // also keeps the connector hops between strokes short). Density nudges it.
    const dn = clamp((parseFloat(controls.density.value) - 0.45) / 0.52, 0, 1);
    const target = 0.82 + 0.15 * dn;
    const CAP = 3500;                       // keep chaining O(n^2) responsive
    const strokes = [];
    let guard = 0;
    while (strokes.length < CAP && guard < CAP * 14) {
      guard++;
      if ((guard & 255) === 0 && coverage() >= target) break;
      const s = buildStroke();
      if (s && s.pts.length >= 2) strokes.push(s.pts);
      else if ((guard & 63) === 0 && coverage() >= target * 0.96) break;
    }
    if (!strokes.length) return [];
    return connectStrokes(strokes);
  }

  // Greedy nearest-endpoint chaining: repeatedly take the unused stroke whose
  // nearer end is closest to the pen, flipping it to draw toward its far end.
  // The short connector segments between strokes read as continuous scribbling.
  function connectStrokes(strokes) {
    const n = strokes.length;
    const used = new Uint8Array(n);
    const path = [];
    let cur = 0;
    for (let i = 1; i < n; i++) if (strokes[i][0].y < strokes[cur][0].y) cur = i; // start up top
    used[cur] = 1;
    for (const p of strokes[cur]) path.push(p);
    let ex = path[path.length - 1].x, ey = path[path.length - 1].y;

    for (let k = 1; k < n; k++) {
      let best = -1, bestD = Infinity, flip = false;
      for (let i = 0; i < n; i++) {
        if (used[i]) continue;
        const p = strokes[i], a = p[0], b = p[p.length - 1];
        const da = (a.x - ex) * (a.x - ex) + (a.y - ey) * (a.y - ey);
        if (da < bestD) { bestD = da; best = i; flip = false; }
        const db = (b.x - ex) * (b.x - ex) + (b.y - ey) * (b.y - ey);
        if (db < bestD) { bestD = db; best = i; flip = true; }
      }
      if (best < 0) break;
      used[best] = 1;
      let p = strokes[best];
      if (flip) p = p.slice().reverse();
      for (const q of p) path.push(q);        // connector to p[0], then the stroke
      ex = p[p.length - 1].x; ey = p[p.length - 1].y;
    }
    return path;
  }

  // Bilinear-ish contour angle lookup.
  function sampleAngle(x, y) {
    const ix = clampInt(x | 0, 0, mapW - 1);
    const iy = clampInt(y | 0, 0, mapH - 1);
    return gradAngle[iy * mapW + ix];
  }

  // Subtract ink demand around a point (small soft disc) so coverage balances.
  function deposit(x, y, amt) {
    const ix = x | 0, iy = y | 0;
    const k = amt * 0.5;
    for (let dy = -1; dy <= 1; dy++) {
      const yy = iy + dy;
      if (yy < 0 || yy >= mapH) continue;
      for (let dx = -1; dx <= 1; dx++) {
        const xx = ix + dx;
        if (xx < 0 || xx >= mapW) continue;
        const idx = yy * mapW + xx;
        const w = (dx === 0 && dy === 0) ? 1 : 0.22;
        residual[idx] = Math.max(0, residual[idx] - k * w);
      }
    }
  }

  // Deposit ink along a segment so long strokes deduct proportionally to their
  // length (otherwise big sweeping strokes under-spend and over-scribble).
  function depositLine(x0, y0, x1, y1, amt) {
    const dist = Math.hypot(x1 - x0, y1 - y0);
    const n = Math.max(1, Math.round(dist));
    for (let s = 1; s <= n; s++) {
      const t = s / n;
      deposit(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, amt);
    }
  }

  function sumResidual() {
    let s = 0;
    // Sample stride for speed on big maps; still representative.
    const stride = residual.length > 120000 ? 3 : 1;
    for (let i = 0; i < residual.length; i += stride) s += residual[i];
    return s * stride;
  }

  function finishSketch() {
    running = false;
    cancelAnimationFrame(rafId);
    penStroke = null;
    clearPen();                  // lift the pen off the finished drawing
    setProgress(1);
    showStatus(false);
    buttons.pause.disabled = true;
    buttons.pause.textContent = 'Pause';
  }

  // ---- Helpers ----
  function boxBlur(src, w, h, radius) {
    const tmp = new Float32Array(src.length);
    const out = new Float32Array(src.length);
    const d = radius * 2 + 1;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        let s = 0;
        for (let k = -radius; k <= radius; k++) s += src[y * w + clampInt(x + k, 0, w - 1)];
        tmp[y * w + x] = s / d;
      }
    }
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        let s = 0;
        for (let k = -radius; k <= radius; k++) s += tmp[clampInt(y + k, 0, h - 1) * w + x];
        out[y * w + x] = s / d;
      }
    }
    return out;
  }

  function angleLerp(a, b, t) {
    let diff = b - a;
    while (diff > Math.PI) diff -= 2 * Math.PI;
    while (diff < -Math.PI) diff += 2 * Math.PI;
    return a + diff * t;
  }
  const clamp = (v, lo, hi) => v < lo ? lo : v > hi ? hi : v;
  const clampInt = (v, lo, hi) => v < lo ? lo : v > hi ? hi : v;

  // ---- UI wiring ----
  function showStatus(show) {
    statusEl.hidden = !show;
    if (show) { statusText.textContent = 'Sketching…'; buttons.pause.disabled = false; }
  }
  function setProgress(p) {
    p = clamp(p, 0, 1);
    barFill.style.width = (p * 100).toFixed(1) + '%';
    statusText.textContent = p >= 1 ? 'Done' : 'Sketching… ' + Math.round(p * 100) + '%';
  }
  // Indeterminate status for slow steps like the first model load.
  function busy(text) {
    statusEl.hidden = false;
    statusText.textContent = text;
    barFill.style.width = '100%';
    buttons.pause.disabled = true;
  }

  function enableControls(on) {
    buttons.redraw.disabled = !on;
    buttons.download.disabled = !on;
    buttons.pause.disabled = !on;
  }

  let redrawTimer = 0;
  function scheduleRedraw() {
    if (!sourceBitmap) return;
    clearTimeout(redrawTimer);
    redrawTimer = setTimeout(startSketch, 220);
  }

  function refreshLabels() {
    const d = parseFloat(controls.density.value);
    labels.density.textContent = d < 0.62 ? 'light' : d < 0.8 ? 'medium' : d < 0.9 ? 'dense' : 'ink-storm';
    labels.contrast.textContent = parseFloat(controls.contrast.value).toFixed(2);
    labels.length.textContent = controls.length.value;
    const f = parseFloat(controls.flow.value);
    labels.flow.textContent = f < 0.33 ? 'chaotic' : f < 0.7 ? 'mixed' : 'contours';
    labels.weight.textContent = parseFloat(controls.weight.value).toFixed(2);
    labels.opacity.textContent = parseFloat(controls.opacity.value).toFixed(2);
    const fv = parseFloat(controls.fill.value);
    labels.fill.textContent = fv === 0 ? 'off' : fv.toFixed(2);
    const sp = parseInt(controls.speed.value, 10);
    labels.speed.textContent = sp <= 3 ? 'slow' : sp <= 7 ? 'medium' : 'fast';
  }

  // Controls handled live (no re-sketch): they tune the in-progress drawing.
  const LIVE = new Set([controls.speed, controls.showPen]);

  // Sliders: update labels live, re-sketch shortly after the user settles.
  Object.entries(controls).forEach(([key, el]) => {
    if (!el || el === controls.removeBg || LIVE.has(el)) return;  // handled separately
    const ev = el.type === 'checkbox' || el.type === 'color' ? 'change' : 'input';
    el.addEventListener(ev, () => {
      if (PRESET_KEYS.includes(key)) { activePreset = null; highlightPreset(); }
      refreshLabels();
      saveSettings();
      scheduleRedraw();
    });
  });

  // Drawing speed + pen visibility apply to the live animation without restarting.
  controls.speed.addEventListener('input', () => { refreshLabels(); saveSettings(); });
  controls.showPen.addEventListener('change', () => {
    saveSettings();
    if (!controls.showPen.checked) clearPen();
  });

  // Background toggle: turning it on may need a (one-time) matte computation.
  controls.removeBg.addEventListener('change', async () => {
    saveSettings();
    if (!sourceBitmap) return;
    if (controls.removeBg.checked && !matteImg) {
      await prepareMatte();
      buildToneMap();
    }
    startSketch();
  });

  // One-tap style presets — set every slider at once.
  const PRESETS = {
    sketch: { density: 0.80, contrast: 1.45, length: 54, flow: 0.82, weight: 0.9, opacity: 0.18, fill: 0.16 },
    bold:   { density: 0.78, contrast: 1.75, length: 78, flow: 0.70, weight: 1.5, opacity: 0.28, fill: 0.10 },
    storm:  { density: 0.91, contrast: 1.35, length: 64, flow: 0.55, weight: 1.0, opacity: 0.22, fill: 0.22 },
  };
  function applyPreset(name) {
    const p = PRESETS[name];
    if (!p) return;
    for (const k in p) controls[k].value = p[k];
    activePreset = name;
    refreshLabels();
    highlightPreset();
    saveSettings();
    if (sourceBitmap) startSketch();
  }
  document.querySelectorAll('[data-preset]').forEach((b) =>
    b.addEventListener('click', () => applyPreset(b.dataset.preset)));

  function highlightPreset() {
    document.querySelectorAll('[data-preset]').forEach((b) =>
      b.classList.toggle('active', b.dataset.preset === activePreset));
  }

  // Persist all control values + the active preset so the app reopens, and
  // sketches each new photo, with your last choice.
  function saveSettings() {
    try {
      const values = {};
      for (const k in controls) {
        const el = controls[k];
        if (el) values[k] = el.type === 'checkbox' ? el.checked : el.value;
      }
      localStorage.setItem(STORE_KEY, JSON.stringify({ values, preset: activePreset }));
    } catch (e) { /* storage unavailable (e.g. private mode) — ignore */ }
  }
  function loadSettings() {
    try {
      const raw = localStorage.getItem(STORE_KEY);
      if (!raw) return;
      const s = JSON.parse(raw);
      for (const k in (s.values || {})) {
        const el = controls[k];
        if (!el) continue;
        if (el.type === 'checkbox') el.checked = !!s.values[k];
        else el.value = s.values[k];
      }
      activePreset = s.preset || null;
    } catch (e) { /* ignore malformed/blocked storage */ }
  }

  buttons.redraw.addEventListener('click', startSketch);
  buttons.download.addEventListener('click', downloadPng);
  buttons.pause.addEventListener('click', () => {
    if (!running) return;
    paused = !paused;
    buttons.pause.textContent = paused ? 'Resume' : 'Pause';
    if (!paused) rafId = requestAnimationFrame(tick);
  });

  function downloadPng() {
    const a = document.createElement('a');
    a.download = 'scribble.png';
    a.href = canvas.toDataURL('image/png');
    a.click();
  }

  // File pickers + drag/drop + paste.
  buttons.pick.addEventListener('click', (e) => { e.stopPropagation(); fileInput.click(); });
  dropzone.addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', (e) => loadFile(e.target.files[0]));

  ['dragenter', 'dragover'].forEach((t) =>
    stage.addEventListener(t, (e) => { e.preventDefault(); dropzone.classList.add('drag'); }));
  ['dragleave', 'drop'].forEach((t) =>
    stage.addEventListener(t, (e) => { e.preventDefault(); if (t === 'dragleave') dropzone.classList.remove('drag'); }));
  stage.addEventListener('drop', (e) => {
    dropzone.classList.remove('drag');
    const f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
    if (f) loadFile(f);
  });
  window.addEventListener('paste', (e) => {
    const items = e.clipboardData && e.clipboardData.items;
    if (!items) return;
    for (const it of items) {
      if (it.type.startsWith('image/')) { loadFile(it.getAsFile()); break; }
    }
  });

  // ---- PWA: install prompt + service worker ----
  let deferredPrompt = null;
  window.addEventListener('beforeinstallprompt', (e) => {
    e.preventDefault();
    deferredPrompt = e;
    buttons.install.hidden = false;
  });
  buttons.install.addEventListener('click', async () => {
    if (!deferredPrompt) return;
    deferredPrompt.prompt();
    await deferredPrompt.userChoice;
    deferredPrompt = null;
    buttons.install.hidden = true;
  });
  window.addEventListener('appinstalled', () => { buttons.install.hidden = true; });

  if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
      navigator.serviceWorker.register('sw.js').catch((e) => console.warn('SW failed', e));
    });
  }

  loadSettings();
  refreshLabels();
  highlightPreset();
})();
