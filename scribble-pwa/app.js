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
  const canvas = $('canvas');
  const ctx = canvas.getContext('2d');
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
    ink: $('inkColor'),
    paper: $('paperColor'),
    animate: $('animateChk'),
  };
  const labels = {
    density: $('densityVal'),
    contrast: $('contrastVal'),
    length: $('lengthVal'),
    flow: $('flowVal'),
    weight: $('weightVal'),
    opacity: $('opacityVal'),
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
  let initialInk = 0;          // sum of residual at start
  let strokeCount = 0;
  let rafId = 0;
  let paused = false;
  let running = false;
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
    const url = URL.createObjectURL(file);
    try {
      const img = await loadImage(url);
      sourceBitmap = img;
      buildToneMap();
      enableControls(true);
      startSketch();
    } catch (err) {
      console.error(err);
      alert('Could not read that image. Try a JPG or PNG.');
    } finally {
      URL.revokeObjectURL(url);
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

    function t(x, y) {
      x = x < 0 ? 0 : x >= mapW ? mapW - 1 : x;
      y = y < 0 ? 0 : y >= mapH ? mapH - 1 : y;
      return tone[y * mapW + x];
    }
  }

  function buildResidual() {
    const gamma = parseFloat(controls.contrast.value);
    residual = new Float32Array(mapW * mapH);
    initialInk = 0;
    for (let i = 0; i < tone.length; i++) {
      // Darkness demand, contrast-shaped. Pure white asks for almost no ink.
      let d = Math.pow(1 - tone[i], gamma);
      if (d < 0.015) d = 0;
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
    paused = false;
    running = true;
    buttons.pause.textContent = 'Pause';

    // Paper.
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = controls.paper.value;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.scale(renderScale, renderScale);
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.strokeStyle = controls.ink.value;

    canvas.hidden = false;
    dropzone.style.display = 'none';
    showStatus(true);

    if (controls.animate.checked) {
      rafId = requestAnimationFrame(tick);
    } else {
      // Draw it all in one shot.
      while (running && !stepBatch(4000)) { /* spin */ }
      finishSketch();
    }
  }

  function tick() {
    if (paused) return;
    const done = stepBatch(550);         // strokes per frame
    const coverage = 1 - sumResidual() / Math.max(1e-6, initialInk);
    setProgress(coverage / coverageTarget());
    if (done) { finishSketch(); return; }
    rafId = requestAnimationFrame(tick);
  }

  function coverageTarget() {
    // Density slider -> how much of the ink demand we satisfy before stopping.
    return parseFloat(controls.density.value);
  }

  // Draw a batch of strokes. Returns true when the sketch is complete.
  function stepBatch(budget) {
    const target = coverageTarget();
    const maxStrokes = 75000;
    const opacity = parseFloat(controls.opacity.value);
    const weight = parseFloat(controls.weight.value);
    const flow = parseFloat(controls.flow.value);
    const baseLen = parseInt(controls.length.value, 10);

    ctx.lineWidth = weight;

    for (let n = 0; n < budget; n++) {
      if (strokeCount >= maxStrokes) return true;

      // Periodic stop check (cheap-ish; sampled to avoid summing every stroke).
      if ((strokeCount & 1023) === 0) {
        const cov = 1 - sumResidual() / Math.max(1e-6, initialInk);
        if (cov >= target) return true;
      }

      if (!drawStroke(opacity, flow, baseLen)) {
        // Couldn't find a worthy dark spot; sketch is effectively done.
        const cov = 1 - sumResidual() / Math.max(1e-6, initialInk);
        if (cov >= target * 0.96) return true;
      }
      strokeCount++;
    }
    return false;
  }

  // Lay a single scribble stroke. Returns false if no dark region was found.
  function drawStroke(opacity, flow, baseLen) {
    // 1) Pick a starting cell, biased toward remaining darkness (rejection).
    let sx = -1, sy = -1, best = 0;
    for (let a = 0; a < 22; a++) {
      const cx = (rnd() * mapW) | 0;
      const cy = (rnd() * mapH) | 0;
      const r = residual[cy * mapW + cx];
      if (r > best) { best = r; sx = cx; sy = cy; }
      if (r > 0.55 && rnd() < 0.6) break; // good enough, stop early
    }
    if (sx < 0 || best < 0.03) return false;

    // 2) Stroke geometry. Strength scales with how dark the start is.
    const strength = clamp(best, 0, 1);
    const segs = Math.max(3, Math.round(baseLen * (0.5 + 0.7 * strength)));
    const step = 0.9 + rnd() * 0.5;       // tone-map units per segment
    let x = sx + (rnd() - 0.5);
    let y = sy + (rnd() - 0.5);

    // Initial heading: contour-following blended with chaos.
    let ang = sampleAngle(x, y) + (rnd() - 0.5) * (1 - flow) * Math.PI * 1.4;
    const wobble = 0.18 + (1 - flow) * 0.5;
    const alpha = clamp(opacity * (0.55 + 0.7 * strength), 0.02, 0.85);

    ctx.globalAlpha = alpha;
    ctx.beginPath();
    ctx.moveTo(x, y);

    let drawn = false;
    for (let i = 0; i < segs; i++) {
      // Steer toward the local contour direction, plus a little jitter.
      const flowAng = sampleAngle(x, y);
      ang = angleLerp(ang, flowAng, flow * 0.5) + (rnd() - 0.5) * wobble;

      const nx = x + Math.cos(ang) * step;
      const ny = y + Math.sin(ang) * step;
      if (nx < 0 || ny < 0 || nx >= mapW || ny >= mapH) break;

      // Stop if we've wandered into a light region this stroke shouldn't ink.
      if (residual[(ny | 0) * mapW + (nx | 0)] < 0.02 && i > 2) break;

      ctx.lineTo(nx, ny);
      deposit(nx, ny, alpha);
      x = nx; y = ny; drawn = true;
    }

    if (drawn) ctx.stroke();
    // Occasional back-scribble over the same path for that hand-drawn density.
    if (drawn && strength > 0.4 && rnd() < 0.25) {
      ctx.stroke();
      deposit(sx, sy, alpha * 0.5);
    }
    return drawn;
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
    const k = amt * 1.35;
    for (let dy = -1; dy <= 1; dy++) {
      const yy = iy + dy;
      if (yy < 0 || yy >= mapH) continue;
      for (let dx = -1; dx <= 1; dx++) {
        const xx = ix + dx;
        if (xx < 0 || xx >= mapW) continue;
        const idx = yy * mapW + xx;
        const w = (dx === 0 && dy === 0) ? 1 : 0.4;
        residual[idx] = Math.max(0, residual[idx] - k * w);
      }
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
  }

  // Sliders: update labels live, re-sketch shortly after the user settles.
  Object.values(controls).forEach((el) => {
    if (!el) return;
    const ev = el.type === 'checkbox' || el.type === 'color' ? 'change' : 'input';
    el.addEventListener(ev, () => { refreshLabels(); scheduleRedraw(); });
  });

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

  refreshLabels();
})();
