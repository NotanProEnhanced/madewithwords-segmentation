# Typortrait engine — container image.
# Build context must be the typography_engine/ directory.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
# glibc allocator tuning for a NumPy/OpenCV/MediaPipe workload. By default glibc
# opens up to (8 * nCPU) malloc arenas and each render thread claims one it never
# returns, so the resident set climbs and stays high. Cap the arenas and trim more
# eagerly so freed render buffers actually go back to the OS -- the single biggest
# lever for fitting multiple brand containers on one box. Pairs with the explicit
# malloc_trim() after each render burst in app/main.py.
ENV MALLOC_ARENA_MAX=2 MALLOC_TRIM_THRESHOLD_=131072
WORKDIR /app

# Native libraries: CairoSVG (cairo/pango/gdk-pixbuf), OpenCV + MediaPipe
# (libGL/GLES/EGL/glib + X libs), plus the fonts the renderer draws with.
# MediaPipe's C bindings dlopen libGLESv2.so.2 / libEGL.so.1 at init -- without
# libgles2 + libegl1 face landmarking and segmentation fail to initialize.
RUN apt-get update && apt-get install -y --no-install-recommends \
      libcairo2 libpango-1.0-0 libpangocairo-1.0-0 libgdk-pixbuf-2.0-0 libffi8 \
      libgl1 libgles2 libegl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
      fonts-dejavu-core fonts-liberation \
      ffmpeg \
      curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-fetch the MediaPipe models so there is no first-request download.
RUN mkdir -p models \
    && curl -fsSL -o models/face_landmarker.task \
      "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task" \
    && curl -fsSL -o models/selfie_segmenter.tflite \
      "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite"

# Optional matting model (RobustVideoMatting, ~15MB). NON-FATAL: a failed fetch must not
# break the build -- the feature is opt-in (TYPO_MATTE_MODEL) and re-fetches at runtime,
# falling back to the guided-filter matte if still unavailable.
RUN curl -fsSL -o models/rvm_mobilenetv3_fp32.onnx \
      "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3_fp32.onnx" \
    || echo "matting model prefetch skipped (optional; fetched at runtime)"

COPY app ./app
COPY static ./static
# memorial_card.py is imported by app.main /download/card to build the print-ready
# prayer-card PDF (reportlab). Top-level module, so it must be copied explicitly.
COPY memorial_card.py ./memorial_card.py
# tools/reel_template.py is imported by app.main /reel endpoint to build the
# shareable 9:16 reel from a paid job.
COPY tools ./tools

EXPOSE 8077
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8077", "--workers", "1"]
