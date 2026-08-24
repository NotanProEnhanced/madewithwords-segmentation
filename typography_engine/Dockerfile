# Typortrait engine — container image.
# Build context must be the typography_engine/ directory.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

# Native libraries: CairoSVG (cairo/pango/gdk-pixbuf), OpenCV + MediaPipe
# (libGL/libGLES/libEGL + glib + X libs), plus the fonts the renderer draws with.
# MediaPipe 0.10.x specifically needs libGLESv2.so.2 and libEGL.so.1 to load
# its C bindings -- without them it silently falls back and the engine renders
# face-landmark-less garbage. Pinning these explicitly keeps builds reproducible.
RUN apt-get update && apt-get install -y --no-install-recommends \
      libcairo2 libpango-1.0-0 libpangocairo-1.0-0 libgdk-pixbuf-2.0-0 libffi8 \
      libgl1 libgles2 libegl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
      fonts-dejavu fonts-liberation \
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

COPY app ./app
COPY static ./static

EXPOSE 8077
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8077", "--workers", "1"]
