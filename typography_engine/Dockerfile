# Typortrait engine — container image.
# Build context must be the typography_engine/ directory.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
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
      curl ca-certificates unzip \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# rtmlib (optional pet face-anchoring, app/pet_landmarks.py) is installed SEPARATELY with
# --no-deps: its own METADATA unconditionally requires "opencv-python" *and*
# "opencv-contrib-python", and installing both packages side by side -- confirmed while
# building this -- silently swaps out which OpenCV build satisfies every `import cv2` in the
# app, changing PNG-encoding output for renders that have nothing to do with pets. rtmlib's
# only OTHER runtime need (besides cv2, already pinned above) is onnxruntime + tqdm, both
# harmless additions -- tqdm is in requirements.txt, onnxruntime is already required.
RUN pip install --no-cache-dir --no-deps rtmlib

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

# Optional pet face-anchoring models (app/pet_landmarks.py, PET_LANDMARKS=1). NON-FATAL, same
# reasoning as the matting model above: this is opt-in and pet_landmarks.py has no runtime
# download fallback at all -- a missing file here just means the feature stays off, exactly
# like a missing RVM model falls back to the guided-filter matte.
#   detector: YOLOX-m (COCO, ~97MB) -- finds the dog/cat bounding box.
#   pose:     RTMPose-m trained on AP-10K (~14MB) -- 17 animal keypoints inside that box; only
#             eyes/nose/neck are actually read (see pet_landmarks.py's docstring for why).
RUN curl -fsSL -o models/yolox_m.onnx \
      "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_m.onnx" \
    || echo "pet detector prefetch skipped (PET_LANDMARKS will stay inactive)"
RUN curl -fsSL -o /tmp/rtmpose_ap10k.zip \
      "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-ap10k_pt-aic-coco_210e-256x256-7a041aa1_20230206.zip" \
    && unzip -o -j /tmp/rtmpose_ap10k.zip "*/end2end.onnx" -d models \
    && mv models/end2end.onnx models/rtmpose_ap10k_m.onnx \
    && rm -f /tmp/rtmpose_ap10k.zip \
    || echo "pet pose-model prefetch skipped (PET_LANDMARKS will stay inactive)"

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
