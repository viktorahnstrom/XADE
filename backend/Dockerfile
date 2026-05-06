# ── Stage 1: builder ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip && \
    # Single resolve so CPU torch (with the +cpu local version, which PEP 440
    # ranks higher than the plain GPU wheel) wins for both direct and
    # transitive resolutions. A second pip-install pass with --prefix
    # doesn't see the first prefix on sys.path and re-resolves torch from
    # PyPI, which pulls in 2.7 GB of nvidia-* and 641 MB of triton.
    pip install --prefix=/install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt && \
    # Defensive cleanup in case the resolver ever drifts back to GPU torch,
    # plus drop the duplicate opencv-python.libs (mediapipe brings in
    # opencv-contrib-python which already ships the cv2 binaries).
    rm -rf /install/lib/python3.11/site-packages/nvidia* \
           /install/lib/python3.11/site-packages/triton* \
           /install/lib/python3.11/site-packages/opencv_python \
           /install/lib/python3.11/site-packages/opencv_python.libs \
           /install/lib/python3.11/site-packages/opencv_python-*.dist-info && \
    find /install -name __pycache__ -type d -prune -exec rm -rf {} +


# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    libgles2 \
    libegl1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local
COPY app/ ./app/
COPY pyproject.toml .

# Download the MediaPipe Face Landmarker model (float16, ~6 MB)
RUN mkdir -p /app/models && \
    wget -q -O /app/models/face_landmarker.task \
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

ENV HF_HOME=/app/.cache/huggingface
# Railway injects PORT; default to 8000 for local docker run
ENV PORT=8000

EXPOSE ${PORT}

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]