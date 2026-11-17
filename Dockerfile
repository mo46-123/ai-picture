FROM python:3.12-slim

# OpenCV and MediaPipe's native library need these shared libraries (libEGL, libGLES).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libegl1 libgles2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Bake the face model into the image. Behind the Great Firewall pass a mirror:
#   docker build --build-arg FACE_MODEL_URL=https://your-mirror/face_landmarker.task .
ARG FACE_MODEL_URL=https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
RUN python scripts/download_models.py --url "$FACE_MODEL_URL" \
    || echo "Face model not downloaded: the app will run in filter-only mode until models/face_landmarker.task exists."

ENV HOST=0.0.0.0 PORT=8000 AUTO_DOWNLOAD_MODEL=0
EXPOSE 8000
# Hosting platforms (Render, etc.) pass the port to listen on in $PORT.
CMD ["sh", "-c", "python -m seoulglow.web --host 0.0.0.0 --port ${PORT:-8000}"]
