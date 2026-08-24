# reels-engine — pipeline + studio in one image.
#
# Two stages so the wheels are built once and the runtime image carries no
# compiler. ffmpeg comes from Debian and is checked for libass at build time:
# a build without it renders blank captions and still exits 0, which is the
# exact failure this project spends the most effort preventing.

FROM python:3.11-slim AS build

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /wheels
COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt


FROM python:3.11-slim

# ffmpeg for every stage; libgles2/libegl1 are what mediapipe needs to load.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgles2 \
        libegl1 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && ffmpeg -hide_banner -filters | grep -q ' ass ' \
       || (echo "ffmpeg has no ass filter (built without libass)" && exit 1)

COPY --from=build /wheels /wheels
COPY requirements.txt .
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt \
    && rm -rf /wheels requirements.txt

WORKDIR /app
COPY pipeline/ ./pipeline/
COPY server/ ./server/
COPY web/ ./web/
COPY assets/ ./assets/
COPY scripts/ ./scripts/
COPY smoke.py ./

# Uploads and rendered clips live here; mount a volume over it in production.
ENV REELS_DATA_DIR=/data \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
RUN mkdir -p /data && useradd --system --uid 10001 reels && chown -R reels /data /app
USER reels

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4).status==200 else 1)"

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
