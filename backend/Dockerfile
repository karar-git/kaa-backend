# ExoTransit Lab API
#
# The image bakes in the precomputed results/ and static/ directories, so the
# container has no runtime dependency on object storage or a mounted volume.
# That is what makes the deploy a single `railway up` with nothing to configure.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

WORKDIR /app

# EXOTIC's dependencies (ultranest, rebound) compile C extensions on Linux.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Dependencies first: this layer is cached until requirements.txt changes, so
# editing application code does not trigger a reinstall.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY data ./data

ENV DATA_DIR=/app/data

# Railway injects PORT. The shell form is required so ${PORT} is expanded, and
# the default keeps `docker run` working locally without setting it.
EXPOSE 8000
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1
