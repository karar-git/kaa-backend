# ExoTransit Lab API
#
# The image carries the precomputed results/ and static/ directories, so the
# container has no runtime dependency on object storage. Those 150 MB are not
# uploaded with `railway up` (the CLI times out on a slow connection); the
# build fetches them from the GitHub repository instead, at the commit given
# by DATA_REV (default: main). Push data changes before deploying.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

WORKDIR /app

# build-essential: EXOTIC's dependencies (ultranest, rebound) compile C
# extensions on Linux. curl + certificates: the data fetch below.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Dependencies first: this layer is cached until requirements.txt changes, so
# editing application code does not trigger a reinstall.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Precomputed results, tiles and the classifier, straight from the repository.
ARG DATA_REPO=https://github.com/karar-git/kaa-backend
ARG DATA_REV=main
RUN curl -fsSL "${DATA_REPO}/archive/${DATA_REV}.tar.gz" \
    | tar xz --strip-components=2 --wildcards '*/backend/data/*' \
    && test -f data/results/planet_catalog.csv \
    && du -sh data

COPY app ./app

ENV DATA_DIR=/app/data

# Railway injects PORT. The shell form is required so ${PORT} is expanded, and
# the default keeps `docker run` working locally without setting it.
EXPOSE 8000
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1
