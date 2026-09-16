"""Runtime configuration.

Everything that differs between a laptop and Railway lives here and is read from
the environment. No secret is ever hard-coded: `OPENROUTER_API_KEY` is read at
call time and, if absent, the AI endpoints report themselves as disabled rather
than failing at import.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path


def _first_existing(*candidates: Path) -> Path:
    for c in candidates:
        if c.exists():
            return c
    return candidates[-1]


class Settings:
    """Application settings, resolved once at import."""

    APP_NAME = "ExoTransit Lab API"
    VERSION = "1.0.0"
    TEAM = "Iraqi Andromeda"
    CHALLENGE = "Hack4Dev Iraq 2026 - Exoplanet Data Challenge, Challenge E"

    # ---------------------------------------------------------------- paths
    ROOT = Path(__file__).resolve().parent.parent          # backend/
    DATA_DIR = Path(os.environ.get("DATA_DIR") or _first_existing(
        ROOT / "data",                                     # baked into the image
        ROOT.parent / "kaggle" / "out",                    # local dev
    ))
    RESULTS_DIR = DATA_DIR / "results"
    STATIC_DIR = DATA_DIR / "static"

    # ------------------------------------------------------------ raw frames
    # Root of the raw FITS archive, laid out observations/<night>/<target>/...
    # and calibration/<night>/... The live `/api/field` routes read from here.
    OBS_ROOT = Path(os.environ.get("OBS_ROOT") or _first_existing(
        ROOT.parent / "database",                          # repo layout
        ROOT / "database",
        DATA_DIR / "raw",
    ))
    # Extra folders a client may point `/api/field` at, separated by the OS path
    # separator (`;` on Windows, `:` elsewhere). Anything outside these roots,
    # OBS_ROOT and UPLOAD_DIR is refused.
    EXTRA_SESSION_ROOTS = [Path(p) for p in
                           os.environ.get("SESSION_ROOTS", "").split(os.pathsep) if p.strip()]
    UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR") or DATA_DIR / "uploads")
    MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", 400))
    FIELD_MAX_FRAMES = int(os.environ.get("FIELD_MAX_FRAMES", 400))
    # POST /api/field/ingest writes into OBS_ROOT. It is closed unless this is
    # set, and is meant to be opened only while an operator loads the archive.
    INGEST_OPEN = os.environ.get("INGEST_OPEN", "").strip().lower() in {"1", "true", "yes"}
    # Enough for every session in the archive to stay warm after its first
    # request; one cached analysis is a few MB.
    FIELD_CACHE_SESSIONS = int(os.environ.get("FIELD_CACHE_SESSIONS", 32))

    # ---------------------------------------------------------------- server
    PORT = int(os.environ.get("PORT", 8000))               # Railway injects PORT
    # Comma-separated list, or "*" for any origin. The dashboard is a separate
    # deployment, so the API must allow cross-origin reads.
    CORS_ORIGINS = [o.strip() for o in
                    os.environ.get("CORS_ORIGINS", "*").split(",") if o.strip()]

    # ---------------------------------------------------------------- caching
    # Everything served here is precomputed and immutable for a given deploy, so
    # it can be cached hard. Images especially: they are content, not state.
    IMAGE_CACHE_SECONDS = int(os.environ.get("IMAGE_CACHE_SECONDS", 60 * 60 * 24 * 30))
    JSON_CACHE_SECONDS = int(os.environ.get("JSON_CACHE_SECONDS", 60 * 5))

    MAX_ROWS = int(os.environ.get("MAX_ROWS", 20000))      # hard cap per response

    # ---------------------------------------------------------------- AI
    OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "google/gemini-2.5-flash")
    OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
    OPENROUTER_TIMEOUT = int(os.environ.get("OPENROUTER_TIMEOUT", 45))

    @property
    def openrouter_key(self) -> str | None:
        """Read at call time, never cached, never logged."""
        return os.environ.get("OPENROUTER_API_KEY") or None

    @property
    def ai_enabled(self) -> bool:
        return self.openrouter_key is not None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
