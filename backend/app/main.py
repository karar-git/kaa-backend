"""ExoTransit Lab API — FastAPI application.

Serves the results of the offline MicroObservatory transit pipeline. Every number
returned here was computed in `kaggle/src/nb01_pipeline.py`; this process reads
files and shapes JSON, and deliberately does no science of its own. That keeps
the analysis reproducible from a notebook rather than buried in a web server.

Interactive docs:  /docs  (Swagger UI)
                   /redoc (ReDoc)
                   /openapi.json
"""

from __future__ import annotations

import logging
import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, RedirectResponse

from . import data
from .config import settings
from .routers import field, images, kpis, meta, ml, physics, science, sessions

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("exotransit")

DESCRIPTION = """
API behind **ExoTransit Lab**, a discovery tool built on 1,741 MicroObservatory
FITS frames for the Hack4Dev Iraq 2026 Exoplanet Data Challenge.

### What you can get from this API

| Ask | Endpoint |
|---|---|
| **All headline KPIs in one call** | `GET /api/kpis` |
| **How far each planet is** (light-years, parsecs) | `GET /api/planets` |
| **How fast each planet moves**, orbit size, period, temperature, starlight received | `GET /api/planets` |
| **One planet in full**, incl. the radius from *our* measured depth | `GET /api/planets/{target}` |
| **Do our numbers match the published ones?** | `GET /api/planets/validation` |
| The equations and constants behind them | `GET /api/planets/equations` |
| **AI: classify a source** (star, hot pixel, cosmic ray, satellite trail, faint source, noise) | `POST /api/classify` |
| **AI: plain-language summary of a night**, English or Arabic | `POST /api/explain` |
| Classifier model card and per-class precision/recall | `GET /api/model/info`, `/api/model/metrics` |
| Light curve of a night, with the predicted transit window | `GET /api/sessions/{id}/lightcurve` |
| Measured transit depth and significance per night | `GET /api/science/depths` |
| Search of every star in every field, with its chance expectation | `GET /api/science/search` |
| Which nights are usable, and why | `GET /api/quality/sessions` |
| Photometric precision vs brightness | `GET /api/quality/noise-floor` |
| Rendered frames and real pixel values | `GET /api/images/...` |
| **Any session, live**: star field with detections ringed, and its light curve | `GET /api/field/frame.png`, `GET /api/field/lightcurve.png` |
| Upload a night of FITS frames and analyse it | `POST /api/field/upload` |
| The 11 ways a dip can fool you | `GET /api/false-positive-cases` |

Distances and speeds are **derived from NASA Exoplanet Archive inputs**, not
measured by our telescope; every response says so. The only per-planet quantity
built on our own frames is the radius from our measured depth, and it is flagged
`inconclusive` wherever that depth is below 3 sigma.

### What this data is

Thirty nights on **Cecilia**, a 6-inch robotic telescope at the Whipple
Observatory in Arizona: 1,681 sixty-second science frames of 8 known
hot-Jupiter hosts across 22 continuous observing sessions, plus 60 dark frames.

### How to read these results

Three things are true of this dataset and shape everything below.

1. **A third of the survey is unusable.** Six of 22 sessions failed quality
   triage and three of the eight targets have no good night at all. Those
   sessions are kept and served, because they are the control sample.
2. **The mount is alt-azimuth with no derotator.** Fields drift up to ~105 px and
   rotate about a degree per session, so the pipeline links detections between
   adjacent frames instead of registering onto a common grid.
3. **There is no WCS solution in these headers.** The host star is therefore
   chosen by position rather than identified astrometrically, and every endpoint
   that depends on that choice says so.

### The one rule

Nothing here is called a confirmed planet. Signals are reported as *candidates
consistent with a transit*, null results are reported as null results, and the
limitations are endpoints in their own right — see `/api/false-positive-cases`.

In particular, `/api/science/search` tests ~2,700 star-nights, so roughly 27 of
them clear `p <= 0.01` with nothing there at all. That expectation is returned
alongside the hit count as `n_expected_by_chance`, because a list of candidates
quoted without it is not a result.
"""

TAGS = [
    {"name": "kpis", "description": "Every headline number in one call. Start here."},
    {"name": "meta", "description": "Dataset, instrument, calibration and the "
                                    "false-positive rulebook."},
    {"name": "sessions", "description": "The 22 observing sessions: frames, "
                                        "tracks, detections, field motion."},
    {"name": "science", "description": "Light curves, ephemerides, transit depths, "
                                       "phase folds and the field-wide search."},
    {"name": "quality", "description": "What governs image quality, and the "
                                       "photometric noise floor."},
    {"name": "physics", "description": "Per planet: orbit size, orbital speed, "
                                       "temperature, radius and distance from Earth, "
                                       "each validated against the NASA Exoplanet "
                                       "Archive."},
    {"name": "images", "description": "Rendered frame tiles and real pixel counts."},
    {"name": "field", "description": "Live quick-look for any folder of FITS frames: "
                                     "dark subtraction, background removal, star "
                                     "detection, differential photometry, and the two "
                                     "pictures (field + light curve). Also accepts uploads."},
    {"name": "ml", "description": "Source classifier and grounded LLM narration."},
]

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description=DESCRIPTION,
    openapi_tags=TAGS,
    contact={"name": settings.TEAM},
    license_info={"name": "MicroObservatory data is public. Credit CfA "
                          "Harvard-Smithsonian."},
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)
# Light curves and detection tables are long, repetitive JSON. Gzip cuts them by
# roughly an order of magnitude.
app.add_middleware(GZipMiddleware, minimum_size=1024)

for r in (kpis.router, meta.router, sessions.router, science.router,
          physics.router, images.router, field.router, ml.router):
    app.include_router(r, prefix="/api")


@app.middleware("http")
async def timing(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Response-Time-ms"] = f"{(time.perf_counter()-t0)*1000:.1f}"
    return response


@app.exception_handler(data.DatasetMissing)
async def dataset_missing_handler(request: Request, exc: data.DatasetMissing):
    """A missing results file is a deployment problem, not a client error."""
    return JSONResponse(
        status_code=503,
        content={"detail": str(exc),
                 "hint": "GET /api/health lists exactly which files are absent."},
    )


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")


@app.on_event("startup")
def warm() -> None:
    """Touch the small tables at boot so the first real request is not the one
    that pays for reading them."""
    inv = data.available()
    missing = [k for k, v in inv.items() if not v]
    log.info("data dir: %s", settings.DATA_DIR)
    log.info("results files present: %d/%d", sum(inv.values()), len(inv))
    if missing:
        log.warning("missing results files: %s", ", ".join(missing))
    log.info("AI narration: %s", "enabled" if settings.ai_enabled else "disabled")
    for fn in (data.sessions, data.session_quality, data.audit):
        try:
            fn()
        except data.DatasetMissing:
            pass
