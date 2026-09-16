# ExoTransit Lab

A discovery tool for exoplanet transits built on MicroObservatory data, for the
**Hack4Dev Iraq 2026 Exoplanet Data Challenge, Challenge E**. Team **Iraqi Andromeda**.

Thirty nights on *Cecilia*, a 6-inch robotic telescope at the Whipple Observatory in
Arizona: 1,681 sixty-second frames of 8 known hot-Jupiter hosts across 22 observing
sessions, plus 60 dark frames. The pipeline calibrates them, finds and follows every
star, measures differential light curves, checks them against published ephemerides,
derives the planets' physics, and says honestly which nights are usable and which are not.

**Live API:** https://exotransit-lab-api-production.up.railway.app/docs

---

## What is in this repository

| Folder | What it is |
|---|---|
| `backend/` | FastAPI service. Serves the pipeline results as JSON and images, classifies sources, and runs a live quick-look on any folder of FITS frames. Deployed on Railway. [Read its README](backend/README.md). |
| `kaggle/` | The science. `src/nb01_pipeline.py` is the whole analysis as one notebook; `src/nb02_train.py` trains the source classifier. `build_notebooks.py` turns them into `notebooks/*.ipynb` for Kaggle. |
| `ExoTransitLab.jsx` | The React dashboard that consumes the API. |
| `database/` | The raw FITS archive (1.1 GB, not committed). Layout: `observations/<night>/<target>/session_01/*.fits` and `calibration/<night>/Dark-*.fits`. `database/metadata/observations.csv` lists every file. |

Nothing in `backend/` computes a scientific result on its own except the `/api/field`
quick-look routes. Every other number comes from the notebook, so the analysis can be
read top to bottom and reproduced.

---

## The pipeline in one paragraph

Frames are dark-subtracted with a per-night master dark and hot pixels are mapped from
it. Sources are detected as connected pixel groups above the local noise and linked
frame to frame, because the alt-azimuth mount drifts up to about 105 px and rotates
about a degree per night, so nothing can be stacked onto one grid. Every linked track is
auto-labelled by its physics (star, hot pixel, cosmic ray, satellite trail, faint source,
noise) and those labels train a small CNN. Aperture photometry on the target is divided
by the summed comparison stars so clouds cancel. Times are converted to BJD_TDB, the
predicted transit window is taken from published ephemerides, and the depth inside the
window is measured and quoted with its significance. Repeat nights are phase-folded.
Nothing is called a confirmed planet: results are *candidates consistent with a transit*,
null results are reported as null results, and the eleven ways a dip can fool you are an
endpoint of their own.

Headline facts the pipeline established:

- Six of 22 sessions failed quality triage, and three of the eight targets have no usable
  night. They are kept and served as the control sample.
- The field-wide search tests about 2,700 star-nights, so roughly 27 pass `p <= 0.01` by
  chance. That expectation is returned with every hit count.
- Distances, speeds and temperatures are derived from NASA Exoplanet Archive inputs. The
  only per-planet quantity built on our own frames is the radius from our measured depth.

---

## Live quick-look for any session

`GET /api/field/frame.png?session=TRES-3__2026-08-10` returns the dark-subtracted,
background-removed field with every detected star ringed. `.../lightcurve.png` returns
the differential light curve. `session` may be a pipeline id, a folder in the archive,
or an id returned by `POST /api/field/upload`, so the same routes work for a night nobody
has processed yet. `.../summary` gives the numbers behind both pictures and a one-line
reading, for example that a 30% drop in the target's raw flux vanishes once divided by
the comparison stars and is therefore cloud, not the star.

**EXOTIC compatible.** Every session can be handed to NASA's
[EXOTIC](https://github.com/rzellem/EXOTIC) (Exoplanet Watch):
`.../exotic/bundle.zip?session=…` returns a filled-in `inits.json`, our light
curve in EXOTIC's pre-reduced format (BJD_TDB, flux, error, airmass) and an
AAVSO Exoplanet Database report, with a README giving the exact commands.

**EXOTIC does the reduction too.** `POST .../exotic/run?session=…` runs the real
EXOTIC on the session's raw frames on the server, as a background job, and
`.../exotic/jobs/{job_id}/lightcurve.png` returns EXOTIC's own light curve and
fit. Nothing in that result is computed by our code: it only writes the
`inits.json` EXOTIC starts from. Our own quick-look stays available beside it.

All 22 archive sessions are loaded on the live server. Example:

```
https://exotransit-lab-api-production.up.railway.app/api/field/frame.png?session=CoRoT-2__2026-08-09
https://exotransit-lab-api-production.up.railway.app/api/field/lightcurve.png?session=CoRoT-2__2026-08-09
https://exotransit-lab-api-production.up.railway.app/api/field/summary?session=CoRoT-2__2026-08-09
```

---

## Running it yourself

**Backend, locally**

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
# http://127.0.0.1:8000/docs
python smoke_test.py          # exercises every route in-process
```

The backend reads precomputed results from `backend/data/` (committed) and raw frames
from `database/` (download separately, or point `OBS_ROOT` elsewhere).

**Pipeline, on Kaggle**

Upload `kaggle/notebooks/exotransit_lab_full.ipynb` with the packaged dataset built by
`kaggle/prepare_data.py`, run all, download `results.zip`, then
`python backend/sync_data.py` to refresh `backend/data/`.

**Deploy**

See [backend/README.md](backend/README.md): `railway up` for the code, and
`bundle_raw.py` plus `push_archive.py` to load the FITS archive onto the service's
volume.

---

## Data credit

MicroObservatory frames are public. Credit the Harvard-Smithsonian Center for
Astrophysics. Planet parameters used for validation come from the NASA Exoplanet Archive.
