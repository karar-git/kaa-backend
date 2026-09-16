# ExoTransit Lab — API

FastAPI service behind the Challenge E discovery tool. It reads the precomputed
output of the analysis pipeline and serves it as JSON and image tiles.

**Interactive docs:** `/docs` (Swagger UI) · `/redoc` · `/openapi.json`

---

## The one architectural rule

**The API does no science.** Every number it returns was computed offline in
`../kaggle/src/nb01_pipeline.py` and written to `results/*.csv`. This process
reads files and shapes JSON.

That is deliberate. The analysis has to be reproducible from a notebook a judge
can read top to bottom, not buried inside a web server. It also means the API
starts in milliseconds, never touches a 1.1 GB FITS archive at request time, and
cannot accidentally produce a different answer than the notebook did.

```
FITS archive (1.1 GB)
        │
        ▼   kaggle/prepare_data.py          once, locally
   packaged .npy + metadata CSVs
        │
        ▼   kaggle/notebooks/01_pipeline    once, on Kaggle
   results/*.csv  +  static/*.png           ~63 MB + tiles
        │
        ▼   backend/sync_data.py            re-encodes PNG → WebP
   backend/data/
        │
        ▼   this API                        reads only
   JSON + image tiles
```

---

## Quick start

```bash
pip install -r requirements.txt
python sync_data.py --no-images     # pulls ../kaggle/out/results into ./data
uvicorn app.main:app --reload
```

Open <http://localhost:8000/docs>.

Verify everything before deploying:

```bash
python smoke_test.py
```

It drives every route in-process through `TestClient`, so it needs no server and
finishes in about a second. A `503` counts as **skip**, not failure — that is the
documented behaviour when a results file has not been generated.

---

## Endpoints

### kpis
| Route | What it gives you |
|---|---|
| `GET /api/kpis` | Every headline number in one call: dataset size, usable nights, best precision, search hits vs chance, nearest/farthest/fastest planet, validation pass rate, AI status |

### physics
| Route | What it gives you |
|---|---|
| `GET /api/planets` | Per planet: orbit size (AU), orbital speed (km/s), period, temperature, starlight received, distance from Earth (ly, pc), all with uncertainties. `?sort=orbital_speed_kms&descending=true` |
| `GET /api/planets/{target}` | One planet in full, including the radius from **our** measured depths and a one-sentence summary. Accepts `HATP-10`, `WASP-11` or `WASP-11 b` |
| `GET /api/planets/validation` | Each derived value against the NASA Exoplanet Archive: % difference, z-score, status. `?uses_our_data=true` for checks on our own measurements |
| `GET /api/planets/equations` | Equations, constants, uncertainty method, external-data disclosure |

Distances and speeds are derived from archive inputs, not measured by our
telescope. Regenerate these files without re-running photometry with
`python kaggle/run_physics.py`.

### meta
| Route | What it gives you |
|---|---|
| `GET /api/health` | Liveness, an inventory of which result files are present, and whether the times are genuinely barycentric |
| `GET /api/meta` | Dataset counts, date range, instrument description |
| `GET /api/calibration` | Master dark summary, hot pixel count, pattern stability |
| `GET /api/calibration/noise-curve` | Measured noise vs number of stacked darks, against the 1/√N prediction |
| `GET /api/calibration/dark-masters` | Per-temperature masters |
| `GET /api/calibration/hot-pixels` | Hot pixel stats and the independent cross-check |
| `GET /api/false-positive-cases` | All 11 ways a dip can fool you, each with its test |
| `GET /api/labels` | Class counts and the rule behind every label |

### sessions
| Route | What it gives you |
|---|---|
| `GET /api/sessions` | All 22 runs. Filter by `?target=` or `?quality=` |
| `GET /api/sessions/{id}` | One session |
| `GET /api/sessions/{id}/frames` | Per-frame telemetry and image statistics |
| `GET /api/sessions/{id}/motion` | Field drift per frame — the evidence against a tracking slip |
| `GET /api/sessions/{id}/tracks` | Objects followed across the night |
| `GET /api/sessions/{id}/detections` | Raw blobs. Pass `?frame=N` for one image's overlay |
| `GET /api/targets` | Per-target usability |

### science
| Route | What it gives you |
|---|---|
| `GET /api/sessions/{id}/lightcurve` | The differential light curve, with its predicted transit window |
| `GET /api/science/photometry` | How well each session was measured |
| `GET /api/science/ephemerides` | Published periods and T0, with source disclosure |
| `GET /api/science/predictions` | Where a transit should fall in each session |
| `GET /api/science/depths` | Measured in-window depth and significance |
| `GET /api/science/phasefold/{target}` | All nights folded on the published period |
| `GET /api/science/search` | Every star tested at the published phase, against a null built from its own field |
| `GET /api/science/field-photometry/{id}` | Light curves for every star in the field |
| `GET /api/science/field-stars/{id}` | Positions, brightness, achieved scatter |

### quality
`GET /api/quality/correlations` · `/noise-floor` · `/sessions`

### images
| Route | What it gives you |
|---|---|
| `GET /api/images/{id}/frame/{n}?stretch=zscale\|asinh\|linear` | Rendered tile |
| `GET /api/images/{id}/values/{n}` | Real ADU counts for the hover readout |
| `GET /api/images/{id}/manifest` | Which tiles exist for a session |
| `GET /api/images/triptych/{kind}` | raw · master_dark · calibrated · hot_pixel_map |

### field — live quick-look for *any* session
The only routes that compute on request. Point them at a folder of FITS frames
and get the two pictures an observer wants first: the star field with every
detection ringed, and the light curve. `session` is a pipeline id
(`TRES-3__2026-08-10`), a folder under the observations root
(`2026-08-10/TRES-3/session_01`), or an upload id (`upload:3f9c1a2b`).
Anything outside `OBS_ROOT`, `UPLOAD_DIR` and `SESSION_ROOTS` is refused.

| Route | What it gives you |
|---|---|
| `GET /api/field/sessions` | Every raw session the server can see, each with ready-made URLs |
| `GET /api/field/frame.png?session=…` | Dark-subtracted, background-removed frame, every detected star ringed: target green, comparison stars blue, others red. `&frame=N`, `&labels=true` |
| `GET /api/field/lightcurve.png?session=…` | Differential light curve vs seconds since start, dip shaded. `&mode=target` for the raw target curve |
| `GET /api/field/lightcurve?session=…` | The same curve as numbers, one row per frame |
| `GET /api/field/summary?session=…` | What was loaded: dark used, frames, timestamps, pixel range, stars, target rule, scatter, dip, and a one-line `reading` |
| `GET /api/field/stars?session=…` | Detected stars with positions and roles |
| `POST /api/field/upload` | Multipart `frames` (FITS files or one zip) and optional `darks`; returns an upload id |
| `GET /api/field/exotic/inits.json?session=…` | **EXOTIC** initialisation file filled in from the analysis: observatory, binning, filter, target and comparison pixels, archive planet parameters. `&obscode=`, `&fits_dir=`, `&darks_dir=` |
| `GET /api/field/exotic/prereduced.csv?session=…` | Our light curve as EXOTIC's `-pre` input: BJD_TDB, flux, uncertainty, airmass |
| `GET /api/field/exotic/aavso.txt?session=…` | AAVSO Exoplanet Database report in the layout EXOTIC writes. `&obscode=` |
| `GET /api/field/exotic/bundle.zip?session=…` | All three plus a README with the exact `exotic` commands |
| `POST /api/field/exotic/run?session=…` | **Run EXOTIC itself** on the session's raw frames as a background job; returns a `job_id`. Same session + options = same job. `&mode=nea|ov`, `&frames=kept|all`, `&force=true` |
| `GET /api/field/exotic/run?session=…` | Status of that job, starting one if none exists (202 while queued/running, 200 when done) |
| `GET /api/field/exotic/jobs` · `/jobs/{job_id}` | All jobs; one job's status, `progress`, `log_tail`, EXOTIC's fitted `results` and file `urls` |
| `GET /api/field/exotic/jobs/{job_id}/lightcurve.png` | EXOTIC's own final light-curve plot. Also `fov.png`, `triangle.png`, `params.json`, `lightcurve.csv`, `aavso.txt`, `log.txt`, `inits.json`, `results.zip` |

**EXOTIC compatibility.** [EXOTIC](https://github.com/rzellem/EXOTIC) is NASA
JPL's Exoplanet Watch reduction code, and its own sample data is a night of
MicroObservatory frames, so it reads this archive's FITS as-is (the raw files are
kept unaltered for that reason). The routes above go further: they let EXOTIC
either re-reduce a session from the frames using the stars we found
(`exotic -red -i inits.json -nea`) or fit its transit model to our photometry
(`exotic -pre -i inits.json -nea`), and they produce the AAVSO submission file.
Pixel positions are zero-based (x = column, y = row) in the first frame with
recovered stars; twilight frames to remove first are listed in the file. Times
are BJD_TDB computed as in the pipeline; airmass is sec(z) from the header
altitude. `/api/field/lightcurve` carries `bjd_tdb` and `airmass` per point too.

**Running EXOTIC on the server.** `POST /api/field/exotic/run?session=…` does
not convert anything of ours: it hands the raw frames and the night's darks to
the real `exotic` package and lets it do the whole reduction (calibration,
alignment, PSF and aperture photometry, limb darkening from PHOENIX models,
nested-sampling transit fit, plot, AAVSO file). Our code only writes the
`inits.json` it starts from (the star pixels we found, the archive planet
parameters) and, by default, leaves out the frames in which no stars were
recoverable, because EXOTIC aligns every image to the first one. A run takes
minutes, so it is a job: poll `urls.status`, then fetch `urls.lightcurve_png`
and `urls.params_json`. Jobs are stored under `EXOTIC_DIR` and are never
computed twice for the same frames and options. The server needs internet for
the NASA Exoplanet Archive lookup and the limb-darkening models.

Optional on every GET: `calibration=<folder of darks>` (default: darks uploaded
with the session, else the archive's darks from the same night, else the nearest
night within a week) and `x=&y=` to pick the target star by position.

**Shipping the raw frames.** The archive is too large for `railway up` (the
CLI upload is capped; 480 MB was refused), so it is *not* in the image. The
service has a Railway volume mounted at `/app/data/raw` (`OBS_ROOT`), and the
archive is loaded into it once through `POST /api/field/ingest`, which files
frames by target and night from their headers. Locally:

```bash
python bundle_raw.py                 # database/ -> data/raw as .fits.fz, 1 GB -> 330 MB
railway variables -s exotransit-lab-api --set INGEST_OPEN=true
python push_archive.py https://exotransit-lab-api-production.up.railway.app
railway variables -s exotransit-lab-api --set INGEST_OPEN=false
```

`data/raw/` is listed in `.railwayignore` and `.dockerignore` on purpose. The
ingest route answers 403 unless `INGEST_OPEN=true`; leave it off except while
loading. On Windows run the `railway` commands from PowerShell: Git Bash rewrites
`/app/...` paths into `C:/Program Files/Git/app/...`.

The first call for a session runs the reduction (about 20 s for 70 frames);
later calls are served from memory. The reference frame is the one where stars
stand out best, not frame 0, because first frames are often twilight. Frames in
which fewer than 30% of the reference stars can be recovered are dropped and
listed in `lost_frames`. Read `warnings` and `reading` before quoting a dip: a
30% drop that vanishes when divided by the comparison stars is cloud, not a
planet, and `reading` says so.

### ml
| Route | What it gives you |
|---|---|
| `GET /api/model/info` | Model card, label provenance, validation protocol |
| `GET /api/model/metrics` | Per-class precision and recall |
| `POST /api/classify` | Classify one 32×32 cutout |
| `POST /api/explain` | Plain-language session summary, grounded on measured numbers |

---

## Images and deploy size

The pipeline renders lossless PNGs: 1,681 frames × 3 stretches is about 1.5 GB,
far too much to bake into a container.

`sync_data.py` re-encodes to WebP at quality 82 and ships only the sessions that
passed quality triage, in two stretches:

```bash
python sync_data.py                    # 15 good sessions, zscale + asinh
python sync_data.py --stretches zscale # half the size
python sync_data.py --all-sessions     # everything
python sync_data.py --no-images        # CSVs only, ~63 MB
```

Anything not shipped still works: if the packaged `.npy` cubes are present the
API renders that frame on demand and caches the result under `static/_cache/`.

---

## Deploying to Railway

```bash
npm i -g @railway/cli
railway login
railway init
railway up
railway domain          # prints the public URL
```

`railway.toml` selects the Dockerfile builder and points the healthcheck at
`/api/health`, which reports the data inventory — so a deploy that built fine but
shipped no data fails the check instead of silently 503-ing later.

### Environment variables

| Variable | Required | Purpose |
|---|---|---|
| `PORT` | injected by Railway | Listen port |
| `DATA_DIR` | no | Defaults to `/app/data` |
| `CORS_ORIGINS` | no | Comma-separated origins. Narrow this to your dashboard domain in production |
| `OPENROUTER_API_KEY` | no | Enables `POST /api/explain` only |
| `OPENROUTER_MODEL` | no | Default `google/gemini-2.5-flash` |
| `OBS_ROOT` | no | Raw FITS archive for `/api/field`, laid out `observations/<night>/<target>/…` and `calibration/<night>/…`. Defaults to the repo's `database/` folder |
| `SESSION_ROOTS` | no | Extra folders `/api/field` may read, separated by the OS path separator |
| `UPLOAD_DIR` | no | Where `POST /api/field/upload` keeps frames. Default `DATA_DIR/uploads` |
| `MAX_UPLOAD_MB` | no | Total size cap per upload, default 400 |
| `FIELD_MAX_FRAMES` | no | Most frames one live session may hold, default 400 |
| `EXOTIC_DIR` | no | Where EXOTIC jobs and their outputs live. Default `DATA_DIR/exotic`; on Railway set it on the volume |
| `EXOTIC_TIMEOUT_S` | no | Kill an EXOTIC run after this many seconds, default 5400 |

**The API key is read from the environment at call time and is never logged,
never returned, and never accepted from a client.** If it is unset, `/api/explain`
returns 503 and every other route works normally. Do not put a key in
`railway.toml`, the Dockerfile, or any committed file — set it in the Railway
dashboard under the service's Variables tab.

---

## Notes for whoever builds the dashboard

- **`GET /api/health` first.** It tells you which routes will have data, and
  whether `timing.barycentric_correction_applied` is true. If it is not, every
  phase and predicted mid-transit on the site is out by up to eight minutes and
  you should say so on screen rather than plot it silently.
- **Read `quality` before plotting anything.** Six of the 22 sessions are
  unusable and three targets have no good night at all. Show that, do not hide it.
- **`improvement_factor` is the trust signal on a light curve.** If it is not
  comfortably above 1, the comparison stars are not working that night.
- **In `/api/science/search`, read `field_p_value`, and compare `n_hits` against
  `n_expected_by_chance` before believing any of it.**
- **Rendered tiles are not data.** For pixel values use `/values/{n}`.
- Long responses are gzipped and image tiles carry a 30-day immutable
  `Cache-Control`, so the frame stepper can run at full speed.
