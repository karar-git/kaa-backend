"""Quick-look pictures for *any* session: the star field and its light curve.

The rest of the API reads precomputed pipeline tables for the 22 sessions the
pipeline was run on. These routes run a small, self-contained reduction on
request, so they also work for a folder of FITS frames nobody has processed
yet — a new night, a new star, or an upload from the dashboard.

A **session** here means what an observer means: one target, one night, one
folder of FITS frames. Point at it with any of:

* a pipeline session id, `TRES-3__2026-08-10`
* a folder under the observations root, `2026-08-10/TRES-3/session_01`
* an upload id returned by `POST /api/field/upload`, `upload:3f9c1a2b`

The first request for a session does the work (a few seconds for 70 frames);
everything after that is served from memory.
"""

from __future__ import annotations

import re
import shutil
import uuid
import zipfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import Response

import json

from .. import exotic, exotic_run, fieldlab
from ..config import settings

router = APIRouter(tags=["field"], prefix="/field")

SessionQ = Query(..., description="Pipeline session id, upload id, or folder under "
                                  "the observations root",
                 examples=["TRES-3__2026-08-10", "2026-08-10/TRES-3/session_01",
                           "upload:3f9c1a2b"])
CalQ = Query(None, description="Folder of dark frames, or `upload:<id>` for darks sent "
                               "with an upload. Default: darks uploaded with the session, "
                               "else the archive's darks from the same night, else the "
                               "nearest night within a week.",
             examples=["2026-08-10", "upload:3f9c1a2b"])
XQ = Query(None, description="Pick the target star nearest this x (column) "
                             "instead of the automatic rule. Needs y too.")
YQ = Query(None, description="Row of the target star; see x.")


def _analysis(session: str, calibration: str | None, x: float | None, y: float | None):
    if (x is None) != (y is None):
        raise HTTPException(422, "Give both x and y, or neither.")
    xy = (x, y) if x is not None else None
    try:
        return fieldlab.get_analysis(session, calibration, xy)
    except fieldlab.SessionNotFound as exc:
        raise HTTPException(404, str(exc))
    except fieldlab.SessionInvalid as exc:
        raise HTTPException(422, str(exc))


def _png(blob: bytes) -> Response:
    # Recomputed from the same files it would be identical, so cache, but not
    # for a month: a user may drop more frames into the folder tonight.
    return Response(content=blob, media_type="image/png",
                    headers={"Cache-Control": "public, max-age=300"})


def _urls(session: str) -> dict:
    q = f"?session={session}"
    return {"summary": f"/api/field/summary{q}",
            "stars": f"/api/field/stars{q}",
            "field_png": f"/api/field/frame.png{q}",
            "lightcurve_png": f"/api/field/lightcurve.png{q}",
            "lightcurve_json": f"/api/field/lightcurve{q}",
            "exotic_inits": f"/api/field/exotic/inits.json{q}",
            "exotic_prereduced": f"/api/field/exotic/prereduced.csv{q}",
            "aavso_report": f"/api/field/exotic/aavso.txt{q}",
            "exotic_bundle": f"/api/field/exotic/bundle.zip{q}"}


@router.get("/sessions", summary="Sessions this server can analyse live")
def sessions() -> dict:
    """Every night/target folder of raw FITS frames visible to the server, plus
    any uploads. The `session` value of each row is what the other routes take."""
    rows = fieldlab.list_raw_sessions()
    for r in rows:
        r["urls"] = _urls(r["session"])
    return {"observations_root": str(settings.OBS_ROOT),
            "upload_dir": str(settings.UPLOAD_DIR),
            "n": len(rows), "sessions": rows}


@router.get("/summary", summary="What was loaded and what was found")
def summary(session: str = SessionQ, calibration: str | None = CalQ,
            x: float | None = XQ, y: float | None = YQ) -> dict:
    """The console read-out of the reduction: which dark was used, how many
    frames loaded, the first timestamp, the pixel range, the stars found, the
    chosen target and comparison stars, the scatter, and any dip.

    `dip` is a *dip consistent with a transit*. Read `warnings` before believing
    it, and `/api/false-positive-cases` before repeating it.
    """
    an = _analysis(session, calibration, x, y)
    out = fieldlab.summary(an)
    out["urls"] = _urls(session)
    return out


@router.get("/stars", summary="Detected stars with their roles")
def stars(session: str = SessionQ, calibration: str | None = CalQ,
          x: float | None = XQ, y: float | None = YQ) -> dict:
    """Every star found on the first frame, brightest first, with the position
    the field picture rings. `role` is target, comparison or other."""
    an = _analysis(session, calibration, x, y)
    return {"session": session, "target": an.target, "frame_shape": list(an.shape),
            "coordinate_note": "x is the column and y the row of the FITS array, "
                               "zero-based, row 0 at the top of frame.png.",
            "target_rule": an.dip.get("target_rule"),
            "stars": [fieldlab.star_record(an, k) for k in range(len(an.stars))]}


@router.get("/frame.png", summary="Background-removed frame with stars ringed",
            response_class=Response,
            responses={200: {"content": {"image/png": {}},
                             "description": "PNG, about 800x600"}})
def frame_png(session: str = SessionQ, calibration: str | None = CalQ,
              frame: int | None = Query(None, ge=0,
                                        description="Zero-based frame index. Default: the "
                                                    "reference frame, where stars stand out best"),
              circles: bool = Query(True, description="Ring the detected stars"),
              labels: bool = Query(False, description="Number every ring"),
              x: float | None = XQ, y: float | None = YQ) -> Response:
    """The picture from the teammate's script, for any session: dark-subtracted,
    background-removed, grey-scaled, with a red ring on every detected star.
    The target is ringed green and the comparison stars blue.

    The grey levels are a display stretch of the residual signal, not counts.
    """
    an = _analysis(session, calibration, x, y)
    if frame is None:
        frame = an.reference_frame
    if frame >= an.n_frames:
        raise HTTPException(404, f"Frame {frame} is out of range; the session has {an.n_frames} frames.")
    key = ("frame", session, calibration, x, y, frame, circles, labels)
    return _png(fieldlab.cached_blob(key, lambda: fieldlab.render_field(an, frame, circles, labels)))


@router.get("/lightcurve.png", summary="Light curve plot",
            response_class=Response,
            responses={200: {"content": {"image/png": {}}}})
def lightcurve_png(session: str = SessionQ, calibration: str | None = CalQ,
                   mode: str = Query("differential",
                                     description="differential (target / comparison stars) "
                                                 "or target (target flux alone)"),
                   x: float | None = XQ, y: float | None = YQ) -> Response:
    """Relative brightness of the target against seconds since the first frame.

    `differential` is the one to look at: dividing by the comparison stars is
    what removes clouds and airmass. `target` is the raw curve, useful only to
    show *why* differential photometry is needed. A shaded band marks the dip
    the summary reports, if any.
    """
    if mode not in {"differential", "target"}:
        raise HTTPException(422, "mode must be 'differential' or 'target'.")
    an = _analysis(session, calibration, x, y)
    key = ("lc", session, calibration, x, y, mode)
    return _png(fieldlab.cached_blob(key, lambda: fieldlab.render_lightcurve(an, mode)))


@router.get("/lightcurve", summary="Light curve as numbers")
def lightcurve(session: str = SessionQ, calibration: str | None = CalQ,
               x: float | None = XQ, y: float | None = YQ) -> dict:
    """The same light curve as the PNG, one row per frame, so a dashboard can
    draw it itself and overlay the pipeline's version where one exists."""
    an = _analysis(session, calibration, x, y)
    s = fieldlab.summary(an)
    points = fieldlab.lightcurve_points(an)
    timing_note = None
    try:
        tm = exotic.timing(an)
        for p, b, a in zip(points, tm["bjd_tdb"], tm["airmass"]):
            p["bjd_tdb"] = round(float(b), 7)
            p["airmass"] = round(float(a), 4)
        timing_note = (f"bjd_tdb: UTC->TDB plus barycentric light travel for RA {tm['ra_deg']:.5f}, "
                       f"Dec {tm['dec_deg']:+.5f} ({tm['coord_source']}); airmass: {tm['airmass_source']}.")
    except Exception as exc:                      # no coordinates, no astropy...
        timing_note = f"bjd_tdb/airmass not available: {exc}"
    return {k: s[k] for k in ("session", "source", "target", "night", "n_frames",
                              "target_star", "comparison_stars", "rms_ppt",
                              "rms_target_only_ppt", "improvement_factor", "dip",
                              "dip_target_only", "reading", "warnings")} | {
        "normalisation": "norm_flux = target_flux / comp_flux_sum, divided by its "
                         "median; target_only_norm = target_flux / its median.",
        "timing": timing_note,
        "points": points,
        "urls": _urls(session),
    }


# -------------------------------------------------------------- EXOTIC

ObsCodeQ = Query("", description="Your AAVSO observer code, if you have one", max_length=16)
_FNAME = re.compile(r"[^A-Za-z0-9_.-]+")


def _attachment(body: bytes | str, media: str, filename: str) -> Response:
    if isinstance(body, str):
        body = body.encode("utf-8")
    return Response(content=body, media_type=media,
                    headers={"Content-Disposition": f'attachment; filename="{filename}"',
                             "Cache-Control": "public, max-age=300"})


def _exotic_ready(an):
    try:
        exotic.timing(an)
    except ValueError as exc:
        raise HTTPException(422, str(exc))


@router.get("/exotic/inits.json", summary="EXOTIC inits.json for a session",
            response_class=Response, responses={200: {"content": {"application/json": {}}}})
def exotic_inits(session: str = SessionQ, calibration: str | None = CalQ,
                 obscode: str = ObsCodeQ,
                 fits_dir: str | None = Query(None, description="Value for 'Directory with FITS files' "
                                                                "on your machine"),
                 darks_dir: str | None = Query(None, description="Value for 'Directory of Darks'"),
                 save_dir: str | None = Query(None, description="Value for 'Directory to Save Plots'"),
                 pixel_frame: int | None = Query(None, ge=0, description="Frame whose pixel coordinates "
                                                                        "to quote. Default: first frame "
                                                                        "with recovered stars"),
                 x: float | None = XQ, y: float | None = YQ) -> Response:
    """NASA's EXOTIC (Exoplanet Watch) initialisation file, filled in from this
    session: observatory, camera, binning, filter, the target and comparison-star
    pixel positions we found, and the planet's archive parameters.

    Save it next to the frames, edit the three directories, and run
    `exotic -red inits.json -nea`. Pixels are zero-based (x = column, y = row).
    """
    an = _analysis(session, calibration, x, y)
    _exotic_ready(an)
    d = exotic.inits(an, obscode=obscode, fits_dir=fits_dir, darks_dir=darks_dir,
                     save_dir=save_dir, pixel_frame=pixel_frame)
    return _attachment(json.dumps(d, indent=4), "application/json",
                       f"inits_{_FNAME.sub('_', session)}.json")


@router.get("/exotic/prereduced.csv", summary="Our light curve in EXOTIC's pre-reduced format",
            response_class=Response, responses={200: {"content": {"text/csv": {}}}})
def exotic_prereduced(session: str = SessionQ, calibration: str | None = CalQ,
                      x: float | None = XQ, y: float | None = YQ) -> Response:
    """Four comma-separated columns, BJD_TDB, flux, uncertainty, airmass, which is
    what EXOTIC's `-pre` mode reads. Lets EXOTIC fit its transit model to the
    photometry this API measured: `exotic -pre inits.json -nea`."""
    an = _analysis(session, calibration, x, y)
    _exotic_ready(an)
    return _attachment(exotic.prereduced(an), "text/csv",
                       f"prereduced_{_FNAME.sub('_', session)}.csv")


@router.get("/exotic/aavso.txt", summary="AAVSO Exoplanet Database report",
            response_class=Response, responses={200: {"content": {"text/plain": {}}}})
def exotic_aavso(session: str = SessionQ, calibration: str | None = CalQ,
                 obscode: str = ObsCodeQ,
                 secondary: str = Query("", description="Secondary observer codes", max_length=64),
                 notes: str = Query("", description="Prepended to #NOTES", max_length=500),
                 x: float | None = XQ, y: float | None = YQ) -> Response:
    """The `#TYPE=EXOPLANET` text format EXOTIC writes and the AAVSO accepts
    (webobs file upload). Priors are the archive parameters; a RESULTS line is
    included only when a dip was found, and is labelled as a dip search rather
    than a model fit. Add your observer code before submitting."""
    an = _analysis(session, calibration, x, y)
    _exotic_ready(an)
    return _attachment(exotic.aavso(an, obscode=obscode, secondary=secondary, notes=notes),
                       "text/plain", f"aavso_{_FNAME.sub('_', session)}.txt")


@router.get("/exotic/bundle.zip", summary="inits.json + pre-reduced curve + AAVSO report + README",
            response_class=Response, responses={200: {"content": {"application/zip": {}}}})
def exotic_bundle(session: str = SessionQ, calibration: str | None = CalQ,
                  obscode: str = ObsCodeQ,
                  fits_dir: str | None = Query(None), darks_dir: str | None = Query(None),
                  x: float | None = XQ, y: float | None = YQ) -> Response:
    """Everything needed to hand a session to EXOTIC, in one zip with a README
    that gives the exact commands."""
    an = _analysis(session, calibration, x, y)
    _exotic_ready(an)
    return _attachment(exotic.bundle(an, obscode=obscode, fits_dir=fits_dir, darks_dir=darks_dir),
                       "application/zip", f"exotic_{_FNAME.sub('_', session)}.zip")


# --------------------------------------------------------- EXOTIC itself

ModeQ = Query("nea", pattern="^(nea|ov)$",
              description="`nea`: EXOTIC takes the planet's parameters from the NASA Exoplanet "
                          "Archive (its default). `ov`: it uses the values in our inits.json.")
AlignQ = Query("wcs", pattern="^(wcs|exotic)$",
               description="`wcs`: each frame handed to EXOTIC carries a WCS built from the target "
                           "position our tracking measured, so EXOTIC finds the stars through it. "
                           "`exotic`: frames as they are; EXOTIC registers images itself (astroalign), "
                           "which fails on most MicroObservatory frames and leaves the run unusable.")
FramesQ = Query("kept", pattern="^(kept|all)$",
                description="`kept`: only frames in which our analysis recovered the stars "
                            "(EXOTIC aligns everything to the first frame, so a twilight "
                            "frame first would sink the run). `all`: every frame as is.")


def _job_response(job: dict, created: bool) -> Response:
    body = exotic_run.public(job)
    return Response(content=json.dumps(body, indent=2), media_type="application/json",
                    status_code=202 if created or job["status"] in exotic_run.LIVE else 200,
                    headers={"Cache-Control": "no-store"})


def _submit(session: str, calibration: str | None, x: float | None, y: float | None,
            mode: str, frames: str, align: str, obscode: str, force: bool) -> tuple[dict, bool]:
    if not exotic_run.available():
        raise HTTPException(503, "EXOTIC is not installed on this server (pip install exotic).")
    an = _analysis(session, calibration, x, y)
    _exotic_ready(an)
    if not an.frame_paths:
        raise HTTPException(422, "This session has no FITS files on disk for EXOTIC to read.")
    try:
        return exotic_run.submit(an, mode=mode, frames=frames, align=align, obscode=obscode, force=force)
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))


@router.post("/exotic/run", summary="Run EXOTIC itself on a session (background job)",
             status_code=202, responses={200: {"description": "A finished or running job already exists"},
                                         202: {"description": "Job queued"}})
def exotic_run_start(session: str = SessionQ, calibration: str | None = CalQ,
                     mode: str = ModeQ, frames: str = FramesQ, align: str = AlignQ, obscode: str = ObsCodeQ,
                     force: bool = Query(False, description="Re-run even if a finished job exists"),
                     x: float | None = XQ, y: float | None = YQ) -> Response:
    """Hands the session's raw frames and darks to NASA's EXOTIC and lets it do
    the whole reduction: calibration, alignment, photometry, limb darkening,
    the transit fit and its plot. Our code only writes the `inits.json` (the
    star positions it found, the planet's archive parameters) and picks the
    frames.

    Returns a job. Poll `urls.status` until `status` is `done`, then fetch
    `urls.lightcurve_png`, `urls.params_json`, `urls.aavso_txt`. The same
    session with the same options is one job: asking again returns it.
    Typical run: 3-10 minutes for 60 frames. Requires internet on the server
    (NASA Exoplanet Archive and limb-darkening models)."""
    job, created = _submit(session, calibration, x, y, mode, frames, align, obscode, force)
    return _job_response(job, created)


@router.get("/exotic/run", summary="Status of the EXOTIC job for a session (starts one if none)",
            responses={200: {"description": "Finished job"}, 202: {"description": "Queued or running"}})
def exotic_run_status(session: str = SessionQ, calibration: str | None = CalQ,
                      mode: str = ModeQ, frames: str = FramesQ, align: str = AlignQ, obscode: str = ObsCodeQ,
                      x: float | None = XQ, y: float | None = YQ) -> Response:
    """Same as POST without `force`: convenient for a browser or a dashboard
    that just wants the answer for a session."""
    job, created = _submit(session, calibration, x, y, mode, frames, align, obscode, False)
    return _job_response(job, created)


@router.get("/exotic/jobs", summary="Every EXOTIC job on this server")
def exotic_jobs() -> dict:
    jobs = [exotic_run.public(j) for j in exotic_run.list_jobs()]
    return {"exotic_installed": exotic_run.available(), "exotic_version": exotic_run.exotic_version(),
            "n": len(jobs), "jobs": jobs}


def _job(job_id: str) -> dict:
    try:
        return exotic_run.load(job_id)
    except exotic_run.JobNotFound:
        raise HTTPException(404, f"No EXOTIC job {job_id!r}.")


@router.get("/exotic/jobs/{job_id}", summary="EXOTIC job status, results and file links")
def exotic_job(job_id: str) -> dict:
    """`status` is queued, running, done or failed. While running, `progress`
    and `log_tail` show where EXOTIC is. When done, `results` is EXOTIC's
    FinalParams (mid-transit time, Rp/Rs, duration, their errors) and `urls`
    lists every file it wrote."""
    return exotic_run.public(_job(job_id))


def _artifact(job_id: str, key: str) -> Response:
    job = _job(job_id)
    try:
        blob, media, name = exotic_run.artifact(job_id, key)
    except exotic_run.NotReady:
        pub = exotic_run.public(job)
        raise HTTPException(409 if job["status"] in exotic_run.LIVE else 404,
                            {"message": f"{key} is not available: job is {job['status']}",
                             "job": pub})
    inline = media.startswith("image/") or media in ("text/plain", "application/json")
    disp = "inline" if inline else "attachment"
    return Response(content=blob, media_type=media,
                    headers={"Content-Disposition": f'{disp}; filename="{name}"',
                             "Cache-Control": "no-store" if job["status"] != "done" else "public, max-age=3600"})


@router.get("/exotic/jobs/{job_id}/lightcurve.png", summary="EXOTIC's final light-curve plot",
            response_class=Response, responses={200: {"content": {"image/png": {}}}})
def exotic_job_lightcurve(job_id: str) -> Response:
    return _artifact(job_id, "lightcurve_png")


@router.get("/exotic/jobs/{job_id}/fov.png", summary="EXOTIC's field-of-view image with target and comps",
            response_class=Response, responses={200: {"content": {"image/png": {}}}})
def exotic_job_fov(job_id: str) -> Response:
    return _artifact(job_id, "fov_png")


@router.get("/exotic/jobs/{job_id}/triangle.png", summary="EXOTIC's posterior corner plot",
            response_class=Response, responses={200: {"content": {"image/png": {}}}})
def exotic_job_triangle(job_id: str) -> Response:
    return _artifact(job_id, "triangle_png")


@router.get("/exotic/jobs/{job_id}/params.json", summary="EXOTIC's FinalParams (fitted transit)",
            response_class=Response, responses={200: {"content": {"application/json": {}}}})
def exotic_job_params(job_id: str) -> Response:
    return _artifact(job_id, "params_json")


@router.get("/exotic/jobs/{job_id}/lightcurve.csv", summary="EXOTIC's final time series",
            response_class=Response, responses={200: {"content": {"text/csv": {}}}})
def exotic_job_csv(job_id: str) -> Response:
    return _artifact(job_id, "lightcurve_csv")


@router.get("/exotic/jobs/{job_id}/aavso.txt", summary="AAVSO report written by EXOTIC",
            response_class=Response, responses={200: {"content": {"text/plain": {}}}})
def exotic_job_aavso(job_id: str) -> Response:
    return _artifact(job_id, "aavso_txt")


@router.get("/exotic/jobs/{job_id}/log.txt", summary="EXOTIC's console output for the job",
            response_class=Response, responses={200: {"content": {"text/plain": {}}}})
def exotic_job_log(job_id: str) -> Response:
    return _artifact(job_id, "log")


@router.get("/exotic/jobs/{job_id}/inits.json", summary="The inits.json the job was run with",
            response_class=Response, responses={200: {"content": {"application/json": {}}}})
def exotic_job_inits(job_id: str) -> Response:
    return _artifact(job_id, "inits")


@router.get("/exotic/jobs/{job_id}/results.zip", summary="Everything EXOTIC wrote, zipped",
            response_class=Response, responses={200: {"content": {"application/zip": {}}}})
def exotic_job_zip(job_id: str) -> Response:
    return _artifact(job_id, "results")


@router.get("/exotic/jobs/{job_id}/{name}", summary="Any other EXOTIC output by key",
            response_class=Response, include_in_schema=False)
def exotic_job_any(job_id: str, name: str) -> Response:
    stem, _, ext = name.rpartition(".")
    key = f"{stem}_{ext}" if stem else name
    if key not in exotic_run.ARTIFACTS:
        raise HTTPException(404, f"Unknown file {name!r}; see the job's urls.")
    return _artifact(job_id, key)


# ------------------------------------------------------------------- upload

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _save_fits(dst: Path, upload: UploadFile, budget: list[int]) -> int:
    """Write one upload (a FITS file or a zip of them) into dst. Returns the
    number of FITS files written. `budget[0]` is the remaining byte allowance."""
    name = _SAFE.sub("_", Path(upload.filename or "file").name)
    raw = upload.file.read()
    budget[0] -= len(raw)
    if budget[0] < 0:
        raise HTTPException(413, f"Upload exceeds {settings.MAX_UPLOAD_MB} MB in total.")
    dst.mkdir(parents=True, exist_ok=True)
    if name.lower().endswith(".zip"):
        import io
        n = 0
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            for info in z.infolist():
                if info.is_dir() or not fieldlab.is_fits_name(info.filename):
                    continue
                budget[0] -= info.file_size
                if budget[0] < 0:
                    raise HTTPException(413, f"Zip contents exceed {settings.MAX_UPLOAD_MB} MB.")
                (dst / _SAFE.sub("_", Path(info.filename).name)).write_bytes(z.read(info))
                n += 1
        return n
    if not fieldlab.is_fits_name(name):
        raise HTTPException(422, f"'{name}' is not a FITS file (.fits/.fit/.fts, optionally "
                                 ".fz or .gz) or a zip.")
    (dst / name).write_bytes(raw)
    return 1


@router.post("/upload", summary="Upload a session of FITS frames", status_code=201)
def upload(frames: list[UploadFile] = File(..., description="Science frames: FITS files, "
                                                             "or one zip of them"),
           darks: list[UploadFile] = File([], description="Optional dark frames, FITS or zip")
           ) -> dict:
    """Send a night's frames and get back a session id you can pass to every
    other `/api/field` route as `?session=upload:<id>`.

    Frames are kept on this server's disk under the upload directory; nothing
    is written anywhere else. Send darks too if you have them — without them the
    reduction still runs, but the dark current is only partly removed.
    """
    sid = uuid.uuid4().hex[:8]
    root = settings.UPLOAD_DIR / sid
    budget = [settings.MAX_UPLOAD_MB * 1024 * 1024]
    try:
        n_frames = sum(_save_fits(root / "frames", f, budget) for f in frames)
        n_darks = sum(_save_fits(root / "darks", f, budget) for f in darks)
        if n_frames < 2:
            raise HTTPException(422, "At least two science frames are needed for a light curve.")
        # Validate now so the client learns about a bad file here, not on the
        # first picture request.
        for p in fieldlab.list_fits(root / "frames")[:3]:
            fieldlab.read_fits(p)
    except HTTPException:
        shutil.rmtree(root, ignore_errors=True)
        raise
    except Exception as exc:
        shutil.rmtree(root, ignore_errors=True)
        raise HTTPException(422, f"Could not read the uploaded FITS: {exc}")
    session = f"upload:{sid}"
    return {"session": session, "n_frames": n_frames, "n_darks": n_darks,
            "calibration": ("uploaded darks are used automatically" if n_darks else
                            "no darks uploaded: the server's archive is searched by night, "
                            "else hot pixels are taken from the frames themselves"),
            "urls": _urls(session)}


@router.post("/ingest", summary="File frames into the archive (operator only)", status_code=201)
def ingest(frames: list[UploadFile] = File([], description="Science frames of ONE target on "
                                                            "ONE night: FITS files or one zip"),
           darks: list[UploadFile] = File([], description="Dark frames of one night, FITS or zip"),
           replace: bool = Query(False, description="Overwrite a session already in the archive")
           ) -> dict:
    """Put frames into the server's archive so they answer to a pipeline-style
    session id (`TARGET__NIGHT`) rather than an upload id.

    Target and night are read from the first frame's header (`OBJECT` and the
    UTC date of `UT-OBS`), so the archive layout is always right. Darks are filed
    by their own UTC date.

    **Closed by default.** Answers 403 unless the service runs with
    `INGEST_OPEN=true`; set that only while loading data, then unset it.
    """
    if not settings.INGEST_OPEN:
        raise HTTPException(403, "Archive ingest is closed. Set INGEST_OPEN=true on the "
                                 "service while loading data, then remove it.")
    if not frames and not darks:
        raise HTTPException(422, "Send frames, darks, or both.")
    tmp = settings.UPLOAD_DIR / f"_ingest_{uuid.uuid4().hex[:8]}"
    budget = [settings.MAX_UPLOAD_MB * 1024 * 1024]
    out: dict = {"filed": []}
    try:
        n_f = sum(_save_fits(tmp / "frames", f, budget) for f in frames)
        n_d = sum(_save_fits(tmp / "darks", f, budget) for f in darks)
        if n_f:
            paths = fieldlab.list_fits(tmp / "frames")
            _, hdr = fieldlab.read_fits(paths[0])
            target = str(hdr.get("OBJECT") or "").strip().replace(" ", "")
            mjd = fieldlab._frame_mjd(hdr, paths[0])
            if not target or mjd is None:
                raise HTTPException(422, "Frames need an OBJECT and a UT-OBS/DATE-OBS header "
                                         "to be filed by target and night.")
            night = fieldlab._mjd_to_iso(mjd)[:10]
            dest = settings.OBS_ROOT / "observations" / night / target / "session_01"
            if dest.exists() and fieldlab.list_fits(dest):
                if not replace:
                    raise HTTPException(409, f"{target}__{night} is already in the archive; "
                                             "pass replace=true to overwrite it.")
                shutil.rmtree(dest)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(tmp / "frames"), str(dest))
            sid = f"{target}__{night}"
            out["filed"].append({"session": sid, "n_frames": n_f,
                                 "path": fieldlab.display_path(dest), "urls": _urls(sid)})
            fieldlab._cache.clear()
            fieldlab._blob_cache.clear()
        if n_d:
            by_night: dict[str, list[Path]] = {}
            for p in fieldlab.list_fits(tmp / "darks"):
                _, hdr = fieldlab.read_fits(p)
                mjd = fieldlab._frame_mjd(hdr, p)
                if mjd is None:
                    raise HTTPException(422, f"Dark {p.name} has no usable time header.")
                by_night.setdefault(fieldlab._mjd_to_iso(mjd)[:10], []).append(p)
            for night, ps in by_night.items():
                dest = settings.OBS_ROOT / "calibration" / night
                dest.mkdir(parents=True, exist_ok=True)
                for p in ps:
                    shutil.move(str(p), str(dest / p.name))
                out["filed"].append({"calibration": night, "n_darks": len(ps),
                                     "path": fieldlab.display_path(dest)})
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(422, f"Could not ingest: {exc}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return out
