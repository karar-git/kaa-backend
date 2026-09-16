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

from .. import fieldlab
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
            "lightcurve_json": f"/api/field/lightcurve{q}"}


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
    return {k: s[k] for k in ("session", "source", "target", "night", "n_frames",
                              "target_star", "comparison_stars", "rms_ppt",
                              "rms_target_only_ppt", "improvement_factor", "dip",
                              "dip_target_only", "reading", "warnings")} | {
        "normalisation": "norm_flux = target_flux / comp_flux_sum, divided by its "
                         "median; target_only_norm = target_flux / its median.",
        "points": fieldlab.lightcurve_points(an),
        "urls": _urls(session),
    }


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
