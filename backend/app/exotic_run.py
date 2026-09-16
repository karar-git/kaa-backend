"""Run NASA's EXOTIC itself on any session and serve what it produces.

`exotic.py` in this package *exports* our own photometry in EXOTIC's file
formats. This module is the other direction: it hands the raw frames to the
real EXOTIC (`exotic -red inits.json -nea`) and returns EXOTIC's own light
curve, fitted parameters and AAVSO report. Nothing in the reduction is ours:
EXOTIC does the dark subtraction, alignment, PSF/aperture photometry, limb
darkening (LDTK), the nested-sampling transit fit and the plot. Our code only
prepares the folder EXOTIC expects and points it at the stars it should use.

A run takes minutes (EXOTIC fits ~380 aperture/annulus combinations, then a
nested-sampling model), so it is a background **job**:

    POST /api/field/exotic/run?session=TRES-3__2026-08-10   -> {job_id, status}
    GET  /api/field/exotic/jobs/{job_id}                     -> status + urls
    GET  /api/field/exotic/jobs/{job_id}/lightcurve.png      -> EXOTIC's plot

Jobs live on disk under EXOTIC_DIR so results survive a restart and are never
computed twice for the same frames and options. One EXOTIC process runs at a
time (they are CPU bound); the rest queue.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from . import exotic, fieldlab
from .config import settings

RUNS_DIR: Path = settings.EXOTIC_DIR
STATUSES = ("queued", "running", "done", "failed")
_LOG_TAIL = 20
_PROGRESS = re.compile(r"Finding transformation (\d+) of (\d+)")
_SPINNER = re.compile(r"(Thinking [|/\\-] \.\.\. ?)+")

# What a finished run exposes, in the order EXOTIC writes them. `glob` is
# relative to EXOTIC's save directory.
ARTIFACTS = {
    "lightcurve_png": ("FinalLightCurve_*.png", "image/png"),
    "lightcurve_csv": ("temp/FinalLightCurve_*.csv", "text/csv"),
    "params_json": ("temp/FinalParams_*.json", "application/json"),
    "aavso_txt": ("AAVSO_*.txt", "text/plain"),
    "fov_png": ("temp/FOV_*.png", "image/png"),
    "triangle_png": ("temp/Triangle_*.png", "image/png"),
    "centroids_pdf": ("temp/CentroidPositions&Distances_*.pdf", "application/pdf"),
    "raw_flux_pdf": ("temp/TargetRawFlux_*.pdf", "application/pdf"),
    "comp_flux_pdf": ("temp/CompRawFlux_*.pdf", "application/pdf"),
    "normalized_pdf": ("temp/NormalizedFluxTime_*.pdf", "application/pdf"),
    "plate_status_csv": ("temp/PlateStatus_*.csv", "text/csv"),
}


class JobNotFound(KeyError):
    pass


class NotReady(RuntimeError):
    """The job exists but has not produced this file (yet, or at all)."""

    def __init__(self, job: dict, what: str):
        super().__init__(what)
        self.job = job


def available() -> bool:
    """True when the `exotic` package is importable by this interpreter."""
    return importlib.util.find_spec("exotic") is not None


def exotic_version() -> str | None:
    try:
        from exotic.version import __version__
        return __version__
    except Exception:
        return None


# ------------------------------------------------------------------ job ids

def job_id(an: fieldlab.SessionAnalysis, options: dict) -> str:
    """Stable id for (these frames, these options): the same night asked twice
    is one job. The frame list carries the folder's fingerprint (names, count)."""
    h = hashlib.sha1()
    h.update(an.ref.encode())
    for p in an.frame_paths:
        h.update(Path(p).name.encode())
    h.update(json.dumps(options, sort_keys=True).encode())
    return h.hexdigest()[:12]


def _job_path(jid: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{12}", jid or ""):
        raise JobNotFound(jid)
    return RUNS_DIR / jid


def _write(job: dict) -> None:
    d = _job_path(job["job_id"])
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / "job.json.tmp"
    tmp.write_text(json.dumps(job, indent=2))
    tmp.replace(d / "job.json")


def load(jid: str) -> dict:
    p = _job_path(jid) / "job.json"
    if not p.is_file():
        raise JobNotFound(jid)
    return json.loads(p.read_text())


def list_jobs() -> list[dict]:
    out = []
    if RUNS_DIR.is_dir():
        for d in RUNS_DIR.iterdir():
            f = d / "job.json"
            if f.is_file():
                try:
                    out.append(json.loads(f.read_text()))
                except Exception:
                    continue
    out.sort(key=lambda j: j.get("created_utc", ""), reverse=True)
    return out


def latest_for(session: str) -> dict | None:
    """Most recent job for a session, preferring finished ones."""
    jobs = [j for j in list_jobs() if j.get("session") == session]
    done = [j for j in jobs if j["status"] == "done"]
    live = [j for j in jobs if j["status"] in ("queued", "running")]
    return (done or live or jobs or [None])[0]


# ----------------------------------------------------------------- prepare

def _link_or_copy(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)             # same volume: free and instant
    except OSError:
        shutil.copyfile(src, dst)


def _prepare(an: fieldlab.SessionAnalysis, job: dict) -> None:
    """Lay out the folder EXOTIC expects: frames/, darks/, save/, inits.json."""
    d = _job_path(job["job_id"])
    frames, darks, save = d / "frames", d / "darks", d / "save"
    for sub in (frames, darks, save):
        shutil.rmtree(sub, ignore_errors=True)
        sub.mkdir(parents=True)
    opts = job["options"]
    skip = set(an.lost_frames) if opts["frames"] == "kept" else set()
    kept = 0
    for i, p in enumerate(an.frame_paths):
        if i in skip:
            continue
        _link_or_copy(Path(p), frames / Path(p).name)
        kept += 1
    n_darks = 0
    if an.calibration_dir:
        for p in sorted(Path(an.calibration_dir).iterdir()):
            if p.is_file() and fieldlab.is_fits_name(p.name):
                _link_or_copy(p, darks / p.name)
                n_darks += 1
    ini = exotic.inits(an, obscode=opts.get("obscode", ""), fits_dir=str(frames),
                       darks_dir=str(darks) if n_darks else None, save_dir=str(save),
                       pixel_frame=None)
    ini["inits_guide"]["Comment"] = (f"Written by ExoTransit Lab API for EXOTIC job {job['job_id']} "
                                     f"(session {an.ref}); paths are inside the job folder.")
    if opts["mode"] == "ov":
        # -ov takes every planetary parameter from this file and EXOTIC turns a
        # null into 0, which breaks the limb-darkening grid. Fill the two the
        # archive tables do not carry with solar-type defaults and say so.
        pp = ini["planetary_parameters"]
        filled = []
        for key, val in (("Star Surface Gravity (log(g))", 4.5), ("Star Metallicity ([FE/H])", 0.0)):
            if pp.get(key) is None:
                pp[key] = val
                filled.append(key)
        for key in list(pp):
            if "Uncertainty" in key and pp[key] is None:
                pp[key] = 0.01 if "Ratio" in key else (100 if "Temperature" in key else 0.1)
        if filled:
            ini["exotransit_lab"]["override_defaults"] = filled
    (d / "inits.json").write_text(json.dumps(ini, indent=2))
    job.update(n_frames_given=kept, n_frames_total=an.n_frames, n_darks=n_darks,
               target_pixel=json.loads(ini["user_info"]["Target Star X & Y Pixel"]),
               comparison_pixels=[c for c in json.loads(ini["user_info"]["Comparison Star(s) X & Y Pixel"]) if c],
               planet=ini["planetary_parameters"]["Planet Name"])


# --------------------------------------------------------------------- run

def _command(job: dict) -> list[str]:
    mode = "-ov" if job["options"]["mode"] == "ov" else "-nea"
    return [sys.executable, "-m", "exotic.exotic", "-red", "inits.json", mode]


def _env() -> dict:
    env = dict(os.environ)
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("PYTHONUNBUFFERED", "1")
    # LDTK downloads PHOENIX limb-darkening spectra on first use; keep them
    # next to the jobs so a redeploy does not fetch them again.
    env.setdefault("LDTK_ROOT", str(RUNS_DIR / "_ldtk"))
    return env


def _collect(job: dict) -> None:
    save = _job_path(job["job_id"]) / "save"
    files = {}
    for key, (pattern, media) in ARTIFACTS.items():
        hits = sorted(save.glob(pattern))
        if hits:
            files[key] = {"path": str(hits[0].relative_to(save)).replace("\\", "/"),
                          "bytes": hits[0].stat().st_size, "media": media}
    job["files"] = files
    params = files.get("params_json")
    if params:
        try:
            job["results"] = json.loads((save / params["path"]).read_text())
        except Exception as exc:                       # keep the file, note the parse
            job["results_error"] = str(exc)


def _tail(path: Path, n: int = _LOG_TAIL) -> list[str]:
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return []
    # EXOTIC prints a spinner ("Thinking | ...") while it downloads catalogues;
    # it is noise in a status response.
    clean = [_SPINNER.sub("", ln).rstrip() for ln in lines]
    return [ln for ln in clean if ln][-n:]


def _run(job: dict) -> None:
    d = _job_path(job["job_id"])
    log = d / "log.txt"
    job.update(status="running", started_utc=_now())
    _write(job)
    with log.open("w", encoding="utf-8") as fh:
        fh.write(f"$ {' '.join(_command(job))}\n")
        fh.flush()
        try:
            proc = subprocess.Popen(_command(job), cwd=d, stdin=subprocess.DEVNULL,
                                    stdout=fh, stderr=subprocess.STDOUT, env=_env())
        except OSError as exc:
            job.update(status="failed", error=f"could not start EXOTIC: {exc}", finished_utc=_now())
            _write(job)
            return
        deadline = time.time() + settings.EXOTIC_TIMEOUT_S
        while proc.poll() is None:
            time.sleep(2)
            if time.time() > deadline:
                proc.kill()
                job.update(status="failed", error=f"EXOTIC exceeded {settings.EXOTIC_TIMEOUT_S} s",
                           finished_utc=_now())
                _write(job)
                return
        rc = proc.returncode
    _collect(job)
    ok = rc == 0 and "lightcurve_png" in job["files"]
    job.update(status="done" if ok else "failed", returncode=rc, finished_utc=_now())
    if not ok:
        job["error"] = (f"EXOTIC exited with code {rc}" if rc else
                        "EXOTIC finished without writing FinalLightCurve; see log")
    job["log_tail"] = _tail(log)
    # The frames were links or copies of the archive; the results are what we keep.
    shutil.rmtree(d / "frames", ignore_errors=True)
    shutil.rmtree(d / "darks", ignore_errors=True)
    _write(job)


# ------------------------------------------------------------------ worker

_queue: "queue.Queue[str]" = queue.Queue()
_worker: threading.Thread | None = None
_lock = threading.Lock()


def _worker_loop() -> None:
    while True:
        jid = _queue.get()
        try:
            job = load(jid)
            if job["status"] == "queued":
                _run(job)
        except Exception as exc:                       # never let the worker die
            try:
                job = load(jid)
                job.update(status="failed", error=repr(exc), finished_utc=_now())
                _write(job)
            except Exception:
                pass
        finally:
            _queue.task_done()


def _ensure_worker() -> None:
    global _worker
    with _lock:
        if _worker is None or not _worker.is_alive():
            _worker = threading.Thread(target=_worker_loop, name="exotic-worker", daemon=True)
            _worker.start()


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def recover() -> None:
    """Jobs left queued/running by a previous process are re-queued; their
    frames folder is rebuilt because it was removed or never finished."""
    for job in list_jobs():
        if job["status"] in ("queued", "running"):
            job.update(status="failed", error="server restarted during the run; submit again",
                       finished_utc=_now())
            _write(job)


def submit(an: fieldlab.SessionAnalysis, *, mode: str = "nea", frames: str = "kept",
           obscode: str = "", force: bool = False) -> tuple[dict, bool]:
    """Start (or find) the EXOTIC job for this analysis. Returns (job, created)."""
    if not available():
        raise RuntimeError("the exotic package is not installed on this server")
    options = {"mode": mode, "frames": frames, "obscode": obscode or ""}
    jid = job_id(an, options)
    try:
        existing = load(jid)
    except JobNotFound:
        existing = None
    if existing and not force and existing["status"] in ("queued", "running", "done"):
        return existing, False
    if existing and existing["status"] == "running":
        return existing, False                        # never kill a live run
    job = {"job_id": jid, "session": an.ref, "target": an.target, "night": an.night,
           "options": options, "status": "queued", "created_utc": _now(),
           "exotic_version": exotic_version(), "command": " ".join(_command({"options": options})),
           "files": {}}
    _prepare(an, job)
    _write(job)
    _ensure_worker()
    _queue.put(jid)
    return job, True


# ----------------------------------------------------------------- serving

def public(job: dict) -> dict:
    """The job as a client sees it: status, what it was given, EXOTIC's
    results if any, and where each file is."""
    out = {k: v for k, v in job.items() if k not in ("files",)}
    d = _job_path(job["job_id"])
    if job["status"] == "running":
        tail = _tail(d / "log.txt")
        out["log_tail"] = tail
        for ln in reversed(tail):
            m = _PROGRESS.search(ln)
            if m:
                out["progress"] = {"phase": "photometry", "frame": int(m.group(1)),
                                   "of": int(m.group(2))}
                break
        else:
            out["progress"] = {"phase": "fitting" if any("Fitting" in t or "sampling" in t.lower()
                                                         for t in tail) else "starting"}
    base = f"/api/field/exotic/jobs/{job['job_id']}"
    urls = {"status": base, "log": f"{base}/log.txt", "inits": f"{base}/inits.json"}
    for key in job.get("files", {}):
        urls[key] = f"{base}/{key.replace('_', '.')}"
    if job.get("files"):
        urls["everything_zip"] = f"{base}/results.zip"
    out["urls"] = urls
    out["files"] = {k: v["path"] for k, v in job.get("files", {}).items()}
    return out


def artifact(jid: str, key: str) -> tuple[bytes, str, str]:
    """(bytes, media type, filename) of one output of a finished job."""
    job = load(jid)
    d = _job_path(jid)
    if key == "log":
        return (d / "log.txt").read_bytes() if (d / "log.txt").is_file() else b"", "text/plain", f"exotic_{jid}_log.txt"
    if key == "inits":
        return (d / "inits.json").read_bytes(), "application/json", f"exotic_{jid}_inits.json"
    if key == "results":
        if not job.get("files"):
            raise NotReady(job, "no results yet")
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(d / "inits.json", "inits.json")
            if (d / "log.txt").is_file():
                z.write(d / "log.txt", "log.txt")
            for p in sorted((d / "save").rglob("*")):
                if p.is_file():
                    z.write(p, "exotic_output/" + str(p.relative_to(d / "save")).replace("\\", "/"))
        return buf.getvalue(), "application/zip", f"exotic_{jid}.zip"
    entry = job.get("files", {}).get(key)
    if not entry:
        raise NotReady(job, key)
    p = d / "save" / entry["path"]
    return p.read_bytes(), entry["media"], f"exotic_{jid}_{Path(entry['path']).name}"
