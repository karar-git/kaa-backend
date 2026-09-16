"""Live session analysis: a folder of FITS frames in, pictures out.

This is the one place the API does science on request, and it exists for a
reason the offline pipeline cannot serve: a user pointing the dashboard at a
*new* session — any directory of FITS frames from one night on one star — and
asking "show me the field, show me the light curve". The pipeline results only
cover the 22 sessions it was run on.

The steps are the ones a first-year observer would do by hand, and they are the
same ones the teammate's `Calobration.py` did on a fixed path:

1. **Master dark.** Median of the night's dark frames, or the nearest night that
   has some. Pixels far above the dark's own noise are hot pixels.
2. **Calibrate.** Subtract the dark, patch the hot pixels with their neighbours.
3. **Remove the background.** A coarse running median of the frame is the sky;
   what is left is the "residual signal" the field picture shows.
4. **Find stars** on the first frame: connected groups of pixels above 4 sigma
   in a lightly smoothed image.
   Single-pixel spikes are rejected, because a star on this camera is never one
   pixel wide and a cosmic ray or hot pixel usually is.
5. **Follow the drift.** The mount is alt-az with no derotator, so the field
   moves tens of pixels a night. Each frame is cross-correlated against the
   first to get the shift, and every star is re-centroided there.
6. **Aperture photometry** with a local sky annulus, on every star, every frame.
7. **Differential light curve.** Target flux divided by the summed flux of the
   comparison stars, so a cloud that dims the whole field cancels out. The
   target is the brightest unsaturated star near the centre of the frame — the
   headers have no WCS, so this is the honest rule and it is stated in every
   response.
8. **Dip search.** The longest run of points sitting well below unity is
   reported as a *dip consistent with a transit*, never as a planet.

Nothing here is a substitute for the pipeline's per-star search, its
ephemerides or its false-positive tests; it is the quick look you want the
moment new frames land.
"""

from __future__ import annotations

import io
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from scipy import ndimage

from .config import settings

FITS_EXT = {".fits", ".fit", ".fts"}
SATURATION = 4000.0            # 12-bit camera: DATAMAX is 4095
APERTURE_R = 4.5               # px; stars are ~2-3 px FWHM at 2x2 binning
ANNULUS = (8.0, 12.0)          # px
DETECT_SIGMA = 4.0             # on the 1-px-smoothed image, in its own sigma
MIN_STAR_PIXELS = 4            # rejects hot pixels and most cosmic rays
MAX_STARS = 40
MAX_COMPS = 8
EDGE_MARGIN = 20               # px; the detector shows bands along its edges
MAX_STEP = 60.0                # px; largest believable shift between neighbours
CENTRE_FRACTION = 0.30         # target must lie within this fraction of the
                               # short axis from the frame centre
_UNIX_EPOCH_MJD = 40587.0


class SessionNotFound(LookupError):
    """The session reference names nothing this server can see."""


class SessionInvalid(ValueError):
    """The session exists but cannot be analysed (no frames, bad FITS...)."""


# --------------------------------------------------------------------- paths

def allowed_roots() -> list[Path]:
    """Directories a client may point at. Anything outside is refused: this
    endpoint reads files named by the request, so the set of readable places
    has to be closed."""
    roots = [settings.OBS_ROOT / "observations", settings.OBS_ROOT / "calibration",
             settings.OBS_ROOT, settings.UPLOAD_DIR, *settings.EXTRA_SESSION_ROOTS]
    out = []
    for r in roots:
        try:
            out.append(r.resolve())
        except OSError:
            pass
    return out


def _inside(p: Path, roots: list[Path]) -> bool:
    try:
        rp = p.resolve()
    except OSError:
        return False
    return any(rp == r or r in rp.parents for r in roots)


def _resolve_dir(ref: str, roots: list[Path]) -> Path | None:
    """Turn a user path into a directory inside one of the roots, or None."""
    ref = ref.strip().replace("\\", "/")
    if not ref or ".." in Path(ref).parts:
        return None
    cand = Path(ref)
    tries = [cand] if cand.is_absolute() else [r / cand for r in roots]
    for t in tries:
        if t.is_dir() and _inside(t, roots):
            return t
    return None


def is_fits_name(name: str) -> bool:
    """`x.fits`, `x.fit`, `x.fts`, and the same with `.fz` (Rice tile
    compression) or `.gz` appended. astropy opens all of them."""
    n = name.lower()
    for z in ("", ".fz", ".gz"):
        if any(n.endswith(ext + z) for ext in FITS_EXT):
            return True
    return False


def list_fits(d: Path) -> list[Path]:
    return sorted(p for p in d.rglob("*") if p.is_file() and is_fits_name(p.name))


def _pipeline_session_dir(session_id: str) -> Path | None:
    """`TRES-3__2026-08-10` -> OBS_ROOT/observations/2026-08-10/TRES-3."""
    m = re.fullmatch(r"([A-Za-z0-9\-]+)__(\d{4}-\d{2}-\d{2})", session_id)
    if not m:
        return None
    d = settings.OBS_ROOT / "observations" / m.group(2) / m.group(1)
    return d if d.is_dir() else None


def resolve_session(ref: str) -> tuple[Path | None, str]:
    """Return (frames_dir, kind). kind is 'fits' or 'cube'.

    `ref` may be a pipeline session id, an upload id (`upload:abc123`), a path
    relative to the observations root such as `2026-08-10/TRES-3/session_01`,
    or an absolute path inside one of the allowed roots.
    """
    ref = ref.strip()
    if not ref:
        raise SessionNotFound("Empty session reference.")

    d = _pipeline_session_dir(ref)
    if d is not None:
        return d, "fits"

    if ref.startswith("upload:"):
        d = settings.UPLOAD_DIR / ref.split(":", 1)[1] / "frames"
        if d.is_dir():
            return d, "fits"
        raise SessionNotFound(f"Upload '{ref}' does not exist on this server.")

    d = _resolve_dir(ref, allowed_roots())
    if d is not None:
        return d, "fits"

    # A pipeline id whose raw frames were not deployed: fall back to the cube.
    if "__" in ref:
        from . import imaging
        if imaging._load_cube(ref) is not None:
            return None, "cube"

    raise SessionNotFound(
        f"Session '{ref}' not found. Give a pipeline session id such as "
        f"TRES-3__2026-08-10, an upload id, or a folder under the observations "
        f"root, e.g. 2026-08-10/TRES-3/session_01. GET /api/field/sessions lists "
        f"what is available.")


def display_path(p: Path | str | None) -> str | None:
    """A path as the client should see it: relative to the observations root or
    the upload directory where possible, so responses describe the archive,
    not the server's disk layout."""
    if p is None:
        return None
    p = Path(p)
    for base, tag in ((settings.OBS_ROOT, ""), (settings.UPLOAD_DIR, "upload:")):
        try:
            rel = p.resolve().relative_to(base.resolve())
            return tag + rel.as_posix()
        except (ValueError, OSError):
            continue
    return str(p)


def resolve_calibration(ref: str | None, night: str | None,
                        session_dir: Path | None = None) -> tuple[Path | None, str]:
    """Explicit calibration dir, else the darks shipped with an upload, else the
    darks of the same night, else the nearest night within a week.
    Returns (dir, note)."""
    # Try the calibration tree first, so "2026-08-10" means that night's darks
    # and not that night's science frames.
    roots = [settings.OBS_ROOT / "calibration", *allowed_roots()]
    if ref:
        if ref.startswith("upload:"):
            d = settings.UPLOAD_DIR / ref.split(":", 1)[1] / "darks"
            if not d.is_dir():
                raise SessionNotFound(f"Upload '{ref}' has no dark frames.")
            return d, f"darks uploaded with {ref}"
        d = _resolve_dir(ref, roots)
        if d is None:
            raise SessionNotFound(f"Calibration folder '{ref}' not found or not allowed.")
        return d, f"explicit: {display_path(d)}"
    if session_dir is not None and session_dir.name == "frames":
        sib = session_dir.parent / "darks"
        if _inside(sib, [settings.UPLOAD_DIR.resolve()]) and sib.is_dir() and list_fits(sib):
            return sib, "darks uploaded with the session"
    cal_root = settings.OBS_ROOT / "calibration"
    if night and cal_root.is_dir():
        exact = cal_root / night
        if exact.is_dir() and list_fits(exact):
            return exact, f"same night: {night}"
        try:
            n0 = datetime.strptime(night, "%Y-%m-%d")
        except ValueError:
            n0 = None
        if n0 is not None:
            best = None
            for d in cal_root.iterdir():
                try:
                    dn = datetime.strptime(d.name, "%Y-%m-%d")
                except ValueError:
                    continue
                gap = abs((dn - n0).days)
                if gap <= 7 and d.is_dir() and list_fits(d) and (best is None or gap < best[0]):
                    best = (gap, d)
            if best:
                return best[1], f"nearest night: {best[1].name} ({best[0]} d away)"
    return None, "none available"


# --------------------------------------------------------------------- FITS

def _parse_ut(s: str | None) -> float | None:
    """ISO-ish header time -> MJD. Handles the '-0000' zone MicroObservatory writes."""
    if not s:
        return None
    s = str(s).strip()
    m = re.fullmatch(r"(\d{4}-\d{2}-\d{2})[T ](\d{2}:\d{2}:\d{2}(?:\.\d+)?)\s*([+-]\d{2}:?\d{2}|Z)?", s)
    if not m:
        return None
    dt = datetime.fromisoformat(f"{m.group(1)}T{m.group(2)}")
    tz = m.group(3)
    if tz and tz != "Z":
        sign = 1 if tz[0] == "+" else -1
        hh, mm = int(tz[1:3]), int(tz[-2:])
        dt = dt - sign * timedelta(hours=hh, minutes=mm)
    dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp() / 86400.0 + _UNIX_EPOCH_MJD


def _mjd_to_iso(mjd: float) -> str:
    dt = datetime.fromtimestamp((mjd - _UNIX_EPOCH_MJD) * 86400.0, tz=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def read_fits(path: Path) -> tuple[np.ndarray, dict]:
    from astropy.io import fits
    with fits.open(path, memmap=False) as hd:
        hdu = next((h for h in hd if h.data is not None and getattr(h.data, "ndim", 0) == 2), None)
        if hdu is None:
            raise SessionInvalid(f"{path.name} holds no 2-D image.")
        data = np.asarray(hdu.data, dtype=np.float32)
        hdr = dict(hdu.header)
    return data, hdr


def _hdr_float(hdr: dict, key: str) -> float | None:
    v = hdr.get(key)
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def _frame_mjd(hdr: dict, path: Path) -> float | None:
    mjd = _parse_ut(hdr.get("UT-OBS")) or _parse_ut(hdr.get("DATE-OBS"))
    if mjd is None and hdr.get("MJD-OBS") is not None:
        try:
            mjd = float(hdr["MJD-OBS"])
        except (TypeError, ValueError):
            mjd = None
    if mjd is None:
        # MicroObservatory names files TARGETyymmddHHMMSS
        m = re.search(r"(\d{12})", path.stem)
        if m:
            dt = datetime.strptime(m.group(1), "%y%m%d%H%M%S").replace(tzinfo=timezone.utc)
            mjd = dt.timestamp() / 86400.0 + _UNIX_EPOCH_MJD
    return mjd


# ----------------------------------------------------------------- analysis

@dataclass
class Star:
    x: float
    y: float
    flux: float
    peak: float
    npix: int
    saturated: bool
    persistence: float = 1.0


@dataclass
class SessionAnalysis:
    ref: str
    source: str
    target: str
    night: str | None
    session_dir: str | None
    frame_paths: list[str]
    calibration_dir: str | None
    calibration_note: str
    n_darks: int
    dark: np.ndarray | None
    hot_mask: np.ndarray
    n_hot: int
    shape: tuple[int, int]
    mjd: np.ndarray
    seconds: np.ndarray
    raw_range: tuple[float, float]
    stars: list[Star]
    target_idx: int
    comp_idx: list[int]
    positions: np.ndarray        # (n_frames, n_stars, 2) as (x, y)
    flux: np.ndarray             # (n_frames, n_stars)
    shifts: np.ndarray           # (n_frames, 2) as (dx, dy)
    norm_flux: np.ndarray
    target_only: np.ndarray
    rms_ppt: float
    rms_target_only_ppt: float
    dip: dict
    residual_ref: np.ndarray
    display_sigma: float
    reference_frame: int = 0
    lost_frames: list[int] = field(default_factory=list)
    dip_target_only: dict = field(default_factory=dict)
    reading: str = ""
    # Header facts the EXOTIC / AAVSO export needs.
    ra_deg: float | None = None          # telescope pointing from the header
    dec_deg: float | None = None
    telalt: np.ndarray | None = None     # per-frame altitude, deg (NaN if absent)
    exptime_s: float | None = None
    binning: str = "1x1"
    filter_name: str = ""
    warnings: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0

    @property
    def n_frames(self) -> int:
        return len(self.mjd)


def _robust_sigma(a: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    a = a[np.isfinite(a)]
    if a.size == 0:
        return 0.0
    med = np.median(a)
    return float(1.4826 * np.median(np.abs(a - med)))


def _background(cal: np.ndarray, ds: int = 4, size: int = 13) -> np.ndarray:
    small = ndimage.median_filter(cal[::ds, ::ds], size=size, mode="reflect")
    zoom = (cal.shape[0] / small.shape[0], cal.shape[1] / small.shape[1])
    bg = ndimage.zoom(small, zoom, order=1)
    if bg.shape != cal.shape:                    # zoom rounds; make it exact
        out = np.empty(cal.shape, dtype=bg.dtype)
        h, w = min(bg.shape[0], cal.shape[0]), min(bg.shape[1], cal.shape[1])
        out[:] = np.median(bg)
        out[:h, :w] = bg[:h, :w]
        bg = out
    return bg


def _calibrate(raw: np.ndarray, dark: np.ndarray | None, hot: np.ndarray) -> np.ndarray:
    cal = raw - dark if dark is not None else raw.copy()
    if hot.any():
        patched = ndimage.median_filter(cal, size=3, mode="reflect")
        cal = np.where(hot, patched, cal)
    return cal


def _residual(cal: np.ndarray) -> np.ndarray:
    return cal - _background(cal)


def _hot_from_dark(dark: np.ndarray) -> np.ndarray:
    return dark > np.median(dark) + 10.0 * max(_robust_sigma(dark), 1.0)


def _hot_from_stack(frames: list[np.ndarray]) -> np.ndarray:
    """No darks: a pixel bright in *every* frame while the stars drift is a
    defect, not a star."""
    mn = np.min(np.stack(frames[: min(len(frames), 12)]), axis=0)
    return mn > np.median(mn) + 10.0 * max(_robust_sigma(mn), 1.0)


def _detect(res: np.ndarray, raw: np.ndarray, hot: np.ndarray) -> tuple[list[Star], float]:
    sigma = max(_robust_sigma(res), 1e-3)
    sm = ndimage.gaussian_filter(res, 1.0)
    # Threshold the smoothed image against *its own* noise: smoothing lowers
    # the pixel scatter roughly threefold, so a cut in raw-sigma units would be
    # far harsher than the number suggests.
    sigma_sm = max(_robust_sigma(sm), 1e-3)
    mask = (sm > DETECT_SIGMA * sigma_sm) & ~hot
    lab, n = ndimage.label(mask)
    stars: list[Star] = []
    if n == 0:
        return stars, sigma
    idx = np.arange(1, n + 1)
    npix = ndimage.sum(np.ones_like(res), lab, idx)
    flux = ndimage.sum(res, lab, idx)
    peak = ndimage.maximum(raw, lab, idx)
    com = ndimage.center_of_mass(np.clip(res, 0, None), lab, idx)
    H, W = res.shape
    for k in range(n):
        if npix[k] < MIN_STAR_PIXELS:
            continue
        cy, cx = com[k]
        if not (EDGE_MARGIN <= cx < W - EDGE_MARGIN and EDGE_MARGIN <= cy < H - EDGE_MARGIN):
            continue
        stars.append(Star(float(cx), float(cy), float(flux[k]), float(peak[k]),
                          int(npix[k]), bool(peak[k] >= SATURATION)))
    stars.sort(key=lambda s: -s.flux)
    return stars[:MAX_STARS], sigma


def _starmap(res: np.ndarray) -> np.ndarray:
    """Stars only: the smoothed residual above 3 sigma, zero elsewhere and zero
    along the edges. Correlating these instead of the raw residuals keeps the
    noise floor and the detector's edge bands out of the answer."""
    sm = ndimage.gaussian_filter(res, 1.5)
    s = max(_robust_sigma(sm), 1e-3)
    m = np.where(sm > 3.0 * s, sm, 0.0).astype(np.float32)
    e = EDGE_MARGIN
    m[:e, :] = 0
    m[-e:, :] = 0
    m[:, :e] = 0
    m[:, -e:] = 0
    return m


def _pick_reference(frames: list[np.ndarray], hot: np.ndarray) -> tuple[int, float, float]:
    """Index of the frame whose brightest pixels stand furthest above its own
    noise, i.e. where the stars are most visible. Returns (index, its score,
    frame 0's score). Samples at most ~40 frames; this is a choice, not a
    measurement."""
    e = EDGE_MARGIN
    hm = hot[e:-e, e:-e]
    step = max(1, len(frames) // 40)
    best, best_s, s0 = 0, -np.inf, 0.0
    for i in range(0, len(frames), step):
        a = frames[i][e:-e, e:-e]
        a = np.where(hm, np.median(a), a)
        sm = ndimage.gaussian_filter(a, 1.0)
        score = float((np.percentile(sm, 99.95) - np.median(sm)) / max(_robust_sigma(sm), 1e-3))
        if i == 0:
            s0 = score
        if score > best_s:
            best, best_s = i, score
    return best, best_s, s0


def _shift(ref_fft: np.ndarray, ref_shape: tuple[int, int], img_map: np.ndarray, ds: int,
           prior: tuple[float, float] = (0.0, 0.0), max_step: float = 1e9) -> tuple[float, float]:
    """(dx, dy) such that img ~= ref moved by (dx, dy), searched within
    `max_step` px of `prior`."""
    b = img_map[::ds, ::ds]
    c = np.fft.irfft2(np.conj(ref_fft) * np.fft.rfft2(b), s=ref_shape)
    H, W = ref_shape
    dy = ((np.arange(H) + H // 2) % H) - H // 2        # signed lag per row
    dx = ((np.arange(W) + W // 2) % W) - W // 2
    ok = ((np.abs(dy * ds - prior[1]) <= max_step)[:, None]
          & (np.abs(dx * ds - prior[0]) <= max_step)[None, :])
    if not ok.any():
        return prior
    cm = np.where(ok, c, -np.inf)
    iy, ix = np.unravel_index(int(np.argmax(cm)), cm.shape)
    return float(dx[ix] * ds), float(dy[iy] * ds)


def _centroid(res: np.ndarray, x: float, y: float, half: int = 4) -> tuple[float, float, bool]:
    H, W = res.shape
    for _ in range(2):
        xi, yi = int(round(x)), int(round(y))
        x0, x1 = max(0, xi - half), min(W, xi + half + 1)
        y0, y1 = max(0, yi - half), min(H, yi + half + 1)
        if x1 - x0 < 3 or y1 - y0 < 3:
            return x, y, False
        box = np.clip(res[y0:y1, x0:x1], 0, None)
        s = box.sum()
        if s <= 0:
            return x, y, False
        yy, xx = np.mgrid[y0:y1, x0:x1]
        x, y = float((box * xx).sum() / s), float((box * yy).sum() / s)
    return x, y, True


def _aperture(cal: np.ndarray, raw: np.ndarray, x: float, y: float) -> tuple[float, float, bool]:
    """(flux above local sky, sky per pixel, saturated?)"""
    H, W = cal.shape
    r_out = int(np.ceil(ANNULUS[1])) + 1
    xi, yi = int(round(x)), int(round(y))
    x0, x1 = max(0, xi - r_out), min(W, xi + r_out + 1)
    y0, y1 = max(0, yi - r_out), min(H, yi + r_out + 1)
    if x1 - x0 < 3 or y1 - y0 < 3:
        return np.nan, np.nan, False
    yy, xx = np.mgrid[y0:y1, x0:x1]
    r = np.hypot(xx - x, yy - y)
    ap = r <= APERTURE_R
    ann = (r >= ANNULUS[0]) & (r < ANNULUS[1])
    sub = cal[y0:y1, x0:x1]
    if ap.sum() < 3 or ann.sum() < 8:
        return np.nan, np.nan, False
    sky = float(np.median(sub[ann]))
    flux = float(sub[ap].sum() - sky * ap.sum())
    sat = bool(raw[y0:y1, x0:x1][ap].max() >= SATURATION)
    return flux, sky, sat


def _find_dip(t: np.ndarray, f: np.ndarray, cadence: float,
              min_depth: float = 0.005, sigma: float | None = None) -> dict:
    """The run of points below unity with the largest depth x length.

    `sigma` is the scatter the threshold is judged against; by default the
    curve's own. For the raw target curve pass the *differential* scatter and a
    larger `min_depth` — judged against its own scatter a curve that swings by
    a factor of two never has a 'significant' dip.
    """
    ok = np.isfinite(f)
    if sigma is None:
        sigma = _robust_sigma(f[ok]) if ok.sum() > 4 else 0.0
    thresh = 1.0 - max(3.0 * sigma, min_depth)
    below = ok & (f < thresh)
    best = (0.0, 0, 0, 0)       # (score, length, start, end)
    i = 0
    n = len(f)
    while i < n:
        if below[i]:
            j = i
            # allow a single point to pop back up inside the dip
            while j + 1 < n and (below[j + 1] or (j + 2 < n and below[j + 2])):
                j += 1
            L = j - i + 1
            score = L * float(1.0 - np.nanmedian(f[i:j + 1]))
            if L >= 3 and score > best[0]:
                best = (score, L, i, j)
            i = j + 1
        else:
            i += 1
    out = {"detected": False, "threshold": float(thresh), "scatter_ppt": float(sigma * 1000),
           "statement": "No run of points sits significantly below unity."}
    if best[1] >= 3:
        _score, L, a, b = best
        depth = float(1.0 - np.nanmedian(f[a:b + 1]))
        signif = float(depth / (sigma / np.sqrt(L))) if sigma > 0 else float("inf")
        dur = float(t[b] - t[a] + cadence)
        out.update(
            detected=True, start_index=int(a), end_index=int(b), n_points=int(L),
            start_s=float(t[a]), end_s=float(t[b]), duration_s=dur,
            duration_min=dur / 60.0, depth=depth, depth_ppt=depth * 1000.0,
            significance_sigma=signif,
            statement=(f"A dip of {depth*1000:.1f} ppt lasting {dur/60:.0f} min "
                       f"({L} points, {signif:.1f} sigma) is consistent with a transit. "
                       "Consistent with, not confirmed: see /api/false-positive-cases."))
    return out


def _load_frames(ref: str) -> tuple[list[np.ndarray], list[dict], list[str], str, str | None, str]:
    """Return (frames, headers, paths, target, night, source)."""
    d, kind = resolve_session(ref)
    if kind == "cube":
        from . import data, imaging
        cube = imaging._load_cube(ref)
        frames = [np.asarray(cube[i], dtype=np.float32) for i in range(len(cube))]
        hdrs: list[dict] = [{} for _ in frames]
        try:
            fq = data.frame_quality()
            fq = fq[fq.session_id == ref].sort_values("frame_index")
            for i, (t, alt) in enumerate(zip(fq.t_utc.tolist(), fq.TELALT.tolist())):
                if i >= len(frames):
                    break
                hdrs[i]["UT-OBS"] = str(t).replace(" ", "T").replace("+00:00", "Z")
                hdrs[i]["TELALT"] = alt
        except Exception:
            pass
        target, night = ref.split("__", 1)
        return frames, hdrs, [], target, night, "cube"

    paths = list_fits(d)
    if not paths:
        raise SessionInvalid(f"'{d}' holds no FITS files (.fits/.fit/.fts).")
    if len(paths) > settings.FIELD_MAX_FRAMES:
        raise SessionInvalid(
            f"'{d}' holds {len(paths)} FITS files; this endpoint handles at most "
            f"{settings.FIELD_MAX_FRAMES} per session.")
    frames, hdrs = [], []
    for p in paths:
        try:
            a, h = read_fits(p)
        except SessionInvalid:
            raise
        except Exception as exc:                      # astropy raises many kinds
            raise SessionInvalid(f"{p.name} is not a readable FITS image: {exc}")
        frames.append(a)
        hdrs.append(h)
    shapes = {f.shape for f in frames}
    if len(shapes) != 1:
        raise SessionInvalid(f"Frames have mixed shapes {sorted(shapes)}; a session is one camera setting.")
    target = str(hdrs[0].get("OBJECT") or d.name).strip() or d.name
    if target.lower() in {"calibration", "dark"}:
        raise SessionInvalid("These are dark frames, not a science session.")
    mjd0 = _frame_mjd(hdrs[0], paths[0])
    night = _mjd_to_iso(mjd0)[:10] if mjd0 else None
    return frames, hdrs, [str(p) for p in paths], target, night, "fits"


def _load_dark(cal_dir: Path | None, exptime: float | None) -> tuple[np.ndarray | None, int, list[str]]:
    warnings: list[str] = []
    if cal_dir is None:
        return None, 0, warnings
    darks = []
    for p in list_fits(cal_dir):
        try:
            a, h = read_fits(p)
        except Exception:
            continue
        if exptime and h.get("EXPTIME") and abs(float(h["EXPTIME"]) - exptime) > 0.5:
            continue
        darks.append(a)
    if not darks:
        warnings.append(f"No usable dark frames in {cal_dir}; frames were not dark-subtracted.")
        return None, 0, warnings
    return np.median(np.stack(darks), axis=0).astype(np.float32), len(darks), warnings


def analyse(ref: str, calibration: str | None = None,
            target_xy: tuple[float, float] | None = None) -> SessionAnalysis:
    t0 = time.perf_counter()
    warnings: list[str] = []
    frames, hdrs, paths, target, night, source = _load_frames(ref)

    # Times: seconds since the first frame, and MJD where the headers have it.
    mjds = [_frame_mjd(h, Path(p) if p else Path(f"frame{i:04d}")) for i, (h, p) in
            enumerate(zip(hdrs, paths or [None] * len(hdrs)))]
    if all(m is not None for m in mjds):
        order = np.argsort(mjds)
        frames = [frames[i] for i in order]
        hdrs = [hdrs[i] for i in order]
        paths = [paths[i] for i in order] if paths else paths
        mjd = np.array([mjds[i] for i in order], dtype=np.float64)
    else:
        exptime = float(hdrs[0].get("EXPTIME", 60.0) or 60.0)
        warnings.append("Some frames carry no time; the time axis assumes a "
                        f"regular cadence of {exptime*3:.0f} s (filename order).")
        mjd = np.arange(len(frames)) * (exptime * 3) / 86400.0
    seconds = (mjd - mjd[0]) * 86400.0
    cadence = float(np.median(np.diff(seconds))) if len(seconds) > 1 else 180.0

    # Pointing, altitude, binning and filter from the headers, for the exports.
    h0 = hdrs[0]
    ra_hdr, dec_hdr = _hdr_float(h0, "RA"), _hdr_float(h0, "DEC")
    telalt = np.array([_hdr_float(h, "TELALT") for h in hdrs], dtype=float)
    xb, yb = _hdr_float(h0, "XBINNING") or 1, _hdr_float(h0, "YBINNING") or 1
    binning = f"{int(xb)}x{int(yb)}"
    filter_name = str(h0.get("FILTER") or "").strip()

    # Calibration.
    exptime = None
    try:
        exptime = float(hdrs[0].get("EXPTIME")) if hdrs[0].get("EXPTIME") is not None else None
    except (TypeError, ValueError):
        pass
    if source == "fits":
        cal_dir, cal_note = resolve_calibration(calibration, night, Path(paths[0]).parent)
    else:
        cal_dir, cal_note = None, "cube mode: raw frames not deployed, no darks"
    dark, n_darks, w = _load_dark(cal_dir, exptime)
    warnings += w
    if dark is not None and dark.shape != frames[0].shape:
        warnings.append("Dark frames have a different shape from the science frames; skipped.")
        dark, n_darks = None, 0
    if dark is not None:
        hot = _hot_from_dark(dark)
    else:
        hot = _hot_from_stack(frames)
        if source == "fits" and cal_dir is None:
            warnings.append("No dark frames found for this night; hot pixels were "
                            "identified from the science frames themselves and the "
                            "dark current was left in (the local sky annulus removes most of it).")

    raw_range = (float(frames[0].min()), float(frames[0].max()))

    # Reference frame: the one where the stars stand out best. Frame 0 is a
    # poor default — on this telescope the first frames of a night are often
    # taken in twilight or haze, with a sky three times brighter than later and
    # no star visible above it. A reference with no stars anchors nothing.
    ref_i, ref_score, score0 = _pick_reference(frames, hot)
    if ref_i != 0:
        warnings.append(f"Frame {ref_i} is the reference, not frame 0: stars stand "
                        f"out {ref_score / max(score0, 1e-3):.1f}x better there "
                        "(frame 0 was probably twilight or haze).")
    cal_ref = _calibrate(frames[ref_i], dark, hot)
    res_ref = _residual(cal_ref)
    stars, sigma0 = _detect(res_ref, frames[ref_i], hot)
    if len(stars) < 2:
        raise SessionInvalid(
            f"Only {len(stars)} star(s) detected on the best frame ({ref_i}); cannot "
            "build a light curve. Is this a science frame of a star field?")

    # Follow the field outward from the reference. Each frame's shift is searched
    # within MAX_STEP px of the previous good frame's, which is how the pipeline
    # chains adjacent frames too: the drift is smooth, so a jump of 200 px
    # between neighbours is a wrong answer, not a discovery.
    ds = 2
    ref_map = _starmap(res_ref)[::ds, ::ds]
    ref_fft = np.fft.rfft2(ref_map)
    n, m = len(frames), len(stars)
    positions = np.full((n, m, 2), np.nan)
    flux = np.full((n, m), np.nan)
    sat = np.zeros((n, m), dtype=bool)
    found = np.zeros((n, m), dtype=bool)
    shifts = np.zeros((n, 2))
    lost = np.zeros(n, dtype=bool)
    min_flux = 3.0 * sigma0 * np.sqrt(np.pi * APERTURE_R**2)

    def measure(i: int, cal: np.ndarray, res: np.ndarray, dx: float, dy: float) -> int:
        n_ok = 0
        for k, s in enumerate(stars):
            x, y, ok = _centroid(res, s.x + dx, s.y + dy)
            if ok and np.hypot(x - (s.x + dx), y - (s.y + dy)) > 4.0:
                ok = False
                x, y = s.x + dx, s.y + dy
            f, _sky, is_sat = _aperture(cal, frames[i], x, y)
            positions[i, k] = (x, y)
            flux[i, k] = f
            sat[i, k] = is_sat
            found[i, k] = ok and np.isfinite(f) and f > min_flux
            n_ok += int(found[i, k])
        return n_ok

    measure(ref_i, cal_ref, res_ref, 0.0, 0.0)
    forward = list(range(ref_i + 1, n))
    backward = list(range(ref_i - 1, -1, -1))
    for chain in (forward, backward):
        prior = (0.0, 0.0)
        for i in chain:
            cal = _calibrate(frames[i], dark, hot)
            res = _residual(cal)
            dx, dy = _shift(ref_fft, ref_map.shape, _starmap(res), ds, prior, MAX_STEP)
            shifts[i] = (dx, dy)
            n_ok = measure(i, cal, res, dx, dy)
            if n_ok < max(2, int(0.3 * m)):
                # Clouds, or the field is not where it should be. Say so rather
                # than record a flux from empty sky.
                lost[i] = True
                positions[i] = np.nan
                flux[i] = np.nan
                found[i] = False
            else:
                prior = (dx, dy)
    if lost.any():
        warnings.append(f"{int(lost.sum())} of {n} frames were dropped: fewer than "
                        "30% of the reference stars could be recovered in them "
                        "(cloud, twilight, or a tracking slip).")

    for k, s in enumerate(stars):
        s.persistence = float(found[:, k].mean())
        s.saturated = bool(s.saturated or sat[:, k].any())

    # Target: brightest unsaturated persistent star near the centre, unless the
    # caller points at one.
    H, W = frames[0].shape
    persistent = [k for k, s in enumerate(stars) if s.persistence >= 0.6]
    if len(persistent) < 2:
        persistent = list(range(m))
        warnings.append("Few stars persist across the night; tracking may be poor.")
    if target_xy is not None:
        tx, ty = target_xy
        target_idx = min(persistent, key=lambda k: np.hypot(stars[k].x - tx, stars[k].y - ty))
        if np.hypot(stars[target_idx].x - tx, stars[target_idx].y - ty) > 15:
            warnings.append(f"No detected star within 15 px of ({tx:.0f}, {ty:.0f}); "
                            "used the nearest one.")
        target_rule = "caller-specified position"
    else:
        rmax = CENTRE_FRACTION * min(H, W)
        central = [k for k in persistent
                   if np.hypot(stars[k].x - W / 2, stars[k].y - H / 2) <= rmax]
        pool = [k for k in central if not stars[k].saturated] or central or persistent
        target_idx = max(pool, key=lambda k: stars[k].flux)
        target_rule = ("brightest unsaturated persistent star within "
                       f"{rmax:.0f} px of the frame centre")
    if stars[target_idx].saturated:
        warnings.append("The chosen target saturates the detector in at least one "
                        "frame; its photometry is unreliable there.")

    # Comparison stars: bright, unsaturated, persistent, and quiet.
    tflux = np.nanmedian(flux[:, target_idx])
    comps = [k for k in persistent
             if k != target_idx and not stars[k].saturated
             and 0.05 * tflux <= np.nanmedian(flux[:, k]) <= 20 * tflux]
    comps.sort(key=lambda k: -np.nanmedian(flux[:, k]))
    comps = comps[: MAX_COMPS + 4]
    if len(comps) >= 3:
        scat = {k: _robust_sigma(flux[:, k] / np.nanmedian(flux[:, k])) for k in comps}
        med = np.median(list(scat.values()))
        comps = [k for k in comps if scat[k] <= 4 * med] or comps
    comps = comps[:MAX_COMPS]

    tgt = flux[:, target_idx]
    target_only = tgt / np.nanmedian(tgt)
    if comps:
        csum = np.nansum(flux[:, comps], axis=1)
        csum[np.all(~np.isfinite(flux[:, comps]), axis=1)] = np.nan
        rel = tgt / csum
        norm = rel / np.nanmedian(rel)
    else:
        norm = target_only.copy()
        warnings.append("No usable comparison star; the light curve is the raw "
                        "target flux and will follow the clouds.")
    rms = _robust_sigma(norm[np.isfinite(norm)]) * 1000
    rms_t = _robust_sigma(target_only[np.isfinite(target_only)]) * 1000
    dip = _find_dip(seconds, norm, cadence)
    dip["target_rule"] = target_rule
    # The same search on the raw target curve. When *that* dips and the
    # differential curve does not, the sky dimmed, not the star — which is the
    # single most common way a first light curve fools its author.
    dip_t = _find_dip(seconds, target_only, cadence, min_depth=0.05, sigma=rms / 1000.0)
    if dip["detected"]:
        reading = (f"A dip of {dip['depth_ppt']:.0f} ppt survives division by the "
                   f"comparison stars, so it is not simple cloud. It is still only "
                   "consistent with a transit; check the predicted window in "
                   "/api/science/predictions and the cases in /api/false-positive-cases.")
    elif dip_t["detected"]:
        mid_h = 0.5 * (dip_t["start_s"] + dip_t["end_s"]) / 3600.0
        reading = (f"The target alone drops {dip_t['depth_ppt']/10:.0f}% around "
                   f"{mid_h:.1f} h after the start, but the comparison stars drop with "
                   f"it and the differential curve stays flat to {rms:.0f} ppt. That "
                   "dip is transparency (cloud, haze or airmass), not the star.")
    else:
        reading = "Neither the differential nor the raw target curve shows a significant dip."

    return SessionAnalysis(
        ref=ref, source=source, target=target, night=night,
        session_dir=str(Path(paths[0]).parent) if paths else None,
        frame_paths=paths, calibration_dir=str(cal_dir) if cal_dir else None,
        calibration_note=cal_note, n_darks=n_darks, dark=dark, hot_mask=hot,
        n_hot=int(hot.sum()), shape=(H, W), mjd=mjd, seconds=seconds,
        raw_range=raw_range, stars=stars, target_idx=target_idx, comp_idx=comps,
        positions=positions, flux=flux, shifts=shifts, norm_flux=norm,
        target_only=target_only, rms_ppt=float(rms), rms_target_only_ppt=float(rms_t),
        dip=dip, residual_ref=res_ref.astype(np.float32), display_sigma=float(sigma0),
        reference_frame=int(ref_i), lost_frames=[int(i) for i in np.flatnonzero(lost)],
        dip_target_only=dip_t, reading=reading,
        ra_deg=ra_hdr, dec_deg=dec_hdr, telalt=telalt, exptime_s=exptime,
        binning=binning, filter_name=filter_name,
        warnings=warnings, elapsed_s=time.perf_counter() - t0,
    )


def residual_for_frame(an: SessionAnalysis, i: int) -> np.ndarray:
    """Background-removed image of frame i. Frame 0 is kept; others are rebuilt
    from disk so a cached analysis stays small."""
    if not (0 <= i < an.n_frames):
        raise IndexError(i)
    if i == an.reference_frame:
        return an.residual_ref
    if an.source == "cube":
        from . import imaging
        raw = np.asarray(imaging._load_cube(an.ref)[i], dtype=np.float32)
    else:
        raw, _ = read_fits(Path(an.frame_paths[i]))
    return _residual(_calibrate(raw, an.dark, an.hot_mask))


# --------------------------------------------------------------------- cache

_lock = threading.Lock()
_cache: dict[tuple, SessionAnalysis] = {}
_blob_cache: dict[tuple, bytes] = {}


def _fingerprint(ref: str) -> tuple:
    d, kind = resolve_session(ref)          # raises SessionNotFound
    if d is None:
        return (ref, "cube")
    files = list_fits(d)
    latest = max((p.stat().st_mtime_ns for p in files), default=0)
    return (str(d), len(files), latest)


def get_analysis(ref: str, calibration: str | None = None,
                 target_xy: tuple[float, float] | None = None) -> SessionAnalysis:
    key = (_fingerprint(ref), calibration, target_xy)
    if key in _cache:
        return _cache[key]
    with _lock:
        if key in _cache:
            return _cache[key]
        an = analyse(ref, calibration, target_xy)
        if len(_cache) >= settings.FIELD_CACHE_SESSIONS:
            _cache.pop(next(iter(_cache)))
            _blob_cache.clear()
        _cache[key] = an
        return an


def cached_blob(key: tuple, make) -> bytes:
    if key in _blob_cache:
        return _blob_cache[key]
    blob = make()
    if len(_blob_cache) > 200:
        _blob_cache.clear()
    _blob_cache[key] = blob
    return blob


# --------------------------------------------------------------------- plots

def _figure(w: float, h: float):
    import matplotlib
    matplotlib.use("Agg", force=False)
    from matplotlib.figure import Figure
    return Figure(figsize=(w, h), dpi=110)


def _png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
    return buf.getvalue()


def render_field(an: SessionAnalysis, frame: int | None = None, circles: bool = True,
                 labels: bool = False) -> bytes:
    """The background-removed frame with every detected star ringed in red and
    the target in green. Default frame: the reference."""
    from matplotlib.patches import Circle
    if frame is None:
        frame = an.reference_frame
    res = residual_for_frame(an, frame)
    s = max(an.display_sigma, 1e-3)
    norm = np.clip((res + 2 * s) / (14 * s), 0, 1)
    fig = _figure(7.2, 5.6)
    ax = fig.add_subplot(111)
    im = ax.imshow(norm, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    if circles:
        dx, dy = an.shifts[frame]
        for k, st in enumerate(an.stars):
            x, y = an.positions[frame, k]
            if not np.isfinite(x):
                x, y = st.x + dx, st.y + dy
            is_t = k == an.target_idx
            is_c = k in an.comp_idx
            col = "lime" if is_t else ("deepskyblue" if is_c else "red")
            ax.add_patch(Circle((x, y), 9 if is_t else 7, fill=False,
                                edgecolor=col, linewidth=1.6 if is_t else 1.1))
            if labels or is_t:
                ax.annotate("target" if is_t else str(k), (x + 10, y - 6),
                            color=col, fontsize=8)
    title = f"{an.target} - background-removed (frame {frame} of {an.n_frames}"
    title += ", reference)" if frame == an.reference_frame else ")"
    if frame in an.lost_frames:
        title += "  [stars not recovered]"
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("x [px]")
    ax.set_ylabel("y [px]")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("residual signal (normalised)")
    if circles:
        fig.text(0.01, -0.02, "green: target   blue: comparison stars   red: other detections",
                 fontsize=7.5, color="0.35")
    return _png(fig)


def render_lightcurve(an: SessionAnalysis, mode: str = "differential") -> bytes:
    y = an.norm_flux if mode == "differential" else an.target_only
    t = an.seconds
    fig = _figure(7.2, 4.4)
    ax = fig.add_subplot(111)
    ax.axhline(1.0, color="0.6", lw=0.8, ls="--")
    ok = np.isfinite(y)
    ax.plot(t[ok], y[ok], marker=".", ms=6, lw=1.0, color="tab:blue")
    d = an.dip
    if d.get("detected") and mode == "differential":
        ax.axvspan(d["start_s"] - 90, d["end_s"] + 90, color="tab:red", alpha=0.12,
                   label=f"dip: {d['depth_ppt']:.0f} ppt, {d['duration_min']:.0f} min")
        ax.legend(loc="lower left", fontsize=8, frameon=False)
    ax.set_title(f"{an.target} light curve"
                 + ("" if mode == "differential" else " (target only, no comparison stars)"))
    ax.set_xlabel("seconds since start")
    ax.set_ylabel("relative brightness")
    if mode == "differential" and an.comp_idx:
        sub = f"target / sum of {len(an.comp_idx)} comparison stars, median-normalised"
        rms = an.rms_ppt
    else:
        sub = "target flux, median-normalised"
        rms = an.rms_target_only_ppt
    n_used = int(np.isfinite(y).sum())
    fig.text(0.01, -0.04, f"{sub}   rms {rms:.1f} ppt   {n_used} of {an.n_frames} frames   "
             f"{_mjd_to_iso(float(an.mjd[0]))[:16]}Z", fontsize=7.5, color="0.35")
    return _png(fig)


# --------------------------------------------------------------------- JSON

def star_record(an: SessionAnalysis, k: int) -> dict:
    s = an.stars[k]
    return {"index": k, "x": round(s.x, 2), "y": round(s.y, 2), "row": round(s.y, 2),
            "col": round(s.x, 2), "flux_frame0": round(s.flux, 1), "peak_adu": s.peak,
            "n_pixels": s.npix, "saturated": s.saturated,
            "persistence": round(s.persistence, 3),
            "median_flux": _f(np.nanmedian(an.flux[:, k]), 1),
            "role": "target" if k == an.target_idx else ("comparison" if k in an.comp_idx else "other")}


def summary(an: SessionAnalysis) -> dict:
    return {
        "session": an.ref,
        "source": an.source,
        "target": an.target,
        "night": an.night,
        "session_dir": display_path(an.session_dir),
        "n_frames": an.n_frames,
        "reference_frame": an.reference_frame,
        "lost_frames": an.lost_frames,
        "frame_shape": list(an.shape),
        "first_timestamp_utc": _mjd_to_iso(float(an.mjd[0])),
        "last_timestamp_utc": _mjd_to_iso(float(an.mjd[-1])),
        "span_hours": round(float(an.seconds[-1]) / 3600.0, 3),
        "median_cadence_s": round(float(np.median(np.diff(an.seconds))), 1) if an.n_frames > 1 else None,
        "pixel_range_first_frame": {"min": an.raw_range[0], "max": an.raw_range[1], "units": "ADU"},
        "calibration": {"dark_path": display_path(an.calibration_dir), "how_chosen": an.calibration_note,
                        "n_dark_frames": an.n_darks, "dark_subtracted": an.dark is not None,
                        "hot_pixels": an.n_hot},
        "field_drift_px": {"max": round(float(np.hypot(an.shifts[:, 0], an.shifts[:, 1]).max()), 1),
                           "final": round(float(np.hypot(*an.shifts[-1])), 1)},
        "n_stars_detected": len(an.stars),
        "target_star": star_record(an, an.target_idx),
        "target_rule": an.dip.get("target_rule"),
        "comparison_stars": [star_record(an, k) for k in an.comp_idx],
        "rms_ppt": round(an.rms_ppt, 2),
        "rms_target_only_ppt": round(an.rms_target_only_ppt, 2),
        "improvement_factor": round(an.rms_target_only_ppt / an.rms_ppt, 2) if an.rms_ppt > 0 else None,
        "dip": an.dip,
        "dip_target_only": an.dip_target_only,
        "reading": an.reading,
        "warnings": an.warnings,
        "compute_seconds": round(an.elapsed_s, 2),
    }


def lightcurve_points(an: SessionAnalysis) -> list[dict]:
    tgt = an.flux[:, an.target_idx]
    if an.comp_idx:
        comp = an.flux[:, an.comp_idx]
        csum = np.nansum(comp, axis=1)
        csum[np.all(~np.isfinite(comp), axis=1)] = np.nan   # nansum of nothing is 0
    else:
        csum = np.full(an.n_frames, np.nan)
    out = []
    for i in range(an.n_frames):
        out.append({
            "frame_index": i,
            "t_utc": _mjd_to_iso(float(an.mjd[i])),
            "mjd": round(float(an.mjd[i]), 6),
            "seconds": round(float(an.seconds[i]), 1),
            "target_flux": _f(tgt[i]), "comp_flux_sum": _f(csum[i]),
            "norm_flux": _f(an.norm_flux[i], 5), "target_only_norm": _f(an.target_only[i], 5),
            "target_x": _f(an.positions[i, an.target_idx, 0], 2),
            "target_y": _f(an.positions[i, an.target_idx, 1], 2),
            "dx": _f(an.shifts[i, 0], 1), "dy": _f(an.shifts[i, 1], 1),
            "file": Path(an.frame_paths[i]).name if an.frame_paths else None,
        })
    return out


def _f(v, nd: int = 1):
    try:
        if v is None or not np.isfinite(v):
            return None
    except TypeError:
        return None
    return round(float(v), nd)


# ------------------------------------------------------------------ catalogue

def list_raw_sessions() -> list[dict]:
    """Every night/target folder under the observations root, plus uploads."""
    out = []
    obs = settings.OBS_ROOT / "observations"
    if obs.is_dir():
        for night in sorted(p for p in obs.iterdir() if p.is_dir()):
            for tgt in sorted(p for p in night.iterdir() if p.is_dir()):
                n = len(list_fits(tgt))
                if n:
                    out.append({"session": f"{tgt.name}__{night.name}", "target": tgt.name,
                                "night": night.name, "path": f"{night.name}/{tgt.name}",
                                "n_frames": n, "kind": "observations"})
    up = settings.UPLOAD_DIR
    if up.is_dir():
        for d in sorted(p for p in up.iterdir() if p.is_dir()):
            fr = d / "frames"
            n = len(list_fits(fr)) if fr.is_dir() else 0
            if n:
                out.append({"session": f"upload:{d.name}", "target": None, "night": None,
                            "path": f"upload:{d.name}/frames", "n_frames": n, "kind": "upload",
                            "n_darks": len(list_fits(d / "darks")) if (d / "darks").is_dir() else 0})
    return out
