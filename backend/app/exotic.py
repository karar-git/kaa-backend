"""EXOTIC and AAVSO compatibility.

EXOTIC (EXOplanet Transit Interpretation Code, NASA JPL / Exoplanet Watch) is
the reference reduction for exactly this kind of data: its own sample dataset is
a night of MicroObservatory frames. Being compatible with it means three things,
and this module does all three for any session the API can analyse:

1. **`inits.json`** — EXOTIC's initialisation file, filled in from our analysis:
   observatory, camera, binning, filter, the target and comparison-star pixel
   positions we found, and the planet's parameters from the NASA Exoplanet
   Archive tables the pipeline already carries. Run EXOTIC on the raw frames
   with `exotic -red inits.json -nea`.
2. **Pre-reduced light curve** — the four-column comma-separated file EXOTIC's
   `-pre` mode reads (BJD_TDB, flux, uncertainty, airmass), so EXOTIC can fit
   its transit model to *our* photometry: `exotic -pre inits.json -nea`.
3. **AAVSO Exoplanet Database report** — the `#TYPE=EXOPLANET` text format
   EXOTIC itself writes and the AAVSO accepts, so a night can be submitted with
   an observer code.

Times are converted to BJD_TDB the same way the pipeline does it (UTC->TDB plus
barycentric light travel, observer at the Whipple Observatory), and airmass is
sec(z) from the header altitude, as in the pipeline.

The formats were taken from the EXOTIC source (`exotic/exotic.py`,
`exotic/api/output_aavso.py`, `inits.json`) at github.com/rzellem/EXOTIC.
"""

from __future__ import annotations

import io
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import data
from .config import settings
from .fieldlab import SessionAnalysis, _mjd_to_iso, display_path

# Matches the FITS headers (LATITUDE/LONGITUD) and the pipeline's WHIPPLE constant.
OBSERVATORY = {"name": "Whipple Observatory, Amado AZ",
               "lat_deg": 31.68, "lon_deg": -110.88, "elev_m": 1268}
# AAVSO filter codes (aavso.org/filters). MicroObservatory's "Clear" is CV:
# clear, reduced to a V zero point.
AAVSO_FILTER = {"clear": "CV", "none": "CV", "": "CV", "v": "V", "b": "B", "r": "R",
                "i": "I", "cv": "CV", "cr": "CR"}


# ------------------------------------------------------------------ timing

def _astropy():
    import warnings
    from astropy import units as u
    from astropy.coordinates import AltAz, EarthLocation, SkyCoord
    from astropy.time import Time
    from astropy.utils import iers
    # Offline is the normal case (Railway has no reason to fetch IERS tables);
    # extrapolating UT1-UTC costs about a microsecond of light travel.
    iers.conf.auto_download = False
    iers.conf.auto_max_age = None
    warnings.filterwarnings("ignore", module="astropy.utils.iers")
    loc = EarthLocation(lat=OBSERVATORY["lat_deg"] * u.deg, lon=OBSERVATORY["lon_deg"] * u.deg,
                        height=OBSERVATORY["elev_m"] * u.m)
    return u, AltAz, SkyCoord, Time, loc


def bjd_tdb(mjd_utc: np.ndarray, ra_deg: float, dec_deg: float) -> np.ndarray:
    u, _AltAz, SkyCoord, Time, loc = _astropy()
    t = Time(np.asarray(mjd_utc, dtype=float), format="mjd", scale="utc", location=loc)
    c = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg)
    return np.asarray((t.tdb + t.light_travel_time(c, kind="barycentric")).jd)


def airmass_from_alt(alt_deg: np.ndarray) -> np.ndarray:
    """Plane-parallel sec(z), clamped at 3 deg altitude, as in the pipeline."""
    alt = np.clip(np.asarray(alt_deg, dtype=float), 3.0, 90.0)
    return 1.0 / np.cos(np.radians(90.0 - alt))


def airmass_from_sky(mjd_utc: np.ndarray, ra_deg: float, dec_deg: float) -> np.ndarray:
    u, AltAz, SkyCoord, Time, loc = _astropy()
    t = Time(np.asarray(mjd_utc, dtype=float), format="mjd", scale="utc", location=loc)
    c = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg)
    alt = c.transform_to(AltAz(obstime=t, location=loc)).alt.deg
    return airmass_from_alt(alt)


# ---------------------------------------------------------------- catalogue

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def catalog_row(target: str) -> dict:
    """Planet and host parameters for a target name as it appears in the FITS
    OBJECT header (TRES-3, HATP-10 ...). Empty dict when unknown."""
    key = _norm(target)
    out: dict = {}
    try:
        pc = data.planet_catalog()
        m = pc[pc.target.map(_norm) == key]
        if m.empty:
            m = pc[pc.hostname.map(_norm) == key]
        if not m.empty:
            out.update({k: (None if (isinstance(v, float) and np.isnan(v)) else v)
                        for k, v in m.iloc[0].to_dict().items()})
    except data.DatasetMissing:
        pass
    try:
        e = data.ephemerides()
        host = out.get("hostname") or target
        m = e[e.hostname.map(_norm) == _norm(host)]
        if m.empty:
            m = e[e.pl_name.map(lambda s: _norm(str(s).replace(" b", ""))) == key]
        if not m.empty:
            r = m.iloc[0]
            out.setdefault("pl_name", r.pl_name)
            out.setdefault("hostname", r.hostname)
            out["pl_tranmid"] = None if np.isnan(r.pl_tranmid) else float(r.pl_tranmid)
            out.setdefault("pl_orbper", None if np.isnan(r.pl_orbper) else float(r.pl_orbper))
            out["pl_trandur_h"] = None if np.isnan(r.pl_trandur) else float(r.pl_trandur)
            out["pl_trandep_pct"] = None if np.isnan(r.pl_trandep) else float(r.pl_trandep)
            out["ephemeris_source"] = r.source
    except data.DatasetMissing:
        pass
    return out


def target_coords(an: SessionAnalysis, cat: dict) -> tuple[float, float, str]:
    """Catalogue position of the host if known, else the telescope pointing."""
    if cat.get("ra") is not None and cat.get("dec") is not None:
        return float(cat["ra"]), float(cat["dec"]), "NASA Exoplanet Archive"
    if an.ra_deg is not None and an.dec_deg is not None:
        return float(an.ra_deg), float(an.dec_deg), "FITS header RA/DEC (telescope pointing)"
    raise ValueError("No coordinates: the headers carry no RA/DEC and the target is not in the catalogue.")


def sexagesimal(ra_deg: float, dec_deg: float) -> tuple[str, str]:
    h = ra_deg / 15.0
    hh = int(h)
    mm = int((h - hh) * 60)
    ss = ((h - hh) * 60 - mm) * 60
    sign = "+" if dec_deg >= 0 else "-"
    d = abs(dec_deg)
    dd = int(d)
    dm = int((d - dd) * 60)
    ds = ((d - dd) * 60 - dm) * 60
    return f"{hh:02d}:{mm:02d}:{ss:05.2f}", f"{sign}{dd:02d}:{dm:02d}:{ds:05.2f}"


# ------------------------------------------------------------------ timing

def timing(an: SessionAnalysis) -> dict:
    """BJD_TDB and airmass per frame, plus the coordinates they were computed for."""
    cat = catalog_row(an.target)
    ra, dec, src = target_coords(an, cat)
    bjd = bjd_tdb(an.mjd, ra, dec)
    if an.telalt is not None and np.isfinite(an.telalt).all():
        am = airmass_from_alt(an.telalt)
        am_src = "sec(z) from header TELALT"
    else:
        am = airmass_from_sky(an.mjd, ra, dec)
        am_src = "sec(z) from the catalogue position and observatory"
    return {"bjd_tdb": bjd, "airmass": am, "ra_deg": ra, "dec_deg": dec,
            "coord_source": src, "airmass_source": am_src, "catalog": cat}


def predicted_midtransit(an: SessionAnalysis, tm: dict) -> tuple[float | None, str]:
    """Published T0 propagated to the epoch nearest this night (BJD_TDB)."""
    try:
        p = data.predictions()
        p = p[p.session_id == an.ref]
        if not p.empty:
            return float(p.iloc[0].pred_mid_bjd), "pipeline transit_predictions.csv"
    except data.DatasetMissing:
        pass
    cat = tm["catalog"]
    t0, per = cat.get("pl_tranmid"), cat.get("pl_orbper")
    if t0 is None or per is None:
        return None, "unknown"
    mid = float(np.mean(tm["bjd_tdb"]))
    n = round((mid - t0) / per)
    return float(t0 + n * per), f"published T0 + {n} x P"


# ------------------------------------------------------------------- pixels

def pixel_positions(an: SessionAnalysis, frame: int | None) -> tuple[int, list[list[int]], list[list[int]], str]:
    """Target and comparison (x, y) in a given frame. Default: the first frame
    whose stars were recovered, because that is the image EXOTIC will use to
    find them. Zero-based column, row; EXOTIC re-centres within a few pixels."""
    good = [i for i in range(an.n_frames) if i not in an.lost_frames]
    if frame is None:
        frame = good[0] if good else an.reference_frame
    note = ""
    if frame in an.lost_frames:
        note = (f"frame {frame} is one where the stars were not recovered; positions are "
                "extrapolated from the field shift and may be off")
    dx, dy = an.shifts[frame]

    def pos(k: int) -> list[int]:
        x, y = an.positions[frame, k]
        if not np.isfinite(x):
            x, y = an.stars[k].x + dx, an.stars[k].y + dy
        return [int(round(x)), int(round(y))]

    target = pos(an.target_idx)
    comps = [pos(k) for k in an.comp_idx][:10]
    return frame, target, comps, note


# --------------------------------------------------------------- inits.json

def _obs_date(an: SessionAnalysis) -> str:
    d = datetime.fromtimestamp((float(an.mjd[0]) - 40587.0) * 86400.0, tz=timezone.utc)
    return f"{d.day}-{d:%B-%Y}"


def _v(x, nd: int | None = None):
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return x
    if not np.isfinite(f):
        return None
    return round(f, nd) if nd is not None else f


def inits(an: SessionAnalysis, obscode: str = "", fits_dir: str | None = None,
          darks_dir: str | None = None, save_dir: str | None = None,
          pixel_frame: int | None = None, prereduced_name: str | None = None) -> dict:
    tm = timing(an)
    cat = tm["catalog"]
    ra_s, dec_s = sexagesimal(tm["ra_deg"], tm["dec_deg"])
    frame, tpix, cpix, pix_note = pixel_positions(an, pixel_frame)
    comps10 = cpix + [[] for _ in range(10 - len(cpix))]
    t0, t0_src = predicted_midtransit(an, tm)
    filt = AAVSO_FILTER.get(an.filter_name.lower(), an.filter_name or "CV")
    sid_safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", an.ref)
    fits_dir = fits_dir or (display_path(an.session_dir) or f"<folder with the {an.n_frames} FITS frames of {an.ref}>")
    darks_dir = darks_dir if darks_dir is not None else display_path(an.calibration_dir)
    save_dir = save_dir or f"exotic_output_{sid_safe}"
    excluded = [Path(an.frame_paths[i]).name for i in an.lost_frames] if an.frame_paths else an.lost_frames
    pixel_file = Path(an.frame_paths[frame]).name if an.frame_paths else f"frame index {frame}"

    notes = (f"MicroObservatory 'Cecilia', {OBSERVATORY['name']}. Frames analysed by "
             f"ExoTransit Lab API v{settings.VERSION} (session {an.ref}). Target/comparison "
             f"pixels are zero-based (x=column, y=row) in {pixel_file}. "
             f"{len(an.stars)} stars detected; differential scatter {an.rms_ppt:.1f} ppt.")
    if excluded:
        notes += (f" {len(excluded)} frames had no recoverable stars (twilight/cloud) and "
                  "are listed under exotransit_lab.frames_to_exclude; move them out of the "
                  "FITS folder before running EXOTIC -red.")
    if pix_note:
        notes += " " + pix_note + "."

    return {
        "inits_guide": {
            "Title": "EXOTIC's Initialization File",
            "Comment": f"Generated by ExoTransit Lab API for session {an.ref}. Edit the three "
                       "directories to match your machine, then run:",
            "Comment1": "  exotic -red inits.json -nea      (reduce the raw frames with EXOTIC)",
            "Comment2": "  exotic -pre inits.json -nea      (fit EXOTIC's model to our light curve)",
            "Comment3": "-nea adopts the NASA Exoplanet Archive parameters; use -ov to force the "
                        "planetary_parameters below instead. null means unknown here.",
            "Comment4": "Formats: see github.com/rzellem/EXOTIC. Do not remove quotes, commas or brackets.",
        },
        "user_info": {
            "Directory with FITS files": fits_dir,
            "Directory to Save Plots": save_dir,
            "Directory of Flats": None,
            "Directory of Darks": darks_dir,
            "Directory of Biases": None,
            "AAVSO Observer Code (blank if none)": obscode or "",
            "Secondary Observer Codes (blank if none)": "",
            "Observation date": _obs_date(an),
            "Obs. Latitude": f"{OBSERVATORY['lat_deg']:+.4f}",
            "Obs. Longitude": f"{OBSERVATORY['lon_deg']:+.4f}",
            "Obs. Elevation (meters)": OBSERVATORY["elev_m"],
            "Camera Type (CCD or DSLR)": "CCD",
            "Pixel Binning": an.binning,
            "Filter Name (aavso.org/filters)": filt,
            "Observing Notes": notes,
            "Plate Solution? (y/n)": "n",
            "Add Comparison Stars from AAVSO? (y/n)": "n",
            "Target Star X & Y Pixel": json.dumps(tpix),
            "Comparison Star(s) X & Y Pixel": json.dumps(comps10),
            "Demosaic Format": None,
            "Demosaic Output": None,
        },
        "planetary_parameters": {
            "Target Star RA": ra_s,
            "Target Star Dec": dec_s,
            "Planet Name": cat.get("pl_name") or f"{an.target} b",
            "Host Star Name": cat.get("hostname") or an.target,
            "Orbital Period (days)": _v(cat.get("pl_orbper")),
            "Orbital Period Uncertainty": _v(cat.get("pl_orbpererr1")),
            "Published Mid-Transit Time (BJD-UTC)": _v(cat.get("pl_tranmid")),
            "Mid-Transit Time Uncertainty": None,
            "Ratio of Planet to Stellar Radius (Rp/Rs)": _v(cat.get("pl_ratror")),
            "Ratio of Planet to Stellar Radius (Rp/Rs) Uncertainty": None,
            "Ratio of Distance to Stellar Radius (a/Rs)": _v(cat.get("pl_ratdor")),
            "Ratio of Distance to Stellar Radius (a/Rs) Uncertainty": None,
            "Orbital Inclination (deg)": _v(cat.get("pl_orbincl")),
            "Orbital Inclination (deg) Uncertainty": None,
            "Orbital Eccentricity (0 if null)": 0,
            "Argument of Periastron (deg)": 0,
            "Star Effective Temperature (K)": _v(cat.get("st_teff")),
            "Star Effective Temperature (+) Uncertainty": _v(cat.get("st_tefferr1")),
            "Star Effective Temperature (-) Uncertainty": (-_v(cat.get("st_tefferr1"))
                                                           if _v(cat.get("st_tefferr1")) is not None else None),
            "Star Metallicity ([FE/H])": None,
            "Star Metallicity (+) Uncertainty": None,
            "Star Metallicity (-) Uncertainty": None,
            "Star Surface Gravity (log(g))": None,
            "Star Surface Gravity (+) Uncertainty": None,
            "Star Surface Gravity (-) Uncertainty": None,
            "Star Distance (pc)": _v(cat.get("sy_dist")),
            "Star Proper Motion RA (mas/yr)": None,
            "Star Proper Motion DEC (mas/yr)": None,
        },
        "optional_info": {
            "Pre-reduced File:": prereduced_name or f"prereduced_{sid_safe}.csv",
            "Pre-reduced File Time Format (BJD_TDB, JD_UTC, MJD_UTC)": "BJD_TDB",
            "Pre-reduced File Units of Flux (flux, magnitude, millimagnitude)": "flux",
            "Filter Minimum Wavelength (nm)": None,
            "Filter Maximum Wavelength (nm)": None,
            "Image Scale (Ex: 5.21 arcsecs/pixel)": "5.0",
            "Exposure Time (s)": _v(an.exptime_s) if an.exptime_s else 60.0,
        },
        "exotransit_lab": {
            "session": an.ref,
            "api": f"/api/field/summary?session={an.ref}",
            "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "pixel_reference_frame": frame,
            "pixel_reference_file": pixel_file,
            "pixel_convention": "zero-based, x = column, y = row of the FITS array",
            "frames_to_exclude": excluded,
            "coordinates_source": tm["coord_source"],
            "predicted_mid_transit_bjd_tdb": t0,
            "predicted_mid_transit_source": t0_src,
            "our_light_curve": {"rms_ppt": round(an.rms_ppt, 2), "n_points": int(np.isfinite(an.norm_flux).sum()),
                                "dip": an.dip.get("statement"), "reading": an.reading},
        },
    }


# ----------------------------------------------------------- pre-reduced

def _curve(an: SessionAnalysis, tm: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ok = np.isfinite(an.norm_flux)
    f = an.norm_flux[ok]
    err = np.full_like(f, max(an.rms_ppt, 0.1) / 1000.0) * f
    return tm["bjd_tdb"][ok], f, err, tm["airmass"][ok]


def prereduced(an: SessionAnalysis) -> str:
    """EXOTIC `-pre` input: comma-separated BJD_TDB, flux, uncertainty, airmass.
    EXOTIC skips any line it cannot parse, so the `#` header is safe."""
    tm = timing(an)
    t, f, e, am = _curve(an, tm)
    lines = [
        f"# ExoTransit Lab pre-reduced light curve for EXOTIC (-pre). Session {an.ref}, target {an.target}.",
        "# Columns: BJD_TDB, normalised differential flux (target / sum of comparison stars, median = 1),",
        f"# uncertainty (robust scatter of the curve, {an.rms_ppt:.1f} ppt, applied per point), airmass ({tm['airmass_source']}).",
        f"# Coordinates for the barycentric correction: RA {tm['ra_deg']:.6f}, Dec {tm['dec_deg']:+.6f} ({tm['coord_source']}).",
        "# BJD_TDB,flux,flux_err,airmass",
    ]
    lines += [f"{ti:.8f},{fi:.7f},{ei:.7f},{ai:.5f}" for ti, fi, ei, ai in zip(t, f, e, am)]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------- AAVSO

def _pm(value, err) -> str:
    if value is None:
        return ""
    return f"{value} +/- {err}" if err is not None else f"{value}"


def aavso(an: SessionAnalysis, obscode: str = "", secondary: str = "", notes: str = "") -> str:
    """AAVSO Exoplanet Database report, in the layout EXOTIC writes."""
    tm = timing(an)
    cat = tm["catalog"]
    t, f, e, am = _curve(an, tm)
    t0, t0_src = predicted_midtransit(an, tm)
    filt = AAVSO_FILTER.get(an.filter_name.lower(), an.filter_name or "CV")
    _frame, tpix, cpix, _n = pixel_positions(an, None)

    priors = {"Period": _pm(_v(cat.get("pl_orbper")), _v(cat.get("pl_orbpererr1"))),
              "a/R*": _pm(_v(cat.get("pl_ratdor")), None),
              "inc": _pm(_v(cat.get("pl_orbincl")), None),
              "Rp/R*": _pm(_v(cat.get("pl_ratror")), None),
              "Tc": _pm(_v(t0, 6), None),
              "ecc": "0"}
    priors_s = ",".join(f"{k}={v}" for k, v in priors.items() if v)

    results_s = ""
    d = an.dip
    if d.get("detected"):
        a, b = d["start_index"], d["end_index"]
        tc = float(0.5 * (tm["bjd_tdb"][a] + tm["bjd_tdb"][b]))
        cad = float(np.median(np.diff(an.seconds))) / 86400.0
        rprs = float(np.sqrt(max(d["depth"], 0.0)))
        rprs_err = float(0.5 * (d["scatter_ppt"] / 1000.0) / max(rprs, 1e-3) / np.sqrt(d["n_points"]))
        results_s = (f"Tc={tc:.6f} +/- {cad/2:.6f},Rp/R*={rprs:.5f} +/- {rprs_err:.5f},"
                     f"Duration_h={d['duration_min']/60:.3f} +/- {cad*24/2:.3f}")

    auto_notes = (f"MicroObservatory Cecilia, {OBSERVATORY['name']}. Differential aperture "
                  f"photometry by ExoTransit Lab API (session {an.ref}); {len(an.comp_idx)} comparison "
                  f"stars; {len(t)} of {an.n_frames} frames kept; scatter {an.rms_ppt:.1f} ppt. "
                  "RESULTS, when present, come from a dip search on the light curve, not from a "
                  "transit-model fit; for a fitted result run EXOTIC -pre on the pre-reduced file. "
                  f"{an.reading}")
    if notes:
        auto_notes = notes + " | " + auto_notes

    hdr = [
        "#TYPE=EXOPLANET",
        f"#OBSCODE={obscode}",
        f"#SECONDARY_OBSCODES={secondary}",
        f"#SOFTWARE=ExoTransit Lab API v{settings.VERSION} (EXOTIC-compatible export)",
        "#DELIM=,",
        "#DATE_TYPE=BJD_TDB",
        "#OBSTYPE=CCD",
        f"#STAR_NAME={cat.get('hostname') or an.target}",
        f"#EXOPLANET_NAME={cat.get('pl_name') or an.target + ' b'}",
        f"#BINNING={an.binning}",
        f"#EXPOSURE_TIME={_v(an.exptime_s) if an.exptime_s else 60.0}",
        "#COMP_STAR-XC=" + json.dumps({"target_xy": tpix, "comparison_xy": cpix,
                                      "note": "zero-based pixel positions in the first usable frame"}),
        f"#NOTES={auto_notes}",
        "#DETREND_PARAMETERS=AIRMASS",
        "#MEASUREMENT_TYPE=Rnflux",
        f"#FILTER={filt}",
        f"#PRIORS={priors_s}",
        "#PRIORS-XC=" + json.dumps({"source": cat.get("catalog_source") or cat.get("ephemeris_source") or "unknown",
                                   "Tc_source": t0_src}),
    ]
    if results_s:
        hdr.append(f"#RESULTS={results_s}")
        hdr.append("#RESULTS-XC=" + json.dumps({"method": "dip search, not a model fit",
                                               "depth_ppt": round(d["depth_ppt"], 1),
                                               "significance_sigma": round(d["significance_sigma"], 1)}))
    hdr.append("#DATE,DIFF,ERR,DETREND_1")
    rows = [f"{ti:.8f},{fi:.7f},{ei:.7f},{ai:.7f}" for ti, fi, ei, ai in zip(t, f, e, am)]
    return "\n".join(hdr + rows) + "\n"


# ------------------------------------------------------------------ bundle

def bundle(an: SessionAnalysis, obscode: str = "", fits_dir: str | None = None,
           darks_dir: str | None = None) -> bytes:
    sid_safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", an.ref)
    pre_name = f"prereduced_{sid_safe}.csv"
    ini = inits(an, obscode=obscode, fits_dir=fits_dir, darks_dir=darks_dir, prereduced_name=pre_name)
    excluded = ini["exotransit_lab"]["frames_to_exclude"]
    readme = f"""ExoTransit Lab -> EXOTIC bundle for session {an.ref} ({an.target})
=====================================================================

Files
  inits.json                EXOTIC initialisation file, filled in from our analysis
  {pre_name:<25} our differential light curve in EXOTIC's pre-reduced format
  aavso_{sid_safe}.txt{' ' * max(1, 19 - len(sid_safe))} AAVSO Exoplanet Database report (needs an observer code)

Run EXOTIC on the raw frames
  1. pip install exotic
  2. Put the session's FITS frames in one folder and its dark frames in another.
     Edit "Directory with FITS files" and "Directory of Darks" in inits.json.
  3. exotic -red inits.json -nea

Fit EXOTIC's transit model to our light curve instead
  exotic -pre inits.json -nea
  ("Pre-reduced File:" in inits.json already names {pre_name}; keep the two files together.)

Notes
  * Target and comparison-star pixels are zero-based (x = column, y = row) in
    {ini['exotransit_lab']['pixel_reference_file']}. EXOTIC re-centres within a few pixels.
  * {len(excluded)} frame(s) had no recoverable stars (twilight or cloud) and are
    listed in inits.json under exotransit_lab.frames_to_exclude. Move them out of
    the FITS folder first, or EXOTIC will fail to find the target in them.
  * Planetary parameters come from the NASA Exoplanet Archive tables the pipeline
    carries; -nea makes EXOTIC fetch the current archive values itself.
  * Our AAVSO report's RESULTS line, when present, is from a dip search, not a
    transit-model fit. Submit EXOTIC's own output for a fitted result.
  * Reading of this night: {an.reading}
"""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("README.txt", readme)
        z.writestr("inits.json", json.dumps(ini, indent=4))
        z.writestr(pre_name, prereduced(an))
        z.writestr(f"aavso_{sid_safe}.txt", aavso(an, obscode=obscode))
    return buf.getvalue()
