#!/usr/bin/env python
"""
prepare_data.py -- package the MicroObservatory FITS archive into a compact,
Kaggle-friendly bundle.

Why this exists
---------------
The raw archive is 1741 individual FITS files (~1.1 GB). Opening 1741 files one
at a time inside a Kaggle notebook is slow, and uploading 1741 loose files is
worse. This script does the file-level work ONCE, locally, and emits:

  frames/<TARGET>__<NIGHT>.npy   uint16 (N, 500, 650)   one array per session
  darks/darks.npy                uint16 (60, 500, 650)  all dark frames
  calib/master_dark.npy          float32 (500, 650)     median-combined dark
  calib/hot_pixel_mask.npy       bool    (500, 650)     >10 sigma in master
  metadata/frames.csv            one row per science frame, full header + stats
  metadata/darks.csv             one row per dark frame
  metadata/sessions.csv          one row per (target, night) observing session

Nothing is thrown away: every header keyword read is preserved in the CSVs, and
the pixel data is bit-identical to the FITS payload. The int16 -> uint16 cast is
a lossless reinterpretation here because DATAMIN = 0 and BZERO = 0; the script
verifies this per frame and aborts if any negative pixel turns up.

Usage:  python prepare_data.py [--src DIR] [--out DIR]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from astropy.io import fits

# Header keywords copied verbatim into the CSVs. Anything not listed is either
# constant across the whole archive or FITS bookkeeping we do not need.
HEADER_KEYS = [
    "OBJECT", "DATE-OBS", "DATE-END", "MJD-OBS", "EXPTIME", "FILTER",
    "RA", "DEC", "TELALT", "TELAZ", "HA",
    "CAMTEMP", "TELTEMP", "TELHUM", "WEATHER", "CAMFOCUS",
    "NAXIS1", "NAXIS2", "XBINNING", "YBINNING", "IM_SCALE",
    "DATAMAX", "DATAMIN", "BSCALE", "BZERO",
    "XPIXSZ", "YPIXSZ", "FOCALLEN",
    "TELESCOP", "OBSERVAT", "LATITUDE", "LONGITUD", "HEIGHT",
    "CREATOR", "ORIGIN",
]

SIGMA_FROM_MAD = 1.4826  # MAD -> Gaussian sigma


def robust_stats(img):
    """Median and MAD-derived sigma. Robust to stars, hot pixels, cosmic rays."""
    med = float(np.median(img))
    mad = float(np.median(np.abs(img.astype(np.float32) - med)))
    return med, mad * SIGMA_FROM_MAD


def airmass_from_alt(alt_deg):
    """Plane-parallel airmass, sec(z). It diverges at the horizon, so clamp the
    altitude at 3 deg -- below that the number is meaningless anyway."""
    if alt_deg is None:
        return None
    alt = max(float(alt_deg), 3.0)
    return float(1.0 / np.cos(np.radians(90.0 - alt)))


def read_frame(path):
    """Return (header dict, uint16 image). Raises if any pixel is negative."""
    with fits.open(path, memmap=False) as hdul:
        hdr = hdul[0].header
        data = hdul[0].data
        row = {k: hdr.get(k) for k in HEADER_KEYS}
    if data is None:
        raise ValueError(path + ": no image data")
    if data.min() < 0:
        raise ValueError(path + ": negative pixel, uint16 cast would be lossy")
    return row, np.ascontiguousarray(data, dtype=np.uint16)


def frame_quality(img, hot_mask):
    """Per-frame image statistics for the dashboard Data Quality page.

    Hot pixels are excluded before measuring peak_contrast and n_source_px.
    Otherwise the 625 permanently-bright pixels dominate every single frame and
    the metrics stop tracking the sky, which is the thing we actually want.
    """
    f = img.astype(np.float32)
    med, sigma = robust_stats(f)
    clean = f[~hot_mask] if hot_mask is not None else f.ravel()
    thresh = med + 5.0 * sigma if sigma > 0 else med + 1.0
    peak = float(clean.max())
    return {
        "sky_level": med,
        "sky_sigma": sigma,
        "img_mean": float(f.mean()),
        "img_min": float(f.min()),
        "img_max": float(f.max()),
        "peak_clean": peak,
        # how far the brightest real source stands above the noise
        "peak_contrast": float((peak - med) / sigma) if sigma > 0 else np.nan,
        "n_source_px": int((clean > thresh).sum()),
        "n_saturated": int((f >= 4095).sum()),
    }


def walk_fits(root):
    out = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith((".fits", ".fit", ".fts")):
                out.append(os.path.join(dirpath, fn))
    return sorted(out)


# ---------------------------------------------------------------- stage: darks

def build_darks(src, out):
    paths = walk_fits(os.path.join(src, "calibration"))
    print("[darks] %d calibration files" % len(paths))
    if not paths:
        sys.exit("[darks] none found -- wrong --src?")

    stack, rows = [], []
    for i, p in enumerate(paths, 1):
        hdr, img = read_frame(p)
        med, sigma = robust_stats(img)
        row = {
            "filename": os.path.basename(p),
            "night": os.path.basename(os.path.dirname(p)),
            "relpath": os.path.relpath(p, src).replace("\\", "/"),
            "dark_index": i - 1,
            "dark_median": med,
            "dark_sigma": sigma,
            "dark_max": float(img.max()),
        }
        row.update(hdr)
        rows.append(row)
        stack.append(img)
        if i % 20 == 0 or i == len(paths):
            print("  %d/%d" % (i, len(paths)))

    darks = np.stack(stack)  # (60, 500, 650) uint16
    master = np.median(darks, axis=0).astype(np.float32)

    # Hot pixels: >10 sigma above the master dark's own robust background.
    m_med, m_sigma = robust_stats(master)
    hot = master > (m_med + 10.0 * m_sigma)
    print("[darks] master median=%.2f sigma=%.3f hot=%d (%.3f%% of chip)"
          % (m_med, m_sigma, hot.sum(), 100 * hot.mean()))

    os.makedirs(os.path.join(out, "darks"), exist_ok=True)
    os.makedirs(os.path.join(out, "calib"), exist_ok=True)
    np.save(os.path.join(out, "darks", "darks.npy"), darks)
    np.save(os.path.join(out, "calib", "master_dark.npy"), master)
    np.save(os.path.join(out, "calib", "hot_pixel_mask.npy"), hot)
    return darks, hot, pd.DataFrame(rows)


# -------------------------------------------------------------- stage: science

def build_science(src, out, hot):
    paths = walk_fits(os.path.join(src, "observations"))
    print("[science] %d science files" % len(paths))
    if not paths:
        sys.exit("[science] none found -- wrong --src?")

    # Group by (target, night) first so we never hold more than one session
    # in memory at a time.
    groups = defaultdict(list)
    for p in paths:
        parts = os.path.normpath(p).split(os.sep)
        # .../observations/<NIGHT>/<TARGET>/<session>/<file>.fits
        night, target = parts[-4], parts[-3]
        groups[(target, night)].append(p)

    os.makedirs(os.path.join(out, "frames"), exist_ok=True)
    frame_rows, session_rows = [], []
    done = 0

    for key in sorted(groups):
        target, night = key
        session_id = target + "__" + night
        cube, rows = [], []

        for idx, p in enumerate(sorted(groups[key])):
            hdr, img = read_frame(p)
            row = {
                "session_id": session_id,
                "target": target,
                "night": night,
                "frame_index": idx,
                "filename": os.path.basename(p),
                "relpath": os.path.relpath(p, src).replace("\\", "/"),
                "airmass": airmass_from_alt(hdr.get("TELALT")),
            }
            row.update(frame_quality(img, hot))
            row.update(hdr)
            rows.append(row)
            cube.append(img)
            done += 1

        arr = np.stack(cube)
        np.save(os.path.join(out, "frames", session_id + ".npy"), arr)

        df = pd.DataFrame(rows)
        t = pd.to_datetime(df["DATE-OBS"], format="ISO8601", utc=True).sort_values()
        span_h = (t.iloc[-1] - t.iloc[0]).total_seconds() / 3600.0
        gaps = t.diff().dt.total_seconds().dropna()
        session_rows.append({
            "session_id": session_id,
            "target": target,
            "night": night,
            "n_frames": len(df),
            "start_utc": t.iloc[0].isoformat(),
            "end_utc": t.iloc[-1].isoformat(),
            "span_hours": round(span_h, 3),
            "median_cadence_s": round(float(gaps.median()), 1) if len(gaps) else None,
            "max_gap_s": round(float(gaps.max()), 1) if len(gaps) else None,
            "mjd_start": float(df["MJD-OBS"].min()),
            "mjd_end": float(df["MJD-OBS"].max()),
            "airmass_min": round(float(df["airmass"].min()), 3),
            "airmass_max": round(float(df["airmass"].max()), 3),
            "median_sky": round(float(df["sky_level"].median()), 2),
            "median_weather": float(df["WEATHER"].median()),
            "npy": "frames/" + session_id + ".npy",
            "shape": str(arr.shape),
            "mb": round(arr.nbytes / 1e6, 1),
        })
        frame_rows.extend(rows)
        print("  %-24s n=%3d span=%4.2fh  (%d/%d)"
              % (session_id, len(df), span_h, done, len(paths)))

    return pd.DataFrame(frame_rows), pd.DataFrame(session_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=r"C:\Users\Lenovo\Downloads\database\database")
    ap.add_argument("--out", default=r"C:\Users\Lenovo\Downloads\database\kaggle\build")
    args = ap.parse_args()

    t0 = time.time()
    md = os.path.join(args.out, "metadata")
    os.makedirs(md, exist_ok=True)

    _, hot, darks_df = build_darks(args.src, args.out)
    frames_df, sessions_df = build_science(args.src, args.out, hot)

    darks_df.to_csv(os.path.join(md, "darks.csv"), index=False)
    frames_df.to_csv(os.path.join(md, "frames.csv"), index=False)
    sessions_df.to_csv(os.path.join(md, "sessions.csv"), index=False)

    # Carry the organisers' own metadata along so the bundle is self-contained.
    for name in ("observations.csv", "observations.json", "download_log.csv"):
        srcp = os.path.join(args.src, "metadata", name)
        if os.path.exists(srcp):
            with open(srcp, "rb") as fi:
                blob = fi.read()
            with open(os.path.join(md, "original_" + name), "wb") as fo:
                fo.write(blob)

    total = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fs in os.walk(args.out) for f in fs
    )
    print("\n[done] %d science + %d darks -> %.1f MB in %.0fs"
          % (len(frames_df), len(darks_df), total / 1e6, time.time() - t0))
    print("[done] %d sessions written to %s" % (len(sessions_df), args.out))


if __name__ == "__main__":
    main()
