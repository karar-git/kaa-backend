# %% [markdown]
# # ExoTransit Lab — 01 · Calibration, Detection, Auto-Labelling, Photometry
#
# **Hack4Dev Iraq 2026 — Exoplanet Data Challenge · Challenge E (Discovery Tool)**
# Team: Iraqi Andromeda
#
# This notebook turns 1,681 raw MicroObservatory frames into everything the web
# dashboard and the ML notebook consume. It is the *only* place science happens —
# the website merely displays what this notebook writes to `/kaggle/working/results/`.
#
# **Pipeline**
#
# | § | Step | Output |
# |---|------|--------|
# | 1 | Load & audit the packaged archive | `dataset_audit.json`, `per_target.csv` |
# | 2 | Master dark, noise-vs-N proof, hot-pixel map | `calibration.json`, `noise_vs_n.csv` |
# | 3 | Session quality triage — which nights are usable at all | `session_quality.csv`, `target_usability.csv` |
# | 4 | Source detection (the baseline the ML must beat) | `detections.csv.gz` |
# | 5 | Frame-to-frame track linking (drift **and** field rotation) | `tracks.csv`, `field_motion.csv` |
# | 6 | Physics-derived auto-labels | `tracks.csv`, `label_counts.csv` |
# | 7 | Run over all 22 sessions | `field_motion_summary.csv` |
# | 8 | 32×32 cutout export for the training notebook | `cutouts.npz`, `cutouts_meta.csv` |
# | 9 | Aperture photometry + differential light curves | `lightcurves.csv`, `photometry_info.csv` |
# | 10 | BJD_TDB timestamps + NASA ephemerides | `ephemerides.csv`, `transit_predictions.csv` |
# | 11 | Depth measurement in the predicted window | `transit_depths.csv` |
# | 12 | Phase folding across repeat nights | `phasefold.csv`, `phasefold_binned.csv` |
# | 13 | Field-wide photometry + ephemeris-guided target search | `field_photometry.csv.gz`, `target_search.csv` |
# | 13b | Planet physics: orbit size, speed, temperature, radius, distance — validated against the archive | `planet_physics.csv`, `physics_validation.csv`, `planet_radius_measurements.csv`, `planet_catalog.csv` |
# | 14 | Observing-condition correlations | `quality_correlations.csv`, `frame_quality.csv` |
# | 15 | Dashboard image assets | `static/` |
#
# **Nothing here is simulated.** Every number comes from the pixels. Where a value
# could not be measured it is written as `NaN`, never invented.

# %%
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy import stats

warnings.filterwarnings("ignore", category=RuntimeWarning)

RNG = np.random.default_rng(42)  # every random choice in this notebook is seeded


def find_dataset_root() -> Path:
    """Locate the packaged bundle whether we are on Kaggle or on a laptop.

    Search rather than guess. Kaggle mounts a dataset at /kaggle/input/<slug>/,
    and depending on how the zip was built it may nest one level deeper again --
    checking only a fixed depth is what made this fail the first time.
    """
    candidates = [
        Path("/kaggle/input"),      # attached dataset
        Path("./build"),            # prepare_data.py output, run from kaggle/
        Path("../build"),
        Path("."),
        Path(".."),
    ]
    for base in candidates:
        if not base.exists():
            continue
        if (base / "metadata" / "frames.csv").exists():
            return base
        for hit in sorted(base.rglob("frames.csv")):
            if hit.parent.name == "metadata":
                return hit.parent.parent

    # Say what we actually saw. "Not found" on its own gives no way to tell an
    # unattached dataset from one mounted at an unexpected depth.
    lines = ["Could not find metadata/frames.csv.", ""]
    inp = Path("/kaggle/input")
    if inp.exists():
        kids = sorted(inp.iterdir())
        if not kids:
            lines += ["/kaggle/input is EMPTY -- the dataset is not attached.",
                      "Right sidebar > Input > + Add Input > your dataset."]
        else:
            lines.append("/kaggle/input contains:")
            for k in kids:
                lines.append(f"  {k.name}/")
                if k.is_dir():
                    for sub in sorted(k.iterdir())[:12]:
                        lines.append(f"    {sub.name}" + ("/" if sub.is_dir() else ""))
    else:
        lines.append("Not on Kaggle. Run prepare_data.py first to build ./build/.")
    raise FileNotFoundError("\n".join(lines))


DATA = find_dataset_root()
ON_KAGGLE = Path("/kaggle/input").exists()
OUT = Path("/kaggle/working") if ON_KAGGLE else Path("./out")
RESULTS = OUT / "results"
STATIC = OUT / "static"
for d in (RESULTS, STATIC):
    d.mkdir(parents=True, exist_ok=True)

print("data   :", DATA)
print("output :", OUT)
print("files  :", sorted(p.name for p in DATA.iterdir()))

# %% [markdown]
# ## 1 · Load and audit
#
# The first thing a judge should see is that we *checked* the archive rather than
# trusting it. Two things are worth stating out loud:
#
# * The challenge brief says **~1,633 records**; the archive as downloaded holds
#   **1,741**. We use the real count and flag the difference.
# * The organisers' own `index.py` reads header key `EXPOSURE`, but MicroObservatory
#   writes `EXPTIME`. Every row of the supplied `dataset_index.csv` therefore has an
#   empty exposure column. Our packaging script reads the correct key.

# %%
frames = pd.read_csv(DATA / "metadata" / "frames.csv")
darks_meta = pd.read_csv(DATA / "metadata" / "darks.csv")
sessions = pd.read_csv(DATA / "metadata" / "sessions.csv")

frames["t_utc"] = pd.to_datetime(frames["DATE-OBS"], format="ISO8601", utc=True)
frames = frames.sort_values(["session_id", "frame_index"]).reset_index(drop=True)

audit = {
    "science_frames": len(frames),
    "dark_frames": len(darks_meta),
    "total_files": len(frames) + len(darks_meta),
    "brief_claims": 1633,
    "discrepancy": len(frames) + len(darks_meta) - 1633,
    "targets": frames["target"].nunique(),
    "sessions": len(sessions),
    "date_first": str(frames["t_utc"].min()),
    "date_last": str(frames["t_utc"].max()),
    "exptime_unique": sorted(frames["EXPTIME"].unique().tolist()),
    "filters_science": sorted(frames["FILTER"].dropna().unique().tolist()),
    "filters_dark": sorted(darks_meta["FILTER"].dropna().unique().tolist()),
    "image_shape": [int(frames["NAXIS2"].iloc[0]), int(frames["NAXIS1"].iloc[0])],
    "arcsec_per_px": float(frames["IM_SCALE"].iloc[0]),
    "camtemp_K_range": [float(frames["CAMTEMP"].min()), float(frames["CAMTEMP"].max())],
    "airmass_range": [round(float(frames["airmass"].min()), 3),
                      round(float(frames["airmass"].max()), 3)],
}
print(json.dumps(audit, indent=2))

# Per-target summary — this becomes the dashboard's Mission Overview table.
per_target = (
    frames.groupby("target")
    .agg(n_frames=("filename", "size"),
         n_nights=("night", "nunique"),
         first_night=("night", "min"),
         last_night=("night", "max"),
         ra_deg=("RA", "median"),
         dec_deg=("DEC", "median"),
         median_sky=("sky_level", "median"),
         median_airmass=("airmass", "median"))
    .reset_index()
    .sort_values("n_frames", ascending=False)
)
per_target.to_csv(RESULTS / "per_target.csv", index=False)
sessions.to_csv(RESULTS / "sessions.csv", index=False)
(RESULTS / "dataset_audit.json").write_text(json.dumps(audit, indent=2))
per_target

# %% [markdown]
# ## 2 · Calibration
#
# ### Why a single dark frame is worse than none
#
# A dark frame contains two very different things:
#
# * **fixed pattern** — hot pixels and dark current, identical every night. Subtracting
#   it *removes* signal-corrupting structure. This is what we want.
# * **random read noise** — different every frame. Subtracting one dark *injects* its
#   read noise into your science frame instead of removing anything.
#
# So we median-combine all 60. The random component then falls as 1/√N while the
# fixed pattern stays. The cell below measures that fall-off directly instead of
# assuming it, using disjoint halves so the two stacks share no frames.

# %%
darks = np.load(DATA / "darks" / "darks.npy")          # (60, 500, 650) uint16
master_dark = np.load(DATA / "calib" / "master_dark.npy")   # float32
hot_mask = np.load(DATA / "calib" / "hot_pixel_mask.npy")   # bool

SIGMA_FROM_MAD = 1.4826


def robust_stats(a):
    """Median and MAD-scaled sigma — unaffected by stars, hot pixels, cosmic rays."""
    a = np.asarray(a, dtype=np.float32)
    med = float(np.median(a))
    return med, float(np.median(np.abs(a - med))) * SIGMA_FROM_MAD


def random_noise_of_stack(cube, n):
    """Random (non-fixed-pattern) noise left in a mean-combined stack of n frames.

    Trick: build TWO stacks of n from disjoint frames and difference them. The fixed
    pattern is identical in both and cancels exactly; only random noise survives.
    Dividing by sqrt(2) undoes the variance doubling from the subtraction.
    """
    idx = RNG.permutation(len(cube))
    a = cube[idx[:n]].astype(np.float32).mean(axis=0)
    b = cube[idx[n:2 * n]].astype(np.float32).mean(axis=0)
    _, s = robust_stats(a - b)
    return s / np.sqrt(2.0)


rows = []
for n in (1, 2, 3, 5, 10, 15, 20, 25, 30):
    trials = [random_noise_of_stack(darks, n) for _ in range(5)]
    rows.append({
        "n_darks": n,
        "random_noise_counts": float(np.mean(trials)),
        "theory_1_over_sqrt_n": float(np.mean([random_noise_of_stack(darks, 1)
                                               for _ in range(5)]) / np.sqrt(n)),
    })
noise_curve = pd.DataFrame(rows)
noise_curve.to_csv(RESULTS / "noise_vs_n.csv", index=False)

m_med, m_sig = robust_stats(master_dark)
one_med, one_sig = robust_stats(darks[0])
corr_first_last = float(np.corrcoef(darks[0].ravel(), darks[-1].ravel())[0, 1])

calib_summary = {
    "n_darks": int(len(darks)),
    "master_median_counts": round(m_med, 3),
    "master_spatial_sigma_counts": round(m_sig, 3),
    "single_dark_spatial_sigma_counts": round(one_sig, 3),
    "random_noise_1_dark": round(noise_curve.iloc[0]["random_noise_counts"], 3),
    "random_noise_full_stack": round(noise_curve.iloc[-1]["random_noise_counts"], 3),
    "hot_pixels": int(hot_mask.sum()),
    "hot_pixel_fraction_pct": round(100 * float(hot_mask.mean()), 4),
    "fixed_pattern_stability_corr_30_nights": round(corr_first_last, 4),
    "camtemp_K_range_darks": [float(darks_meta["CAMTEMP"].min()),
                              float(darks_meta["CAMTEMP"].max())],
}
(RESULTS / "calibration.json").write_text(json.dumps(calib_summary, indent=2))
print(json.dumps(calib_summary, indent=2))
noise_curve

# %% [markdown]
# The measured curve should track the 1/√N line closely. Where it flattens at large
# N you are hitting the 12-bit quantisation floor, not a mistake — the ADC simply
# cannot represent a difference smaller than a fraction of a count.
#
# ### Temperature matching
#
# Dark current roughly doubles every ~6 °C, so a dark taken at 285 K does not
# describe a frame taken at 276 K. We therefore build **per-temperature masters**
# and pick the nearest one at calibration time.

# %%
temp_masters = {}
for temp, grp in darks_meta.groupby("CAMTEMP"):
    idx = grp["dark_index"].to_numpy()
    temp_masters[float(temp)] = np.median(darks[idx].astype(np.float32), axis=0)

temp_table = pd.DataFrame([
    {"camtemp_K": t, "n_darks": int((darks_meta["CAMTEMP"] == t).sum()),
     "master_median": round(float(np.median(m)), 3)}
    for t, m in sorted(temp_masters.items())
])
temp_table.to_csv(RESULTS / "dark_masters_by_temp.csv", index=False)
print("dark masters built for temperatures:", sorted(temp_masters))
temp_table

# %%
HOT_Y, HOT_X = np.where(hot_mask)


def nearest_master(camtemp):
    """Pick the temperature-matched master dark; fall back to the global one."""
    if camtemp is None or not np.isfinite(camtemp):
        return master_dark
    temps = np.array(sorted(temp_masters))
    return temp_masters[float(temps[np.argmin(np.abs(temps - camtemp))])]


def repair_hot_pixels(img):
    """Replace hot pixels with the median of their 3x3 neighbourhood.

    Done AFTER dark subtraction. Dark subtraction removes a hot pixel's mean level
    but not its excess shot noise, so the pixel stays unreliable and is better
    interpolated over than trusted.
    """
    out = img.copy()
    smooth = ndi.median_filter(img, size=3)
    out[hot_mask] = smooth[hot_mask]
    return out


def calibrate(img, camtemp, repair=True):
    cal = img.astype(np.float32) - nearest_master(camtemp)
    return repair_hot_pixels(cal) if repair else cal


def load_session(session_id):
    return np.load(DATA / "frames" / f"{session_id}.npy")


demo_id = "TRES-5__2026-08-26"
demo_cube = load_session(demo_id)
demo_meta = frames[frames.session_id == demo_id].reset_index(drop=True)
demo_raw = demo_cube[len(demo_cube) // 2]
demo_cal = calibrate(demo_raw, demo_meta["CAMTEMP"].iloc[len(demo_cube) // 2])

print(f"{demo_id}: {demo_cube.shape}, {demo_cube.nbytes/1e6:.0f} MB")
print("raw       median %.1f  sigma %.2f" % robust_stats(demo_raw))
print("calibrated median %.1f  sigma %.2f" % robust_stats(demo_cal))

# %% [markdown]
# ## 3 · Session quality triage — which nights are actually usable?
#
# Before measuring anything we ask a blunt question: **is this night worth
# analysing at all?** The per-frame statistics computed during packaging answer it
# without opening a single image again.
#
# Two measurements do the work:
#
# * `n_source_px` — how many pixels sit above sky + 5σ. A proxy for how many stars
#   are punching through.
# * `peak_contrast` — how far the brightest real (non-hot) pixel stands above the
#   noise. A proxy for how well the *target* is doing.
#
# ### What `WEATHER` actually means
#
# The header keyword `WEATHER` is not a warning flag. Correlating it against our own
# pixel measurements gives **r ≈ +0.75** with `peak_contrast`: it is a *transparency
# score where 100 means clear*. Good nights read 100, the washed-out ones read 0–20.
# We did not have to take that on faith — the pixels confirm it, and that
# cross-check is worth more than the keyword itself.
#
# The thresholds below are deliberately crude and stated up front. Being explicit
# about a cut beats hiding a subtle one.

# %%
GOOD_CONTRAST, GOOD_SOURCES = 300.0, 800.0
MARGINAL_CONTRAST = 100.0

q = (frames.groupby(["session_id", "target"])
     .agg(n_frames=("frame_index", "size"),
          median_contrast=("peak_contrast", "median"),
          median_sources=("n_source_px", "median"),
          min_sources=("n_source_px", "min"),
          median_sky=("sky_level", "median"),
          max_sky=("sky_level", "max"),
          median_sky_sigma=("sky_sigma", "median"),
          median_weather=("WEATHER", "median"),
          median_airmass=("airmass", "median"))
     .reset_index())


def tier(r):
    if r.median_contrast >= GOOD_CONTRAST and r.median_sources >= GOOD_SOURCES:
        return "good"
    if r.median_contrast >= MARGINAL_CONTRAST:
        return "marginal"
    return "unusable"


q["quality"] = q.apply(tier, axis=1)
# how badly the sky brightened during the night -- the signature of moon or cloud
q["sky_swing"] = (q.max_sky / q.median_sky).round(2)
q = q.sort_values(["quality", "median_contrast"], ascending=[True, False])
q.to_csv(RESULTS / "session_quality.csv", index=False)

SESSION_QUALITY = dict(zip(q.session_id, q.quality))
print(q[["session_id", "target", "n_frames", "median_contrast", "median_sources",
         "median_sky", "sky_swing", "median_weather", "quality"]]
      .to_string(index=False))

# %% [markdown]
# ### The headline result of the triage

# %%
print("sessions by quality:\n" + q.quality.value_counts().to_string())

by_target = (q.groupby("target")
             .agg(nights=("session_id", "size"),
                  good=("quality", lambda s: int((s == "good").sum())),
                  marginal=("quality", lambda s: int((s == "marginal").sum())),
                  unusable=("quality", lambda s: int((s == "unusable").sum())))
             .reset_index()
             .sort_values("good", ascending=False))
by_target.to_csv(RESULTS / "target_usability.csv", index=False)
print()
print(by_target.to_string(index=False))

lost = by_target[by_target.good == 0].target.tolist()
print(f"\nTargets with NO usable night: {', '.join(lost) if lost else 'none'}")
print("These cannot be analysed at all, and saying so is a result -- not a gap.")
print(f"\nUsable frames: {int(q[q.quality == 'good'].n_frames.sum())} of {len(frames)}"
      f" ({100*q[q.quality=='good'].n_frames.sum()/len(frames):.0f}%)")

# %% [markdown]
# Roughly a third of this survey is unusable, and three of the eight targets have no
# good night at all. That is not a disappointing outcome — it is the single most
# important thing to establish before drawing a light curve, because a dip measured
# on one of those nights would be pure noise dressed up as a discovery.
#
# **Everything downstream still runs on every session.** Bad nights are excellent
# training data for the classifier (a model that has only seen clear skies is
# useless), and their light curves become the *control sample*: any "transit" the
# pipeline reports on an unusable night is a false positive we can point at.

# %% [markdown]
# ## 4 · Source detection — the baseline the ML has to beat
#
# A machine-learning result means nothing without a dumb baseline beside it. This
# is the dumb baseline: estimate a smooth background, threshold at 8σ above it,
# label connected components, measure each blob. Transparent, fast, and close to
# what an astronomer would actually do by hand.
#
# Two decisions worth defending:
#
# **Background subtraction is not optional.** The raw frames have a 16-count
# gradient across the chip while the pixel-to-pixel noise is only ~3 counts. A
# single global threshold would therefore be far stricter on one side of the image
# than the other. We build a coarse block-median background and subtract it, which
# makes the detection threshold mean the same thing everywhere.
#
# **Detection runs on RAW frames on purpose.** Dark subtraction removes most hot
# pixels — which would delete the very thing we want the classifier to learn to
# reject. The discrimination problem lives in the raw data, so that is where we look.

# %%
from scipy.spatial import cKDTree

DET_SIGMA = 8.0
MIN_PIX = 4
MAX_PIX = 400
BG_BOX = 32


def background(im, box=BG_BOX):
    """Coarse block-median background, smoothed and resampled to full size.

    A median filter over a 32 px box would cost seconds per frame. Taking block
    medians, median-filtering the small grid, then bilinearly zooming back up
    gives essentially the same surface for ~1 ms.
    """
    im = np.asarray(im, dtype=np.float32)
    h, w = im.shape
    ph, pw = (-h) % box, (-w) % box
    p = np.pad(im, ((0, ph), (0, pw)), mode="edge")
    H, W = p.shape[0] // box, p.shape[1] // box
    coarse = np.median(
        p.reshape(H, box, W, box).transpose(0, 2, 1, 3).reshape(H, W, -1), axis=2)
    coarse = ndi.median_filter(coarse, size=3)
    return ndi.zoom(coarse, (p.shape[0] / H, p.shape[1] / W), order=1)[:h, :w]


DET_COLS = ["x", "y", "flux", "peak", "npix", "snr", "bbox_h", "bbox_w",
            "elongation", "fill_factor", "on_hot_pixel", "edge"]


def detect_frame(img, det_sigma=DET_SIGMA):
    """Detect sources in one raw frame. Returns (array of DET_COLS, sky, sigma)."""
    raw = np.asarray(img, dtype=np.float32)
    flat = raw - background(raw)
    med, sig = robust_stats(flat)
    if sig <= 0:
        return np.zeros((0, len(DET_COLS))), med, sig

    mask = flat > med + det_sigma * sig
    lbl, n = ndi.label(mask)
    if n == 0:
        return np.zeros((0, len(DET_COLS))), med, sig

    idx = np.arange(1, n + 1)
    sizes = np.asarray(ndi.sum(mask, lbl, idx))
    keep = (sizes >= MIN_PIX) & (sizes <= MAX_PIX)
    idx, sizes = idx[keep], sizes[keep]
    if len(idx) == 0:
        return np.zeros((0, len(DET_COLS))), med, sig

    cm = np.asarray(ndi.center_of_mass(flat, lbl, idx))          # (n, 2) as (y, x)
    flux = np.asarray(ndi.sum(flat, lbl, idx))
    peak = np.asarray(ndi.maximum(raw, lbl, idx))                # peak in RAW counts
    slices = ndi.find_objects(lbl)
    h_ = np.array([slices[i - 1][0].stop - slices[i - 1][0].start for i in idx], float)
    w_ = np.array([slices[i - 1][1].stop - slices[i - 1][1].start for i in idx], float)

    y, x = cm[:, 0], cm[:, 1]
    yi = np.clip(np.round(y).astype(int), 0, raw.shape[0] - 1)
    xi = np.clip(np.round(x).astype(int), 0, raw.shape[1] - 1)

    out = np.column_stack([
        x, y, flux, peak, sizes,
        (peak - med - background(raw)[yi, xi]) / sig,             # SNR of the peak
        h_, w_,
        np.maximum(h_, w_) / np.maximum(1.0, np.minimum(h_, w_)),  # >3 = streak
        sizes / np.maximum(1.0, h_ * w_),                          # sparse = cosmic ray
        hot_mask[yi, xi].astype(float),
        ((x < 8) | (x > raw.shape[1] - 9) | (y < 8) | (y > raw.shape[0] - 9)).astype(float),
    ])
    return out, med, sig


t0 = time.time()
_d, _med, _sig = detect_frame(demo_raw)
print("demo frame: %d sources in %.2fs  (background-subtracted sky sigma %.2f)"
      % (len(_d), time.time() - t0, _sig))
_ddf = pd.DataFrame(_d, columns=DET_COLS)
print("  on a known hot pixel : %d" % int(_ddf.on_hot_pixel.sum()))
print("  elongated (>3)       : %d" % int((_ddf.elongation > 3).sum()))
print("  median npix          : %.0f" % _ddf.npix.median())
_ddf.sort_values("flux", ascending=False).head(6).round(2)

# %% [markdown]
# ~400 sources per frame is not a bug. At 5 arcsec/pixel the 650×500 chip covers
# **54′ × 42′ — nearly a square degree** — and every one of these targets lies in
# or near the Milky Way. A few hundred stars in that field is exactly right.
#
# ## 5 · Tracking sources across a night
#
# Here is the thing that makes this dataset harder than it looks. Measuring the
# telescope's motion over one session:
#
# * the field **drifts by ~105 pixels**, and
# * it **rotates by ~1.1°**.
#
# That rotation is the fingerprint of an **alt-azimuth mount with no field
# derotator** — the sky turns relative to the camera as the target climbs. The
# consequence is important: *no single shift can align the first frame to the last*.
# A rotation of 1° puts the corners 5 px out however you translate. Any pipeline
# that assumes pure translation will silently mismatch stars.
#
# So we never build a common grid. Instead we **chain detections between
# consecutive frames**, which are only 180 s and about 1.4 px apart. Over that tiny
# step the match is unambiguous, and rotation is far too small to matter. Chaining
# those steps follows each star through the whole run, drift and rotation included.

# %%
MATCH_R = 4.0     # px; ~3x the frame-to-frame motion
MISS_TOL = 5      # frames a track may vanish for (clouds) before it is dropped


def estimate_offset(a, b, max_shift=30.0):
    """Modal pairwise offset between two point sets.

    Every pair of unrelated stars contributes a random difference, but the true
    offset is the one value that hundreds of pairs agree on, so it stands out as
    a sharp histogram peak. Robust to stars entering, leaving, or dropping out.
    """
    if len(a) < 5 or len(b) < 5:
        return np.zeros(2), 0
    diffs = []
    for i, js in enumerate(cKDTree(b).query_ball_point(a, r=max_shift)):
        for j in js:
            diffs.append(b[j] - a[i])
    if len(diffs) < 5:
        return np.zeros(2), 0
    diffs = np.asarray(diffs)
    bins = np.arange(-max_shift, max_shift + 1, 1.0)
    H, xe, ye = np.histogram2d(diffs[:, 0], diffs[:, 1], bins=[bins, bins])
    i, j = np.unravel_index(np.argmax(H), H.shape)
    cx, cy = (xe[i] + xe[i + 1]) / 2, (ye[j] + ye[j + 1]) / 2
    near = diffs[np.hypot(diffs[:, 0] - cx, diffs[:, 1] - cy) < 2.0]
    return near.mean(axis=0), len(near)


def track_session(cube, det_sigma=DET_SIGMA):
    """Detect on every frame and link detections into tracks.

    Returns a detections DataFrame carrying a `track_id`, plus the per-frame
    offsets we used for prediction (useful evidence on the dashboard: a sudden
    jump in the offset is a tracking slip, one of our false-positive cases).
    """
    n_frames = len(cube)
    per_frame, sky_rows = [], []
    for i, img in enumerate(cube):
        arr, med, sig = detect_frame(img, det_sigma)
        per_frame.append(arr)
        sky_rows.append((i, med, sig, len(arr)))

    track_of = [np.full(len(a), -1, dtype=int) for a in per_frame]
    last_xy, last_seen = [], []          # per track
    offsets = np.zeros((n_frames, 2))
    active = []

    for i, arr in enumerate(per_frame):
        if len(arr) == 0:
            continue
        xy = arr[:, :2]
        if not active:
            for k in range(len(arr)):
                last_xy.append(xy[k].copy())
                last_seen.append(i)
                track_of[i][k] = len(last_xy) - 1
            active = list(range(len(last_xy)))
            continue

        pred = np.array([last_xy[t] for t in active])
        off, _ = estimate_offset(pred, xy)
        offsets[i] = off
        pred = pred + off

        dist, j = cKDTree(xy).query(pred, distance_upper_bound=MATCH_R)
        taken, still = set(), []
        for k, t in enumerate(active):
            if np.isfinite(dist[k]) and j[k] not in taken:
                taken.add(int(j[k]))
                last_xy[t] = xy[j[k]].copy()
                last_seen[t] = i
                track_of[i][int(j[k])] = t
                still.append(t)
            else:
                last_xy[t] = last_xy[t] + off   # coast on the prediction
                if i - last_seen[t] <= MISS_TOL:
                    still.append(t)
        for k in range(len(arr)):
            if k not in taken:
                last_xy.append(xy[k].copy())
                last_seen.append(i)
                track_of[i][k] = len(last_xy) - 1
                still.append(len(last_xy) - 1)
        active = still

    frames_list = []
    for i, arr in enumerate(per_frame):
        if len(arr) == 0:
            continue
        df = pd.DataFrame(arr, columns=DET_COLS)
        df["frame_index"] = i
        df["track_id"] = track_of[i]
        df["sky"] = sky_rows[i][1]
        df["sky_sigma"] = sky_rows[i][2]
        frames_list.append(df)

    det = pd.concat(frames_list, ignore_index=True)
    det["on_hot_pixel"] = det["on_hot_pixel"].astype(bool)
    det["edge"] = det["edge"].astype(bool)
    off_df = pd.DataFrame(offsets, columns=["offset_dx", "offset_dy"])
    off_df["frame_index"] = np.arange(n_frames)
    off_df["cum_dx"] = off_df["offset_dx"].cumsum()
    off_df["cum_dy"] = off_df["offset_dy"].cumsum()
    return det, off_df


t0 = time.time()
demo_det, demo_off = track_session(demo_cube)
print("%s: %d detections, %d tracks, %.0fs"
      % (demo_id, len(demo_det), demo_det.track_id.nunique(), time.time() - t0))
print("cumulative field motion over the session: dx=%+.1f dy=%+.1f px"
      % (demo_off.cum_dx.iloc[-1], demo_off.cum_dy.iloc[-1]))

# %% [markdown]
# ## 6 · Physics-derived auto-labels
#
# **This is what makes the machine learning honest.** We never hand-draw a box.
# Every label follows from a property of the data we can state as a rule:
#
# | Label | Rule | Why it is trustworthy |
# |---|---|---|
# | `star` | seen in ≥ 60 % of frames **and** moves with the field | Real sources do not blink out, and they share the field's motion |
# | `hot_pixel` | a pixel flagged in the master dark, sampled directly from the mask (section 8) | Comes from 60 shutter-closed frames over 30 nights — the calibration data never saw a star, so there is no circularity |
# | `cosmic_ray` | one frame only, ≤ 8 px, compact, not on a hot pixel | A particle hit cannot recur at the same pixel |
# | `satellite_trail` | one frame only, elongation > 3 | Geometry: nothing astronomical is a straight streak in 60 s |
# | `faint_source` | seen in 10–60 % of frames | Real but near the detection limit — kept separate rather than mislabelled |
# | `noise` | random position with nothing detected within 8 px | Negative class |
#
# The 60 % threshold is deliberately below 100 %: a genuine star *does* drop out
# when a cloud passes, and demanding perfection would relabel every cloudy-night
# star as junk.
#
# ### A free cross-check
#
# The field drifts ~105 px, but a hot pixel is a defect in the *silicon* — it does
# not move at all. So "persistent but stationary" identifies hot pixels using only
# the science frames, completely independently of the dark frames. Two unrelated
# methods, one answer. The cell below reports how often they agree, and that
# agreement rate is a validation result in its own right.

# %%
PERSIST_FRAC = 0.60
FAINT_FRAC = 0.10
STATIONARY_PX = 3.0     # a track that moves less than this is stuck to the chip


def summarise_tracks(det, n_frames):
    g = det.groupby("track_id")
    tr = g.agg(
        n_points=("frame_index", "nunique"),
        first_frame=("frame_index", "min"),
        last_frame=("frame_index", "max"),
        x_first=("x", "first"), y_first=("y", "first"),
        x_last=("x", "last"), y_last=("y", "last"),
        x_med=("x", "median"), y_med=("y", "median"),
        median_flux=("flux", "median"),
        median_peak=("peak", "median"),
        max_peak=("peak", "max"),
        median_snr=("snr", "median"),
        median_npix=("npix", "median"),
        max_elongation=("elongation", "max"),
        median_fill=("fill_factor", "median"),
        any_hot=("on_hot_pixel", "any"),
        hot_frac=("on_hot_pixel", "mean"),
        any_edge=("edge", "any"),
    ).reset_index()
    tr["persistence"] = tr["n_points"] / n_frames
    tr["motion_px"] = np.hypot(tr.x_last - tr.x_first, tr.y_last - tr.y_first)
    tr["span_frames"] = tr.last_frame - tr.first_frame + 1
    return tr


def label_tracks(tr, field_motion_px):
    """Apply the rule table. Order matters: most specific rule wins."""
    lab = np.full(len(tr), "unknown", dtype=object)
    persistent = tr.persistence.to_numpy() >= PERSIST_FRAC
    single = tr.n_points.to_numpy() == 1
    elong = tr.max_elongation.to_numpy()
    npix = tr.median_npix.to_numpy()
    fill = tr.median_fill.to_numpy()

    lab[(tr.persistence.to_numpy() >= FAINT_FRAC) & ~persistent] = "faint_source"
    lab[persistent] = "star"
    lab[single & (elong > 3.0)] = "satellite_trail"
    lab[single & (elong <= 3.0) & (npix <= 8) & (fill >= 0.5)] = "cosmic_ray"
    # A hot pixel is fixed to the silicon, so a hot-pixel track must sit on the
    # mask for most of its life AND not move. The first version used "touched a
    # hot pixel in any frame", which relabelled every star that drifted across one:
    # 996 of 1,322 "hot_pixel" tracks moved more than 20 px with the field. The
    # classifier trained on those labels learned that hot pixels look like stars.
    on_mask = tr.hot_frac.to_numpy() >= 0.5
    still = tr.motion_px.to_numpy() < STATIONARY_PX
    lab[on_mask & still] = "hot_pixel"
    tr = tr.copy()
    tr["label"] = lab

    # Independent, science-frame-only hot-pixel test (see markdown above).
    tr["stationary"] = (tr.motion_px < STATIONARY_PX) & (field_motion_px > 20)
    tr["hot_by_motion"] = tr.stationary & persistent
    return tr


demo_tr = summarise_tracks(demo_det, len(demo_cube))
field_motion = float(np.hypot(demo_off.cum_dx.iloc[-1], demo_off.cum_dy.iloc[-1]))
demo_tr = label_tracks(demo_tr, field_motion)

print("tracks by label:")
print(demo_tr.label.value_counts().to_string())

a = demo_tr.hot_by_motion.to_numpy()
b = demo_tr.any_hot.to_numpy()
print("\nHot pixels, two independent methods:")
print("  dark-frame mask only        : %d" % int((b & ~a).sum()))
print("  stationary-track only       : %d" % int((a & ~b).sum()))
print("  BOTH agree                  : %d" % int((a & b).sum()))
if a.sum():
    print("  -> %.0f%% of stationary persistent tracks are confirmed by the darks"
          % (100 * (a & b).sum() / a.sum()))
print("\nmedian motion, stars %.1f px vs hot pixels %.1f px"
      % (demo_tr.loc[demo_tr.label == "star", "motion_px"].median(),
         demo_tr.loc[demo_tr.label == "hot_pixel", "motion_px"].median()))

# %% [markdown]
# ## 7 · Run over all 22 sessions
#
# About 5 s per session. Set `SESSION_LIMIT` to a small number while iterating.

# %%
SESSION_LIMIT = int(os.environ.get("SESSION_LIMIT", "0")) or None

all_det, all_tr, all_off = [], [], []
ids = sessions["session_id"].tolist()[:SESSION_LIMIT]

t0 = time.time()
for k, sid in enumerate(ids, 1):
    cube = load_session(sid)
    det, off = track_session(cube)
    fm = float(np.hypot(off.cum_dx.iloc[-1], off.cum_dy.iloc[-1]))
    tr = label_tracks(summarise_tracks(det, len(cube)), fm)
    det["session_id"] = sid
    tr["session_id"] = sid
    off["session_id"] = sid
    off["field_motion_px"] = fm
    all_det.append(det)
    all_tr.append(tr)
    all_off.append(off)
    n_star = int((tr.label == "star").sum())
    print("[%2d/%2d] %-24s det=%6d tracks=%5d stars=%4d drift=%5.1fpx (%.0fs)"
          % (k, len(ids), sid, len(det), len(tr), n_star, fm, time.time() - t0))
    del cube

detections = pd.concat(all_det, ignore_index=True)
tracks = pd.concat(all_tr, ignore_index=True)
offsets = pd.concat(all_off, ignore_index=True)

# Give every detection its track's label, so cutouts can be cut straight from it.
detections = detections.merge(
    tracks[["session_id", "track_id", "label", "persistence", "n_points"]],
    on=["session_id", "track_id"], how="left")

detections.to_csv(RESULTS / "detections.csv.gz", index=False, compression="gzip")
tracks.to_csv(RESULTS / "tracks.csv", index=False)
offsets.to_csv(RESULTS / "field_motion.csv", index=False)

print("\ntotal: %d detections / %d tracks in %.0fs" % (len(detections), len(tracks), time.time() - t0))
print("\ntracks by label:\n" + tracks.label.value_counts().to_string())
print("\ndetections by label:\n" + detections.label.value_counts().to_string())
tracks.label.value_counts().to_frame("n_tracks").to_csv(RESULTS / "label_counts.csv")

# %% [markdown]
# ### Field motion is itself a science product
#
# The per-session drift we just measured is not bookkeeping — it is the evidence
# that rules out false-positive case 3 (tracking slip). A dip that coincides with a
# jump in the field offset is a pointing error, not a planet. This table goes on
# the dashboard next to every light curve.

# %%
motion_summary = (offsets.groupby("session_id")
                  .agg(field_motion_px=("field_motion_px", "first"),
                       max_step_px=("offset_dx", lambda s: float(np.abs(s).max())),
                       total_dx=("cum_dx", "last"),
                       total_dy=("cum_dy", "last"))
                  .reset_index()
                  .sort_values("field_motion_px", ascending=False))
motion_summary.to_csv(RESULTS / "field_motion_summary.csv", index=False)
motion_summary.round(2)

# %% [markdown]
# ### Class balance — say it before a judge has to ask
#
# The classes are wildly imbalanced, and that is a fact about the sky rather than a
# flaw in the pipeline: stars are detected in every frame, cosmic rays are rare,
# satellites rarer still. It is also precisely why **accuracy is a useless metric
# here**, and why the training notebook reports per-class precision and recall.

# %%
share = (detections.label.value_counts(normalize=True) * 100).round(3)
print("detection-level class share (%):\n" + share.to_string())
print("\nA model that answered 'star' every time would score %.1f%% accuracy "
      "and be worthless." % share.get("star", 0.0))

# %% [markdown]
# ## 8 · Cutout export for the training notebook
#
# 32×32 postage stamps around each detection, cut from the **raw** frame. Each class
# is capped so no single class can swamp a batch, and every stamp records its
# session so the training notebook can **split by night rather than at random** —
# frames from one night are heavily correlated and a random split would leak.

# %%
CUT = 32
HALF = CUT // 2
PER_CLASS_CAP = 12000
NOISE_PER_FRAME = 4
# hot_pixel is NOT taken from detections: the detector mostly ignores single-pixel
# spikes, and the few hot-pixel detections it does make are too rare to train on.
# Hot-pixel stamps are cut directly at dark-mask positions below instead, which is
# also the most trustworthy label in the set.
TRAIN_LABELS = ["star", "cosmic_ray", "satellite_trail", "faint_source"]
HOT_PER_FRAME = 10
HOT_FRAMES_PER_SESSION = 40


def extract_cutout(img, x, y):
    """32x32 stamp, zero-padded when the source sits near an edge."""
    h, w = img.shape
    y0, x0 = int(round(y)) - HALF, int(round(x)) - HALF
    out = np.zeros((CUT, CUT), dtype=np.float32)
    ys, xs = max(0, y0), max(0, x0)
    ye, xe = min(h, y0 + CUT), min(w, x0 + CUT)
    if ye > ys and xe > xs:
        out[ys - y0:ye - y0, xs - x0:xe - x0] = img[ys:ye, xs:xe]
    return out


pool = detections[detections.label.isin(TRAIN_LABELS)]
sel = pd.concat([
    g if len(g) <= PER_CLASS_CAP else g.sample(PER_CLASS_CAP, random_state=42)
    for _, g in pool.groupby("label")
], ignore_index=True)
print("selected per class:\n" + sel.label.value_counts().to_string())

stamps, meta_rows = [], []
for sid, grp in sel.groupby("session_id"):
    cube = load_session(sid)
    fi = grp.frame_index.to_numpy()
    xs, ys = grp.x.to_numpy(), grp.y.to_numpy()
    for k in range(len(grp)):
        stamps.append(extract_cutout(cube[fi[k]], xs[k], ys[k]))
    meta_rows.append(grp[["session_id", "frame_index", "x", "y", "label",
                          "snr", "npix", "elongation", "peak", "persistence"]])
    # negative class: random spots where the detector found nothing
    neg = []
    by_frame = {f: grp[grp.frame_index == f][["x", "y"]].to_numpy()
                for f in np.unique(fi)}
    for i in RNG.choice(len(cube), size=min(len(cube), 40), replace=False):
        same = by_frame.get(i, np.zeros((0, 2)))
        for _ in range(NOISE_PER_FRAME):
            x = float(RNG.integers(HALF, cube.shape[2] - HALF))
            y = float(RNG.integers(HALF, cube.shape[1] - HALF))
            if len(same) and np.hypot(same[:, 0] - x, same[:, 1] - y).min() < 8:
                continue
            stamps.append(extract_cutout(cube[i], x, y))
            neg.append({"session_id": sid, "frame_index": int(i), "x": x, "y": y,
                        "label": "noise", "snr": np.nan, "npix": 0,
                        "elongation": np.nan, "peak": np.nan, "persistence": 0.0})
    if neg:
        meta_rows.append(pd.DataFrame(neg))

    # hot-pixel class: stamps centred on dark-mask pixels, skipping any that sit
    # within 8 px of a detected source in that frame so no star shares the stamp
    hot_yx = np.argwhere(hot_mask)
    det_sess = detections[(detections.session_id == sid)
                          & (detections.label != "hot_pixel")]
    det_by_frame = {f: g[["x", "y"]].to_numpy() for f, g in det_sess.groupby("frame_index")}
    hot = []
    for i in RNG.choice(len(cube), size=min(len(cube), HOT_FRAMES_PER_SESSION),
                        replace=False):
        near = det_by_frame.get(int(i), np.zeros((0, 2)))
        for j in RNG.choice(len(hot_yx), size=HOT_PER_FRAME, replace=False):
            y, x = (float(v) for v in hot_yx[j])
            if len(near) and np.hypot(near[:, 0] - x, near[:, 1] - y).min() < 8:
                continue
            stamps.append(extract_cutout(cube[i], x, y))
            hot.append({"session_id": sid, "frame_index": int(i), "x": x, "y": y,
                        "label": "hot_pixel", "snr": np.nan, "npix": 1,
                        "elongation": np.nan,
                        "peak": float(cube[i][int(y), int(x)]), "persistence": 1.0})
    if hot:
        meta_rows.append(pd.DataFrame(hot))
    del cube
    print("  %-24s running total %d" % (sid, len(stamps)))

X = np.stack(stamps).astype(np.uint16)
cut_meta = pd.concat(meta_rows, ignore_index=True)
assert len(X) == len(cut_meta), "cutout/metadata length mismatch"

np.savez_compressed(RESULTS / "cutouts.npz",
                    X=X,
                    y=cut_meta.label.to_numpy().astype("U16"),
                    session=cut_meta.session_id.to_numpy().astype("U40"))
cut_meta.to_csv(RESULTS / "cutouts_meta.csv", index=False)
print("\ncutouts: %s  (%.0f MB raw)" % (str(X.shape), X.nbytes / 1e6))
print(cut_meta.label.value_counts().to_string())

# %% [markdown]
# ## 9 · Aperture photometry and differential light curves
#
# The measurement the whole challenge is about.
#
# **Choosing the target.** MicroObservatory centres the requested star, so we take
# the brightest unsaturated persistent track that starts near the centre of frame 0.
# Its own track then gives its position in every frame — no registration needed,
# which is what saves us from the field rotation.
#
# **Why divide by comparison stars.** A cloud dims *everything*. Airmass dims
# everything. Only a transit dims the target alone. Dividing the target's flux by
# the summed comparison flux cancels every effect common to the whole field — the
# single step that turns an undetectable 2 % dip into a measurable one.

# %%
R_APER, R_IN, R_OUT = 5.0, 9.0, 14.0
_BOX = int(np.ceil(R_OUT)) + 1
_by, _bx = np.mgrid[-_BOX:_BOX + 1, -_BOX:_BOX + 1]
SATURATION = 4000.0
MIN_PERSIST_PHOT = 0.90


def aperture_flux(img, x, y):
    """Circular aperture minus a median-annulus sky. Returns (flux, sky, npix).

    Cut a small box around the star instead of masking the whole 500x650 frame:
    ~20x faster, and we call this tens of thousands of times.
    """
    h, w = img.shape
    xi, yi = int(round(x)), int(round(y))
    if not (_BOX <= xi < w - _BOX and _BOX <= yi < h - _BOX):
        return np.nan, np.nan, 0        # too close to the edge for a clean annulus
    box = img[yi - _BOX:yi + _BOX + 1, xi - _BOX:xi + _BOX + 1]
    r2 = (_bx - (x - xi)) ** 2 + (_by - (y - yi)) ** 2   # sub-pixel centring
    ap = r2 <= R_APER ** 2
    ann = (r2 > R_IN ** 2) & (r2 <= R_OUT ** 2)
    sky = float(np.median(box[ann]))
    return float(box[ap].sum() - sky * ap.sum()), sky, int(ap.sum())


def pick_stars(tr, det, shape, max_comps=8, centre_r=80.0):
    """Return (target_track_id, [comparison_track_ids], persistence_used, flags).

    The persistence bar is relaxed step by step. On a clear night almost every
    star is detected in every frame and 90% is easily met; on a cloudy night
    (CoRoT-2 2026-08-23 loses 80% of its detections) insisting on 90% would throw
    the whole session away. Recording which threshold was used keeps that honest —
    a session measured at 0.45 is flagged as lower quality on the dashboard.

    Saturation is treated asymmetrically, and deliberately. A *comparison* star
    must be unsaturated or it cannot track transparency: once pixels hit the 4095
    ceiling the measured flux stops responding to the sky. But the *target* is not
    ours to choose — MicroObservatory pointed at it, and on these bright hosts it
    often does saturate. Excluding it would silently substitute some unrelated
    neighbour and quietly invalidate the whole light curve, so instead we keep it
    and raise `target_saturated`, which the dashboard shows as a warning.
    """
    for bar in (MIN_PERSIST_PHOT, 0.75, 0.6, 0.45):
        cand = tr[(tr.label == "star")
                  & (tr.persistence >= bar)
                  & (~tr.any_edge)].copy()
        if (cand.max_peak < SATURATION).sum() >= 3:
            break
    else:
        return None, [], np.nan, {}
    if len(cand) < 2:
        return None, [], np.nan, {}
    cy, cx = shape[0] / 2, shape[1] / 2
    cand["r0"] = np.hypot(cand.x_first - cx, cand.y_first - cy)

    near = cand[cand.r0 <= centre_r]
    # MicroObservatory centres the requested star, so the target is the brightest
    # thing near the middle of the first frame. This is an a-priori rule: we do NOT
    # choose whichever candidate happens to show the nicest dip, which would be a
    # false-positive generator rather than a detection.
    target = (near.sort_values("median_flux", ascending=False).iloc[0]
              if len(near) else cand.sort_values("r0").iloc[0])

    flags = {
        "target_saturated": bool(target.max_peak >= SATURATION),
        "target_r_from_centre": round(float(target.r0), 1),
        "n_central_candidates": int(len(near)),
    }
    comps = cand[(cand.track_id != target.track_id) & (cand.max_peak < SATURATION)]
    # Comparison stars should be comparably bright: much fainter adds noise,
    # much brighter risks non-linearity near the 4095 ceiling.
    lo, hi = target.median_flux * 0.2, target.median_flux * 8.0
    good = comps[(comps.median_flux >= lo) & (comps.median_flux <= hi)]
    if len(good) < 3:
        good = comps
    good = good.reindex(
        (good.median_flux - target.median_flux).abs().sort_values().index)
    return int(target.track_id), [int(t) for t in good.track_id.head(max_comps)], bar, flags


def session_photometry(sid, cube, det, tr):
    tgt_id, comp_ids, bar, flags = pick_stars(tr, det, cube.shape[1:])
    if tgt_id is None or not comp_ids:
        return pd.DataFrame(), None

    wanted = [tgt_id] + comp_ids
    sub = det[det.track_id.isin(wanted)]
    # position of every wanted star in every frame it was detected in
    pos = {t: g.set_index("frame_index")[["x", "y"]] for t, g in sub.groupby("track_id")}

    meta = frames[frames.session_id == sid].set_index("frame_index")
    rows = []
    for i in range(len(cube)):
        cal = calibrate(cube[i], meta["CAMTEMP"].get(i))
        if i not in pos[tgt_id].index:
            continue
        tx, ty = pos[tgt_id].loc[i]
        tf, tsky, _ = aperture_flux(cal, tx, ty)
        cs = []
        for c in comp_ids:
            if i in pos[c].index:
                cx_, cy_ = pos[c].loc[i]
                cs.append(aperture_flux(cal, cx_, cy_)[0])
        cs = np.array(cs, dtype=float)
        rows.append({"session_id": sid, "frame_index": i,
                     "target_x": float(tx), "target_y": float(ty),
                     "target_flux": tf, "target_sky": tsky,
                     "comp_flux_sum": float(np.nansum(cs)),
                     "n_comps_ok": int(np.isfinite(cs).sum())})

    lc = pd.DataFrame(rows)
    if len(lc) < 10:
        return pd.DataFrame(), None
    lc = lc.merge(frames[["session_id", "frame_index", "t_utc", "MJD-OBS", "airmass",
                          "TELALT", "WEATHER", "sky_level", "CAMTEMP", "target"]],
                  on=["session_id", "frame_index"], how="left")

    lc["rel_flux"] = lc.target_flux / lc.comp_flux_sum
    # Normalise to the out-of-transit level. The median is the right anchor: a
    # transit occupies well under half the run, so it cannot drag the median.
    lc["norm_flux"] = lc.rel_flux / lc.rel_flux.median()
    lc["target_only_norm"] = lc.target_flux / lc.target_flux.median()

    info = {
        "session_id": sid,
        "target": lc.target.iloc[0],
        "target_track": tgt_id,
        "target_x0": float(lc.target_x.iloc[0]), "target_y0": float(lc.target_y.iloc[0]),
        "n_comparison_stars": len(comp_ids),
        "persistence_bar_used": round(float(bar), 2),
        "quality": SESSION_QUALITY.get(sid, "unknown"),
        **flags,
        "n_points": len(lc),
        "rms_ppt": round(float(lc.norm_flux.std() * 1000), 3),
        "rms_target_only_ppt": round(float(lc.target_only_norm.std() * 1000), 3),
    }
    info["improvement_factor"] = round(
        info["rms_target_only_ppt"] / max(info["rms_ppt"], 1e-9), 2)
    return lc, info


t0 = time.time()
lcs, infos = [], []
for k, sid in enumerate(ids, 1):
    cube = load_session(sid)
    det = detections[detections.session_id == sid]
    tr = tracks[tracks.session_id == sid]
    lc, info = session_photometry(sid, cube, det, tr)
    del cube
    if info is None:
        print("[%2d/%2d] %-24s NO PHOTOMETRY -- too few reference stars (quality: %s)"
              % (k, len(ids), sid, SESSION_QUALITY.get(sid, "?")))
        continue
    lcs.append(lc)
    infos.append(info)
    print("[%2d/%2d] %-24s comps=%d  rms %6.2f ppt  (raw %7.2f, x%.1f better)  (%.0fs)"
          % (k, len(ids), sid, info["n_comparison_stars"], info["rms_ppt"],
             info["rms_target_only_ppt"], info["improvement_factor"], time.time() - t0))

lightcurves = pd.concat(lcs, ignore_index=True)
photometry_info = pd.DataFrame(infos)
lightcurves.to_csv(RESULTS / "lightcurves.csv", index=False)
photometry_info.to_csv(RESULTS / "photometry_info.csv", index=False)
photometry_info[["session_id", "target", "quality", "target_saturated",
                 "target_r_from_centre", "n_comparison_stars", "n_points",
                 "rms_ppt", "rms_target_only_ppt", "improvement_factor"]]

# %% [markdown]
# **Read that last table carefully.** `rms_target_only_ppt` is the scatter before
# differential correction and `rms_ppt` is after. If `improvement_factor` is not
# comfortably above 1 on a given night, the comparison stars are not doing their
# job there and no dip from that night can be trusted. That comparison is itself a
# result, and it belongs on the dashboard beside every light curve.

# %% [markdown]
# ## 10 · Proper timestamps and published ephemerides
#
# Two corrections stand between a header timestamp and a transit prediction:
#
# 1. **UTC → TDB.** Leap seconds; a few tens of seconds.
# 2. **Geocentric → barycentric.** Earth moves ±8 light-minutes across its orbit,
#    so the same event is seen up to 16 minutes apart depending on the date. At our
#    180 s cadence this is a real, several-frame error if ignored.
#
# `astropy` does both. Published ephemerides are quoted in BJD_TDB, so we must too.

# %%
try:
    from astropy import units as u
    from astropy.coordinates import EarthLocation, SkyCoord
    from astropy.time import Time
    from astropy.utils import iers

    # Kaggle has no internet by default, and the IERS Earth-rotation table bundled
    # with its astropy predates our observations. Without the two lines below
    # astropy refuses to extrapolate ("predictive values more than 30 days old").
    #
    # Extrapolating is safe here. The table only supplies UT1-UTC, which is capped
    # at 0.9 s by definition. That shifts where the observatory sits on the
    # rotating Earth by at most 0.46 km/s x 0.9 s ~ 0.4 km, i.e. ~1 microsecond of
    # light travel. Our cadence is 180 s. The barycentric term itself (~500 s)
    # comes from the planetary ephemeris, which needs no download.
    iers.conf.auto_download = False
    iers.conf.auto_max_age = None
    try:
        iers.conf.iers_degraded_accuracy = "warn"   # astropy >= 5.3
    except Exception:
        pass
    import warnings
    warnings.filterwarnings("ignore", module="astropy.utils.iers")
    WHIPPLE = EarthLocation(lat=31.68 * u.deg, lon=-110.88 * u.deg, height=1268 * u.m)
    HAVE_ASTROPY_COORDS = True
except Exception as exc:
    print("!! astropy.coordinates unavailable (%s: %s)" % (type(exc).__name__, exc))
    print("!! falling back to JD_UTC -- timings may be off by up to +/-8 minutes.")
    print("!! On Kaggle this import works; do not ship results computed this way.")
    HAVE_ASTROPY_COORDS = False


def to_bjd_tdb(mjd_utc, ra_deg, dec_deg):
    """Barycentric Julian Date in TDB.

    Two corrections separate a header timestamp from a usable transit time:
    UTC->TDB (leap seconds) and geocentric->barycentric (Earth's orbit moves the
    observer by up to +/-8 light-minutes). At our 180 s cadence the second one is
    worth several frames, so it is not optional.
    """
    if not HAVE_ASTROPY_COORDS:
        return np.asarray(mjd_utc, dtype=float) + 2400000.5   # JD_UTC, uncorrected
    t = Time(np.asarray(mjd_utc), format="mjd", scale="utc", location=WHIPPLE)
    coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg)
    return (t.tdb + t.light_travel_time(coord, kind="barycentric")).jd


bjd = []
for sid, grp in lightcurves.groupby("session_id"):
    m = frames[frames.session_id == sid]
    ra, dec = float(m["RA"].median()), float(m["DEC"].median())
    bjd.append(pd.DataFrame({
        "session_id": sid,
        "frame_index": grp["frame_index"].to_numpy(),
        "bjd_tdb": to_bjd_tdb(grp["MJD-OBS"].to_numpy(), ra, dec),
    }))
lightcurves = lightcurves.merge(pd.concat(bjd, ignore_index=True),
                                on=["session_id", "frame_index"], how="left")

_chk = lightcurves.dropna(subset=["bjd_tdb"]).iloc[0]
print("barycentric correction on the first frame: %.1f seconds"
      % ((_chk.bjd_tdb - (_chk["MJD-OBS"] + 2400000.5)) * 86400))

# Rewrite the file: section 9 saved it before bjd_tdb existed, and the API and
# dashboard both need the corrected timestamps.
lightcurves.to_csv(RESULTS / "lightcurves.csv", index=False)
print("lightcurves.csv rewritten with bjd_tdb (%d rows)" % len(lightcurves))

# Record whether the correction actually ran. Without this the column is named
# bjd_tdb either way and a run that silently fell back to JD_UTC is
# indistinguishable downstream from one that did the work -- exactly the kind of
# quiet mislabelling the API must not pass on to a reader.
import json

json.dump({
    "generated_utc": pd.Timestamp.now("UTC").isoformat(),
    "barycentric_correction_applied": bool(HAVE_ASTROPY_COORDS),
    "time_column": "bjd_tdb",
    "time_scale": ("BJD_TDB (UTC->TDB and barycentric light travel applied)"
                   if HAVE_ASTROPY_COORDS else
                   "JD_UTC -- UNCORRECTED. astropy.coordinates was unavailable."),
    "max_timing_error_s": 0.0 if HAVE_ASTROPY_COORDS else 480.0,
    "observatory": "Whipple Observatory, Amado AZ (31.68 N, 110.88 W, 1268 m)",
}, open(RESULTS / "provenance.json", "w"), indent=1)
print("provenance.json: barycentric_correction_applied = %s" % HAVE_ASTROPY_COORDS)

# %% [markdown]
# ### Ephemerides
#
# **Primary source: the NASA Exoplanet Archive** (requires Kaggle Internet = ON).
# If the query fails we fall back to a small table of published literature values.
# The fallback is clearly flagged in the output and **must be verified against the
# archive before anything is submitted** — a prediction is only as good as its
# ephemeris, and a stale `T0` drifts by hours over a decade.

# %%
TARGET_TO_HOST = {
    "TRES-1": "TrES-1", "TRES-3": "TrES-3", "TRES-5": "TrES-5",
    "CoRoT-2": "CoRoT-2", "Qatar-1": "Qatar-1",
    "WASP-2": "WASP-2", "WASP-10": "WASP-10",
    "HATP-10": "HAT-P-10",  # also catalogued as WASP-11 — same system, two names
}

# Literature fallback. Values are approximate and rounded; treat as provisional.
FALLBACK = {
    "TrES-1":   dict(pl_orbper=3.0300722, pl_tranmid=2453186.80603, pl_trandur=2.50, pl_trandep=1.90),
    "TrES-3":   dict(pl_orbper=1.3061864, pl_tranmid=2454185.91110, pl_trandur=1.33, pl_trandep=2.50),
    "TrES-5":   dict(pl_orbper=1.4822446, pl_tranmid=2455443.25160, pl_trandur=1.70, pl_trandep=2.10),
    "CoRoT-2":  dict(pl_orbper=1.7429964, pl_tranmid=2454237.53562, pl_trandur=2.28, pl_trandep=2.80),
    "Qatar-1":  dict(pl_orbper=1.4200242, pl_tranmid=2455518.41094, pl_trandur=1.60, pl_trandep=2.40),
    "WASP-2":   dict(pl_orbper=2.1522254, pl_tranmid=2453991.51530, pl_trandur=1.80, pl_trandep=1.70),
    "WASP-10":  dict(pl_orbper=3.0927616, pl_tranmid=2454357.85808, pl_trandur=2.30, pl_trandep=2.90),
    "HAT-P-10": dict(pl_orbper=3.7224690, pl_tranmid=2454759.68683, pl_trandur=2.60, pl_trandep=2.00),
}

TAP = ("https://exoplanetarchive.ipac.caltech.edu/TAP/sync?format=csv&query="
       "select+hostname,pl_name,pl_orbper,pl_tranmid,pl_trandur,pl_trandep,"
       "default_flag+from+ps+where+hostname+in+({hosts})")


# The archive files HAT-P-10 under its other name. Querying only "HAT-P-10"
# returned nothing and silently sent that target to the unverified fallback.
ARCHIVE_ALIAS = {"HAT-P-10": "WASP-11"}
EPH_FIELDS = ["pl_orbper", "pl_tranmid", "pl_trandur", "pl_trandep"]


def fetch_ephemerides(hosts):
    import io
    import urllib.request
    names = [ARCHIVE_ALIAS.get(h, h) for h in hosts]
    q = TAP.format(hosts=",".join("'%s'" % h for h in names))
    with urllib.request.urlopen(q.replace(" ", "+"), timeout=60) as r:
        df = pd.read_csv(io.StringIO(r.read().decode()))
    df = df[df.default_flag == 1]
    df["hostname"] = df.hostname.replace({v: k for k, v in ARCHIVE_ALIAS.items()})
    df = df.groupby("hostname").first().reset_index()

    # A planet's "default" row is one paper, and that paper does not always quote
    # every field -- TrES-1's has no mid-transit time at all, which left a NaN epoch
    # and crashed the prediction cell. Fill only the gaps, from the archive's
    # composite table, and say per row which fields were filled.
    comp_q = ("https://exoplanetarchive.ipac.caltech.edu/TAP/sync?format=csv&query="
              "select+hostname,pl_orbper,pl_tranmid,pl_trandur,pl_trandep+from+"
              "pscomppars+where+hostname+in+({hosts})").format(
        hosts=",".join("'%s'" % h for h in names))
    with urllib.request.urlopen(comp_q, timeout=60) as r:
        comp = pd.read_csv(io.StringIO(r.read().decode()))
    comp["hostname"] = comp.hostname.replace({v: k for k, v in ARCHIVE_ALIAS.items()})
    comp = comp.set_index("hostname")
    filled = []
    for i, row in df.iterrows():
        gaps = [c for c in EPH_FIELDS
                if not np.isfinite(row[c]) and row.hostname in comp.index
                and np.isfinite(comp.loc[row.hostname, c])]
        for c in gaps:
            df.at[i, c] = comp.loc[row.hostname, c]
        filled.append(",".join(gaps))
    df["filled_from_composite"] = filled
    return df


hosts = sorted(set(TARGET_TO_HOST.values()))
try:
    eph = fetch_ephemerides(hosts)
    eph["source"] = np.where(eph.filled_from_composite == "",
                             "NASA Exoplanet Archive (live)",
                             "NASA Exoplanet Archive (live; gaps filled from pscomppars: "
                             + eph.filled_from_composite + ")")
    print("ephemerides: fetched from NASA Exoplanet Archive")
except Exception as exc:  # no internet on Kaggle by default
    print(f"!! archive query failed ({type(exc).__name__}: {exc})")
    print("!! FALLING BACK to literature values -- VERIFY THESE BEFORE SUBMITTING")
    eph = pd.DataFrame([dict(hostname=h, **FALLBACK[h]) for h in hosts])
    eph["source"] = "literature fallback (UNVERIFIED)"

missing = [h for h in hosts if h not in set(eph.hostname)]
for h in missing:
    eph = pd.concat([eph, pd.DataFrame([dict(hostname=h, **FALLBACK[h],
                                             source="literature fallback (UNVERIFIED)")])],
                    ignore_index=True)
for i, row in eph.iterrows():
    gaps = [c for c in EPH_FIELDS if not np.isfinite(row[c]) and row.hostname in FALLBACK]
    for c in gaps:
        eph.at[i, c] = FALLBACK[row.hostname][c]
    if gaps:
        eph.at[i, "source"] = "%s; %s from literature fallback (UNVERIFIED)" % (
            row.source, ",".join(gaps))
eph.to_csv(RESULTS / "ephemerides.csv", index=False)
eph[["hostname", "pl_orbper", "pl_tranmid", "pl_trandur", "pl_trandep", "source"]]

# %% [markdown]
# ### Which of our 22 nights should contain a transit?
#
# Fold each session's time span against the ephemeris and check whether a predicted
# mid-transit lands inside it. **A night with no predicted transit is not a failure —
# it is a control.** Any dip found there is by definition a false positive, which
# makes those nights the most valuable validation data we have.

# %%
eph_by_host = eph.set_index("hostname").to_dict("index")
pred_rows = []

for sid, grp in lightcurves.groupby("session_id"):
    tgt = grp["target"].iloc[0]
    host = TARGET_TO_HOST.get(tgt)
    e = eph_by_host.get(host)
    t = grp["bjd_tdb"].dropna()
    if (e is None or len(t) == 0 or not np.isfinite(e.get("pl_orbper", np.nan))
            or not np.isfinite(e.get("pl_tranmid", np.nan))):
        print("  %s: no usable ephemeris (period or T0 missing) -- skipped" % sid)
        continue
    t0_bjd, period = e["pl_tranmid"], e["pl_orbper"]
    dur_h = e.get("pl_trandur")
    dur_h = 2.0 if not np.isfinite(dur_h or np.nan) else float(dur_h)

    n = np.round((t.mean() - t0_bjd) / period)
    mid = t0_bjd + n * period
    half = (dur_h / 24.0) / 2.0
    pred_rows.append({
        "session_id": sid, "target": tgt, "host": host,
        "epoch": int(n),
        "pred_mid_bjd": mid,
        "pred_ingress_bjd": mid - half,
        "pred_egress_bjd": mid + half,
        "obs_start_bjd": float(t.min()), "obs_end_bjd": float(t.max()),
        "duration_h": dur_h,
        "expected_depth_pct": e.get("pl_trandep"),
        # how much of the predicted transit our window actually covers
        "coverage_frac": float(
            max(0.0, min(t.max(), mid + half) - max(t.min(), mid - half)) / (2 * half)),
        "mid_in_window": bool(t.min() <= mid <= t.max()),
        "ephemeris_source": e["source"],
    })

predictions = pd.DataFrame(pred_rows).sort_values(["target", "session_id"])
predictions.to_csv(RESULTS / "transit_predictions.csv", index=False)
print("sessions with the predicted mid-transit inside the observed window: "
      f"{int(predictions.mid_in_window.sum())} / {len(predictions)}")
predictions[["session_id", "epoch", "duration_h", "coverage_frac",
             "mid_in_window", "expected_depth_pct"]]

# %% [markdown]
# ## 11 · Measure the depth where a transit is predicted
#
# For each session with coverage, compare the mean normalised flux inside the
# predicted window against the mean outside it. This is deliberately the simplest
# possible estimator — no model fitting, nothing that could manufacture a signal
# that is not in the data.

# %%
def measure_depth(lc, mid, half):
    t = lc["bjd_tdb"].to_numpy()
    f = lc["norm_flux"].to_numpy()
    ok = np.isfinite(t) & np.isfinite(f)
    t, f = t[ok], f[ok]
    inw = (t >= mid - half) & (t <= mid + half)
    out = ~inw
    if inw.sum() < 3 or out.sum() < 5:
        return None
    fin, fout = f[inw].mean(), f[out].mean()
    # error on the difference of two means
    err = np.hypot(f[inw].std() / np.sqrt(inw.sum()),
                   f[out].std() / np.sqrt(out.sum()))
    return {
        "n_in": int(inw.sum()), "n_out": int(out.sum()),
        "depth_pct": float((fout - fin) / fout * 100),
        "depth_err_pct": float(err / fout * 100),
        "significance_sigma": float((fout - fin) / err) if err > 0 else np.nan,
        "out_of_transit_rms_ppt": float(f[out].std() * 1000),
    }


rows = []
for _, p in predictions.iterrows():
    lc = lightcurves[lightcurves.session_id == p.session_id]
    half = (p.duration_h / 24.0) / 2.0
    m = measure_depth(lc, p.pred_mid_bjd, half)
    if m is None:
        continue
    rows.append({"session_id": p.session_id, "target": p.target,
                 "epoch": p.epoch, "coverage_frac": round(p.coverage_frac, 3),
                 "expected_depth_pct": p.expected_depth_pct, **m})

depths = pd.DataFrame(rows).sort_values("significance_sigma", ascending=False)
depths.to_csv(RESULTS / "transit_depths.csv", index=False)
print("Measured depth in the predicted window, most significant first.")
print("A NEGATIVE depth means the star got BRIGHTER -- that is a null result,")
print("and reporting it is the point.\n")
depths.round(3)

# %% [markdown]
# ## 12 · Phase folding
#
# TRES-5 has six independent nights and TRES-3 and CoRoT-2 have four each. Folding
# them on the published period is the strongest test available to us: a real transit
# stacks up at the same phase every time, whereas noise scatters. This is one plot
# that a judge can evaluate in three seconds.

# %%
fold_rows = []
for target, grp in lightcurves.groupby("target"):
    host = TARGET_TO_HOST.get(target)
    e = eph_by_host.get(host)
    if e is None or not np.isfinite(e.get("pl_orbper", np.nan)):
        continue
    period, t0_bjd = e["pl_orbper"], e["pl_tranmid"]
    ph = ((grp["bjd_tdb"] - t0_bjd) / period) % 1.0
    ph = np.where(ph > 0.5, ph - 1.0, ph)  # centre the transit on phase 0
    fold_rows.append(pd.DataFrame({
        "target": target,
        "session_id": grp["session_id"].to_numpy(),
        "phase": ph,
        "phase_hours": ph * period * 24.0,
        "norm_flux": grp["norm_flux"].to_numpy(),
        "bjd_tdb": grp["bjd_tdb"].to_numpy(),
    }))

phasefold = pd.concat(fold_rows, ignore_index=True)
phasefold.to_csv(RESULTS / "phasefold.csv", index=False)

# Binned version — this is what actually gets plotted.
bins = np.arange(-3.0, 3.01, 0.15)  # hours from mid-transit
binned = []
for target, g in phasefold.groupby("target"):
    g = g[np.abs(g.phase_hours) <= 3.0]
    if len(g) < 20:
        continue
    idx = np.digitize(g.phase_hours, bins) - 1
    for b in range(len(bins) - 1):
        m = idx == b
        if m.sum() < 3:
            continue
        v = g.norm_flux.to_numpy()[m]
        binned.append({"target": target,
                       "phase_hours": (bins[b] + bins[b + 1]) / 2,
                       "flux": float(np.median(v)),
                       "err": float(v.std() / np.sqrt(len(v))),
                       "n": int(m.sum()),
                       "n_nights": int(g.session_id[m].nunique())})

phasefold_binned = pd.DataFrame(binned)
phasefold_binned.to_csv(RESULTS / "phasefold_binned.csv", index=False)
print(phasefold.groupby("target").size().to_string())
phasefold_binned.head(10).round(4)

# %% [markdown]
# ## 13 · Field-wide photometry — and an honest problem
#
# ### The problem
#
# Section 9 picked the target by an a-priori rule: brightest persistent star near
# the centre of frame 0. The diagnostics say that rule is **not reliable here**. The
# chosen star sits 44–85 px (4–7 arcmin) from the centre, and there are ~20 other
# persistent stars inside the same radius. The archive has no WCS solution, the
# header `RA`/`DEC` describe the telescope pointing rather than a plate solution, and
# at 5 arcsec/pixel we cannot resolve our way out of it.
#
# Guessing harder would be dishonest. So we do the opposite: **measure every star in
# the field**, and let the published ephemeris say which one is the planet host.
#
# ### Why this is a search, not a fishing expedition
#
# Testing hundreds of stars for a dip would normally be a false-positive machine —
# with 300 stars, some will show a 3σ dip by chance alone. Three things keep it
# defensible, and all three must hold:
#
# 1. **The phase is not free.** The dip has to fall at phase 0 of a *published*
#    period we did not fit. That is one specific place out of the whole cycle.
#    (`track_id` is per-session, so this runs within a night — stacking nights
#    would need an astrometric cross-match and these headers carry no WCS.)
# 2. **It must survive its own control.** For every star we slide the same window
#    to other phases within that night and count how often noise alone fakes it.
# 3. **We calibrate the noise.** We run the identical test at randomised phases and
#    report how often it produces a "detection" of the same strength. That number
#    is the honest false-alarm rate, and it is printed below.
#
# If the candidate does not beat its own control, we say so and report a null result.

# %%
ENSEMBLE_N = 25          # brightest unsaturated stars used as the comparison ensemble
MIN_PERSIST_FIELD = 0.70


def session_field_photometry(sid, cube, det, tr):
    """Aperture photometry for every persistent star in the field.

    One pass over the frames: each frame is calibrated once and then every star is
    measured in it. Doing it per-star instead would recalibrate the same frame
    hundreds of times.
    """
    stars = tr[(tr.label == "star")
               & (tr.persistence >= MIN_PERSIST_FIELD)
               & (~tr.any_edge)].copy()
    if len(stars) < 5:
        return pd.DataFrame(), pd.DataFrame()

    ids = stars.track_id.to_numpy()
    pos = {t: g.set_index("frame_index")[["x", "y"]]
           for t, g in det[det.track_id.isin(ids)].groupby("track_id")}
    meta = frames[frames.session_id == sid].set_index("frame_index")

    n_f, n_s = len(cube), len(ids)
    flux = np.full((n_f, n_s), np.nan, dtype=np.float64)
    for i in range(n_f):
        cal = calibrate(cube[i], meta["CAMTEMP"].get(i))
        for k, t in enumerate(ids):
            pt = pos.get(t)
            if pt is None or i not in pt.index:
                continue
            x, y = pt.loc[i]
            flux[i, k] = aperture_flux(cal, x, y)[0]

    # Comparison ensemble: the brightest unsaturated stars, which carry the most
    # photons and therefore define the transparency curve most precisely.
    unsat = stars.max_peak.to_numpy() < SATURATION
    med_flux = np.nanmedian(flux, axis=0)
    order = np.argsort(-np.where(unsat, med_flux, -np.inf))
    ens_idx = order[:min(ENSEMBLE_N, int(unsat.sum()))]
    ensemble = np.nansum(flux[:, ens_idx], axis=1)
    ensemble[ensemble <= 0] = np.nan

    rel = flux / ensemble[:, None]
    norm = rel / np.nanmedian(rel, axis=0)[None, :]

    long = pd.DataFrame({
        "session_id": sid,
        "track_id": np.repeat(ids, n_f),
        "frame_index": np.tile(np.arange(n_f), n_s),
        "flux": flux.T.ravel(),
        "norm_flux": norm.T.ravel(),
    }).dropna(subset=["norm_flux"])

    summary = stars[["track_id", "x_first", "y_first", "median_flux",
                     "max_peak", "persistence"]].copy()
    summary["session_id"] = sid
    summary["in_ensemble"] = np.isin(np.arange(n_s), ens_idx)
    summary["rms_ppt"] = np.nanstd(norm, axis=0) * 1000
    summary["n_points"] = np.isfinite(norm).sum(axis=0)
    return long, summary


good_ids = [s for s in ids if SESSION_QUALITY.get(s) == "good"]
print("running field photometry on %d good sessions" % len(good_ids))

t0 = time.time()
field_rows, field_sum = [], []
for k, sid in enumerate(good_ids, 1):
    cube = load_session(sid)
    lg, sm = session_field_photometry(sid, cube,
                                      detections[detections.session_id == sid],
                                      tracks[tracks.session_id == sid])
    del cube
    if len(lg) == 0:
        print("[%2d/%2d] %-24s skipped" % (k, len(good_ids), sid))
        continue
    field_rows.append(lg)
    field_sum.append(sm)
    print("[%2d/%2d] %-24s %4d stars, median rms %6.2f ppt, best %5.2f ppt (%.0fs)"
          % (k, len(good_ids), sid, len(sm), sm.rms_ppt.median(),
             sm.rms_ppt.min(), time.time() - t0))

field = pd.concat(field_rows, ignore_index=True)
field_stars = pd.concat(field_sum, ignore_index=True)

# Attach the barycentric timestamps from section 10. Field photometry is indexed
# by (session_id, frame_index) and so is the target light curve, so the same
# corrected clock serves both -- no second Time computation, no chance of the two
# tables disagreeing about when a frame was taken.
field = field.merge(
    lightcurves[["session_id", "frame_index", "bjd_tdb", "airmass"]],
    on=["session_id", "frame_index"], how="left")
_miss = field.bjd_tdb.isna().mean() * 100
print("field rows without a barycentric time: %.2f%%" % _miss)

field.to_csv(RESULTS / "field_photometry.csv.gz", index=False, compression="gzip")
field_stars.to_csv(RESULTS / "field_stars.csv", index=False)
print("\n%d star-frame measurements over %d stars"
      % (len(field), len(field_stars)))

# %% [markdown]
# ### The photometric noise floor
#
# Plotting scatter against brightness gives the instrument's real performance curve.
# The floor tells us the smallest transit this telescope could ever detect on one
# night — a hard limit that no amount of processing can talk its way past, and a
# number every claim later has to respect.

# %%
fs = field_stars[field_stars.n_points > 20].copy()
fs["mag_inst"] = -2.5 * np.log10(fs.median_flux.clip(lower=1))
bins = np.percentile(fs.mag_inst, np.linspace(0, 100, 11))
fs["mag_bin"] = pd.cut(fs.mag_inst, bins, include_lowest=True)
curve = (fs.groupby("mag_bin", observed=True)
         .agg(n_stars=("track_id", "size"),
              median_flux=("median_flux", "median"),
              median_rms_ppt=("rms_ppt", "median"),
              best_rms_ppt=("rms_ppt", "min"))
         .reset_index().drop(columns="mag_bin"))
curve.to_csv(RESULTS / "noise_floor.csv", index=False)
print("Photometric scatter vs brightness (all good sessions pooled):\n")
print(curve.round(2).to_string(index=False))
print("\nBest single-star precision achieved: %.2f ppt (%.3f%%)"
      % (fs.rms_ppt.min(), fs.rms_ppt.min() / 10))
print("A 2% transit is 20 ppt, so any star whose rms exceeds ~20 ppt")
print("cannot yield a single-night detection no matter what we do.")

# %% [markdown]
# ### Searching every star at the predicted phase
#
# For every star we fold on the published period and compare the mean flux inside
# the predicted transit window against the mean outside it.
#
# ### Where the null comes from
#
# The obvious control — slide the same window to a different phase on the same
# star — **does not work here**, and it is worth saying why rather than quietly
# dropping it. A session spans about 3.6 hours and the transit occupies 2 of them,
# so there is no room left for a non-overlapping control window. Our first attempt
# produced zero valid trials on every star.
#
# So the null comes from the **field** instead. Each frame contains 100-450 stars,
# essentially none of which host a transiting planet at *this* target's period and
# phase. Their depths at phase 0 are therefore a direct, empirical measurement of
# what this instrument, this night and this analysis produce from noise alone.
#
# A candidate has to stand out **against the other stars in its own image**. That
# controls for cloud, airmass, focus and every other thing the night did, because
# every comparison star lived through exactly the same night.
#
# The resolution floor is 1/N: with 250 stars the smallest p-value obtainable is
# 0.004, so `field_p_value = 0.004` means "nothing in this field beat it", not
# "one in a thousand".

# %%
def fold_depth(phase_h, flux, half_h):
    """Depth and significance for one star, folded on the published ephemeris.

    Identical estimator to `measure_depth` -- mean inside the window against mean
    outside, error on the difference of two means -- but takes arrays in hours
    from mid-transit instead of a light-curve frame. Using the same arithmetic for
    the target and for the field is the whole point: the null is only a null if it
    was measured the same way as the thing it is a null for.

    Returns (depth_pct, significance_sigma), both NaN when the star does not have
    enough points on either side of the window to support the comparison.
    """
    ok = np.isfinite(phase_h) & np.isfinite(flux)
    ph, f = phase_h[ok], flux[ok]
    inw = np.abs(ph) <= half_h
    out = ~inw
    if inw.sum() < 5 or out.sum() < 15:
        return np.nan, np.nan
    fin, fout = f[inw].mean(), f[out].mean()
    err = np.hypot(f[inw].std() / np.sqrt(inw.sum()),
                   f[out].std() / np.sqrt(out.sum()))
    if not (err > 0) or not np.isfinite(fout) or fout == 0:
        return np.nan, np.nan
    return float((fout - fin) / fout * 100), float((fout - fin) / err)


target_of = sessions.set_index("session_id").target.to_dict()
field["target"] = field.session_id.map(target_of)

search_rows = []
for target, grp in field.dropna(subset=["bjd_tdb"]).groupby("target"):
    host = TARGET_TO_HOST.get(target)
    e = eph_by_host.get(host)
    if e is None or not np.isfinite(e.get("pl_orbper", np.nan)):
        continue
    period, t0_bjd = e["pl_orbper"], e["pl_tranmid"]
    dur = e.get("pl_trandur")
    half_h = (2.0 if not np.isfinite(dur or np.nan) else float(dur)) / 2.0

    ph = ((grp.bjd_tdb - t0_bjd) / period) % 1.0
    ph = np.where(ph > 0.5, ph - 1.0, ph)
    grp = grp.assign(phase_h=ph * period * 24.0)

    # track_id is assigned per session, so a star is identified by the PAIR
    # (session_id, track_id). Stacking nights would need an astrometric
    # cross-match, and these headers carry no WCS solution.
    for (sid, tid), g in grp.groupby(["session_id", "track_id"]):
        if len(g) < 40:
            continue
        d, sg = fold_depth(g.phase_h.to_numpy(), g.norm_flux.to_numpy(), half_h)
        if not np.isfinite(sg):
            continue
        search_rows.append({
            "target": target, "session_id": sid, "track_id": int(tid),
            "n_points": len(g),
            "depth_pct": d, "significance_sigma": sg,
            "median_flux": float(g.flux.median()),
            "rms_ppt": float(g.norm_flux.std() * 1000),
            "expected_depth_pct": e.get("pl_trandep"),
        })

search = pd.DataFrame(search_rows)

if len(search):
    # Empirical null, computed within each session against its own field.
    out = []
    for sid, g in search.groupby("session_id"):
        sig = g.significance_sigma.to_numpy()
        g = g.copy()
        g["n_field_stars"] = len(g)
        g["field_p_value"] = [(sig >= v).sum() / len(sig) for v in sig]
        g["field_sigma_95"] = float(np.percentile(sig, 95))
        g["field_sigma_median"] = float(np.median(sig))
        out.append(g)
    search = pd.concat(out, ignore_index=True)

    # Secondary, NOT used for selection: how close the measured depth is to the
    # published one. Selecting on this would be circular.
    search["depth_ratio"] = search.depth_pct / search.expected_depth_pct
    search = search.sort_values(["target", "significance_sigma"],
                                ascending=[True, False])
    search.to_csv(RESULTS / "target_search.csv", index=False)

    print("%d star-nights searched across %d sessions.\n"
          % (len(search), search.session_id.nunique()))
    print("Top 3 per target by significance at the published phase:\n")
    cols = ["target", "session_id", "track_id", "depth_pct", "significance_sigma",
            "n_field_stars", "field_p_value", "field_sigma_95",
            "expected_depth_pct", "depth_ratio"]
    print(search.groupby("target").head(3)[cols].round(3).to_string(index=False))
else:
    print("no star had enough coverage to search")

# %% [markdown]
# ### How to read that table
#
# `significance_sigma` on its own means nothing — it is the maximum over hundreds
# of stars, and the maximum of hundreds of noisy draws is always large.
#
# **`field_p_value` is the column that matters.** It is the fraction of stars in
# the same image that produced a dip at least as strong at the same phase. If it
# is not tiny, the candidate is simply what this field does.
#
# `field_sigma_95` says how large a dip the top 5 % of ordinary stars reach. A
# candidate must clear that before it deserves another sentence.
#
# `depth_ratio` compares the measured depth to the published one. It is reported
# **after** selection and never used for it — picking whichever star best matched
# the expected depth would manufacture the very agreement we are testing for.
#
# **And one more correction, the one it is easiest to skip.** We test roughly
# 2,700 star-nights. A p-value of 0.01 means "one star in a hundred does this",
# so about **27 stars should clear that bar with nothing there at all**. Any
# count of hits has to be quoted against that expectation, and the next cell
# does exactly that. A list of 20 hits out of 2,700 is not 20 discoveries; it is
# a ranking of the most transit-like stars in the archive.

# %%
if len(search):
    hits = search[(search.field_p_value <= 0.01)
                  & (search.significance_sigma > search.field_sigma_95)
                  & (search.depth_pct > 0)].copy()

    # The look-elsewhere effect. field_p_value is a rank within its own session,
    # so by construction about 1 % of all stars tested land at p <= 0.01 whether
    # or not anything is there. Quoting the count of hits without the count we
    # expected from noise would be the single easiest way to fool ourselves here.
    n_tested = len(search)
    n_expected = 0.01 * n_tested
    print("%d star-nights tested. At p <= 0.01 we expect ~%.0f by chance alone."
          % (n_tested, n_expected))
    print("Stars beating their own field at p <= 0.01: %d" % len(hits))
    print("-> %s\n" % ("FEWER than chance predicts. The hit list is a ranking, "
                       "not a detection." if len(hits) <= n_expected else
                       "an excess of %.0f over chance." % (len(hits) - n_expected)))
    if len(hits):
        cols = ["target", "session_id", "track_id", "depth_pct",
                "significance_sigma", "field_sigma_95", "field_p_value",
                "n_field_stars", "expected_depth_pct", "depth_ratio", "rms_ppt"]
        print(hits[cols].round(3).to_string(index=False))
        hits.to_csv(RESULTS / "candidates.csv", index=False)

        # Depth agreement was never used to select these stars, so asking whether
        # the selected ones agree with the published depth more often than the
        # field does is a fair test. If the hits are just noise, the two rates
        # should match.
        near = hits[hits.depth_ratio.between(0.5, 2.0)]
        pool = search[np.isfinite(search.depth_ratio)]
        r_hits = len(near) / max(len(hits[np.isfinite(hits.depth_ratio)]), 1)
        r_pool = pool.depth_ratio.between(0.5, 2.0).mean() if len(pool) else np.nan
        print("\nOf those, %d have a depth within a factor of 2 of the published "
              "value." % len(near))
        print("depth agrees within 2x:  %.0f%% of hits  vs  %.0f%% of all stars "
              "tested" % (r_hits * 100, r_pool * 100))
        print("(depth was never used to select, so this comparison is fair)")

        # How surprising is that enrichment? If the hits were ordinary stars,
        # each would agree with probability r_pool, so the count of agreements is
        # binomial. This is the only number in the notebook that is evidence FOR
        # anything, so it gets a real test rather than an adjective.
        n_h = int(np.isfinite(hits.depth_ratio).sum())
        if n_h and np.isfinite(r_pool):
            p_enrich = float(stats.binom.sf(len(near) - 1, n_h, r_pool))
            print("binomial p(%d or more of %d agreeing at a base rate of %.2f) "
                  "= %.2g" % (len(near), n_h, r_pool, p_enrich))
            print("-> %s" % ("the stars our field test picked out do land on the "
                             "published depth more often than the field does."
                             if p_enrich < 0.01 else
                             "not significant; consistent with chance."))

        print("\nThese are CANDIDATES CONSISTENT WITH A TRANSIT.")
        print("They are NOT confirmed planets, for four specific reasons:")
        print("  1. The number of hits does not exceed what chance predicts, so")
        print("     the list is a RANKING of the most transit-like stars, not a")
        print("     population of detections.")
        print("  2. The host star was never identified astrometrically -- there is")
        print("     no WCS solution, so we cannot say this is the catalogued host.")
        print("  3. A blended eclipsing binary inside the 5 arcsec/px aperture")
        print("     cannot be excluded.")
        print("  4. Each detection is from a single night. Confirming a period")
        print("     needs the same sky position across nights, which needs the")
        print("     cross-match we cannot do.")
    else:
        print("NULL RESULT. No star stands out against its own field at the")
        print("published phase. Reporting that honestly is worth more than")
        print("forcing a detection out of a %.0f ppt noise floor."
              % fs.rms_ppt.min())

# %% [markdown]
# ## 13b · Planet physics — how far, how fast, how big
#
# The frames tell us **when** the star dims and **by how much**. Turning that into
# distances and speeds needs a few textbook equations and a few numbers our
# telescope cannot measure (the star's mass and size, the distance to the system).
# Those come from the **NASA Exoplanet Archive** (`pscomppars` table) and are
# external data, disclosed as such in every output row.
#
# | Quantity | Equation | From our frames | From the archive |
# |---|---|---|---|
# | Orbit size *a* | Kepler III: $a^3 = G(M_\star+M_p)P^2/4\pi^2$ | — | $P, M_\star, M_p$ |
# | Orbital speed *v* | $v = 2\pi a/P$ | — | via *a* |
# | Orbit in star radii | $a/R_\star$ | — | $R_\star$ |
# | Transit duration | $T_{14} = \frac{P}{\pi}\arcsin\!\left(\frac{\sqrt{(1+k)^2-b^2}}{(a/R_\star)\sin i}\right)$ | — | $k, b$ |
# | Planet temperature | $T_{eq} = T_\star\sqrt{R_\star/2a}$ (albedo 0) | — | $T_\star$ |
# | Starlight received | $S/S_\oplus = (R_\star/R_\odot)^2 (T_\star/5772)^4 / a_{AU}^2$ | — | $T_\star, R_\star$ |
# | **Planet radius** | $R_p = R_\star\sqrt{\delta}$ | **depth δ** | $R_\star$ |
# | Distance from Earth | parsecs → light-years | — | Gaia distance |
#
# **Validation.** Every derived value is compared with the archive's own published
# value for the same planet: percentage difference, and a z-score where both sides
# have uncertainties. This checks our equations and inputs, not the planet — the
# archive values were themselves derived from transits much better than ours.
#
# **What is honestly ours.** Only the planet radius from *our* depth uses our
# measurements, and only where the depth was measured. Everything else is derived
# from catalogue inputs; we say so rather than let a distance look like a discovery.
# The distance from Earth is Gaia's number, converted — a 6-inch telescope with no
# astrometric solution cannot measure parallax.

# %%
# Physical constants. IAU 2015 nominal values where they exist.
G_SI = 6.67430e-11
GM_SUN = 1.3271244e20            # m^3 s^-2
GM_JUP = 1.2668653e17
R_SUN = 6.957e8                  # m
R_JUP = 7.1492e7
AU = 1.495978707e11
PARSEC = 3.0856775814913673e16
LIGHT_YEAR = 9.4607304725808e15
DAY = 86400.0
T_SUN = 5772.0                   # K
V_EARTH_KMS = 29.78              # Earth's mean orbital speed, for scale
MERCURY_AU = 0.387

# Our target names -> archive host names. HAT-P-10 is catalogued there as WASP-11.
TARGET_TO_ARCHIVE = {
    "TRES-1": "TrES-1", "TRES-3": "TrES-3", "TRES-5": "TrES-5",
    "CoRoT-2": "CoRoT-2", "Qatar-1": "Qatar-1", "WASP-2": "WASP-2",
    "WASP-10": "WASP-10", "HATP-10": "WASP-11",
}

PHYS_COLS = ["hostname", "pl_name", "pl_orbper", "pl_orbpererr1", "pl_orbsmax",
             "pl_orbsmaxerr1", "pl_radj", "pl_radjerr1", "pl_bmassj",
             "pl_bmassjerr1", "pl_orbincl", "pl_imppar", "pl_ratdor", "pl_ratror",
             "pl_trandep", "pl_trandur", "pl_eqt", "pl_insol", "pl_dens",
             "st_mass", "st_masserr1", "st_rad", "st_raderr1", "st_teff",
             "st_tefferr1", "sy_dist", "sy_disterr1", "sy_vmag", "ra", "dec"]

# Snapshot of the same query, taken 2026-09-16, used only if the archive is
# unreachable. Marked as a snapshot in every output row.
CATALOG_SNAPSHOT = [
    dict(hostname="TrES-1", pl_name="TrES-1 b", pl_orbper=3.03007, pl_orbpererr1=8e-06, pl_orbsmax=0.03925, pl_orbsmaxerr1=0.00056, pl_radj=1.13, pl_radjerr1=0.06, pl_bmassj=0.84, pl_orbincl=90.0, pl_imppar=0.191, pl_ratdor=10.48, pl_ratror=0.1358, pl_trandep=1.8, pl_trandur=2.508, pl_eqt=1140.0, pl_insol=318.4582, pl_dens=0.712, st_mass=1.04, st_masserr1=0.19, st_rad=0.85, st_raderr1=0.05, st_teff=5230.0, sy_dist=159.658, sy_disterr1=0.736, sy_vmag=11.424, ra=286.040876, dec=36.632536),
    dict(hostname="CoRoT-2", pl_name="CoRoT-2 b", pl_orbper=1.742994, pl_orbpererr1=1e-06, pl_orbsmax=0.02798, pl_orbsmaxerr1=0.00076, pl_radj=1.466, pl_radjerr1=0.042, pl_bmassj=3.47, pl_orbincl=88.08, pl_imppar=0.221, pl_ratdor=6.7, pl_ratror=0.1667, pl_trandep=2.75, pl_trandur=2.26704, pl_eqt=1521.0, pl_insol=932.9076, pl_dens=1.47, st_mass=0.96, st_masserr1=0.08, st_rad=0.906, st_raderr1=0.026, st_teff=5625.0, sy_dist=213.283, sy_disterr1=2.485, sy_vmag=12.516, ra=291.777046, dec=1.383663),
    dict(hostname="TrES-5", pl_name="TrES-5 b", pl_orbper=1.482247, pl_orbpererr1=6e-06, pl_orbsmax=0.02459, pl_orbsmaxerr1=0.0007, pl_radj=1.194, pl_radjerr1=0.015, pl_bmassj=1.79, pl_orbincl=84.27, pl_imppar=0.577, pl_ratdor=6.1, pl_ratror=0.143, pl_trandep=2.192, pl_trandur=1.595, pl_eqt=1480.0, pl_insol=442.993, pl_dens=1.31, st_mass=0.901, st_masserr1=0.03, st_rad=0.868, st_raderr1=0.013, st_teff=5171.0, sy_dist=360.313, sy_disterr1=1.921, sy_vmag=13.677, ra=305.221945, dec=59.448903),
    dict(hostname="Qatar-1", pl_name="Qatar-1 b", pl_orbper=1.420024, pl_orbpererr1=0.0, pl_orbsmax=0.02332, pl_orbsmaxerr1=0.0004, pl_radj=1.143, pl_radjerr1=0.026, pl_bmassj=1.294, pl_orbincl=84.08, pl_imppar=0.645, pl_ratdor=6.247, pl_ratror=0.14629, pl_trandep=2.14, pl_trandur=1.66104, pl_eqt=1418.0, pl_insol=382.328, pl_dens=1.076, st_mass=0.838, st_masserr1=0.043, st_rad=0.803, st_raderr1=0.016, st_teff=5013.0, sy_dist=185.615, sy_disterr1=0.805, sy_vmag=12.692, ra=303.38187, dec=65.162331),
    dict(hostname="WASP-11", pl_name="WASP-11 b", pl_orbper=3.72247, pl_orbpererr1=7e-06, pl_orbsmax=0.0435, pl_orbsmaxerr1=0.0006, pl_radj=1.11, pl_radjerr1=0.1, pl_bmassj=0.79, pl_orbincl=89.8, pl_imppar=0.054, pl_ratdor=12.71, pl_ratror=0.131, pl_trandep=1.6, pl_trandur=2.556, pl_eqt=992.0, pl_insol=184.988, pl_dens=0.632, st_mass=1.42, st_masserr1=0.43, st_rad=0.89, st_raderr1=0.08, st_teff=4800.0, sy_dist=124.73, sy_disterr1=2.112, sy_vmag=11.567, ra=47.368947, dec=30.673382),
    dict(hostname="WASP-2", pl_name="WASP-2 b", pl_orbper=2.152175, pl_orbpererr1=1.2e-05, pl_orbsmax=0.03144, pl_orbsmaxerr1=0.00088, pl_radj=1.081, pl_radjerr1=0.041, pl_bmassj=0.931, pl_orbincl=84.49, pl_imppar=0.749, pl_ratdor=7.81, pl_ratror=0.12831, pl_trandep=1.646, pl_trandur=1.78824, pl_eqt=1311.0, pl_insol=492.0, pl_dens=0.914, st_mass=0.895, st_masserr1=0.077, st_rad=0.866, st_raderr1=0.031, st_teff=5180.0, sy_dist=153.242, sy_disterr1=1.64, sy_vmag=11.728, ra=307.725559, dec=6.42933),
    dict(hostname="TrES-3", pl_name="TrES-3 b", pl_orbper=1.306186, pl_orbpererr1=np.nan, pl_orbsmax=0.02282, pl_orbsmaxerr1=0.00023, pl_radj=1.336, pl_radjerr1=0.031, pl_bmassj=1.91, pl_orbincl=81.85, pl_imppar=0.84, pl_ratdor=5.926, pl_ratror=0.1655, pl_trandep=2.739, pl_trandur=1.417775, pl_eqt=1638.0, pl_insol=1025.5022, pl_dens=0.994, st_mass=0.928, st_masserr1=0.028, st_rad=0.829, st_raderr1=0.015, st_teff=5650.0, sy_dist=231.337, sy_disterr1=1.303, sy_vmag=12.362, ra=268.029111, dec=37.546327),
    dict(hostname="WASP-10", pl_name="WASP-10 b", pl_orbper=3.092762, pl_orbpererr1=1.1e-05, pl_orbsmax=0.03781, pl_orbsmaxerr1=0.00067, pl_radj=1.08, pl_radjerr1=0.02, pl_bmassj=3.15, pl_orbincl=88.49, pl_imppar=0.299, pl_ratdor=11.65, pl_ratror=0.15918, pl_trandep=2.525, pl_trandur=2.2271, pl_eqt=1370.0, pl_insol=174.0323, pl_dens=4.14, st_mass=0.75, st_masserr1=0.04, st_rad=0.698, st_raderr1=0.012, st_teff=4675.0, sy_dist=140.998, sy_disterr1=0.75, sy_vmag=12.413, ra=348.993046, dec=31.462751),
]


def fetch_planet_catalog(hosts):
    import io
    import urllib.parse
    import urllib.request
    q = ("select %s from pscomppars where hostname in (%s)"
         % (",".join(PHYS_COLS), ",".join("'%s'" % h for h in hosts)))
    url = ("https://exoplanetarchive.ipac.caltech.edu/TAP/sync?format=csv&query="
           + urllib.parse.quote(q))
    with urllib.request.urlopen(url, timeout=60) as r:
        return pd.read_csv(io.StringIO(r.read().decode()))


try:
    catalog = fetch_planet_catalog(sorted(TARGET_TO_ARCHIVE.values()))
    catalog["catalog_source"] = ("NASA Exoplanet Archive pscomppars (live, %s)"
                                 % pd.Timestamp.now("UTC").date())
    print("planet catalog: fetched live, %d planets" % len(catalog))
except Exception as exc:
    print("!! archive query failed (%s) -- using the 2026-09-16 snapshot"
          % type(exc).__name__)
    catalog = pd.DataFrame(CATALOG_SNAPSHOT)
    catalog["catalog_source"] = "NASA Exoplanet Archive pscomppars (snapshot 2026-09-16)"

catalog = catalog.reindex(columns=PHYS_COLS + ["catalog_source"])
archive_to_target = {v: k for k, v in TARGET_TO_ARCHIVE.items()}
catalog.insert(0, "target", catalog.hostname.map(archive_to_target))
catalog.to_csv(RESULTS / "planet_catalog.csv", index=False)


# %% [markdown]
# ### Derive, with uncertainties
#
# Uncertainties are propagated by Monte Carlo: draw every catalogue input 20,000
# times from a normal distribution with its published error, push each draw through
# the equations, and report the median and the 16–84 % half-width. That handles the
# cube root and arcsine honestly, where the linear error formula would not. Inputs
# with no published error are held fixed, and the output says which.

# %%
RNG = np.random.default_rng(42)
N_MC = 20000


def _draw(value, err):
    if not np.isfinite(value):
        return np.full(N_MC, np.nan)
    if not (np.isfinite(err) and err > 0):
        return np.full(N_MC, float(value))
    return np.clip(RNG.normal(value, err, N_MC), value * 1e-3, None)


def _summ(x):
    x = x[np.isfinite(x)]
    if not len(x):
        return np.nan, np.nan
    lo, med, hi = np.percentile(x, [16, 50, 84])
    return float(med), float((hi - lo) / 2)


# Header pointing, to measure how far the telescope aimed from the catalogued host.
pt = pd.read_csv(RESULTS / "per_target.csv").set_index("target")
_sess = pd.read_csv(RESULTS / "sessions.csv")
_mjd = _sess.groupby("target").mjd_start.mean()


def _pointing_offset(c):
    """Arcminutes between the header RA/DEC and the catalogued host.

    The headers are in coordinates of the date, not J2000: compared naively they
    sit 5-21 arcmin from the archive position, which is exactly 26 years of
    precession. Precess the catalogue position to the observing epoch first.
    """
    if c.target not in pt.index:
        return np.nan
    try:
        from astropy import units as u
        from astropy.coordinates import FK5, SkyCoord
        from astropy.time import Time
        now = FK5(equinox=Time(_mjd[c.target], format="mjd"))
        cat_now = SkyCoord(c.ra * u.deg, c.dec * u.deg, frame="icrs").transform_to(now)
        hdr = SkyCoord(pt.loc[c.target, "ra_deg"] * u.deg,
                       pt.loc[c.target, "dec_deg"] * u.deg, frame=now)
        return float(cat_now.separation(hdr).arcmin)
    except Exception:
        return np.nan

phys_rows = []
for _, c in catalog.iterrows():
    P = _draw(c.pl_orbper, c.pl_orbpererr1) * DAY
    m_star = _draw(c.st_mass, c.st_masserr1) * GM_SUN
    m_pl = _draw(c.pl_bmassj, c.pl_bmassjerr1) * GM_JUP
    r_star = _draw(c.st_rad, c.st_raderr1) * R_SUN
    teff = _draw(c.st_teff, c.st_tefferr1)

    gm = m_star + np.nan_to_num(m_pl)            # planet mass adds ~0.1-0.3 %
    a = np.cbrt(gm * P ** 2 / (4 * np.pi ** 2))  # metres
    v = 2 * np.pi * a / P                         # m/s
    a_rs = a / r_star
    k, b = c.pl_ratror, c.pl_imppar
    sin_i = np.sqrt(np.clip(1 - (b / a_rs) ** 2, 0, 1))
    arg = np.sqrt(np.clip((1 + k) ** 2 - b ** 2, 0, None)) / (a_rs * sin_i)
    t14_h = P / np.pi * np.arcsin(np.clip(arg, 0, 1)) / 3600
    teq = teff * np.sqrt(r_star / (2 * a))
    insol = (r_star / R_SUN) ** 2 * (teff / T_SUN) ** 4 / (a / AU) ** 2
    depth_cat = c.pl_trandep / 100.0
    rp_cat_depth = r_star * np.sqrt(depth_cat) / R_JUP

    row = dict(target=c.target, planet=c.pl_name, archive_host=c.hostname,
               catalog_source=c.catalog_source)
    for name, arr in [("a_au", a / AU), ("orbital_speed_kms", v / 1000),
                      ("a_over_rstar", a_rs), ("transit_duration_h", t14_h),
                      ("teq_k", teq), ("insolation_earth", insol),
                      ("rp_from_catalog_depth_rjup", rp_cat_depth)]:
        row[name], row[name + "_err"] = _summ(arr)

    row["orbital_period_days"] = float(c.pl_orbper)
    row["orbit_in_mercury_orbits"] = row["a_au"] / MERCURY_AU
    row["orbital_speed_vs_earth"] = row["orbital_speed_kms"] / V_EARTH_KMS
    row["orbit_circumference_million_km"] = 2 * np.pi * row["a_au"] * AU / 1e9
    row["distance_pc"] = float(c.sy_dist)
    row["distance_pc_err"] = float(c.sy_disterr1)
    row["distance_ly"] = float(c.sy_dist) * PARSEC / LIGHT_YEAR
    row["distance_ly_err"] = float(c.sy_disterr1) * PARSEC / LIGHT_YEAR
    row["star_vmag"] = float(c.sy_vmag)
    row["inputs_without_errors"] = ",".join(
        n for n, e in [("pl_orbper", c.pl_orbpererr1), ("st_mass", c.st_masserr1),
                       ("pl_bmassj", c.pl_bmassjerr1), ("st_rad", c.st_raderr1),
                       ("st_teff", c.st_tefferr1)]
        if not (np.isfinite(e) and e > 0)) or None

    row["header_pointing_offset_arcmin"] = _pointing_offset(c)
    row["header_pointing_offset_px"] = row["header_pointing_offset_arcmin"] * 60 / 5.0
    phys_rows.append(row)

physics = pd.DataFrame(phys_rows)
physics["distance_source"] = "Gaia via NASA Exoplanet Archive (external, not measured here)"
physics["speed_distance_basis"] = ("Derived: Kepler III from archive P, M_star, M_planet. "
                                   "No input from our frames.")

# %% [markdown]
# ### Planet radius from *our* measured depth
#
# The one derived quantity that uses our photometry. Two sources of depth:
#
# * **target window** — the star chosen by position, measured where the ephemeris
#   predicts the transit (section 11);
# * **best field candidate** — the most significant star at the published phase that
#   passed the field test (section 13). Its identity as the host is **unverified**
#   (no WCS), so it is reported separately and never merged with the first.
#
# A radius from a depth below 3σ is arithmetic on noise. It is still reported,
# because hiding it would hide the failure, but it is flagged `inconclusive`.
#
# Limb darkening makes the true depth slightly larger than $k^2$ at mid-transit, so
# $R_\star\sqrt{\delta}$ runs a few percent high against the archive radius even with
# perfect data. That bias is small next to our error bars.

# %%
cat_by_target = catalog.set_index("target")
meas = []
dep = pd.read_csv(RESULTS / "transit_depths.csv")
for _, d in dep.iterrows():
    meas.append(dict(target=d.target, session_id=d.session_id,
                     depth_source="target window (position-chosen host)",
                     track_id=None, depth_pct=d.depth_pct, depth_err_pct=d.depth_err_pct,
                     significance_sigma=d.significance_sigma))
cand_path = RESULTS / "candidates.csv"
if cand_path.exists():
    cand = pd.read_csv(cand_path)
    for tgt, g in cand.groupby("target"):
        best = g.sort_values("significance_sigma", ascending=False).iloc[0]
        meas.append(dict(target=tgt, session_id=best.session_id,
                         depth_source="best field candidate (host identity unverified)",
                         track_id=int(best.track_id), depth_pct=best.depth_pct,
                         depth_err_pct=best.depth_pct / best.significance_sigma,
                         significance_sigma=best.significance_sigma))
radius_meas = pd.DataFrame(meas)

rp, rp_err, cat_dep, cat_rp = [], [], [], []
for _, m in radius_meas.iterrows():
    c = cat_by_target.loc[m.target]
    if m.depth_pct > 0:
        r = c.st_rad * np.sqrt(m.depth_pct / 100) * R_SUN / R_JUP
        frac = np.hypot(np.nan_to_num(c.st_raderr1 / c.st_rad),
                        m.depth_err_pct / (2 * m.depth_pct))
        rp.append(r)
        rp_err.append(r * frac)
    else:
        rp.append(np.nan)
        rp_err.append(np.nan)
    cat_dep.append(c.pl_trandep)
    cat_rp.append(c.pl_radj)
radius_meas["rp_rjup"] = rp
radius_meas["rp_err_rjup"] = rp_err
radius_meas["catalog_depth_pct"] = cat_dep
radius_meas["catalog_rp_rjup"] = cat_rp
radius_meas["depth_z_vs_catalog"] = ((radius_meas.depth_pct - radius_meas.catalog_depth_pct)
                                     / radius_meas.depth_err_pct)
radius_meas["status"] = np.where(
    radius_meas.depth_pct <= 0, "null (star brightened)",
    np.where(radius_meas.significance_sigma < 3, "inconclusive (below 3 sigma)",
             "measured"))
radius_meas.to_csv(RESULTS / "planet_radius_measurements.csv", index=False)
print(radius_meas[["target", "session_id", "depth_source", "depth_pct",
                   "significance_sigma", "rp_rjup", "rp_err_rjup",
                   "catalog_rp_rjup", "status"]].round(3).to_string(index=False))

# %% [markdown]
# ### Validation against the published values

# %%
def _check(target, planet, quantity, unit, derived, derived_err, published,
           published_err, equation, uses_our_data=False, note=None):
    diff = derived - published
    pct = 100 * diff / published if published else np.nan
    sig = np.hypot(np.nan_to_num(derived_err), np.nan_to_num(published_err))
    z = diff / sig if sig > 0 else np.nan
    if not np.isfinite(derived) or not np.isfinite(published):
        status = "not checkable"
    elif np.isfinite(z) and abs(z) <= 2:
        status = "agrees"
    elif abs(pct) <= 10:
        status = "close"
    else:
        status = "disagrees"
    return dict(target=target, planet=planet, quantity=quantity, unit=unit,
                derived=derived, derived_err=derived_err, published=published,
                published_err=published_err, pct_diff=pct, z_score=z,
                status=status, equation=equation, uses_our_data=uses_our_data,
                note=note)


checks = []
for _, p in physics.iterrows():
    c = cat_by_target.loc[p.target]
    t, pl = p.target, p.planet
    checks += [
        _check(t, pl, "semi_major_axis", "AU", p.a_au, p.a_au_err, c.pl_orbsmax,
               c.pl_orbsmaxerr1, "a^3 = G(M*+Mp) P^2 / 4 pi^2"),
        _check(t, pl, "a_over_rstar", "stellar radii", p.a_over_rstar, p.a_over_rstar_err,
               c.pl_ratdor, np.nan, "a / R*"),
        _check(t, pl, "transit_duration", "h", p.transit_duration_h,
               p.transit_duration_h_err, c.pl_trandur, np.nan,
               "T14 = P/pi * asin( sqrt((1+k)^2 - b^2) / ((a/R*) sin i) )"),
        _check(t, pl, "equilibrium_temperature", "K", p.teq_k, p.teq_k_err,
               c.pl_eqt, np.nan, "Teq = T* sqrt(R*/2a), albedo 0",
               note="Published Teq assumptions (albedo, heat redistribution) vary by paper."),
        _check(t, pl, "insolation", "Earth=1", p.insolation_earth,
               p.insolation_earth_err, c.pl_insol, np.nan,
               "S = (R*/Rsun)^2 (T*/5772)^4 / a^2"),
        _check(t, pl, "radius_from_catalog_depth", "R_jup",
               p.rp_from_catalog_depth_rjup, p.rp_from_catalog_depth_rjup_err,
               c.pl_radj, c.pl_radjerr1, "Rp = R* sqrt(depth)",
               note="Limb darkening biases this a few percent away from k*R*."),
    ]

for _, m in radius_meas.iterrows():
    note = "%s, %s, %.1f sigma" % (m.depth_source, m.session_id, m.significance_sigma)
    row = _check(m.target, cat_by_target.loc[m.target, "pl_name"], "transit_depth_ours",
                 "%", m.depth_pct, m.depth_err_pct, m.catalog_depth_pct, np.nan,
                 "in-window mean vs out-of-window mean", uses_our_data=True, note=note)
    row2 = _check(m.target, cat_by_target.loc[m.target, "pl_name"],
                  "radius_from_our_depth", "R_jup", m.rp_rjup, m.rp_err_rjup,
                  m.catalog_rp_rjup, cat_by_target.loc[m.target, "pl_radjerr1"],
                  "Rp = R* sqrt(depth_ours)", uses_our_data=True, note=note)
    # A negative depth has no radius; one "null" row says it without a duplicate.
    for r in ((row, row2) if m.depth_pct > 0 else (row,)):
        if m.status != "measured":
            r["status"] = m.status   # agreement with noise is not validation
        checks.append(r)

validation = pd.DataFrame(checks)
validation.to_csv(RESULTS / "physics_validation.csv", index=False)
physics.to_csv(RESULTS / "planet_physics.csv", index=False)

print("\nDerived vs published:")
print(validation.groupby(["uses_our_data", "status"]).size().to_string())
print()
print(physics[["target", "a_au", "orbital_speed_kms", "orbital_period_days",
               "teq_k", "distance_ly", "header_pointing_offset_arcmin"]]
      .round(3).to_string(index=False))
bad = validation[(~validation.uses_our_data) & (validation.status == "disagrees")]
if len(bad):
    print("\nDisagreements (catalogue-only checks), reported not hidden:")
    print(bad[["target", "quantity", "derived", "published", "pct_diff"]]
          .round(3).to_string(index=False))

# %% [markdown]
# ### Reading the validation
#
# * **Catalogue-only checks** (`uses_our_data = False`) test the equations. When
#   one disagrees, recompute it by hand from the archive's own inputs before
#   blaming the equation: `pscomppars` stitches together parameters from different
#   papers, so a published insolation or temperature is not always consistent
#   with the published stellar radius, temperature and orbit sitting next to it.
#   In the 2026-09-16 run every disagreement traced back to that, not to our maths.
#   A large stellar-mass error (WASP-11: 1.42 ± 0.43 M☉) shows up as a wide error
#   bar rather than a disagreement — which is the correct behaviour.
# * **Checks on our data** test the measurement. Most target-window depths are
#   below 3σ or negative and are flagged `inconclusive` or `null`: we do not claim
#   to have measured these planets' radii, and the table says so line by line.
#   The field-candidate radii come from stars whose identity as the host is
#   unverified, so an agreement there is suggestive, not a measurement of the planet.
# * `header_pointing_offset_arcmin` is how far the telescope's recorded pointing
#   sits from the catalogued host, **after** precessing the catalogue to the date
#   (the headers use coordinates of the date; skipping that step fakes a 5–21′
#   pointing error). The result is under an arcminute — a few pixels — for every
#   target, so the host is near the pointing centre even though, without a WCS,
#   we still cannot say which star it is.

# %% [markdown]
# ## 14 · Data-quality correlations
#
# Which observing conditions actually govern image quality? We have the header
# telemetry and we have per-frame measurements, so we can answer this instead of
# guessing. The airmass result is the interesting one.

# %%
qcols = ["WEATHER", "airmass", "TELALT", "CAMTEMP", "TELTEMP", "TELHUM",
         "sky_level", "sky_sigma", "peak_contrast", "n_source_px"]
corr = frames[qcols].corr(method="pearson").round(3)
corr.to_csv(RESULTS / "quality_correlations.csv")

print("Correlations against image quality across all 1681 frames:\n")
for a, b in [("WEATHER", "peak_contrast"), ("WEATHER", "n_source_px"),
             ("sky_level", "n_source_px"), ("airmass", "sky_level"),
             ("airmass", "n_source_px"), ("airmass", "peak_contrast"),
             ("CAMTEMP", "sky_sigma")]:
    print(f"  corr({a:12s}, {b:14s}) = {corr.loc[a, b]:+.3f}")

frames[["session_id", "frame_index", "t_utc", "target", "airmass", "TELALT",
        "WEATHER", "sky_level", "sky_sigma", "peak_contrast", "n_source_px",
        "n_saturated", "CAMTEMP"]].to_csv(RESULTS / "frame_quality.csv", index=False)
corr

# %% [markdown]
# ## 15 · Dashboard image assets
#
# The website must never touch a FITS file or a 1.1 GB array. We pre-render PNGs
# here, once, and the site just serves them.
#
# Three stretches per frame, because no single stretch shows everything: `linear`
# is honest but hides faint stars, `zscale` is what astronomers actually use, and
# `asinh` shows faint structure and bright cores at the same time.

# %%
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def zscale(img, contrast=0.25, n_samples=10000):
    """IRAF zscale: fit a line to the sorted sample and take its central slope."""
    s = np.sort(RNG.choice(np.asarray(img, dtype=np.float32).ravel(),
                           size=min(n_samples, img.size), replace=False))
    n = len(s)
    x = np.arange(n) - n // 2
    slope = np.polyfit(x, s, 1)[0] / max(contrast, 1e-6)
    mid = s[n // 2]
    return mid + slope * x[0], mid + slope * x[-1]


def stretch(img, mode="zscale"):
    f = np.asarray(img, dtype=np.float32)
    if mode == "linear":
        lo, hi = np.percentile(f, [1, 99.5])
    elif mode == "asinh":
        lo, hi = np.percentile(f, [1, 99.9])
        f = np.arcsinh((f - lo) / max(hi - lo, 1e-6) * 10.0)
        lo, hi = float(f.min()), float(f.max())
    else:
        lo, hi = zscale(f)
    return np.clip((f - lo) / max(hi - lo, 1e-6), 0, 1)


def save_png(img, path, mode="zscale"):
    plt.imsave(path, stretch(img, mode), cmap="gray", vmin=0, vmax=1, origin="lower")


# A calibration triptych for the dashboard's Calibration Lab page.
tri = STATIC / "triptych"
tri.mkdir(parents=True, exist_ok=True)
save_png(demo_raw, tri / "raw.png", "zscale")
save_png(nearest_master(demo_meta["CAMTEMP"].iloc[len(demo_cube) // 2]),
         tri / "master_dark.png", "zscale")
save_png(demo_cal, tri / "calibrated.png", "zscale")

hot_rgb = np.zeros((500, 650, 3), dtype=np.float32)
hot_rgb[..., 0] = hot_mask.astype(np.float32)
plt.imsave(tri / "hot_pixel_map.png", hot_rgb, origin="lower")
print("triptych written to", tri)

# %% [markdown]
# Rendering every frame in three stretches produces ~5,000 PNGs, about 1.5 GB. That
# is minutes of wall clock, so the scope is a switch: `"sample"` for one session
# while iterating, `"good"` for the deploy build, `"all"` if you want the unusable
# nights too.
#
# These PNGs are lossless and far too large to ship. `backend/sync_data.py`
# re-encodes them to WebP, which came out at **12 % of the PNG size** on the first
# session — that is the step that makes the image viewer deployable at all.

# %%
# "sample" is the shipped default: one session is enough to show the viewer works,
# and rendering all 15 costs ~25 minutes of a Kaggle session for assets the
# deployed site already has. Set "good" when rebuilding the deploy's image tiles.
RENDER_SCOPE = "sample"    # "sample" | "good" (deploy build) | "all"
SAMPLE_SESSIONS = ["TRES-5__2026-08-26"]

if RENDER_SCOPE == "all":
    todo = list(ids)
elif RENDER_SCOPE == "good":
    # The six unusable nights are not worth 300 MB of tiles. The dashboard shows
    # that they exist and why they failed; nobody needs to page through them.
    todo = [s for s in ids if SESSION_QUALITY.get(s) == "good"]
else:
    todo = [s for s in SAMPLE_SESSIONS if s in ids]
FORCE_RERENDER = False   # True to redo tiles after changing a stretch

MODES = ("zscale", "asinh", "linear")
print("rendering %d session(s) [scope=%s]" % (len(todo), RENDER_SCOPE))
t0 = time.time()
manifest = []
for sid in todo:
    n_frames = int((frames.session_id == sid).sum())

    # Skip a session whose tiles are already on disk. Rendering is by far the
    # slowest cell in the notebook and its output depends only on the frame and
    # the stretch, so re-running the analysis should not cost 25 minutes of PNGs.
    done = (not FORCE_RERENDER
            and all(len(list((STATIC / "frames" / sid / m).glob("*.png"))) == n_frames
                    for m in MODES)
            and (STATIC / "frames" / sid / "values_4x.npy").exists())
    if done:
        manifest.append({"session_id": sid, "n_frames": n_frames,
                         "modes": ",".join(MODES), "values_downsample": 4})
        print(f"  {sid} already rendered, skipped")
        continue

    cube = load_session(sid)
    for mode in MODES:
        d = STATIC / "frames" / sid / mode
        d.mkdir(parents=True, exist_ok=True)
        for i, img in enumerate(cube):
            save_png(img, d / f"{i:04d}.png", mode)
    # Downsampled pixel values so the web viewer can show real counts on hover
    # without shipping the full array.
    small = cube[:, ::4, ::4].astype(np.uint16)
    np.save(STATIC / "frames" / sid / "values_4x.npy", small)
    manifest.append({"session_id": sid, "n_frames": len(cube),
                     "modes": ",".join(MODES), "values_downsample": 4})
    del cube
    print(f"  {sid} rendered ({time.time()-t0:.0f}s)")

pd.DataFrame(manifest).to_csv(STATIC / "render_manifest.csv", index=False)
print(f"rendered {len(todo)} session(s); RENDER_SCOPE=\"all\" includes the unusable nights")

# %% [markdown]
# ## 16 · Summary of what this notebook produced

# %%
produced = []
for base in (RESULTS, STATIC):
    for p in sorted(base.rglob("*")):
        if p.is_file():
            produced.append({"file": p.relative_to(OUT).as_posix(),
                             "dir": p.parent.relative_to(OUT).as_posix(),
                             "mb": round(p.stat().st_size / 1e6, 3)})
produced = pd.DataFrame(produced)
produced.to_csv(RESULTS / "manifest.csv", index=False)
print(f"{len(produced)} files, {produced.mb.sum():.1f} MB total\n")

# Roll the image tiles up by directory -- listing five thousand PNGs by name
# buries the eight CSVs that actually matter.
roll = (produced.groupby("dir")
        .agg(files=("file", "size"), mb=("mb", "sum"))
        .sort_values("mb", ascending=False).reset_index())
print(roll[roll.mb > 0.05].round(2).to_string(index=False))

# %% [markdown]
# ### Hand-off
#
# * `results/cutouts.npz` → **notebook 02** trains the CNN / YOLO classifiers on it.
# * `results/*.csv` → the **web dashboard** reads these directly. No FITS in the browser.
# * `static/` → pre-rendered PNG tiles for the image viewer.
#
# ### What we can and cannot claim
#
# We can rule out clouds (comparison stars move together), tracking slips (the per-frame
# field offset is recorded), cosmic rays (single-frame), hot pixels (independent
# dark-frame mask) and airmass trends (measured per frame).
#
# We **cannot** rule out a blended eclipsing binary or starspot activity with this
# data. At 5 arcsec/pixel and a single Clear filter there is no way to separate a
# faint companion inside the aperture, and CoRoT-2 in particular is a known spotted
# star. Any signal we report is therefore a **candidate consistent with a transit**,
# never a confirmed planet — which is also exactly what the challenge brief demands.
