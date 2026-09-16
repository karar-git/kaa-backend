"""Launch EXOTIC with one substitution: image registration from the WCS.

Run as a script, from the job folder, with EXOTIC's own arguments:

    python exotic_driver.py -red inits.json -nea

Everything is EXOTIC's (`exotic.exotic.main`), except that its
`transformation()` -- the astroalign / imreg_dft step that maps the first
frame's star positions onto a later frame -- is replaced by the offset between
the two frames' WCS reference pixels. ExoTransit Lab writes those WCS
headers from the target position it tracked in each frame. EXOTIC's own
registration fails on nearly every MicroObservatory frame (too few bright
stars for astroalign; the imreg_dft fallback raises on numpy >= 1.24) and
then silently uses the identity, which puts its apertures on empty sky.

EXOTIC calls transformation() only when its WCS path raises, which it does
whenever a star's fitted amplitude changes by more than half between
consecutive frames or a comparison star's integer distance to the target
moves by more than a pixel; on faint stars that is most frames. This driver
makes that fallback correct instead of empty. Photometry, comparison-star
selection, aperture optimisation, limb darkening, the fit and every output
file remain EXOTIC's.

This file is standalone on purpose (no imports from the API package), so the
subprocess needs nothing but the `exotic` package.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def _wcs_ref(path: Path) -> tuple[float, float] | None:
    from astropy.io import fits
    try:
        h = fits.getheader(path)
    except Exception:
        return None
    if "CRPIX1" in h and "CRPIX2" in h:
        return float(h["CRPIX1"]), float(h["CRPIX2"])
    return None


def _obs_time(path: Path) -> float:
    from astropy.io import fits
    try:
        h = fits.getheader(path)
    except Exception:
        return float("inf")
    for key in ("MJD-OBS", "JD", "JULIAN"):
        if key in h:
            try:
                return float(h[key])
            except (TypeError, ValueError):
                pass
    for key in ("DATE-OBS", "UT-OBS"):
        if key in h:
            try:
                from astropy.time import Time
                return Time(str(h[key]).replace("-0700", "").replace("+0000", ""), format="isot").mjd
            except Exception:
                pass
    return float("inf")


def first_frame(folder: Path) -> Path | None:
    """The earliest frame in the folder: the image EXOTIC aligns everything to."""
    files = [p for p in folder.iterdir() if p.is_file()
             and p.name.lower().endswith((".fits", ".fit", ".fts", ".fz", ".gz"))]
    if not files:
        return None
    return min(files, key=_obs_time)


def wcs_translation(first: Path, current: Path) -> tuple[float, float] | None:
    """Pixel shift that takes a position in `first` to the same star in
    `current`, from the WCS reference pixels (both frames share CRVAL)."""
    a, b = _wcs_ref(first), _wcs_ref(current)
    if a is None or b is None:
        return None
    return b[0] - a[0], b[1] - a[1]


def install(exotic_module, frames_dir: Path) -> None:
    from skimage.transform import SimilarityTransform
    original = exotic_module.transformation
    state = {"first": first_frame(frames_dir)}

    def transformation(image_data, file_name, roi=1):
        first = state["first"]
        shift = wcs_translation(first, Path(file_name)) if first else None
        if shift is None:
            return original(image_data, file_name, roi)
        exotic_module.log.debug(f"WCS registration for {file_name}: shift {shift[0]:+.2f}, {shift[1]:+.2f} px")
        return SimilarityTransform(scale=1, rotation=0, translation=[shift[0], shift[1]])

    exotic_module.transformation = transformation


def main() -> None:
    import json
    args = sys.argv[1:]
    inits = next((Path(a) for a in args if a.endswith(".json")), Path("inits.json"))
    frames_dir = Path(json.loads(inits.read_text())["user_info"]["Directory with FITS files"])
    import exotic.exotic as ex
    install(ex, frames_dir)
    sys.argv = ["exotic"] + args
    ex.main()


if __name__ == "__main__":
    main()
