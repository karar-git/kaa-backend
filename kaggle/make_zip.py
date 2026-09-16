#!/usr/bin/env python
"""
make_zip.py -- turn build/ into a single Kaggle-uploadable archive.

Kaggle auto-extracts an uploaded .zip, so the directory tree inside the archive
is exactly what the notebook will see under /kaggle/input/<slug>/.

Deflate level 6 was chosen by measurement, not by habit: on this 12-bit-in-
16-bit data it hits ratio 0.29 in 4 s per 59 MB, while level 9 is *worse*
(0.296) and three times slower. LZMA reaches 0.27 but Kaggle only auto-extracts
zip, and 2% is not worth the special handling.

Usage:  python make_zip.py [--build DIR] [--out FILE]
"""

from __future__ import annotations

import argparse
import os
import time
import zipfile

DATA_CARD = """# ExoTransit Lab -- MicroObservatory transit survey (packaged)

Repackaged from the Hack4Dev Iraq 2026 Exoplanet Data Challenge archive.
Pixel data is bit-identical to the source FITS; only the container changed.

## Provenance

| | |
|---|---|
| Source | MicroObservatory Image Directory, Harvard-Smithsonian CfA |
| Telescope | `Cecilia` -- 6" (150 mm) Maksutov-Newtonian, f = 560 mm |
| Site | Fred Lawrence Whipple Observatory, Amado AZ (31.68 N, -110.88 W, 1268 m) |
| Detector | 650 x 500 px, 2x2 binned, 13.6 um pixels, 5.0 arcsec/px, 12-bit (0-4095) |
| Filter | `Clear` on science frames, `Opaque` (shutter closed = dark) on calibration |
| Exposure | 60.0 s on every frame |
| Date range | 2026-08-06 to 2026-09-05 |
| Contents | 1681 science frames + 60 dark frames = 1741 |

Note: the challenge brief states ~1,633 records. The archive as downloaded
contains 1,741. We use the actual file count and flag the discrepancy.

## Layout

```
frames/<TARGET>__<NIGHT>.npy    uint16 (N, 500, 650)  one array per session
darks/darks.npy                 uint16 (60, 500, 650) all dark frames
calib/master_dark.npy           float32 (500, 650)    median of the 60 darks
calib/hot_pixel_mask.npy        bool    (500, 650)    >10 sigma in master dark
metadata/frames.csv             1681 rows -- header + per-frame image stats
metadata/darks.csv              60 rows
metadata/sessions.csv           22 rows -- one per observing session
metadata/original_*.csv|json    the organisers' own manifests, unmodified
```

`frames.csv` joins to the arrays on `(session_id, frame_index)`:

```python
import numpy as np, pandas as pd
f = pd.read_csv("metadata/frames.csv")
row = f.iloc[0]
cube = np.load("frames/" + row.session_id + ".npy")   # (N, 500, 650)
img  = cube[row.frame_index]                          # (500, 650) uint16
```

## The 22 observing sessions

Each session is one continuous run: 51-93 frames, 2.5-5.2 h, ~180 s cadence.
Targets: CoRoT-2 (4 nights), TRES-5 (6), TRES-3 (4), Qatar-1 (3), WASP-2 (2),
TRES-1 (1), WASP-10 (1), HATP-10 (1).

## Derived columns in frames.csv

| Column | Meaning |
|---|---|
| `sky_level` | median counts (robust background) |
| `sky_sigma` | 1.4826 x MAD -- noise estimate immune to stars |
| `peak_contrast` | (brightest non-hot pixel - sky) / sky_sigma |
| `n_source_px` | pixels above sky + 5 sigma, hot pixels excluded |
| `n_saturated` | pixels at the 4095 ceiling |
| `airmass` | sec(z) from `TELALT`, altitude clamped at 3 deg |

Hot pixels are excluded from `peak_contrast` and `n_source_px` on purpose:
the chip has ~690 permanently-bright pixels that would otherwise dominate
every frame and make the metrics blind to the sky.

## Licence / attribution

MicroObservatory data is public and free to use. Please credit the
Center for Astrophysics | Harvard-Smithsonian.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default=r"C:\Users\Lenovo\Downloads\database\kaggle\build")
    ap.add_argument("--out", default=r"C:\Users\Lenovo\Downloads\database\kaggle\exotransit-microobs.zip")
    args = ap.parse_args()

    card = os.path.join(args.build, "DATA_CARD.md")
    with open(card, "w", encoding="utf-8") as fh:
        fh.write(DATA_CARD)

    files = []
    for dp, _, fs in os.walk(args.build):
        for f in fs:
            full = os.path.join(dp, f)
            files.append((full, os.path.relpath(full, args.build).replace("\\", "/")))
    files.sort(key=lambda t: t[1])

    raw = sum(os.path.getsize(f) for f, _ in files)
    print("zipping %d files, %.1f MB raw" % (len(files), raw / 1e6))

    t0 = time.time()
    done = 0
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for full, arc in files:
            z.write(full, arc)
            done += os.path.getsize(full)
            print("  %-42s %5.1f%%  %.0fs" % (arc[:42], 100 * done / raw, time.time() - t0))

    got = os.path.getsize(args.out)
    print("\n[done] %s" % args.out)
    print("[done] %.1f MB  (ratio %.3f)  in %.0fs" % (got / 1e6, got / raw, time.time() - t0))


if __name__ == "__main__":
    main()
