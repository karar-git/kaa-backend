#!/usr/bin/env python
"""
bundle_raw.py -- pack the raw FITS archive into the image, compressed.

The live `/api/field` routes read raw frames from OBS_ROOT. In the container
that is `data/raw/`, which this script fills from the laptop's `database/`
folder, keeping the layout:

    observations/<night>/<target>/<session>/*.fits.fz
    calibration/<night>/*.fits.fz

Frames are Rice-compressed (FITS tile compression, lossless for integer data),
which takes the 1.1 GB archive to roughly a quarter of that. astropy reads
`.fits.fz` transparently, so nothing else changes.

Usage:
    python bundle_raw.py                       # everything
    python bundle_raw.py --only TRES-3         # one target, all its nights
    python bundle_raw.py --only 2026-08-10     # one night
    python bundle_raw.py --no-compress         # plain copies instead
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_SRC = HERE.parent / "database"
DEFAULT_DST = HERE / "data" / "raw"
FITS_EXT = {".fits", ".fit", ".fts"}


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def compress(src: Path, dst: Path) -> None:
    from astropy.io import fits
    with fits.open(src, memmap=False) as hd:
        hdu = next(h for h in hd if h.data is not None and h.data.ndim == 2)
        comp = fits.CompImageHDU(data=hdu.data, header=hdu.header,
                                 compression_type="RICE_1")
        fits.HDUList([fits.PrimaryHDU(), comp]).writeto(dst, overwrite=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--dst", type=Path, default=DEFAULT_DST)
    ap.add_argument("--only", default=None,
                    help="substring a path must contain (target name or night)")
    ap.add_argument("--no-compress", action="store_true")
    ap.add_argument("--clean", action="store_true",
                    help="delete dst first")
    a = ap.parse_args()

    if not (a.src / "observations").is_dir():
        sys.exit(f"{a.src} has no observations/ folder")
    if a.clean and a.dst.exists():
        shutil.rmtree(a.dst)

    files = [p for sub in ("observations", "calibration")
             for p in (a.src / sub).rglob("*") if p.suffix.lower() in FITS_EXT]
    if a.only:
        # Darks always come along: a session without its night's darks is
        # analysed with no dark subtraction.
        files = [p for p in files
                 if a.only in str(p.relative_to(a.src)) or p.parts[-3] == "calibration"]
    if not files:
        sys.exit("nothing matched")

    t0 = time.time()
    n_in = n_out = 0
    for i, p in enumerate(sorted(files), 1):
        rel = p.relative_to(a.src)
        out = a.dst / rel
        if not a.no_compress:
            out = out.with_name(out.name + ".fz")
        out.parent.mkdir(parents=True, exist_ok=True)
        if not out.exists():
            if a.no_compress:
                shutil.copy2(p, out)
            else:
                compress(p, out)
        n_in += p.stat().st_size
        n_out += out.stat().st_size
        if i % 100 == 0 or i == len(files):
            print(f"  {i}/{len(files)}  {human(n_in)} -> {human(n_out)}  "
                  f"{time.time()-t0:.0f}s", flush=True)
    print(f"done: {len(files)} files, {human(n_in)} -> {human(n_out)} "
          f"({n_out/max(n_in,1):.0%}) in {a.dst}")


if __name__ == "__main__":
    main()
