#!/usr/bin/env python
"""
sync_data.py -- copy pipeline output into the API's data directory.

The pipeline writes `kaggle/out/{results,static}`. The API reads
`backend/data/{results,static}`. This script moves one to the other and, in the
process, solves the only real deployment problem: image size.

The pipeline renders PNGs, which are lossless and enormous — 1,681 frames in
three stretches is roughly 1.5 GB, far too much to bake into a container image.
Re-encoding to WebP at quality 82 cuts that by ~90% with no visible difference on
a greyscale star field, and by default we ship only the sessions that passed
quality triage in two stretches rather than three.

Usage:
    python sync_data.py                     # good sessions, zscale + asinh
    python sync_data.py --all-sessions      # every session
    python sync_data.py --stretches zscale  # one stretch only, smallest image
    python sync_data.py --no-images         # results CSVs only (~50 MB)
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_SRC = HERE.parent / "kaggle" / "out"
DEFAULT_DST = HERE / "data"


def human(n: int) -> str:
    return f"{n/1e6:.1f} MB" if n >= 1e6 else f"{n/1e3:.0f} kB"


def dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.exists() else 0


# Pipeline outputs that exist for notebook 02, not for the API. No endpoint reads
# them, and cutouts.npz alone is 32 MB of training stamps that would ride along in
# every container build for nothing.
TRAINING_ONLY = {"cutouts.npz", "cutouts_meta.csv"}


def copy_results(src: Path, dst: Path) -> None:
    s, d = src / "results", dst / "results"
    if not s.exists():
        sys.exit(f"no results/ in {src} -- run the pipeline notebook first")
    if d.exists():
        shutil.rmtree(d)
    shutil.copytree(s, d, ignore=shutil.ignore_patterns(*TRAINING_ONLY))
    skipped = [f.name for f in s.iterdir() if f.name in TRAINING_ONLY]
    n = len(list(d.glob("*")))
    print(f"results : {n} files, {human(dir_size(d))}")
    if skipped:
        print(f"          skipped (training data, not served): {', '.join(skipped)}")


def copy_triptych(src: Path, dst: Path) -> None:
    s, d = src / "static" / "triptych", dst / "static" / "triptych"
    if not s.exists():
        print("triptych: none found (skipping)")
        return
    d.mkdir(parents=True, exist_ok=True)
    for f in s.glob("*.png"):
        shutil.copy2(f, d / f.name)
    print(f"triptych: {len(list(d.glob('*')))} images, {human(dir_size(d))}")


def convert_frames(src: Path, dst: Path, sessions: list[str],
                   stretches: list[str], quality: int) -> None:
    try:
        from PIL import Image
    except ImportError:
        sys.exit("Pillow is required to re-encode frames: pip install Pillow")

    s_root = src / "static" / "frames"
    if not s_root.exists():
        print("frames  : none rendered. Set RENDER_ALL=True in the notebook, or "
              "let the API render on demand from the .npy bundle.")
        return

    total_in = total_out = 0
    for sid in sessions:
        s_sess = s_root / sid
        if not s_sess.exists():
            continue
        for stretch in stretches:
            s_dir = s_sess / stretch
            if not s_dir.exists():
                continue
            d_dir = dst / "static" / "frames" / sid / stretch
            d_dir.mkdir(parents=True, exist_ok=True)
            for f in sorted(s_dir.glob("*.png")):
                total_in += f.stat().st_size
                out = d_dir / (f.stem + ".webp")
                Image.open(f).convert("L").save(out, "WEBP", quality=quality, method=4)
                total_out += out.stat().st_size
        # the hover-readout value grid is small and worth carrying along
        vals = s_sess / "values_4x.npy"
        if vals.exists():
            dv = dst / "static" / "frames" / sid
            dv.mkdir(parents=True, exist_ok=True)
            shutil.copy2(vals, dv / vals.name)
        print(f"  {sid:24s} -> webp")

    if total_in:
        print(f"frames  : {human(total_in)} PNG -> {human(total_out)} WebP "
              f"({100*total_out/total_in:.0f}% of original)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--dst", type=Path, default=DEFAULT_DST)
    ap.add_argument("--stretches", nargs="+", default=["zscale", "asinh"],
                    choices=["zscale", "asinh", "linear"])
    ap.add_argument("--quality", type=int, default=82)
    ap.add_argument("--all-sessions", action="store_true",
                    help="Include sessions that failed quality triage")
    ap.add_argument("--no-images", action="store_true")
    args = ap.parse_args()

    args.dst.mkdir(parents=True, exist_ok=True)
    copy_results(args.src, args.dst)

    if args.no_images:
        print(f"\ntotal   : {human(dir_size(args.dst))} (no images)")
        return

    copy_triptych(args.src, args.dst)

    import pandas as pd
    q = args.dst / "results" / "session_quality.csv"
    if q.exists() and not args.all_sessions:
        df = pd.read_csv(q)
        sessions = df[df.quality == "good"].session_id.tolist()
        print(f"frames  : {len(sessions)} good sessions "
              f"(--all-sessions for all {len(df)})")
    else:
        sessions = pd.read_csv(args.dst / "results" / "sessions.csv").session_id.tolist()

    convert_frames(args.src, args.dst, sessions, args.stretches, args.quality)
    print(f"\ntotal   : {human(dir_size(args.dst))} in {args.dst}")
    if dir_size(args.dst) > 400e6:
        print("WARNING: over 400 MB. Consider --stretches zscale to halve it.")


if __name__ == "__main__":
    main()
