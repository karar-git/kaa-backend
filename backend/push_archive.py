#!/usr/bin/env python
"""
push_archive.py -- load the raw FITS archive into a running API's volume.

`railway up` cannot carry the archive (the CLI upload is capped well below the
482 MB it would need), so the container keeps its archive on a Railway volume
and this script fills it through `POST /api/field/ingest`, one session per
request. Frames are sent Rice-compressed (`.fits.fz`, from `bundle_raw.py`)
when that bundle exists, else raw from `database/`.

The ingest route only answers while the service has `INGEST_OPEN=true`:

    railway variables --set INGEST_OPEN=true --service exotransit-lab-api
    python push_archive.py https://exotransit-lab-api-production.up.railway.app
    railway variables --set INGEST_OPEN=false --service exotransit-lab-api

Usage:
    python push_archive.py <base_url> [--only TRES-3] [--replace] [--src PATH]
"""

from __future__ import annotations

import argparse
import io
import sys
import time
import zipfile
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent


def is_fits(p: Path) -> bool:
    n = p.name.lower()
    return any(n.endswith(e + z) for e in (".fits", ".fit", ".fts") for z in ("", ".fz", ".gz"))


def zip_dir(files: list[Path]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as z:
        for f in files:
            z.write(f, arcname=f.name)
    return buf.getvalue()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("base_url")
    ap.add_argument("--src", type=Path, default=None,
                    help="archive root; default data/raw if present, else ../database")
    ap.add_argument("--only", default=None, help="substring filter on night/target")
    ap.add_argument("--replace", action="store_true")
    ap.add_argument("--timeout", type=float, default=600)
    a = ap.parse_args()

    src = a.src or (HERE / "data" / "raw" if (HERE / "data" / "raw" / "observations").is_dir()
                    else HERE.parent / "database")
    base = a.base_url.rstrip("/")
    client = httpx.Client(timeout=a.timeout)

    r = client.get(f"{base}/api/field/sessions")
    r.raise_for_status()
    have = {s["session"] for s in r.json()["sessions"]}
    print(f"server has {len(have)} sessions; source {src}")

    t0 = time.time()
    sent = skipped = 0
    # Darks first, one request per night, so every session finds its darks.
    cal = src / "calibration"
    nights = sorted(p for p in cal.iterdir() if p.is_dir()) if cal.is_dir() else []
    for night in nights:
        if a.only and a.only not in night.name:
            continue
        files = [p for p in night.iterdir() if is_fits(p)]
        if not files:
            continue
        r = client.post(f"{base}/api/field/ingest", params={"replace": "true"},
                        files=[("darks", (f"darks_{night.name}.zip", zip_dir(files), "application/zip"))])
        print(f"  darks {night.name}: {r.status_code} {r.json() if r.status_code != 201 else len(files)} files")
        if r.status_code == 403:
            sys.exit("ingest is closed: set INGEST_OPEN=true on the service first")
        r.raise_for_status()

    for night in sorted(p for p in (src / "observations").iterdir() if p.is_dir()):
        for tgt in sorted(p for p in night.iterdir() if p.is_dir()):
            sid = f"{tgt.name}__{night.name}"
            if a.only and a.only not in f"{night.name}/{tgt.name}":
                continue
            if sid in have and not a.replace:
                skipped += 1
                print(f"  {sid}: already on server, skipped")
                continue
            files = sorted(p for p in tgt.rglob("*") if is_fits(p))
            blob = zip_dir(files)
            t1 = time.time()
            r = client.post(f"{base}/api/field/ingest",
                            params={"replace": str(a.replace).lower()},
                            files=[("frames", (f"{sid}.zip", blob, "application/zip"))])
            ok = r.status_code == 201
            sent += ok
            print(f"  {sid}: {r.status_code} {len(files)} files {len(blob)/1e6:.0f} MB "
                  f"{time.time()-t1:.0f}s" + ("" if ok else f" {r.text[:200]}"))
            if r.status_code == 403:
                sys.exit("ingest is closed: set INGEST_OPEN=true on the service first")

    r = client.get(f"{base}/api/field/sessions")
    print(f"done in {time.time()-t0:.0f}s: sent {sent}, skipped {skipped}; "
          f"server now has {r.json()['n']} sessions")


if __name__ == "__main__":
    main()
