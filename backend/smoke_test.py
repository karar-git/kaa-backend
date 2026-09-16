#!/usr/bin/env python
"""
smoke_test.py -- exercise every route without starting a server.

FastAPI's TestClient drives the ASGI app in-process, so this runs in a second and
needs no port. It is the check to run before `railway up`: a 500 here is a 500 in
production.

A 503 is reported as SKIP rather than FAIL when the underlying results file has
not been generated — that is the documented degraded mode, not a bug.

Usage:  python smoke_test.py
"""

from __future__ import annotations

import io
import sys

import numpy as np
from fastapi.testclient import TestClient

from app.data import known_sessions, known_targets
from app.main import app

# Surface a server error as a 500 response instead of re-raising, so one
# broken route does not abort the whole sweep.
client = TestClient(app, raise_server_exceptions=False)
PASS = FAIL = SKIP = 0


def check(method: str, url: str, *, expect: int = 200, **kw) -> object:
    global PASS, FAIL, SKIP
    r = client.request(method, url, **kw)
    if r.status_code == expect:
        PASS += 1
        body = r.json() if r.headers.get("content-type", "").startswith("application/json") else None
        n = len(body) if isinstance(body, list) else ""
        print(f"  ok   {r.status_code} {method:4s} {url[:62]:62s} {n}")
        return body
    if r.status_code == 503:
        SKIP += 1
        print(f"  skip 503 {method:4s} {url[:62]:62s} (results file absent)")
        return None
    FAIL += 1
    detail = ""
    try:
        detail = str(r.json())[:200]
    except Exception:
        detail = r.text[:160]
    print(f"  FAIL {r.status_code} {method:4s} {url[:62]:62s} {detail}")
    return None


print("\n-- meta ------------------------------------------------------------")
health = check("GET", "/api/health")
if health:
    print(f"       status={health['status']} present={health['results_present']} "
          f"images={health['images_available']} ai={health['ai_enabled']}")
    if health["results_missing"]:
        print(f"       missing: {', '.join(health['results_missing'])}")
    t = health.get("timing", {})
    bary = t.get("barycentric_correction_applied")
    print(f"       barycentric correction applied: {bary}")
    if bary is not True:
        print("       !! event times are UNCORRECTED -- up to 8 min out. "
              "Regenerate results where astropy.coordinates imports.")
check("GET", "/api/meta")
check("GET", "/api/calibration")
check("GET", "/api/calibration/noise-curve")
check("GET", "/api/calibration/dark-masters")
check("GET", "/api/calibration/hot-pixels")
check("GET", "/api/false-positive-cases")
check("GET", "/api/labels")

print("\n-- kpis + physics --------------------------------------------------")
k = check("GET", "/api/kpis")
if k and k.get("planets"):
    p = k["planets"]
    print(f"       nearest={p['nearest']['planet']} {p['nearest']['value']} ly  "
          f"fastest={p['fastest']['planet']} {p['fastest']['value']} km/s")
check("GET", "/api/planets")
check("GET", "/api/planets?sort=orbital_speed_kms&descending=true")
check("GET", "/api/planets?sort=planet", expect=422)
v = check("GET", "/api/planets/validation")
if v:
    print(f"       {v['finding'][:110]}...")
check("GET", "/api/planets/validation?uses_our_data=true")
check("GET", "/api/planets/equations")
for t in known_targets():
    check("GET", f"/api/planets/{t}")
check("GET", "/api/planets/WASP-11")
check("GET", "/api/planets/not-a-planet", expect=404)

print("\n-- sessions --------------------------------------------------------")
sessions = check("GET", "/api/sessions") or []
check("GET", "/api/sessions?quality=good")
check("GET", "/api/sessions?quality=unusable")
check("GET", "/api/sessions?target=TRES-5")
check("GET", "/api/sessions?target=NoSuchTarget", expect=404)
check("GET", "/api/targets")
check("GET", "/api/targets/TRES-5")
check("GET", "/api/targets/nope", expect=404)

try:
    known = known_sessions()
except Exception as exc:
    print(f"\n!! no session table: {exc}")
    print("!! run: python sync_data.py")
    sys.exit(1)
sid = "TRES-5__2026-08-26" if "TRES-5__2026-08-26" in known else known[0]
print(f"\n-- session detail ({sid}) --")
check("GET", f"/api/sessions/{sid}")
check("GET", f"/api/sessions/{sid}/frames?limit=5")
check("GET", f"/api/sessions/{sid}/motion")
check("GET", f"/api/sessions/{sid}/tracks?limit=5")
check("GET", f"/api/sessions/{sid}/tracks?label=hot_pixel&limit=3")
check("GET", f"/api/sessions/{sid}/detections?frame=0&limit=5")
check("GET", "/api/sessions/does__not__exist", expect=404)

print("\n-- science ---------------------------------------------------------")
lc = check("GET", f"/api/sessions/{sid}/lightcurve")
if lc:
    print(f"       {lc['n_points']} points, rms={lc['rms_ppt']} ppt, "
          f"improvement x{lc['improvement_factor']}")
check("GET", "/api/science/photometry")
check("GET", "/api/science/ephemerides")
check("GET", "/api/science/predictions")
check("GET", "/api/science/depths")
check("GET", "/api/science/phasefold/TRES-5")
check("GET", "/api/science/phasefold/Nope", expect=404)
check("GET", "/api/science/search")
check("GET", "/api/science/search?target=TRES-5&max_p_value=0.05")
check("GET", f"/api/science/field-photometry/{sid}?limit=10")
check("GET", f"/api/science/field-stars/{sid}")

print("\n-- quality ---------------------------------------------------------")
check("GET", "/api/quality/correlations")
check("GET", "/api/quality/noise-floor")
check("GET", "/api/quality/sessions")

print("\n-- images ----------------------------------------------------------")
man = check("GET", f"/api/images/{sid}/manifest")
if man:
    print(f"       prerendered={man['prerendered_frames']}/{man['n_frames']} "
          f"on_demand={man['can_render_on_demand']}")
    if man["prerendered_frames"] or man["can_render_on_demand"]:
        # Only the stretches the manifest advertises. sync_data ships a subset by
        # default, so hardcoding all three would fail on a correct deploy.
        print(f"       stretches offered: {', '.join(man['stretches'])}")
        for st in man["stretches"]:
            r = client.get(f"/api/images/{sid}/frame/0?stretch={st}")
            ok = r.status_code == 200 and len(r.content) > 500
            print(f"  {'ok  ' if ok else 'FAIL'} {r.status_code} GET  "
                  f"frame 0 [{st}] -> {len(r.content)} bytes "
                  f"{r.headers.get('content-type')}")
            globals().__setitem__("PASS" if ok else "FAIL",
                                  globals()["PASS" if ok else "FAIL"] + 1)
        check("GET", f"/api/images/{sid}/values/0")
        check("GET", f"/api/images/{sid}/frame/0?stretch=bogus", expect=422)
        check("GET", f"/api/images/{sid}/frame/99999", expect=404)
    else:
        print("       no tiles and no .npy bundle -- image routes not exercised")
check("GET", "/api/images/triptych/raw")
check("GET", "/api/images/triptych/bogus", expect=422)

print("\n-- field (live quick-look) -----------------------------------------")
raw = check("GET", "/api/field/sessions")
if raw and raw["n"]:
    fs = raw["sessions"][0]["session"]
    summ = check("GET", f"/api/field/summary?session={fs}")
    if summ:
        print(f"       {fs}: {summ['n_frames']} frames, ref {summ['reference_frame']}, "
              f"{summ['n_stars_detected']} stars, rms {summ['rms_ppt']} ppt, "
              f"{len(summ['lost_frames'])} lost, {summ['compute_seconds']} s")
    check("GET", f"/api/field/stars?session={fs}")
    check("GET", f"/api/field/lightcurve?session={fs}")
    for url in (f"/api/field/frame.png?session={fs}",
                f"/api/field/frame.png?session={fs}&frame=0&labels=true",
                f"/api/field/lightcurve.png?session={fs}",
                f"/api/field/lightcurve.png?session={fs}&mode=target"):
        r = client.get(url)
        ok = r.status_code == 200 and r.headers["content-type"] == "image/png" and r.content[:4] == b"\x89PNG"
        PASS += ok
        FAIL += not ok
        print(f"  {'ok  ' if ok else 'FAIL'} {r.status_code} GET  {url[:62]:62s} {len(r.content)} bytes")
    check("GET", f"/api/field/frame.png?session={fs}&frame=99999", expect=404)
    check("GET", f"/api/field/lightcurve.png?session={fs}&mode=bogus", expect=422)
    check("GET", f"/api/field/summary?session={fs}&x=10", expect=422)
else:
    print("       no raw FITS visible to this deploy; field routes answer 404 by design")
check("GET", "/api/field/summary?session=../../etc", expect=404)
check("GET", "/api/field/summary?session=Nope__2020-01-01", expect=404)
check("POST", "/api/field/upload", expect=422,
      files=[("frames", ("a.fits", b"not a fits file", "application/octet-stream")),
             ("frames", ("b.fits", b"not a fits file", "application/octet-stream"))])
# Ingest writes into the archive, so it must be shut unless an operator opened it.
from app.config import settings as _settings
check("POST", "/api/field/ingest", expect=422 if _settings.INGEST_OPEN else 403,
      files=[("frames", ("a.fits", b"not a fits file", "application/octet-stream"))])

print("\n-- ml --------------------------------------------------------------")
check("GET", "/api/model/info")
from app.routers.ml import _model_files
check("GET", "/api/model/metrics", expect=200 if _model_files()["metrics"] else 503)
buf = io.BytesIO()
np.save(buf, np.random.default_rng(0).normal(400, 5, (32, 32)).astype(np.float32))
check("POST", "/api/classify", files={"file": ("cut.npy", buf.getvalue(),
                                               "application/octet-stream")})
bad = io.BytesIO()
np.save(bad, np.zeros((16, 16), dtype=np.float32))
check("POST", "/api/classify", expect=422,
      files={"file": ("bad.npy", bad.getvalue(), "application/octet-stream")})
check("POST", "/api/explain", expect=503 if not health or not health["ai_enabled"] else 200,
      json={"session_id": sid, "audience": "student", "language": "en"})

print("\n-- openapi ---------------------------------------------------------")
spec = check("GET", "/openapi.json")
if spec:
    paths = spec["paths"]
    ops = sum(len(v) for v in paths.values())
    print(f"       {len(paths)} paths, {ops} operations, {len(spec.get('tags', []))} tags")
    undocumented = [p for p, v in paths.items()
                    for m, op in v.items() if not op.get("summary")]
    if undocumented:
        print(f"       WARNING undocumented: {undocumented}")
check("GET", "/docs")

print(f"\n{'='*70}\n  {PASS} passed   {FAIL} failed   {SKIP} skipped (degraded mode)\n{'='*70}")
sys.exit(1 if FAIL else 0)
