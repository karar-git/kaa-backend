"""Light curves, ephemerides, transit measurements and the target search."""

from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Path, Query

from .. import data
from ..config import settings
from ..schemas import LightCurve

router = APIRouter(tags=["science"])

SessionId = Path(..., examples=["TRES-5__2026-08-26"])


@router.get("/sessions/{session_id}/lightcurve", response_model=LightCurve,
            tags=["science"], summary="Differential light curve for a session")
def lightcurve(session_id: str = SessionId) -> dict:
    """The differential light curve of the nominal target.

    `norm_flux` is target flux divided by the summed comparison flux, normalised
    to the session median. Dividing by comparison stars is the step that matters:
    a cloud dims every star at once, so only a real transit survives the ratio.

    Compare `rms_ppt` with `rms_target_only_ppt` to see how much that helped.
    Where `improvement_factor` is not comfortably above 1 the comparison stars
    are not working on that night, and no dip from it should be believed.

    **Caveat worth reading.** The target is chosen by an a-priori rule (brightest
    persistent star near the centre of frame 0) because these headers carry no
    WCS solution. `target_saturated` and `target_r_from_centre` in
    `/api/science/photometry` quantify how confident that choice is.
    """
    lc = data.lightcurves()
    lc = lc[lc.session_id == session_id].sort_values("frame_index")
    if lc.empty:
        raise HTTPException(
            404,
            f"No light curve for '{session_id}'. Sessions that failed quality "
            f"triage have too few reference stars to measure. See "
            f"GET /api/sessions/{session_id}.")

    info_df = data.photometry_info()
    info_df = info_df[info_df.session_id == session_id]
    info = data.to_records(info_df)[0] if not info_df.empty else {}

    pred = None
    try:
        p = data.predictions()
        p = p[p.session_id == session_id]
        if not p.empty:
            pred = data.to_records(p)[0]
    except data.DatasetMissing:
        pass

    cols = ["frame_index", "t_utc", "bjd_tdb", "target_flux", "comp_flux_sum",
            "norm_flux", "target_only_norm", "airmass", "sky_level", "WEATHER"]
    return {
        "session_id": session_id,
        "target": lc.target.iloc[0],
        "quality": info.get("quality"),
        "n_points": len(lc),
        "rms_ppt": info.get("rms_ppt"),
        "rms_target_only_ppt": info.get("rms_target_only_ppt"),
        "improvement_factor": info.get("improvement_factor"),
        "target_saturated": info.get("target_saturated"),
        "prediction": pred,
        "points": data.to_records(lc[[c for c in cols if c in lc.columns]]),
    }


@router.get("/science/photometry", summary="Photometry quality for every session")
def photometry() -> list[dict]:
    """How well each session was measured, and how the target was chosen.

    `persistence_bar_used` records how far the pipeline had to relax its
    reference-star requirement to measure the night at all — 0.9 on a clear
    night, lower when clouds cost detections. A session measured at 0.45 is
    weaker evidence than one measured at 0.9, and the number says so.
    """
    return data.to_records(data.photometry_info())


@router.get("/science/ephemerides", summary="Published orbital ephemerides")
def ephemerides() -> dict:
    """Periods and reference mid-transit times used to predict transit windows.

    Sourced live from the **NASA Exoplanet Archive** where the pipeline had
    internet, otherwise from a literature fallback table. The `source` column
    says which, per row, and a fallback row is explicitly marked UNVERIFIED — a
    stale `T0` drifts by hours over a decade, so this distinction is not cosmetic.
    """
    df = data.ephemerides()
    return {
        "external_data_disclosure":
            "Ephemerides are external data, not derived from the MicroObservatory "
            "frames. Primary source: NASA Exoplanet Archive (TAP service). "
            "Target name mapping: HATP-10 in this archive is HAT-P-10, also "
            "catalogued as WASP-11 -- one system, two names.",
        "rows": data.to_records(df),
    }


@router.get("/science/predictions", summary="Predicted transit windows")
def predictions() -> list[dict]:
    """Where a transit should fall in each session, given the ephemeris.

    `coverage_frac` is how much of the predicted event our observing window
    actually spans. A session with no predicted transit is not a failure — it is
    a control, because any dip found there is a false positive by definition.
    """
    return data.to_records(data.predictions())


@router.get("/science/depths", summary="Measured depth in the predicted window")
def depths() -> dict:
    """In-transit mean versus out-of-transit mean, per session.

    Deliberately the simplest possible estimator: no model fitting, nothing that
    could manufacture a signal the data does not contain.

    A **negative** depth means the star got brighter during the predicted window.
    That is a null result, and reporting it is the point.
    """
    df = data.depths()
    return {
        "interpretation": {
            "depth_pct": "Positive means the star dimmed. Negative means it "
                         "brightened, which is a null result.",
            "significance_sigma": "Depth divided by its own uncertainty. Below "
                                  "about 3 this is noise.",
            "expected_depth_pct": "Published depth for the known planet. Absent "
                                  "where the archive has no value.",
        },
        "rows": data.to_records(df),
    }


@router.get("/science/phasefold/{target}", summary="Phase-folded light curve")
def phasefold(
    target: str = Path(..., examples=["TRES-5"]),
    binned: bool = Query(True, description="Return binned points (recommended)"),
) -> dict:
    """All of a target's nights folded onto the published orbital period.

    A real transit stacks at the same phase every time; noise scatters. TRES-5
    has six good nights and TRES-3 four, which makes this the strongest visual
    test available from this dataset.
    """
    df = data.phasefold_binned() if binned else data.phasefold()
    df = df[df.target.str.lower() == target.lower()]
    if df.empty:
        raise HTTPException(
            404, f"No folded data for '{target}'. See GET /api/targets.")
    return {"target": target, "binned": binned, "n_points": len(df),
            "points": data.to_records(df, limit=settings.MAX_ROWS)}


@router.get("/science/search", summary="Ephemeris-guided search over every star")
def search(
    target: str | None = Query(None, examples=["TRES-5"]),
    max_p_value: float = Query(1.0, ge=0.0, le=1.0,
                               description="Keep only rows at or below this field p-value"),
    limit: int = Query(200, le=settings.MAX_ROWS),
) -> dict:
    """Every measured star tested for a dip at the published transit phase.

    Because the header carries no WCS solution we cannot identify the host star
    from its coordinates, so instead of guessing we measure the whole field and
    let the ephemeris pick.

    **Read `field_p_value`, not `significance_sigma`.** The maximum of hundreds of
    noisy stars is always large, so a big sigma means nothing on its own. The null
    comes from the field: `field_p_value` is the fraction of stars *in the same
    image* that produced a dip at least as deep at the same phase. Those stars
    lived through the same cloud, airmass and focus, so they measure exactly what
    this night does to noise.

    The sliding-window control usually used for this does not work on these data.
    A session spans ~3.6 h and the transit occupies ~2 h, leaving no room for a
    non-overlapping control window — our first attempt produced zero valid trials
    on every star, so the field replaced it.

    **`field_p_value` floors at 1/`n_field_stars`.** With 250 stars the smallest
    obtainable value is 0.004, which means "nothing in this field beat it", not
    "one in a thousand".

    `n_expected_by_chance` is the look-elsewhere correction: across ~2,700
    star-nights, about 1 % clear p <= 0.01 with nothing there at all. Compare
    `n_hits` against it before reading anything into the list.

    The search runs **within a night**, not across nights: track identifiers are
    assigned per session, and stacking would need an astrometric cross-match that
    these headers do not support.
    """
    try:
        full = data.target_search()
    except data.DatasetMissing:
        raise HTTPException(503, "Target search has not been generated yet.")

    df = full
    if target:
        df = df[df.target.str.lower() == target.lower()]
        if df.empty:
            raise HTTPException(404, f"No search rows for target {target!r}.")

    hits = df[(df.field_p_value <= 0.01)
              & (df.significance_sigma > df.field_sigma_95)
              & (df.depth_pct > 0)]
    n_expected = 0.01 * len(df)

    if hits.empty:
        verdict = ("No star stands out against its own field at the published "
                   "phase. This is a null result, and it is the honest one.")
    elif len(hits) <= n_expected:
        verdict = (f"{len(hits)} stars clear p <= 0.01, but ~{n_expected:.0f} were "
                   "expected by chance across this many tests. The rows below are "
                   "a RANKING of the most transit-like stars, not a set of "
                   "detections, and none is a confirmed planet.")
    else:
        verdict = (f"{len(hits)} stars clear p <= 0.01 against ~{n_expected:.0f} "
                   "expected by chance. Candidates consistent with a transit. "
                   "NOT confirmed planets.")

    df = df[df.field_p_value <= max_p_value].sort_values(
        "significance_sigma", ascending=False)
    return {
        "n_star_nights_searched": int(len(full)),
        "n_hits": int(len(hits)),
        "n_expected_by_chance": round(float(n_expected), 1),
        "verdict": verdict,
        "rows": data.to_records(df, limit=limit),
    }


@router.get("/science/field-photometry/{session_id}",
            summary="Light curves for every star in the field")
def field_photometry(
    session_id: str = SessionId,
    track_id: int | None = Query(None, description="Restrict to one star"),
    limit: int = Query(5000, le=settings.MAX_ROWS),
) -> dict:
    """Per-star differential photometry for a whole field.

    Powers the dashboard's field browser: click any star, see its curve. It is
    also the evidence for false-positive case 2 — if the target dips and so do
    two hundred other stars, that was a cloud.
    """
    try:
        df = data.field_photometry()
    except data.DatasetMissing:
        raise HTTPException(503, "Field photometry has not been generated yet.")
    df = df[df.session_id == session_id]
    if df.empty:
        raise HTTPException(404, f"No field photometry for '{session_id}'.")
    if track_id is not None:
        df = df[df.track_id == track_id]
    return {"session_id": session_id, "n_rows": len(df),
            "points": data.to_records(df, limit=limit)}


@router.get("/science/field-stars/{session_id}", summary="Measured stars in a field")
def field_stars(session_id: str = SessionId) -> list[dict]:
    """Position, brightness and achieved scatter for every star measured.

    `in_ensemble` marks the stars used as the comparison ensemble.
    """
    try:
        df = data.field_stars()
    except data.DatasetMissing:
        raise HTTPException(503, "Field star table has not been generated yet.")
    df = df[df.session_id == session_id].sort_values("median_flux", ascending=False)
    if df.empty:
        raise HTTPException(404, f"No field stars for '{session_id}'.")
    return data.to_records(df)


@router.get("/quality/correlations", tags=["quality"],
            summary="What actually governs image quality")
def correlations() -> dict:
    """Pearson correlations across all 1,681 frames.

    Two results are worth pointing at:

    * `WEATHER` correlates **+0.75** with peak contrast. The keyword is a
      transparency score where 100 means clear, not a warning flag — and the
      pixels prove it rather than the documentation.
    * **Airmass barely matters** (about -0.13). That is genuinely surprising:
      everyone expects airmass to dominate. Over the 1.0-4.1 range sampled here,
      sky transparency swamps it.
    """
    m = data.quality_correlations()
    return {
        "matrix": {r: {c: (None if pd.isna(v) else float(v))
                       for c, v in m.loc[r].items()} for r in m.index},
        "highlights": [
            {"pair": ["WEATHER", "peak_contrast"], "r": float(m.loc["WEATHER", "peak_contrast"]),
             "note": "Transparency dominates image quality."},
            {"pair": ["airmass", "peak_contrast"], "r": float(m.loc["airmass", "peak_contrast"]),
             "note": "Surprisingly weak. Airmass is not the limiting factor here."},
            {"pair": ["sky_level", "n_source_px"], "r": float(m.loc["sky_level", "n_source_px"]),
             "note": "A bright sky washes out faint stars."},
        ],
    }


@router.get("/quality/noise-floor", tags=["quality"],
            summary="Photometric precision versus brightness")
def noise_floor() -> dict:
    """The instrument's real performance curve, measured from the data.

    This sets a hard limit on what can be claimed: a 2% transit is 20 ppt, so any
    star whose scatter exceeds ~20 ppt cannot yield a single-night detection no
    matter how the analysis is done.
    """
    df = data.noise_floor()
    best = float(df.best_rms_ppt.min())
    return {
        "best_precision_ppt": best,
        "detectable_depth_pct_1night": round(3 * best / 10, 3),
        "note": "A 3-sigma single-night detection needs a depth about three "
                "times the scatter.",
        "curve": data.to_records(df),
    }


@router.get("/quality/sessions", tags=["quality"], summary="Session quality triage")
def session_quality() -> dict:
    """Which nights are usable, and why.

    Thresholds are stated rather than tuned: `good` needs median peak contrast
    >= 300 and >= 800 source pixels; `marginal` needs contrast >= 100; anything
    below is `unusable`. Being explicit about a crude cut beats hiding a subtle
    one.
    """
    df = data.session_quality()
    return {
        "thresholds": {"good_contrast": 300.0, "good_sources": 800.0,
                       "marginal_contrast": 100.0},
        "counts": df.quality.value_counts().to_dict(),
        "rows": data.to_records(df),
    }
