"""Dataset-level metadata, calibration, and the false-positive rulebook."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from .. import data, imaging
from ..config import settings
from ..schemas import (CalibrationSummary, DatasetMeta, FalsePositiveCase,
                       Health, Instrument)

router = APIRouter(tags=["meta"])

INSTRUMENT = Instrument(
    telescope="Cecilia",
    aperture_mm=150,
    focal_length_mm=560,
    observatory="Fred Lawrence Whipple Observatory, Amado AZ",
    latitude_deg=31.68,
    longitude_deg=-110.88,
    height_m=1268.0,
    detector_width_px=650,
    detector_height_px=500,
    arcsec_per_px=5.0,
    field_of_view_arcmin=[54.2, 41.7],
    bit_depth=12,
    exposure_s=60.0,
    filters=["Clear", "Opaque"],
)

# The eleven ways a dip can appear in a light curve. Ordered cheapest test
# first, which is also the order the pipeline applies them.
FALSE_POSITIVE_CASES: list[FalsePositiveCase] = [
    FalsePositiveCase(
        id=1, name="Planet transit",
        description="A planet crosses the disc of its host star.",
        signature="Flat-bottomed U shape, 1-3 h long, recurring on a fixed period "
                  "with the same depth every time.",
        test="Falls at the predicted time AND recurs across nights.",
        testable_with_our_data=True, endpoint="/api/science/depths"),
    FalsePositiveCase(
        id=2, name="Cloud or haze",
        description="Sky transparency drops across the whole field.",
        signature="Every star dims together; the shape is irregular.",
        test="Check whether the comparison stars dip at the same time.",
        testable_with_our_data=True,
        endpoint="/api/sessions/{session_id}/lightcurve"),
    FalsePositiveCase(
        id=3, name="Tracking slip",
        description="The star drifts partly out of the measuring aperture.",
        signature="Sudden and sharp; the star's measured position jumps.",
        test="Compare against the per-frame field offset.",
        testable_with_our_data=True, endpoint="/api/sessions/{session_id}/motion"),
    FalsePositiveCase(
        id=4, name="Cosmic ray",
        description="A charged particle strikes the sensor.",
        signature="One frame only, a sharp 1-2 px spike.",
        test="Single-frame outlier; no track persistence.",
        testable_with_our_data=True, endpoint="/api/sessions/{session_id}/tracks"),
    FalsePositiveCase(
        id=5, name="Hot pixel in the aperture",
        description="A permanently bright detector defect drifts into the aperture.",
        signature="Step change that coincides with the field drift.",
        test="Cross-check against the dark-frame hot pixel mask.",
        testable_with_our_data=True, endpoint="/api/calibration/hot-pixels"),
    FalsePositiveCase(
        id=6, name="Airmass / extinction",
        description="The star sinks towards the horizon and reddens.",
        signature="A slow smooth trend rather than a dip.",
        test="Correlate with TELALT; detrend.",
        testable_with_our_data=True, endpoint="/api/quality/correlations"),
    FalsePositiveCase(
        id=7, name="Starspots",
        description="Dark patches on the star's own surface rotate in and out of view.",
        signature="A slow wave over hours to days, changing shape between epochs.",
        test="Needs multi-epoch, multi-colour photometry.",
        testable_with_our_data=False, endpoint=None),
    FalsePositiveCase(
        id=8, name="Eclipsing binary",
        description="Two stars orbit each other and one eclipses the other.",
        signature="V-shaped rather than flat-bottomed, and often very deep.",
        test="Shape test only -- partial discrimination.",
        testable_with_our_data=True, endpoint="/api/science/depths"),
    FalsePositiveCase(
        id=9, name="Blended eclipsing binary",
        description="A faint eclipsing pair sits inside the same aperture as the target.",
        signature="A diluted dip; the measured centroid shifts during the event.",
        test="Cannot be resolved at 5 arcsec/pixel.",
        testable_with_our_data=False, endpoint=None),
    FalsePositiveCase(
        id=10, name="Odd/even depth mismatch",
        description="An eclipsing binary mistaken for a planet at half its true period.",
        signature="Alternating depths between consecutive events.",
        test="Compare depths across repeat nights.",
        testable_with_our_data=True, endpoint="/api/science/phasefold/{target}"),
    FalsePositiveCase(
        id=11, name="Pure noise",
        description="Random scatter that happens to look like a dip.",
        signature="Wrong duration, no recurrence, no fixed phase.",
        test="Compare the dip against the other 100-450 stars in the same image "
             "at the same phase, then against the number of hits chance predicts "
             "over ~2,700 star-nights. The sliding-window control normally used "
             "here is impossible: a 3.6 h session with a 2 h transit leaves no "
             "room for a non-overlapping window.",
        testable_with_our_data=True, endpoint="/api/science/search"),
]


@router.get("/health", response_model=Health, summary="Liveness and data inventory")
def health() -> Health:
    """Reports which result files this deployment actually has.

    Useful in its own right: a deploy missing `lightcurves.csv` will answer
    metadata requests perfectly well and then 404 on the science, so the
    inventory is worth surfacing rather than discovering endpoint by endpoint.
    """
    inv = data.available()
    missing = [k for k, v in inv.items() if not v]
    return Health(
        status="ok" if not missing else "degraded",
        version=settings.VERSION,
        team=settings.TEAM,
        challenge=settings.CHALLENGE,
        data_dir=str(settings.DATA_DIR),
        results_present=sum(inv.values()),
        results_missing=missing,
        images_available=imaging.images_available(),
        ai_enabled=settings.ai_enabled,
        timing=_timing(),
    )


def _timing() -> dict:
    """Whether the times in these results are really barycentric.

    Surfaced on /health rather than buried, because every phase, prediction and
    depth downstream is computed from `bjd_tdb` and an uncorrected run mislabels
    all of them by up to eight minutes.
    """
    try:
        p = data.provenance()
    except data.DatasetMissing:
        return {"barycentric_correction_applied": None,
                "note": "provenance.json absent -- results predate this record. "
                        "Treat the timing as unverified."}
    return {
        "barycentric_correction_applied": p["barycentric_correction_applied"],
        "time_scale": p["time_scale"],
        "max_timing_error_s": p["max_timing_error_s"],
        "generated_utc": p["generated_utc"],
    }


@router.get("/meta", response_model=DatasetMeta, summary="Dataset and instrument")
def meta() -> DatasetMeta:
    a = dict(data.audit())
    a["instrument"] = INSTRUMENT
    return DatasetMeta(**a)


@router.get("/calibration", response_model=CalibrationSummary,
            summary="Master dark and hot-pixel summary")
def calibration() -> CalibrationSummary:
    return CalibrationSummary(**data.calibration())


@router.get("/calibration/noise-curve", summary="Read noise vs number of darks")
def noise_curve() -> list[dict]:
    """Measured random noise left in a master dark built from N frames.

    Compare `random_noise_counts` against `theory_1_over_sqrt_n`: they should
    track. This is the evidence that stacking darks works, and that subtracting a
    single dark actively injects noise instead of removing it.
    """
    return data.to_records(data.noise_curve())


@router.get("/calibration/dark-masters", summary="Per-temperature dark masters")
def dark_masters() -> list[dict]:
    """Dark current roughly doubles every ~6 C, so one master dark does not fit
    every night. The pipeline builds one per camera temperature and picks the
    nearest at calibration time."""
    return data.to_records(data.dark_masters())


@router.get("/calibration/hot-pixels", summary="Hot pixel statistics")
def hot_pixels() -> dict:
    c = data.calibration()
    out = {
        "count": c["hot_pixels"],
        "fraction_pct": c["hot_pixel_fraction_pct"],
        "definition": "Above 10 sigma in the median-combined master dark, where "
                      "sigma is MAD-derived from the master itself.",
        "derived_from": "60 Opaque (shutter-closed) frames spanning 30 nights",
        "independent_check": "A hot pixel is a silicon defect, so it stays put "
                             "while the star field drifts ~100 px. Tracks that "
                             "persist but do not move are hot pixels found "
                             "without using the darks at all.",
        "map_endpoint": "/api/images/triptych/hot_pixel_map",
    }
    try:
        tr = data.tracks()
        both = int((tr.hot_by_motion & tr.any_hot).sum())
        motion_only = int((tr.hot_by_motion & ~tr.any_hot).sum())
        out["cross_check"] = {
            "confirmed_by_both_methods": both,
            "stationary_track_only": motion_only,
            "agreement_pct": round(100 * both / max(both + motion_only, 1), 1),
        }
    except data.DatasetMissing:
        pass
    return out


@router.get("/false-positive-cases", response_model=list[FalsePositiveCase],
            summary="Every way a dip can fool you")
def false_positive_cases() -> list[FalsePositiveCase]:
    """The eleven cases, each with the test that rules it out and the endpoint
    holding the evidence.

    Cases 7 and 9 are marked `testable_with_our_data: false` on purpose. At
    5 arcsec/pixel with a single Clear filter we cannot separate a blended
    eclipsing binary or model starspot activity, and saying so is part of the
    result rather than a gap in it.
    """
    return FALSE_POSITIVE_CASES


@router.get("/labels", summary="Source class counts and how each label is derived")
def labels() -> dict:
    try:
        counts = data.to_records(data.label_counts())
    except data.DatasetMissing:
        counts = []
    return {
        "counts": counts,
        "rules": {
            "star": "Detected in >=60% of a session's frames and moving with the field.",
            "hot_pixel": "Flagged in the master dark built from 60 shutter-closed frames.",
            "cosmic_ray": "Present in exactly one frame, <=8 px, compact, not on a hot pixel.",
            "satellite_trail": "Present in exactly one frame with elongation > 3.",
            "faint_source": "Detected in 10-60% of frames -- real but near the limit.",
            "noise": "Random position with no detection within 8 px. Negative class.",
        },
        "note": "No label was drawn by hand. Every one follows from a stated "
                "physical rule, which is what makes the training set auditable.",
    }
