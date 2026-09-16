"""Headline KPIs: the numbers a status report or a dashboard landing page needs.

Aggregates only. Each group points at the endpoint that holds the detail, so a
number on a slide can always be traced back to a table.
"""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter

from .. import data
from ..config import settings
from ..schemas import KPIs
from .ml import _classes, _metrics, _model_files, _reading

router = APIRouter(tags=["kpis"])


def _safe(fn, default=None):
    try:
        return fn()
    except data.DatasetMissing:
        return default


def _r(x, n=2):
    return None if x is None or not np.isfinite(x) else round(float(x), n)


@router.get("/kpis", response_model=KPIs, summary="Headline KPIs for managers and dashboards")
def kpis() -> dict:
    """Every headline number in one call, grouped, each with the endpoint behind it.

    Groups: dataset size, data quality, photometric precision, the transit search
    (with its chance expectation), per-planet distance and speed, physics
    validation, and AI status. A group whose results file is absent comes back
    empty rather than failing the whole response.
    """
    out: dict = {}

    a = _safe(data.audit, {})
    out["dataset"] = {
        "science_frames": a.get("science_frames"), "dark_frames": a.get("dark_frames"),
        "observing_sessions": a.get("sessions"), "targets": a.get("targets"),
        "first_night": str(a.get("date_first", ""))[:10] or None,
        "last_night": str(a.get("date_last", ""))[:10] or None,
        "detail": "/api/meta",
    }

    q = _safe(data.session_quality)
    u = _safe(data.target_usability)
    if q is not None:
        counts = q.quality.value_counts().to_dict()
        out["data_quality"] = {
            "sessions_good": int(counts.get("good", 0)),
            "sessions_marginal": int(counts.get("marginal", 0)),
            "sessions_unusable": int(counts.get("unusable", 0)),
            "usable_session_pct": _r(100 * counts.get("good", 0) / len(q), 1),
            "targets_without_a_good_night":
                sorted(u[u.good == 0].target.tolist()) if u is not None else None,
            "detail": "/api/quality/sessions",
        }
    else:
        out["data_quality"] = {}

    nf = _safe(data.noise_floor)
    pi = _safe(data.photometry_info)
    phot = {"detail": "/api/quality/noise-floor, /api/science/photometry"}
    if nf is not None:
        best = float(nf.best_rms_ppt.min())
        phot.update(best_precision_ppt=_r(best),
                    smallest_3sigma_depth_one_night_pct=_r(3 * best / 10, 3))
    if pi is not None:
        good = pi[pi.quality == "good"]
        phot.update(median_light_curve_rms_ppt_good_sessions=_r(good.rms_ppt.median()),
                    median_comparison_star_improvement=_r(
                        good.improvement_factor.median()))
    out["photometry"] = phot

    ts = _safe(data.target_search)
    dep = _safe(data.depths)
    srch = {"detail": "/api/science/search, /api/science/depths"}
    if ts is not None:
        hits = ts[(ts.field_p_value <= 0.01) & (ts.significance_sigma > ts.field_sigma_95)
                  & (ts.depth_pct > 0)]
        srch.update(star_nights_searched=int(len(ts)), hits_p_le_0_01=int(len(hits)),
                    hits_expected_by_chance=_r(0.01 * len(ts), 1),
                    excess_over_chance=bool(len(hits) > 0.01 * len(ts)))
    if dep is not None:
        srch.update(target_windows_measured=int(len(dep)),
                    target_windows_at_3_sigma=int((dep.significance_sigma >= 3).sum()),
                    target_windows_negative_depth=int((dep.depth_pct <= 0).sum()))
    srch["confirmed_planets"] = 0
    srch["reading"] = ("No detection is claimed as a confirmed planet. Compare hits "
                       "with hits_expected_by_chance before quoting either.")
    out["transit_search"] = srch

    ph = _safe(data.planet_physics)
    if ph is not None:
        per = ph[["target", "planet", "distance_ly", "a_au", "orbital_speed_kms",
                  "orbital_period_days", "teq_k"]].round(
            {"distance_ly": 0, "a_au": 4, "orbital_speed_kms": 1,
             "orbital_period_days": 3, "teq_k": 0})
        out["planets"] = {
            "count": int(len(ph)),
            "nearest": _pick(ph, "distance_ly", "min", "light-years"),
            "farthest": _pick(ph, "distance_ly", "max", "light-years"),
            "fastest": _pick(ph, "orbital_speed_kms", "max", "km/s"),
            "slowest": _pick(ph, "orbital_speed_kms", "min", "km/s"),
            "hottest": _pick(ph, "teq_k", "max", "K"),
            "closest_to_its_star": _pick(ph, "a_au", "min", "AU"),
            "per_planet": data.to_records(per.sort_values("distance_ly")),
            "basis": "Derived from NASA Exoplanet Archive inputs; distance is Gaia's. "
                     "Not measured from our frames.",
            "detail": "/api/planets",
        }
    else:
        out["planets"] = {}

    v = _safe(data.physics_validation)
    if v is not None:
        eq, ms = v[~v.uses_our_data], v[v.uses_our_data]
        out["physics_validation"] = {
            "equation_checks": int(len(eq)),
            "equation_checks_agree_pct": _r(100 * (eq.status == "agrees").mean(), 1),
            "equation_checks_disagree": int((eq.status == "disagrees").sum()),
            "median_abs_pct_diff_semi_major_axis": _r(
                eq[eq.quantity == "semi_major_axis"].pct_diff.abs().median()),
            "measurement_checks": int(len(ms)),
            "measurement_checks_at_3_sigma": int(
                ms.status.isin(["agrees", "close", "disagrees"]).sum()),
            "detail": "/api/planets/validation",
        }
    else:
        out["physics_validation"] = {}

    files = _model_files()
    m = _metrics() or {}
    best = m.get("best_model")
    rep = m.get("models", {}).get(best, {}) if best else {}
    out["ai"] = {
        "source_classifier_trained": files["weights"],
        "source_classifier_classes": _classes(),
        "source_classifier_model": best,
        "test_accuracy": _r(rep.get("accuracy"), 3) if rep else None,
        "test_macro_f1": _r(rep.get("macro_f1"), 3) if rep else None,
        "baseline_macro_f1": _r(m.get("baseline", {}).get("macro_f1"), 3) if m else None,
        "majority_class_accuracy": m.get("majority_class_accuracy"),
        "classifier_reading": _reading(rep.get("per_class", {})) if rep else None,
        "llm_narration_enabled": settings.ai_enabled,
        "llm_model": settings.OPENROUTER_MODEL if settings.ai_enabled else None,
        "detail": "/api/model/info, /api/model/metrics, /api/classify, /api/explain",
    }

    prov = _safe(data.provenance, {})
    out["generated_utc"] = prov.get("generated_utc")
    return out


def _pick(df, col, how, unit):
    row = df.loc[df[col].idxmin() if how == "min" else df[col].idxmax()]
    return {"planet": row.planet, "target": row.target,
            "value": _r(row[col], 4 if unit == "AU" else 1), "unit": unit}
