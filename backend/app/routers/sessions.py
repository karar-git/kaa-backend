"""Per-session endpoints: frames, tracks, detections, field motion."""

from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Path, Query

from .. import data
from ..config import settings
from ..schemas import SessionSummary, SourceLabel, TargetSummary

router = APIRouter(tags=["sessions"])

SessionId = Path(..., description="Session identifier, e.g. TRES-5__2026-08-26",
                 examples=["TRES-5__2026-08-26"])


def _require_session(session_id: str) -> None:
    if session_id not in data.known_sessions():
        raise HTTPException(
            404, f"Unknown session '{session_id}'. See GET /api/sessions.")


def _sessions_joined() -> pd.DataFrame:
    """sessions.csv enriched with quality tier and total field drift."""
    df = data.sessions().copy()
    try:
        q = data.session_quality()[
            ["session_id", "quality", "median_contrast", "median_sources"]]
        df = df.merge(q, on="session_id", how="left")
    except data.DatasetMissing:
        pass
    try:
        m = data.field_motion_summary()[["session_id", "field_motion_px"]]
        df = df.merge(m, on="session_id", how="left")
    except data.DatasetMissing:
        pass
    return df


@router.get("/sessions", response_model=list[SessionSummary],
            summary="All 22 observing sessions")
def list_sessions(
    target: str | None = Query(None, description="Filter to one target"),
    quality: str | None = Query(
        None, description="Filter by tier: good, marginal or unusable"),
) -> list[dict]:
    """One row per continuous observing run (target + night).

    Each session is 51-93 frames over 2.5-5.2 hours at ~180 s cadence. The
    `quality` field is the triage verdict: 15 sessions are good, 1 marginal and
    6 unusable, and the unusable ones are kept rather than deleted because they
    serve as the control sample.
    """
    df = _sessions_joined()
    if target:
        df = df[df.target.str.lower() == target.lower()]
    if quality:
        df = df[df.get("quality", pd.Series(dtype=str)) == quality.lower()]
    if df.empty:
        raise HTTPException(404, "No sessions match that filter.")
    return data.to_records(df)


@router.get("/sessions/{session_id}", response_model=SessionSummary,
            summary="One session")
def get_session(session_id: str = SessionId) -> dict:
    _require_session(session_id)
    df = _sessions_joined()
    return data.to_records(df[df.session_id == session_id])[0]


@router.get("/sessions/{session_id}/frames", summary="Per-frame telemetry and quality")
def session_frames(
    session_id: str = SessionId,
    limit: int = Query(2000, le=settings.MAX_ROWS),
    offset: int = 0,
) -> list[dict]:
    """Every frame with its telescope telemetry and measured image statistics.

    This is what the dashboard plots underneath a light curve so a dip can be
    read against airmass, sky level and transparency at the same instant.
    """
    _require_session(session_id)
    df = data.frame_quality()
    df = df[df.session_id == session_id].sort_values("frame_index")
    return data.to_records(df, limit=limit, offset=offset)


@router.get("/sessions/{session_id}/motion", summary="Field drift per frame")
def session_motion(session_id: str = SessionId) -> dict:
    """Frame-to-frame offset of the star field, and its running total.

    The mount is alt-azimuth with no derotator, so a session drifts up to ~105 px
    *and* rotates about a degree. That is why the pipeline chains detections
    between adjacent frames rather than registering everything onto one grid.

    A sudden jump here is false-positive case 3, a tracking slip.
    """
    _require_session(session_id)
    df = data.field_motion()
    df = df[df.session_id == session_id].sort_values("frame_index")
    if df.empty:
        raise HTTPException(404, "No motion data for that session.")
    return {
        "session_id": session_id,
        "total_drift_px": float(np.hypot(df.cum_dx.iloc[-1], df.cum_dy.iloc[-1])),
        "max_single_step_px": float(np.hypot(df.offset_dx, df.offset_dy).max()),
        "points": data.to_records(
            df[["frame_index", "offset_dx", "offset_dy", "cum_dx", "cum_dy"]]),
    }


@router.get("/sessions/{session_id}/tracks", summary="Linked sources for a session")
def session_tracks(
    session_id: str = SessionId,
    label: SourceLabel | None = Query(None, description="Filter by class"),
    min_persistence: float = Query(0.0, ge=0.0, le=1.0),
    limit: int = Query(2000, le=settings.MAX_ROWS),
    offset: int = 0,
) -> list[dict]:
    """One row per physical object followed across the night.

    `persistence` is the fraction of frames the object was detected in, and
    `motion_px` is how far it travelled. Together they separate stars (persistent,
    moving with the field) from hot pixels (persistent, stationary) without
    reference to the calibration frames.
    """
    _require_session(session_id)
    df = data.tracks()
    df = df[df.session_id == session_id]
    if label:
        df = df[df.label == label]
    df = df[df.persistence >= min_persistence].sort_values("median_flux",
                                                           ascending=False)
    return data.to_records(df, limit=limit, offset=offset)


@router.get("/sessions/{session_id}/detections", summary="Raw detections")
def session_detections(
    session_id: str = SessionId,
    frame: int | None = Query(None, description="Restrict to one frame index"),
    label: SourceLabel | None = None,
    limit: int = Query(2000, le=settings.MAX_ROWS),
    offset: int = 0,
) -> list[dict]:
    """Individual blob detections, one row per source per frame.

    Pass `frame` to get the overlay for a single image — that is the intended
    use, since a whole session is tens of thousands of rows.
    """
    _require_session(session_id)
    df = data.detections()
    df = df[df.session_id == session_id]
    if frame is not None:
        df = df[df.frame_index == frame]
    if label:
        df = df[df.label == label]
    return data.to_records(df, limit=limit, offset=offset)


@router.get("/targets", response_model=list[TargetSummary], summary="The 8 targets")
def list_targets() -> list[dict]:
    """Per-target usability.

    Three of the eight targets have no good night at all. That is a headline
    result of the triage, not a missing feature: TRES-1, WASP-10 and WASP-2 each
    got a single washed-out run, so nothing can honestly be measured for them.
    """
    df = data.target_usability().copy()
    try:
        pt = data.per_target()[["target", "n_frames", "ra_deg", "dec_deg"]]
        df = df.merge(pt, on="target", how="left")
    except data.DatasetMissing:
        pass
    df["usable"] = df.good > 0
    return data.to_records(df)


@router.get("/targets/{target}", response_model=TargetSummary, summary="One target")
def get_target(target: str = Path(..., examples=["TRES-5"])) -> dict:
    rows = [r for r in list_targets() if r["target"].lower() == target.lower()]
    if not rows:
        raise HTTPException(404, f"Unknown target '{target}'. See GET /api/targets.")
    return rows[0]
