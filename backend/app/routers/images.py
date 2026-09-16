"""Image tiles for the frame viewer."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Path, Query
from fastapi.responses import Response

from .. import data, imaging
from ..config import settings
from ..schemas import Stretch

router = APIRouter(tags=["images"], prefix="/images")

SessionId = Path(..., examples=["TRES-5__2026-08-26"])


def _image_response(blob: bytes, media: str) -> Response:
    # Tiles are content, not state: a given (session, frame, stretch) never
    # changes within a deploy, so it can be cached indefinitely.
    return Response(
        content=blob, media_type=media,
        headers={"Cache-Control": f"public, max-age={settings.IMAGE_CACHE_SECONDS}, immutable"},
    )


@router.get("/{session_id}/frame/{frame}", summary="One rendered frame",
            response_class=Response,
            responses={200: {"content": {"image/webp": {}},
                             "description": "Greyscale frame, stretched"}})
def frame_image(
    session_id: str = SessionId,
    frame: int = Path(..., ge=0, description="Zero-based frame index"),
    stretch: Stretch = Query("zscale", description="Display stretch"),
) -> Response:
    """A single 650x500 frame, contrast-stretched for display.

    Three stretches, because no single one shows everything:

    * `zscale` — the IRAF algorithm astronomers actually use. Best default.
    * `asinh` — compresses the bright end, so faint stars and bright cores are
      visible together.
    * `linear` — honest 1-99.5 percentile clip. Hides faint stars.

    The returned image is 8-bit and stretched, so its grey levels are **not**
    data. For real ADU counts use `/values/{frame}`.
    """
    if session_id not in data.known_sessions():
        raise HTTPException(404, f"Unknown session '{session_id}'.")
    try:
        out = imaging.get_frame_image(session_id, frame, stretch)
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    if out is None:
        raise HTTPException(
            404,
            f"Frame {frame} of '{session_id}' is not available. Either the index "
            f"is out of range, or this deploy ships no tiles for that session.")
    return _image_response(*out)


@router.get("/{session_id}/values/{frame}", summary="Real pixel counts")
def frame_values(
    session_id: str = SessionId,
    frame: int = Path(..., ge=0),
) -> dict:
    """Downsampled grid of actual ADU counts, for the viewer's hover readout.

    Sent at 1/4 resolution because the full array is 325,000 integers per frame
    and the readout only needs to be locally accurate.
    """
    out = imaging.get_pixel_values(session_id, frame)
    if out is None:
        raise HTTPException(404, f"No pixel values for frame {frame} of '{session_id}'.")
    out["units"] = "ADU (12-bit, 0-4095)"
    out["note"] = ("Values are raw, before dark subtraction. Divide the pixel "
                   "index by `downsample` to map image coordinates onto this grid.")
    return out


@router.get("/{session_id}/manifest", summary="What tiles exist for a session")
def manifest(session_id: str = SessionId) -> dict:
    if session_id not in data.known_sessions():
        raise HTTPException(404, f"Unknown session '{session_id}'.")
    row = data.sessions()
    row = row[row.session_id == session_id].iloc[0]
    n = int(row.n_frames)
    have = [i for i in range(n)
            if imaging.prerendered_path(session_id, i, "zscale") is not None]

    # Report the stretches that will actually answer, not the three this service
    # knows how to make. `sync_data.py` ships a subset to keep the image the
    # right size, and a dashboard that offers a button for a stretch we did not
    # ship would just 404 in the user's face.
    on_demand = imaging._load_cube(session_id) is not None
    probe = have[0] if have else 0
    stretches = [s for s in imaging.STRETCHES
                 if on_demand
                 or imaging.prerendered_path(session_id, probe, s) is not None]

    return {
        "session_id": session_id,
        "n_frames": n,
        "stretches": stretches,
        "prerendered_frames": len(have),
        "can_render_on_demand": on_demand,
        "frame_url": f"/api/images/{session_id}/frame/{{frame}}?stretch=zscale",
        "values_url": f"/api/images/{session_id}/values/{{frame}}",
    }


@router.get("/triptych/{kind}", summary="Calibration comparison images",
            response_class=Response,
            responses={200: {"content": {"image/png": {}}}})
def triptych(
    kind: str = Path(..., description="raw, master_dark, calibrated or hot_pixel_map"),
) -> Response:
    """The calibration story in four pictures: the raw frame, the master dark
    subtracted from it, the result, and the map of 690-odd hot pixels."""
    allowed = {"raw", "master_dark", "calibrated", "hot_pixel_map"}
    if kind not in allowed:
        raise HTTPException(422, f"kind must be one of {sorted(allowed)}")
    for ext, media in ((".png", "image/png"), (".webp", "image/webp")):
        p = settings.STATIC_DIR / "triptych" / f"{kind}{ext}"
        if p.exists():
            return _image_response(p.read_bytes(), media)
    raise HTTPException(404, f"Triptych image '{kind}' was not generated.")
