"""Frame rendering.

Two ways an image can be served, tried in order:

1. **Pre-rendered tile** written by the pipeline into `static/frames/...`. This is
   what production uses — rendering is done once, offline.
2. **Rendered on demand** from the packaged `.npy` cubes, if those happen to be
   present. This exists so a developer can run the API against a raw bundle
   without first generating five thousand PNGs, and so a deploy that ships only
   some sessions still answers for the rest.

Results are cached on disk under `static/_cache/`, so an on-demand render happens
at most once per (session, frame, stretch).
"""

from __future__ import annotations

import io
import threading
from functools import lru_cache
from pathlib import Path

import numpy as np

from .config import settings

_render_lock = threading.Lock()
STRETCHES = ("zscale", "asinh", "linear")


@lru_cache(maxsize=8)
def _load_cube(session_id: str) -> np.ndarray | None:
    """Memory-map a session cube if the raw bundle was deployed alongside."""
    for base in (settings.DATA_DIR, settings.DATA_DIR.parent / "build"):
        p = base / "frames" / f"{session_id}.npy"
        if p.exists():
            return np.load(p, mmap_mode="r")
    return None


def _zscale(img: np.ndarray, contrast: float = 0.25, n: int = 10000):
    """IRAF zscale: fit a line through the sorted sample, take its central slope.

    This is what astronomers actually look at. A plain min/max stretch is ruined
    by one hot pixel; percentile clipping loses the faint end.
    """
    flat = img.ravel()
    step = max(1, flat.size // n)
    s = np.sort(flat[::step].astype(np.float32))
    k = len(s)
    if k < 8:
        return float(img.min()), float(img.max())
    x = np.arange(k) - k // 2
    slope = np.polyfit(x, s, 1)[0] / max(contrast, 1e-6)
    mid = float(s[k // 2])
    return mid + slope * x[0], mid + slope * x[-1]


def stretch_to_uint8(img: np.ndarray, mode: str = "zscale") -> np.ndarray:
    f = np.asarray(img, dtype=np.float32)
    if mode == "linear":
        lo, hi = np.percentile(f, [1.0, 99.5])
    elif mode == "asinh":
        lo, hi = np.percentile(f, [1.0, 99.9])
        f = np.arcsinh((f - lo) / max(float(hi - lo), 1e-6) * 10.0)
        lo, hi = float(f.min()), float(f.max())
    else:
        lo, hi = _zscale(f)
    out = np.clip((f - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0)
    return (out * 255).astype(np.uint8)


def encode(img_u8: np.ndarray, fmt: str = "WEBP", quality: int = 82) -> bytes:
    from PIL import Image
    # FITS row 0 is the bottom of the sky; PNG row 0 is the top. Flip so the
    # image is oriented the way every other astronomy tool shows it.
    im = Image.fromarray(np.flipud(img_u8), mode="L")
    buf = io.BytesIO()
    if fmt.upper() == "WEBP":
        im.save(buf, format="WEBP", quality=quality, method=4)
    elif fmt.upper() == "JPEG":
        im.save(buf, format="JPEG", quality=quality, optimize=True)
    else:
        im.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def prerendered_path(session_id: str, frame: int, stretch: str) -> Path | None:
    base = settings.STATIC_DIR / "frames" / session_id / stretch
    for ext in (".webp", ".png", ".jpg"):
        p = base / f"{frame:04d}{ext}"
        if p.exists():
            return p
    return None


def cached_path(session_id: str, frame: int, stretch: str) -> Path:
    return settings.STATIC_DIR / "_cache" / session_id / stretch / f"{frame:04d}.webp"


def get_frame_image(session_id: str, frame: int, stretch: str = "zscale"
                    ) -> tuple[bytes, str] | None:
    """Return (bytes, media_type), or None when the frame cannot be produced."""
    if stretch not in STRETCHES:
        raise ValueError(f"stretch must be one of {STRETCHES}")

    pre = prerendered_path(session_id, frame, stretch)
    if pre is not None:
        media = {".webp": "image/webp", ".png": "image/png",
                 ".jpg": "image/jpeg"}[pre.suffix]
        return pre.read_bytes(), media

    cache = cached_path(session_id, frame, stretch)
    if cache.exists():
        return cache.read_bytes(), "image/webp"

    cube = _load_cube(session_id)
    if cube is None or not (0 <= frame < len(cube)):
        return None

    with _render_lock:
        if cache.exists():
            return cache.read_bytes(), "image/webp"
        blob = encode(stretch_to_uint8(np.asarray(cube[frame]), stretch))
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(blob)
    return blob, "image/webp"


def get_pixel_values(session_id: str, frame: int, downsample: int = 4
                     ) -> dict | None:
    """Real ADU counts for the hover readout.

    The rendered image is 8-bit and stretched, so its grey levels are not data.
    The viewer needs the actual counts, and shipping a downsampled integer grid
    is far cheaper than the full array.
    """
    pre = settings.STATIC_DIR / "frames" / session_id / "values_4x.npy"
    if pre.exists():
        arr = np.load(pre, mmap_mode="r")
        if 0 <= frame < len(arr):
            a = np.asarray(arr[frame])
            return {"session_id": session_id, "frame_index": frame,
                    "downsample": 4, "shape": list(a.shape),
                    "values": a.astype(int).tolist()}

    cube = _load_cube(session_id)
    if cube is None or not (0 <= frame < len(cube)):
        return None
    a = np.asarray(cube[frame])[::downsample, ::downsample]
    return {"session_id": session_id, "frame_index": frame,
            "downsample": downsample, "shape": list(a.shape),
            "values": a.astype(int).tolist()}


def images_available() -> bool:
    if (settings.STATIC_DIR / "frames").exists():
        return True
    return (settings.DATA_DIR / "frames").exists()
