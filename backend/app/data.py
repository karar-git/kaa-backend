"""Data access layer.

Every artefact the API serves was produced offline by the Kaggle pipeline
(`kaggle/src/nb01_pipeline.py`). Nothing is computed on request — the API is a
reader. That is a deliberate architectural choice: the science must be
reproducible from a notebook, not hidden inside a web server.

Tables are loaded lazily on first use and then kept in memory. The whole
`results/` set is a few tens of megabytes, so this costs little and removes disk
I/O from the request path entirely.
"""

from __future__ import annotations

import json
import threading
from typing import Any

import numpy as np
import pandas as pd

from .config import settings

_lock = threading.Lock()
_cache: dict[str, Any] = {}


class DatasetMissing(RuntimeError):
    """Raised when a results file the endpoint needs was never generated."""


def _read(name: str, **kwargs) -> pd.DataFrame:
    """Load and memoise one results file. Thread-safe."""
    if name in _cache:
        return _cache[name]
    with _lock:
        if name in _cache:                       # another thread won the race
            return _cache[name]
        path = settings.RESULTS_DIR / name
        if not path.exists():
            raise DatasetMissing(
                f"{name} is not present in {settings.RESULTS_DIR}. "
                "Run the pipeline notebook and copy its results/ directory here."
            )
        if name.endswith(".json"):
            df = json.loads(path.read_text(encoding="utf-8"))
        else:
            df = pd.read_csv(path, **kwargs)
        _cache[name] = df
        return df


def available() -> dict[str, bool]:
    """Which results files this deployment actually has. Drives /health."""
    names = [
        "dataset_audit.json", "calibration.json", "per_target.csv", "sessions.csv",
        "session_quality.csv", "target_usability.csv", "noise_vs_n.csv",
        "dark_masters_by_temp.csv", "tracks.csv", "detections.csv.gz",
        "field_motion.csv", "field_motion_summary.csv", "label_counts.csv",
        "lightcurves.csv", "photometry_info.csv", "ephemerides.csv",
        "transit_predictions.csv", "transit_depths.csv", "phasefold.csv",
        "phasefold_binned.csv", "field_photometry.csv.gz", "field_stars.csv",
        "noise_floor.csv", "target_search.csv", "quality_correlations.csv",
        "frame_quality.csv", "provenance.json", "planet_catalog.csv",
        "planet_physics.csv", "physics_validation.csv",
        "planet_radius_measurements.csv",
    ]
    return {n: (settings.RESULTS_DIR / n).exists() for n in names}


# ------------------------------------------------------------------ accessors

def audit() -> dict:
    return _read("dataset_audit.json")


def calibration() -> dict:
    return _read("calibration.json")


def sessions() -> pd.DataFrame:
    return _read("sessions.csv")


def session_quality() -> pd.DataFrame:
    return _read("session_quality.csv")


def per_target() -> pd.DataFrame:
    return _read("per_target.csv")


def target_usability() -> pd.DataFrame:
    return _read("target_usability.csv")


def noise_curve() -> pd.DataFrame:
    return _read("noise_vs_n.csv")


def dark_masters() -> pd.DataFrame:
    return _read("dark_masters_by_temp.csv")


def tracks() -> pd.DataFrame:
    return _read("tracks.csv")


def detections() -> pd.DataFrame:
    return _read("detections.csv.gz")


def field_motion() -> pd.DataFrame:
    return _read("field_motion.csv")


def field_motion_summary() -> pd.DataFrame:
    return _read("field_motion_summary.csv")


def label_counts() -> pd.DataFrame:
    return _read("label_counts.csv")


def lightcurves() -> pd.DataFrame:
    return _read("lightcurves.csv")


def photometry_info() -> pd.DataFrame:
    return _read("photometry_info.csv")


def ephemerides() -> pd.DataFrame:
    return _read("ephemerides.csv")


def predictions() -> pd.DataFrame:
    return _read("transit_predictions.csv")


def depths() -> pd.DataFrame:
    return _read("transit_depths.csv")


def phasefold() -> pd.DataFrame:
    return _read("phasefold.csv")


def phasefold_binned() -> pd.DataFrame:
    return _read("phasefold_binned.csv")


def field_photometry() -> pd.DataFrame:
    return _read("field_photometry.csv.gz")


def field_stars() -> pd.DataFrame:
    return _read("field_stars.csv")


def noise_floor() -> pd.DataFrame:
    return _read("noise_floor.csv")


def target_search() -> pd.DataFrame:
    return _read("target_search.csv")


def provenance() -> dict:
    """How the results currently on disk were produced.

    Chiefly: did the barycentric correction actually run? The column is called
    `bjd_tdb` whether or not astropy was importable, so without this record a
    fallback to plain JD_UTC -- worth up to eight minutes of timing error -- is
    invisible to anything downstream.
    """
    return _read("provenance.json")


def planet_catalog() -> pd.DataFrame:
    """Archive inputs to the physics section. External data, not ours."""
    return _read("planet_catalog.csv")


def planet_physics() -> pd.DataFrame:
    return _read("planet_physics.csv")


def physics_validation() -> pd.DataFrame:
    return _read("physics_validation.csv")


def planet_radius_measurements() -> pd.DataFrame:
    return _read("planet_radius_measurements.csv")


def quality_correlations() -> pd.DataFrame:
    return _read("quality_correlations.csv", index_col=0)


def frame_quality() -> pd.DataFrame:
    return _read("frame_quality.csv")


# ------------------------------------------------------------------ helpers

def to_records(df: pd.DataFrame, limit: int | None = None,
               offset: int = 0) -> list[dict]:
    """DataFrame -> JSON-safe records.

    pandas uses NaN for missing numbers, which is not valid JSON. We convert to
    None so a missing measurement arrives as `null` rather than as the string
    "NaN" or a silently coerced zero — the difference between "we did not
    measure this" and "we measured zero" matters here.
    """
    if offset:
        df = df.iloc[offset:]
    if limit is not None:
        df = df.iloc[:limit]
    df = df.replace({np.nan: None, np.inf: None, -np.inf: None})
    return df.to_dict(orient="records")


def known_sessions() -> list[str]:
    return sessions()["session_id"].tolist()


def known_targets() -> list[str]:
    return sorted(sessions()["target"].unique().tolist())
