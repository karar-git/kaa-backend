"""Model inference and LLM narration.

Two very different things live here, and the split is deliberate:

* `/classify` runs **our** trained classifier on a 32x32 cutout. It produces a
  measurement.
* `/explain` asks a language model to *phrase* numbers this API already computed.
  It is never asked to produce a number, and the exact figures it was given come
  back in the response so any claim can be checked against them.

The challenge brief prohibits submitting AI output the team does not understand.
Keeping generation strictly downstream of measurement is how that line is held.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
from pathlib import Path as FsPath

import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile

from .. import data
from ..config import settings
from ..schemas import ClassifyResult, ExplainRequest, ExplainResponse

log = logging.getLogger("exotransit.ml")

router = APIRouter(tags=["ml"])

# Output order of the trained network, exactly as nb02_train.py defines it. The
# logits carry no names, so a different order here silently relabels every
# prediction -- the first version of this file had one, and it would have called
# faint sources hot pixels. metrics.json records the order used in training and
# overrides this whenever it is present.
CLASSES = ["star", "faint_source", "hot_pixel", "cosmic_ray", "satellite_trail", "noise"]
MODEL_DIR = settings.DATA_DIR / "model"
CLIP = (-10.0, 60.0)          # nb02_train.py clips normalised stamps to this range


def _metrics() -> dict | None:
    p = MODEL_DIR / "metrics.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        log.exception("metrics.json unreadable")
        return None


def _classes() -> list[str]:
    m = _metrics()
    return list(m["classes"]) if m and m.get("classes") else CLASSES


def _model_files() -> dict[str, bool]:
    return {
        "weights": (MODEL_DIR / "classifier.onnx").exists(),
        "metrics": (MODEL_DIR / "metrics.json").exists(),
        "card": (MODEL_DIR / "model_card.json").exists(),
    }


@router.get("/model/info", summary="Model card")
def model_info() -> dict:
    """What the classifier is, how it was trained, and how to read its numbers."""
    files = _model_files()
    card = {}
    if files["card"]:
        card = json.loads((MODEL_DIR / "model_card.json").read_text(encoding="utf-8"))
    m = _metrics() or {}
    best = m.get("best_model")
    rep = m.get("models", {}).get(best, {}) if best else {}
    headline = None
    if rep:
        pc = rep.get("per_class", {})
        headline = {
            "model": best,
            "test_accuracy": round(rep["accuracy"], 4),
            "test_macro_f1": round(rep["macro_f1"], 4),
            "baseline_macro_f1": round(m["baseline"]["macro_f1"], 4),
            "majority_class_accuracy": m.get("majority_class_accuracy"),
            "test_nights": len(m.get("split", {}).get("test_sessions", [])),
            "test_cutouts": m.get("split", {}).get("n_test"),
            "recall_by_class": {c: round(v["recall"], 3) for c, v in pc.items()},
            "precision_by_class": {c: round(v["precision"], 3) for c, v in pc.items()},
            "reading": _reading(pc),
        }
    return {
        "task": "Classify a 32x32 raw-frame cutout as one of six source classes.",
        "classes": _classes(),
        "test_set_performance": headline,
        "trained": files["weights"],
        "artifacts_present": files,
        "label_provenance":
            "Labels were derived from physical rules, never hand-drawn. Hot pixels "
            "come from 60 shutter-closed dark frames; stars from persistence plus "
            "shared motion with the field; cosmic rays and satellite trails from "
            "single-frame morphology. See GET /api/labels.",
        "validation_protocol":
            "Split by NIGHT, never at random. Frames within one night are heavily "
            "correlated, so a random split leaks and inflates the score.",
        "metrics_note":
            "Accuracy is not reported as a headline. The classes are extremely "
            "imbalanced, so per-class precision and recall are what matter.",
        "baseline":
            "The classical sigma-clip plus connected-components detector in the "
            "pipeline. A learned model that does not beat it is not worth shipping.",
        **card,
    }


def _reading(per_class: dict) -> str:
    """Plain statement of where the model can and cannot be trusted, from the
    test-set numbers. Computed, so it cannot drift from the metrics."""
    if not per_class:
        return ""
    good = [c for c, v in per_class.items() if v["f1"] >= 0.7]
    weak = [c for c, v in per_class.items() if v["recall"] < 0.2]
    over = [c for c, v in per_class.items() if v["precision"] < 0.25 and v["recall"] >= 0.5]
    parts = []
    if good:
        parts.append("Reliable on unseen nights for: %s." % ", ".join(good))
    if weak:
        parts.append("Rarely finds: %s (recall below 20%%); do not rely on it for "
                     "these." % ", ".join(weak))
    if over:
        parts.append("Over-predicts: %s (most predictions of these are wrong)."
                     % ", ".join(over))
    return " ".join(parts)


@router.get("/model/metrics", summary="Per-class precision and recall")
def model_metrics() -> dict:
    p = MODEL_DIR / "metrics.json"
    if not p.exists():
        raise HTTPException(
            503,
            "No metrics published yet. Train with kaggle/notebooks/02_train.ipynb "
            "and copy its model/ output into the API's data directory.")
    return json.loads(p.read_text(encoding="utf-8"))


def _decode_cutout(raw: bytes) -> np.ndarray:
    """Accept a .npy array or any common image, return a 32x32 float array."""
    try:
        arr = np.load(io.BytesIO(raw), allow_pickle=False)
    except Exception:
        try:
            from PIL import Image
            arr = np.asarray(Image.open(io.BytesIO(raw)).convert("L"))
        except Exception:
            raise HTTPException(
                422, "Could not read that file. Send a 32x32 .npy array or a "
                     "greyscale PNG/JPEG.")
    arr = np.asarray(arr, dtype=np.float32).squeeze()
    if arr.ndim != 2:
        raise HTTPException(422, f"Expected a 2-D cutout, got shape {arr.shape}.")
    if arr.shape != (32, 32):
        raise HTTPException(
            422, f"Expected 32x32, got {arr.shape}. Cutouts must match the "
                 f"training geometry or the prediction is meaningless.")
    return arr


def _normalise(arr: np.ndarray) -> np.ndarray:
    """Same normalisation the training notebook applies: subtract the stamp's own
    median, divide by its MAD-derived sigma. Makes the model insensitive to sky
    level, which varies by a factor of five between nights."""
    med = float(np.median(arr))
    sig = float(np.median(np.abs(arr - med))) * 1.4826
    return np.clip((arr - med) / max(sig, 1e-3), *CLIP).astype(np.float32)


@router.post("/classify", response_model=ClassifyResult,
             summary="Classify one 32x32 cutout")
async def classify(file: UploadFile = File(
        ..., description="32x32 cutout as .npy, or a greyscale image")) -> dict:
    """Run the trained classifier on a single cutout.

    Until a model is published this returns a **transparent rule-based fallback**
    using the same morphology features the pipeline measures, and says so in
    `note`. It is not a stand-in pretending to be a network — the label tells you
    which produced the answer.
    """
    arr = _decode_cutout(await file.read())
    x = _normalise(arr)

    weights = MODEL_DIR / "classifier.onnx"
    have_ort = True
    if weights.exists():
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            # A model is present but the runtime is not. Degrade to the rule-based
            # answer with a note saying why, rather than 500 on every request --
            # but say it out loud, because a silently degraded classifier is worse
            # than a broken one.
            have_ort = False
            log.error("classifier.onnx is present but onnxruntime is not "
                      "installed; falling back to the morphology rules.")
    if weights.exists() and have_ort:
        try:
            sess = _ort_session(str(weights))
            inp = sess.get_inputs()[0].name
            logits = sess.run(None, {inp: x[None, None].astype(np.float32)})[0][0]
            e = np.exp(logits - logits.max())
            probs = e / e.sum()
            k = int(np.argmax(probs))
            classes = _classes()
            m = _metrics() or {}
            best = m.get("best_model", "cnn")
            pc = m.get("models", {}).get(best, {}).get("per_class", {}).get(classes[k])
            note = None
            if pc:
                # A softmax confidence is not a hit rate. Say how often this label
                # was actually right on nights the model never saw.
                note = ("On unseen test nights, %.0f%% of '%s' predictions were correct "
                        "and %.0f%% of true '%s' sources were found. `confidence` is "
                        "the network's own score, not that rate."
                        % (100 * pc["precision"], classes[k], 100 * pc["recall"],
                           classes[k]))
            return {"label": classes[k], "confidence": float(probs[k]),
                    "probabilities": {c: float(p) for c, p in zip(classes, probs)},
                    "model_name": "exotransit-%s" % best, "note": note}
        except Exception as exc:
            raise HTTPException(500, f"Inference failed: {type(exc).__name__}: {exc}")

    peak = float(x.max())
    above = x > 5.0
    npix = int(above.sum())
    if npix == 0:
        label, conf = "noise", 0.6
    elif npix <= 3 and peak > 15:
        label, conf = "cosmic_ray", 0.5
    elif npix > 40:
        label, conf = "satellite_trail", 0.4
    elif peak > 20:
        label, conf = "star", 0.5
    else:
        label, conf = "faint_source", 0.4
    return {
        "label": label, "confidence": conf,
        "probabilities": {c: (conf if c == label else (1 - conf) / 5) for c in _classes()},
        "model_name": "rule-based-fallback",
        "note": ("classifier.onnx IS deployed but onnxruntime is missing from this "
                 "image, so this answer came from the morphology rules, not the "
                 "network. Add onnxruntime to requirements.txt and redeploy."
                 if weights.exists() else
                 "No trained model is deployed. This is the transparent morphology "
                 "fallback, not a neural network. Publish model/classifier.onnx to "
                 "replace it."),
    }


_ORT_CACHE: dict[str, object] = {}


def _ort_session(path: str):
    if path not in _ORT_CACHE:
        import onnxruntime as ort
        _ORT_CACHE[path] = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return _ORT_CACHE[path]


# --------------------------------------------------------------------- explain

SYSTEM_PROMPT = """You are helping present results from a student astronomy project.

Absolute rules:
- Use ONLY the numbers in the MEASUREMENTS block. Never invent, round misleadingly,
  or add figures from your own knowledge.
- The `verdict` field states what this session showed. It was computed from the
  measurements before you were asked. Your job is to PHRASE it. You may not reach
  a different conclusion, soften it, or hedge it into something stronger.
- In particular: "candidate consistent with a transit" is permitted ONLY if the
  verdict says so. If the verdict says NOT A DETECTION, NULL RESULT or NO RESULT,
  write it as exactly that. A weak signal is a non-detection, not a weak candidate.
- Never call anything a confirmed planet, whatever the verdict says.
- If `field_context` is present, explain that the signal was compared against the
  other stars in the same image. That comparison is the whole basis of the claim.
- Mention the limitation given, if one is present.
- Two short paragraphs maximum. Plain language, no marketing tone.
"""


def _session_facts(session_id: str) -> dict:
    sess = data.sessions()
    row = sess[sess.session_id == session_id]
    if row.empty:
        raise HTTPException(404, f"Unknown session '{session_id}'.")
    facts: dict = data.to_records(row)[0]
    for name, fn, key in (
        ("quality", data.session_quality, "session_id"),
        ("photometry", data.photometry_info, "session_id"),
        ("prediction", data.predictions, "session_id"),
        ("depth", data.depths, "session_id"),
    ):
        try:
            df = fn()
            sub = df[df[key] == session_id]
            if not sub.empty:
                facts[name] = data.to_records(sub)[0]
        except data.DatasetMissing:
            pass
    # Significance means nothing without the field it was measured against. The
    # same night's other 100-450 stars say what this instrument produces from
    # noise alone, so hand that over too -- otherwise a 2.0 sigma dip reads as a
    # result when the field routinely fakes 2.1.
    try:
        srch = data.target_search()
        sub = srch[srch.session_id == session_id]
        if not sub.empty:
            facts["field_context"] = {
                "n_field_stars": int(sub.n_field_stars.iloc[0]),
                "field_sigma_95": float(sub.field_sigma_95.iloc[0]),
                "field_sigma_median": float(sub.field_sigma_median.iloc[0]),
                "meaning": "field_sigma_95 is the significance reached by the top "
                           "5% of ORDINARY stars in this same image at this same "
                           "phase. A signal below it is not distinguishable from "
                           "what this night does to noise.",
            }
    except data.DatasetMissing:
        pass

    # The verdict is computed here, not by the model. Generation stays strictly
    # downstream of measurement: the model chooses words, never the conclusion.
    facts["verdict"] = _verdict(facts)

    facts["limitation"] = (
        "At 5 arcsec/pixel with a single Clear filter, a blended eclipsing binary "
        "or starspot activity cannot be excluded. The host star was also chosen by "
        "position rather than identified astrometrically, because these headers "
        "carry no WCS solution.")
    return facts


def _verdict(facts: dict) -> str:
    """Decide what this session actually showed, in one sentence.

    Deterministic and ours. The language model is told to phrase this verdict and
    forbidden to reach a different one, which is what stops a 2-sigma dip being
    narrated as a candidate.
    """
    if facts.get("quality", {}).get("quality") == "unusable":
        return ("NO RESULT. This night failed quality triage; no photometry from "
                "it can be trusted. Say that and stop.")
    d = facts.get("depth")
    if not d or d.get("significance_sigma") is None:
        return ("NO MEASUREMENT. No transit depth could be measured for this "
                "session. Report that plainly as a null result.")

    sig = float(d["significance_sigma"])
    floor = facts.get("field_context", {}).get("field_sigma_95")

    if sig <= 0:
        return (f"NULL RESULT. The star was brighter in the predicted window, not "
                f"fainter ({sig:.2f} sigma). There is no dip here. Report the null.")
    if floor is not None and sig <= float(floor):
        return (f"NOT A DETECTION. The dip reaches {sig:.2f} sigma, but the top 5% "
                f"of ordinary stars in this same image reach {float(floor):.2f} "
                f"sigma at the same phase. This is indistinguishable from noise. "
                f"Do NOT call it a candidate.")
    if sig < 3.0:
        # State the field relationship explicitly. Leaving it implicit invites the
        # narrator to guess at it, and it guessed wrong: it once wrote "below the
        # 2.35 sigma level" about a 2.74 sigma signal.
        rel = ""
        if floor is not None:
            rel = (f" It does exceed the field 95th percentile of "
                   f"{float(floor):.2f} sigma -- say so accurately, do NOT state "
                   f"the reverse -- but clearing the field is necessary, not "
                   f"sufficient.")
        return (f"NOT A DETECTION. The dip reaches {sig:.2f} sigma, which is too "
                f"weak to claim a transit on a single night.{rel} Report it as a "
                f"non-detection.")
    if floor is not None:
        return (f"CANDIDATE CONSISTENT WITH A TRANSIT, and nothing stronger. The "
                f"dip reaches {sig:.2f} sigma against a field 95th percentile of "
                f"{float(floor):.2f}. It is NOT a confirmed planet.")
    return (f"CANDIDATE CONSISTENT WITH A TRANSIT at {sig:.2f} sigma, and nothing "
            f"stronger. It is NOT a confirmed planet.")


@router.post("/explain", response_model=ExplainResponse,
             summary="Plain-language summary of one session")
async def explain(req: ExplainRequest) -> dict:
    """Ask a language model to phrase this session's measured numbers.

    The model receives the measurements and nothing else, and the block it was
    given is returned in `grounded_on` so every sentence can be checked against
    the figures behind it. It is a writing aid, not a source of results.

    Requires `OPENROUTER_API_KEY` on the server. The key is never accepted from
    the client and never returned.
    """
    if not settings.ai_enabled:
        raise HTTPException(
            503,
            "AI narration is disabled: OPENROUTER_API_KEY is not set on the "
            "server. Every other endpoint works without it.")

    facts = _session_facts(req.session_id)
    lang = "Arabic" if req.language == "ar" else "English"
    user = (f"Write for a {req.audience} audience, in {lang}.\n\n"
            f"MEASUREMENTS:\n{json.dumps(facts, indent=2, default=str)}")

    import httpx
    try:
        async with httpx.AsyncClient(timeout=settings.OPENROUTER_TIMEOUT) as client:
            r = await client.post(
                settings.OPENROUTER_URL,
                headers={"Authorization": f"Bearer {settings.openrouter_key}",
                         "Content-Type": "application/json",
                         "X-Title": "ExoTransit Lab"},
                json={"model": settings.OPENROUTER_MODEL,
                      "temperature": 0.2,
                      "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                                   {"role": "user", "content": user}]},
            )
    except Exception as exc:
        raise HTTPException(502, f"Upstream request failed: {type(exc).__name__}")
    if r.status_code >= 400:
        # Never echo the response body: it can contain request headers.
        raise HTTPException(502, f"OpenRouter returned HTTP {r.status_code}.")
    body = r.json()
    try:
        text = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        raise HTTPException(502, "Unexpected response shape from OpenRouter.")

    return {"session_id": req.session_id, "language": req.language,
            "audience": req.audience, "text": text,
            "model_name": settings.OPENROUTER_MODEL, "grounded_on": facts}
