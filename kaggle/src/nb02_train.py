# %% [markdown]
# # ExoTransit Lab — 02 · Source Classifier
#
# **Hack4Dev Iraq 2026 · Challenge E · Team Iraqi Andromeda**
#
# Trains the model behind `POST /api/classify`. Input is `cutouts.npz` from
# notebook 01; output is an ONNX file plus honest metrics for the dashboard.
#
# ### The task, stated precisely
#
# > Given a 32×32 cutout from a **raw** frame, decide whether it is a real
# > astronomical source and, if so, what kind.
#
# Six classes: `star`, `faint_source`, `hot_pixel`, `cosmic_ray`,
# `satellite_trail`, `noise`. "Real object or not" is not a testable question —
# this is.
#
# ### Three commitments that decide whether the result means anything
#
# 1. **Split by night, never at random.** Frames from one night share the same
#    stars, the same optics and the same sky. A random split puts near-duplicate
#    cutouts in train and test and inflates every score. This is the single
#    easiest way to fake a good number, so we do the opposite.
# 2. **Beat the classical baseline or admit it.** Notebook 01 already detects and
#    classifies sources with threshold-plus-morphology rules. A network that does
#    not beat that has earned nothing, and the comparison is printed side by side.
# 3. **Never report accuracy as the headline.** `star` alone is over half the
#    data, so a model that answers "star" every time already scores ~55%. Per-class
#    precision and recall are the real numbers.
#
# ### Runtime
#
# Kaggle's **T4 ×2** accelerator. The models here are small, so instead of
# splitting one model across both GPUs we train **a different architecture on each
# GPU at the same time** — which is the useful kind of parallelism when the models
# are this size.

# %%
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
np.random.seed(SEED)


def find_cutouts() -> Path:
    """Locate cutouts.npz from notebook 01, on Kaggle or locally."""
    for base in [Path("/kaggle/input"), Path("/kaggle/working"),
                 Path("../out"), Path("./out"), Path(".")]:
        if not base.exists():
            continue
        for p in sorted(base.rglob("cutouts.npz")):
            return p
    raise FileNotFoundError(
        "cutouts.npz not found. Run notebook 01 first, then attach its output "
        "as a dataset (Add Data -> Notebook Output).")


CUTOUTS = find_cutouts()
ON_KAGGLE = Path("/kaggle/input").exists()
OUT = Path("/kaggle/working") if ON_KAGGLE else Path("./out")
MODEL_DIR = OUT / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("cutouts :", CUTOUTS)
print("output  :", MODEL_DIR)

# %%
import torch

print("torch", torch.__version__, "| CUDA", torch.cuda.is_available())
N_GPU = torch.cuda.device_count()
for i in range(N_GPU):
    p = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {p.name}, {p.total_memory/1e9:.1f} GB")
if N_GPU == 0:
    print("  no GPU -- training on CPU. Enable T4 x2 under Settings > Accelerator.")

# %% [markdown]
# ## 1 · Load and inspect
#
# Before training anything, look at the class balance. It decides which metrics
# are meaningful and which are theatre.

# %%
z = np.load(CUTOUTS, allow_pickle=False)
X_all = z["X"].astype(np.float32)          # (N, 32, 32) raw ADU
y_all = z["y"].astype(str)
sess_all = z["session"].astype(str)

CLASSES = ["star", "faint_source", "hot_pixel", "cosmic_ray",
           "satellite_trail", "noise"]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}

keep = np.isin(y_all, CLASSES)
X_all, y_all, sess_all = X_all[keep], y_all[keep], sess_all[keep]
labels = np.array([CLASS_TO_ID[c] for c in y_all])

counts = pd.Series(y_all).value_counts()
share = (counts / len(y_all) * 100).round(2)
print("cutouts:", X_all.shape, "| sessions:", len(set(sess_all)))
print()
print(pd.DataFrame({"n": counts, "share_%": share}).to_string())
print()
maj = share.max()
print("Majority class is %.1f%% of the data." % maj)
print("Any model scoring below %.1f%% accuracy is worse than a constant." % maj)

# %% [markdown]
# ## 2 · Split by night
#
# Sessions go entirely into train, validation or test — never split within one.
# The test set is held out until the very end and scored once.

# %%
sessions_all = sorted(set(sess_all))
rng = np.random.default_rng(SEED)
order = rng.permutation(sessions_all)
n = len(order)
n_test = max(2, int(round(0.25 * n)))
n_val = max(2, int(round(0.15 * n)))

test_s = set(order[:n_test])
val_s = set(order[n_test:n_test + n_val])
train_s = set(order[n_test + n_val:])

tr = np.isin(sess_all, list(train_s))
va = np.isin(sess_all, list(val_s))
te = np.isin(sess_all, list(test_s))

print("train %2d sessions, %6d cutouts" % (len(train_s), tr.sum()))
print("val   %2d sessions, %6d cutouts" % (len(val_s), va.sum()))
print("test  %2d sessions, %6d cutouts" % (len(test_s), te.sum()))
assert not (train_s & val_s) and not (train_s & test_s) and not (val_s & test_s)
print("\nno session appears in more than one split -- no leakage")
print("\ntest sessions:", ", ".join(sorted(test_s)))

print("\nclass share per split (%):")
print(pd.DataFrame({
    "train": pd.Series(y_all[tr]).value_counts(normalize=True) * 100,
    "val": pd.Series(y_all[va]).value_counts(normalize=True) * 100,
    "test": pd.Series(y_all[te]).value_counts(normalize=True) * 100,
}).round(2).to_string())

# %% [markdown]
# ## 3 · Normalisation
#
# Each stamp is normalised **by its own statistics**: subtract its median, divide
# by its MAD-derived sigma.
#
# This matters more than it looks. Sky level ranges from 399 to 2249 ADU across
# these nights — a factor of five. A model trained on absolute counts would learn
# the sky brightness of the training nights and fall apart on a new one. After
# per-stamp normalisation the model sees shape and contrast, which is what
# actually distinguishes a star from a hot pixel.
#
# The API applies exactly this transform before inference.

# %%
def normalise(batch: np.ndarray) -> np.ndarray:
    med = np.median(batch, axis=(1, 2), keepdims=True)
    mad = np.median(np.abs(batch - med), axis=(1, 2), keepdims=True) * 1.4826
    return (batch - med) / np.maximum(mad, 1e-3)


t0 = time.time()
Xn = normalise(X_all)
Xn = np.clip(Xn, -10, 60).astype(np.float32)   # cosmic rays reach absurd values
print("normalised %s in %.1fs" % (str(Xn.shape), time.time() - t0))
print("value range after clipping: %.1f .. %.1f" % (Xn.min(), Xn.max()))

# %% [markdown]
# ## 4 · The baseline
#
# Notebook 01's morphology rules, applied to the same test set. Everything below
# is measured against this.

# %%
from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_recall_fscore_support)


def baseline_predict(batch: np.ndarray) -> np.ndarray:
    """Threshold-and-morphology rules -- no learning of any kind."""
    out = np.empty(len(batch), dtype=int)
    for i, img in enumerate(batch):
        above = img > 5.0
        npix = int(above.sum())
        peak = float(img.max())
        if npix == 0:
            out[i] = CLASS_TO_ID["noise"]
        elif npix <= 3 and peak > 15:
            out[i] = CLASS_TO_ID["cosmic_ray"]
        elif npix > 40:
            out[i] = CLASS_TO_ID["satellite_trail"]
        elif peak > 20:
            out[i] = CLASS_TO_ID["star"]
        else:
            out[i] = CLASS_TO_ID["faint_source"]
    return out


base_pred = baseline_predict(Xn[te])
base_true = labels[te]
base_acc = float((base_pred == base_true).mean())
base_p, base_r, base_f1, _ = precision_recall_fscore_support(
    base_true, base_pred, average="macro", zero_division=0)
print("BASELINE on the held-out test sessions")
print("  accuracy %.4f | macro precision %.4f | macro recall %.4f | macro F1 %.4f"
      % (base_acc, base_p, base_r, base_f1))
print()
print(classification_report(base_true, base_pred, labels=range(len(CLASSES)),
                            target_names=CLASSES, zero_division=0, digits=3))

# %% [markdown]
# ## 5 · Two architectures, one per GPU
#
# * **SmallCNN** — four conv blocks. The workhorse, and the one most likely to ship.
# * **WideCNN** — fewer, wider layers with global pooling. Different inductive bias,
#   so its errors are not the same errors, which makes the comparison informative.
#
# Both are tiny; the point of two GPUs here is to train two models at once, not to
# shard one.

# %%
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class SmallCNN(nn.Module):
    def __init__(self, n_classes=len(CLASSES)):
        super().__init__()
        def block(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                nn.MaxPool2d(2))
        self.features = nn.Sequential(block(1, 32), block(32, 64), block(64, 128))
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(0.3), nn.Linear(128, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


class WideCNN(nn.Module):
    def __init__(self, n_classes=len(CLASSES)):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 5, padding=2), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.AdaptiveMaxPool2d(1), nn.Flatten(),
            nn.Dropout(0.4), nn.Linear(256, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


ARCHS = {"small_cnn": SmallCNN, "wide_cnn": WideCNN}
for name, cls in ARCHS.items():
    n_par = sum(p.numel() for p in cls().parameters())
    print("%-12s %7d parameters" % (name, n_par))

# %% [markdown]
# ### Class weighting
#
# `satellite_trail` is about 0.03% of the data. Left alone the optimiser will
# ignore it entirely, because doing so costs almost nothing. Inverse-frequency
# weights in the loss make a trail miss expensive.
#
# Weights are **capped at 50×**. Uncapped inverse frequency would multiply the
# rarest class by several thousand and let a handful of stamps dominate every
# gradient step.

# %%
freq = np.bincount(labels[tr], minlength=len(CLASSES)).astype(np.float64)
freq[freq == 0] = 1.0
weights = len(labels[tr]) / (len(CLASSES) * freq)
weights = np.clip(weights, 0.2, 50.0)
print(pd.DataFrame({"class": CLASSES, "n_train": freq.astype(int),
                    "loss_weight": weights.round(2)}).to_string(index=False))

# %% [markdown]
# ## 6 · Training

# %%
EPOCHS = int(os.environ.get("EPOCHS", 12))
BATCH = 256
LR = 2e-3


def make_loader(mask, shuffle, augment=False):
    x = torch.from_numpy(Xn[mask]).unsqueeze(1)
    y = torch.from_numpy(labels[mask]).long()
    return DataLoader(TensorDataset(x, y), batch_size=BATCH, shuffle=shuffle,
                      num_workers=int(os.environ.get("NUM_WORKERS", 2)),
                      pin_memory=torch.cuda.is_available(), drop_last=False)


def augment_batch(x):
    """Flips and 90-degree rotations only.

    These are the transforms the sky is genuinely invariant under -- a star looks
    the same rotated. Brightness or contrast jitter would be wrong: it would
    destroy the very contrast cue that separates a star from a faint source.
    """
    if torch.rand(1).item() < 0.5:
        x = torch.flip(x, dims=[3])
    if torch.rand(1).item() < 0.5:
        x = torch.flip(x, dims=[2])
    k = int(torch.randint(0, 4, (1,)).item())
    return torch.rot90(x, k, dims=[2, 3]) if k else x


def train_one(name, device, epochs=EPOCHS, log=True):
    torch.manual_seed(SEED)
    model = ARCHS[name]().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device))
    tl, vl = make_loader(tr, True), make_loader(va, False)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR, epochs=epochs, steps_per_epoch=max(1, len(tl)))
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_f1, best_state, history = -1.0, None, []
    for ep in range(1, epochs + 1):
        model.train()
        run = 0.0
        for xb, yb in tl:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            xb = augment_batch(xb)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                loss = crit(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            run += loss.item() * len(xb)

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for xb, yb in vl:
                out = model(xb.to(device))
                preds.append(out.argmax(1).cpu().numpy())
                trues.append(yb.numpy())
        pr, tv = np.concatenate(preds), np.concatenate(trues)
        _, _, f1, _ = precision_recall_fscore_support(
            tv, pr, average="macro", zero_division=0)
        history.append({"epoch": ep, "train_loss": run / max(1, tr.sum()),
                        "val_macro_f1": float(f1),
                        "val_accuracy": float((pr == tv).mean())})
        if f1 > best_f1:
            best_f1 = float(f1)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if log:
            print("  [%s] epoch %2d/%d loss %.4f  val macro-F1 %.4f  acc %.4f"
                  % (name, ep, epochs, history[-1]["train_loss"], f1,
                     history[-1]["val_accuracy"]))

    model.load_state_dict(best_state)
    # Selection used validation only; the test set is still untouched.
    return model, best_f1, history


# %% [markdown]
# ### Launch both models at once
#
# With two GPUs each architecture gets its own device and they train
# concurrently in separate threads. Python's GIL is not a problem here because
# CUDA work releases it. On one GPU or on CPU they simply run in sequence.

# %%
import threading

devices = ([torch.device(f"cuda:{i}") for i in range(N_GPU)] if N_GPU
           else [torch.device("cpu")])
assign = {name: devices[i % len(devices)] for i, name in enumerate(ARCHS)}
print("device assignment:", {k: str(v) for k, v in assign.items()})

results: dict = {}
t0 = time.time()

if len(devices) > 1:
    threads = []

    def worker(nm):
        m, f1, h = train_one(nm, assign[nm])
        results[nm] = {"model": m, "val_f1": f1, "history": h}

    for nm in ARCHS:
        t = threading.Thread(target=worker, args=(nm,), daemon=False)
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
else:
    for nm in ARCHS:
        m, f1, h = train_one(nm, assign[nm])
        results[nm] = {"model": m, "val_f1": f1, "history": h}

print("\ntrained %d models in %.0fs" % (len(results), time.time() - t0))
for nm, r in results.items():
    print("  %-12s best val macro-F1 %.4f" % (nm, r["val_f1"]))

# %% [markdown]
# ## 7 · Evaluate on the held-out nights
#
# The test sessions have not been touched. This is the number that counts, and it
# is computed once.

# %%
def evaluate(model, device, mask):
    model.eval()
    x = torch.from_numpy(Xn[mask]).unsqueeze(1)
    preds, probs = [], []
    with torch.no_grad():
        for i in range(0, len(x), 512):
            out = model(x[i:i + 512].to(device))
            probs.append(torch.softmax(out, 1).cpu().numpy())
            preds.append(out.argmax(1).cpu().numpy())
    return np.concatenate(preds), np.concatenate(probs)


y_test = labels[te]
report = {}
for nm, r in results.items():
    pred, prob = evaluate(r["model"], assign[nm], te)
    acc = float((pred == y_test).mean())
    p, rc, f1, _ = precision_recall_fscore_support(
        y_test, pred, average="macro", zero_division=0)
    per = precision_recall_fscore_support(
        y_test, pred, labels=range(len(CLASSES)), zero_division=0)
    report[nm] = {
        "accuracy": acc, "macro_precision": float(p),
        "macro_recall": float(rc), "macro_f1": float(f1),
        "per_class": {c: {"precision": float(per[0][i]),
                          "recall": float(per[1][i]),
                          "f1": float(per[2][i]),
                          "support": int(per[3][i])}
                      for i, c in enumerate(CLASSES)},
        "confusion_matrix": confusion_matrix(
            y_test, pred, labels=range(len(CLASSES))).tolist(),
    }
    r["test_pred"] = pred
    print("\n=== %s ===" % nm)
    print("accuracy %.4f | macro P %.4f | macro R %.4f | macro F1 %.4f"
          % (acc, p, rc, f1))
    print(classification_report(y_test, pred, labels=range(len(CLASSES)),
                                target_names=CLASSES, zero_division=0, digits=3))

# %% [markdown]
# ### The comparison that decides whether any of this shipped value

# %%
rows = [{"model": "classical baseline", "accuracy": base_acc,
         "macro_precision": base_p, "macro_recall": base_r, "macro_f1": base_f1}]
for nm, m in report.items():
    rows.append({"model": nm, "accuracy": m["accuracy"],
                 "macro_precision": m["macro_precision"],
                 "macro_recall": m["macro_recall"], "macro_f1": m["macro_f1"]})
cmp = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
print(cmp.round(4).to_string(index=False))

best_name = cmp.iloc[0]["model"]
if best_name == "classical baseline":
    print("\nThe classical baseline WON. Neither network beat threshold-and-"
          "morphology rules on held-out nights.")
    print("That is a real result and it goes in the README. Ship the baseline.")
else:
    gain = cmp.iloc[0]["macro_f1"] - base_f1
    print("\nBest model: %s, macro-F1 %.4f vs baseline %.4f (+%.4f)"
          % (best_name, cmp.iloc[0]["macro_f1"], base_f1, gain))
    if gain < 0.02:
        print("The margin is under 0.02. That is thin -- report it as thin.")

# %% [markdown]
# ### Where it fails
#
# The confusion matrix and a sample of real mistakes. Hiding failed cases is
# explicitly prohibited by the challenge brief, and it is also the most useful
# page on the dashboard: it shows exactly which distinctions the model cannot make.

# %%
if best_name != "classical baseline":
    best = results[best_name]
    cm = np.array(report[best_name]["confusion_matrix"])
    print("Confusion matrix for %s (rows = truth, columns = prediction):\n" % best_name)
    print(pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_string())

    print("\nMost common confusions:")
    pairs = [(cm[i, j], CLASSES[i], CLASSES[j])
             for i in range(len(CLASSES)) for j in range(len(CLASSES)) if i != j]
    for n_err, a, b in sorted(pairs, reverse=True)[:6]:
        if n_err:
            print("  %5d  %-16s predicted as %-16s" % (n_err, a, b))

    wrong = np.where(best["test_pred"] != y_test)[0]
    print("\n%d of %d test cutouts misclassified (%.2f%%)"
          % (len(wrong), len(y_test), 100 * len(wrong) / len(y_test)))
    if len(wrong):
        idx_all = np.where(te)[0]
        sample = wrong[:200]
        pd.DataFrame({
            "session": sess_all[idx_all[sample]],
            "true": [CLASSES[i] for i in y_test[sample]],
            "predicted": [CLASSES[i] for i in best["test_pred"][sample]],
        }).to_csv(MODEL_DIR / "misclassified_sample.csv", index=False)
        np.save(MODEL_DIR / "misclassified_cutouts.npy",
                X_all[idx_all[sample]].astype(np.uint16))
        print("saved 200 failure cases for the dashboard's 'where it fails' page")

# %% [markdown]
# ## 8 · Export for the API
#
# ONNX, so the API needs `onnxruntime` rather than a full PyTorch install — a much
# smaller container, and inference on the CPU Railway gives us is fast enough for
# 32×32 input.

# %%
metrics = {
    "generated_utc": pd.Timestamp.now("UTC").isoformat(),
    "classes": CLASSES,
    "n_cutouts": int(len(X_all)),
    "split": {"strategy": "by observing night, never within one",
              "train_sessions": sorted(train_s),
              "val_sessions": sorted(val_s),
              "test_sessions": sorted(test_s),
              "n_train": int(tr.sum()), "n_val": int(va.sum()), "n_test": int(te.sum())},
    "class_share_pct": share.to_dict(),
    "majority_class_accuracy": float(maj / 100),
    "baseline": {"name": "classical sigma-clip + morphology",
                 "accuracy": base_acc, "macro_precision": base_p,
                 "macro_recall": base_r, "macro_f1": base_f1},
    "models": report,
    "best_model": best_name,
    "epochs": EPOCHS,
    "caveats": [
        "Labels are derived from physical rules, not human annotation. A "
        "systematic error in a rule becomes a systematic error in the labels.",
        "Split is by night, so scores reflect generalisation to unseen nights.",
        "Accuracy is not the headline: the majority class is %.1f%% of the data."
        % maj,
        "satellite_trail has very few examples; its metrics are correspondingly "
        "uncertain.",
    ],
}
(MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
print("wrote metrics.json")

if best_name != "classical baseline":
    model = results[best_name]["model"].cpu().eval()
    # Save the weights first: if the export below fails, the training is not lost.
    torch.save(model.state_dict(), MODEL_DIR / "classifier.pt")
    dummy = torch.zeros(1, 1, 32, 32)
    onnx_kw = dict(input_names=["cutout"], output_names=["logits"],
                   dynamic_axes={"cutout": {0: "batch"}, "logits": {0: "batch"}},
                   opset_version=17)

    # Recent PyTorch defaults to the dynamo exporter, which needs `onnxscript`, and
    # even the classic exporter now needs `onnx`. Kaggle ships neither, so install
    # them first (Internet must be ON, as it already is for the archive query).
    import importlib.util
    import subprocess
    import sys
    need = [m for m in ("onnx", "onnxscript") if importlib.util.find_spec(m) is None]
    if need:
        print("installing", " ".join(need))
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *need], check=True)
        importlib.invalidate_caches()

    out_path = str(MODEL_DIR / "classifier.onnx")
    try:
        # Classic TorchScript exporter: simplest graph for a small CNN.
        torch.onnx.export(model, dummy, out_path, dynamo=False, **onnx_kw)
    except TypeError:
        torch.onnx.export(model, dummy, out_path, **onnx_kw)   # old PyTorch, no `dynamo`
    except Exception as exc:
        print("classic exporter failed (%s: %s); using the dynamo exporter"
              % (type(exc).__name__, exc))
        torch.onnx.export(model, dummy, out_path, dynamo=True, **onnx_kw)
    size = (MODEL_DIR / "classifier.onnx").stat().st_size
    print("wrote classifier.onnx (%.1f kB)" % (size / 1e3))

    card = {
        "architecture": best_name,
        "input": "32x32 float32, per-stamp median subtracted and MAD normalised, "
                 "clipped to [-10, 60]",
        "output": "6 logits in the order given by `classes`",
        "trained_on": "%d cutouts from %d nights" % (int(tr.sum()), len(train_s)),
        "beats_baseline_macro_f1_by": round(
            float(report[best_name]["macro_f1"] - base_f1), 4),
    }
    (MODEL_DIR / "model_card.json").write_text(json.dumps(card, indent=2))
    print("wrote model_card.json")

pd.DataFrame([h for r in results.values() for h in r["history"]]).to_csv(
    MODEL_DIR / "training_history.csv", index=False)
print("\nfiles in", MODEL_DIR)
for f in sorted(MODEL_DIR.iterdir()):
    print("  %-32s %8.1f kB" % (f.name, f.stat().st_size / 1e3))

# %% [markdown]
# ### Verify the export matches
#
# An ONNX file that disagrees with the PyTorch model it came from is worse than no
# model, because it fails silently. Check before shipping.

# %%
if (MODEL_DIR / "classifier.onnx").exists():
    try:
        import onnxruntime as ort
        sess_ort = ort.InferenceSession(str(MODEL_DIR / "classifier.onnx"),
                                        providers=["CPUExecutionProvider"])
        probe = Xn[te][:256]
        with torch.no_grad():
            torch_out = model(torch.from_numpy(probe).unsqueeze(1)).numpy()
        onnx_out = sess_ort.run(None, {"cutout": probe[:, None].astype(np.float32)})[0]
        agree = (torch_out.argmax(1) == onnx_out.argmax(1)).mean()
        print("ONNX vs PyTorch: max |diff| %.2e, predictions agree on %.1f%%"
              % (np.abs(torch_out - onnx_out).max(), 100 * agree))
        assert agree > 0.999, "ONNX export does not reproduce the PyTorch model"
        print("export verified")
    except ImportError:
        print("onnxruntime not installed here -- installing it to verify the export")
        import subprocess
        import sys
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "onnxruntime"],
                       check=False)
        print("re-run this cell to verify. Do NOT deploy an unverified export.")

# %% [markdown]
# ## 9 · YOLO (optional)
#
# A note on why this is optional rather than central.
#
# YOLO solves **detection**: find objects in a large image and draw boxes. We have
# already solved detection — classically, with sigma-clipping and connected
# components, which on a 650×500 frame of point sources is both faster and more
# accurate than a learned detector. Notebook 01 finds ~400 sources per frame in
# about 60 ms.
#
# What is genuinely hard here is **classification**: deciding whether a 3-pixel
# blob is a star, a hot pixel or a cosmic ray. That is what the CNN above does.
#
# So YOLO is included for comparison and completeness, not because it is the right
# tool. Reporting that honestly is worth more than adding it because it sounds
# impressive.

# %%
RUN_YOLO = bool(int(os.environ.get("RUN_YOLO", "0")))

if RUN_YOLO:
    try:
        from ultralytics import YOLO
        print("Exporting detections to YOLO format...")
        # Boxes come from the detection table: each source becomes a box sized by
        # its own footprint, with the class taken from the physics-derived label.
        print("See kaggle/src/yolo_export.py for the converter.")
        print("Train: yolo detect train data=yolo/data.yaml model=yolov8n.pt "
              "epochs=50 imgsz=672 device=0,1")
    except ImportError:
        print("ultralytics not installed. pip install ultralytics")
else:
    print("YOLO skipped. Set RUN_YOLO=1 to enable.")
    print()
    print("Justification for the README: classical detection already finds ~400")
    print("sources per frame in ~60 ms with no training data and no failure modes")
    print("we cannot inspect. The hard part of this problem is classification, and")
    print("that is where the model effort went.")

# %% [markdown]
# ## 10 · Hand-off
#
# Copy `model/` into the API's data directory:
#
# ```bash
# cp -r /kaggle/working/model backend/data/model
# ```
#
# Then `GET /api/model/info` reports the card, `GET /api/model/metrics` serves the
# per-class numbers, and `POST /api/classify` switches from its rule-based
# fallback to the trained network automatically — the API checks for
# `model/classifier.onnx` at request time.
#
# ### What this model can and cannot be trusted for
#
# It can separate stars, hot pixels and cosmic rays on nights it has never seen,
# and its per-class numbers above say how well.
#
# It **cannot** tell you whether a source is a planet host. It classifies pixels,
# not astrophysics. Nothing in this notebook constitutes evidence for or against a
# transit — that lives entirely in notebook 01's photometry.
