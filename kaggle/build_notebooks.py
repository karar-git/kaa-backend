#!/usr/bin/env python
"""
build_notebooks.py -- turn the `# %%` scripts in src/ into .ipynb files.

Keeping notebooks as plain Python in src/ means they diff cleanly in git, can be
linted, and can be imported for testing. This script is the one-way build step.

Cell markers (jupytext "percent" format):
    # %%              -> code cell
    # %% [markdown]   -> markdown cell (leading "# " stripped from each line)

Usage:  python build_notebooks.py
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "src"
NB = HERE / "notebooks"

KERNEL = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
}


def split_cells(text: str):
    cells, kind, buf = [], "code", []

    def flush():
        if not buf:
            return
        body = "\n".join(buf).strip("\n")
        if body.strip():
            cells.append((kind, body))

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# %%"):
            flush()
            buf = []
            kind = "markdown" if "[markdown]" in stripped else "code"
            continue
        buf.append(line)
    flush()
    return cells


def to_source(kind: str, body: str) -> list[str]:
    if kind == "markdown":
        lines = []
        for ln in body.splitlines():
            lines.append(ln[2:] if ln.startswith("# ") else (ln[1:] if ln == "#" else ln))
        body = "\n".join(lines)
    out = body.splitlines(keepends=True)
    if out and not out[-1].endswith("\n"):
        pass
    return out


def build(py_path: Path, ipynb_path: Path) -> None:
    cells = split_cells(py_path.read_text(encoding="utf-8"))
    nb = {
        "cells": [
            {
                "cell_type": kind,
                "metadata": {},
                "source": to_source(kind, body),
                **({"outputs": [], "execution_count": None} if kind == "code" else {}),
            }
            for kind, body in cells
        ],
        "metadata": KERNEL,
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    ipynb_path.parent.mkdir(parents=True, exist_ok=True)
    ipynb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    n_code = sum(1 for k, _ in cells if k == "code")
    print("%-28s -> %-34s %2d cells (%d code, %d md)"
          % (py_path.name, ipynb_path.name, len(cells), n_code, len(cells) - n_code))


# Markdown inserted where the two scripts are joined in the combined build, so a
# reader can see exactly where the analysis ends and the training begins.
BRIDGE = """\
# %% [markdown]
# ---
#
# # Part 2 — Training the source classifier
#
# Everything above produced measurements. From here we train the model behind
# `POST /api/classify` on the cutouts section 8 just wrote.
#
# This is the same code as the standalone `02_train.ipynb`; it is included here
# so the whole project runs in one session, with no dataset-chaining step between
# the two halves. `cutouts.npz` is already in `/kaggle/working/results/`, so the
# lookup below finds it without anything being attached.
#
# **Set the accelerator to T4 x2 before running this part.** Part 1 is pure NumPy
# and ignores the GPU; part 2 trains one network on each.
"""


# The first script introduces itself as "notebook 01 of two". In the combined
# build that is simply untrue, so the few sentences that say it are rewritten.
HEADER_FIXUPS = [
    ("# # ExoTransit Lab — 01 · Calibration, Detection, Auto-Labelling, Photometry",
     "# # ExoTransit Lab — Calibration, Photometry, Transit Search, Classifier\n"
     "#\n"
     "# *Part 1 measures. Part 2 (further down) trains. One session, top to bottom.*"),
    ("# dashboard and the ML notebook consume. It is the *only* place science happens —",
     "# dashboard and the classifier in part 2 consume. Part 1 is the *only* place\n"
     "# science happens —"),
    ("# | 8 | 32×32 cutout export for the training notebook | `cutouts.npz`, `cutouts_meta.csv` |",
     "# | 8 | 32×32 cutout export, consumed by part 2 | `cutouts.npz`, `cutouts_meta.csv` |"),
]


def combine(parts: list[Path], out_path: Path) -> None:
    """Concatenate the scripts into one notebook.

    Safe because the second script reassigns every name it shares with the first
    (ON_KAGGLE, OUT, n, rows, share, t0) before reading it -- checked, not assumed.
    """
    text = parts[0].read_text(encoding="utf-8")
    for old, new in HEADER_FIXUPS:
        if old not in text:
            raise SystemExit("combine: header fixup no longer matches:\n  " + old[:80])
        text = text.replace(old, new, 1)
    for p in parts[1:]:
        text += "\n\n" + BRIDGE + "\n" + p.read_text(encoding="utf-8")
    tmp = SRC / "_combined_generated.py"
    tmp.write_text(text, encoding="utf-8")
    try:
        compile(text, str(tmp), "exec")      # catch a bad join before shipping it
        build(tmp, out_path)
    finally:
        tmp.unlink(missing_ok=True)


def main() -> None:
    scripts = sorted(SRC.glob("nb*.py"))
    if not scripts:
        raise SystemExit("no nb*.py found in " + str(SRC))
    for p in scripts:
        build(p, NB / (p.stem.replace("nb01_", "01_").replace("nb02_", "02_") + ".ipynb"))
    combine(scripts, NB / "exotransit_lab_full.ipynb")


if __name__ == "__main__":
    main()
