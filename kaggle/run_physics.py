#!/usr/bin/env python
"""
run_physics.py -- re-run only section 13b (planet physics) of the pipeline.

Section 13b reads nothing but finished results CSVs, so it can be refreshed
without re-running photometry. The code is not duplicated here: this script
cuts the section out of src/nb01_pipeline.py and executes it, so the notebook
stays the single source of truth.

Usage:  python run_physics.py [results_dir]      (default: out/results)
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
src = (HERE / "src" / "nb01_pipeline.py").read_text(encoding="utf-8")
start = src.index("# %% [markdown]\n# ## 13b")
end = src.index("# %% [markdown]\n# ## 14 ")
warnings.filterwarnings("ignore")

RESULTS = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "out" / "results"
exec(compile(src[start:end], "nb01_pipeline.py:13b", "exec"),
     {"np": np, "pd": pd, "RESULTS": RESULTS, "__name__": "__main__"})
