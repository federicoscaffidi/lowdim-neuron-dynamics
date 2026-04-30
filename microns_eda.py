"""microns_eda — helper module for the MICrONS exploratory data analysis notebook.

The module is organized in four sections:

(a) Loading       — wrappers around MicronsFunctionalReader plus h5py fallback.
(b) Aggregation   — per-session and cross-session summary statistics.
(c) Analysis      — deep-dive diagnostics (per-neuron stats, correlation, drift).
(d) Plotting      — thin matplotlib wrappers; all accept an optional `ax`.

Design constraints (see docs/specs/2026-04-30-microns-eda-design.md):
- Data functions are pure: no plotting, no file writes.
- Plotting functions accept `ax` so plots compose.
- Every public function has a NumPy-style docstring with Args, Returns, Notes.
- No magic constants — anything tunable is a parameter with a sensible default.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# microns_datacleaner is imported lazily inside functions that use it,
# so the module can still be imported in environments without the library
# (e.g. for type checking or partial use).


# =============================================================================
# (a) Loading
# =============================================================================


# (loaders go here)


# =============================================================================
# (b) Aggregation (cross-session)
# =============================================================================


# (aggregators go here)


# =============================================================================
# (c) Analysis (deep-dive diagnostics)
# =============================================================================


# (analyzers go here)


# =============================================================================
# (d) Plotting
# =============================================================================


# (plotters go here)
