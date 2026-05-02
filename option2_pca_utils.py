"""option2_pca_utils — helper module for the trial-averaged PCA notebook.

The module is organized in four sections:

(a) Preprocessing and matrix construction — preprocess_responses,
    build_per_area_matrices.
(b) Per-area analysis — fit_pca, balanced silhouette + null,
    cross-validated logistic regression + null, population matching,
    run_area_pipeline.
(c) Plotting — per-area scatters / scree / null histograms, cross-area
    comparison plots.
(d) Cross-session aggregation — read per-session CSVs and plot.

Design constraints (see docs/specs/2026-05-02-pca-design.md):
- Data functions are pure: no plotting, no file writes.
- Plotting functions accept ``ax`` (or ``fig`` for Plotly) so plots compose.
- Every public function has a NumPy-style docstring with Args, Returns, Notes.
- No magic constants — anything tunable is a parameter with a sensible default.
- All randomness is seeded via the ``seed`` (or ``random_state``) parameter.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_score


# Shared visual conventions for stim-class colouring across all plotters.
# Okabe–Ito-inspired, colour-blind safe.
STIM_COLORS = {
    "Clip": "#0072B2",      # blue
    "Monet2": "#E69F00",    # orange
    "Trippy": "#009E73",    # green
}
STIM_MARKERS = {
    "Clip": "o",       # circle
    "Monet2": "^",     # triangle
    "Trippy": "s",     # square
}
STIM_ORDER = ("Clip", "Monet2", "Trippy")


# =============================================================================
# (a) Preprocessing and matrix construction
# =============================================================================


# (preprocessing functions go here)


# =============================================================================
# (b) Per-area analysis
# =============================================================================


# (analysis functions go here)


# =============================================================================
# (c) Plotting
# =============================================================================


# (plotters go here)


# =============================================================================
# (d) Cross-session aggregation
# =============================================================================


# (aggregation functions go here)
