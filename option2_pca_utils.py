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


def preprocess_responses(
    responses: np.ndarray,
    *,
    apply_log: bool = False,
) -> np.ndarray:
    """Apply the full-timeseries preprocessing pipeline (Stage A).

    Three steps in order:

    1. Optional ``log(1 + x)`` for variance stabilisation. Applied first
       because step 2 produces values that can be negative, which would
       break log.
    2. Per-neuron linear detrend on the full session timeseries — fits a
       line ``a + b * t`` to each neuron's trace against time index, subtracts.
       Removes the photobleaching drift detected in EDA (~45% across session).
    3. Per-neuron z-score using post-detrend mean and standard deviation,
       so every neuron contributes on the same scale.

    Args:
        responses: shape ``(n_neurons, total_timesteps)`` — the full neuron ×
            time matrix returned by ``microns_eda.load_session_responses``.
        apply_log: if ``True``, apply ``log1p`` before detrending. The
            primary pipeline runs with ``apply_log=False``; the log variant
            is run only as a sensitivity check (see Phase 7 of the notebook).

    Returns:
        Preprocessed array of the same shape as ``responses``. Pure: input
        is not modified.

    Notes:
        Detrending uses ``np.polyfit`` with degree 1 per neuron, vectorised
        via the numpy least-squares route (``np.linalg.lstsq``). Z-scoring
        guards against zero-std neurons (none expected after EDA's silent-
        neuron check, but defensive: zero-std rows are returned as zeros).
    """
    n_neurons, n_timesteps = responses.shape

    # Step 1: optional log transform.
    x = np.log1p(responses) if apply_log else responses.astype(np.float64, copy=True)

    # Step 2: per-neuron linear detrend.
    # Fit y = a + b * t for each neuron, then subtract.
    # We compute slope and intercept analytically with closed-form OLS to
    # avoid a Python-level loop over neurons.
    t = np.arange(n_timesteps, dtype=np.float64)
    t_mean = t.mean()
    t_centered = t - t_mean
    t_var = (t_centered * t_centered).sum()  # scalar
    x_mean = x.mean(axis=1, keepdims=True)
    # slope_i = sum((t - t_mean) * (x_i - x_i_mean)) / sum((t - t_mean) ** 2)
    slopes = ((x - x_mean) * t_centered).sum(axis=1) / t_var  # shape (n_neurons,)
    intercepts = x_mean.ravel() - slopes * t_mean
    fit = intercepts[:, None] + slopes[:, None] * t[None, :]
    x_detrended = x - fit

    # Step 3: per-neuron z-score.
    mu = x_detrended.mean(axis=1, keepdims=True)
    sigma = x_detrended.std(axis=1, keepdims=True, ddof=0)
    # Defensive: avoid division by zero for any constant neurons.
    sigma_safe = np.where(sigma > 0, sigma, 1.0)
    x_z = (x_detrended - mu) / sigma_safe
    # Restore zero rows for zero-std neurons (rather than NaN or huge values).
    x_z = np.where(sigma > 0, x_z, 0.0)

    return x_z


def build_per_area_matrices(
    responses_preprocessed: np.ndarray,
    trial_boundaries: np.ndarray,
    clean_trial_indices: np.ndarray,
    brain_areas: np.ndarray,
    *,
    n_frames: int = 75,
) -> dict[str, np.ndarray]:
    """Build trial-averaged matrices per cortical area (Stage B).

    Three operations applied in order to the preprocessed response matrix:

    1. Truncate each trial to its first ``n_frames`` timesteps (Clip is
       already 75 frames; Monet2 and Trippy are 113 and need truncation).
    2. Apply the trial mask: keep only trials in ``clean_trial_indices``
       (the running-outlier-filtered set from EDA).
    3. Per cortical area: index out the columns belonging to that area, and
       for each surviving trial average across the time axis. The result
       is one row per trial.

    Args:
        responses_preprocessed: shape ``(n_neurons, total_timesteps)``,
            output of :func:`preprocess_responses`.
        trial_boundaries: shape ``(n_trials + 1,)``, cumulative timestep
            counts as returned by ``microns_eda.load_session_responses``;
            ``trial_boundaries[i]`` is the start timestep of trial ``i``,
            ``trial_boundaries[i+1]`` is the end (exclusive).
        clean_trial_indices: shape ``(n_clean_trials,)``, indices of trials
            to keep, in original trial order.
        brain_areas: shape ``(n_neurons,)`` of bytes-or-string area labels
            per neuron, as returned by ``microns_eda.get_session_meta``.
            Decoded inside this function if bytes.
        n_frames: number of leading frames to keep per trial. Defaults to
            75, matching the Clip trial length.

    Returns:
        Dict mapping area name to trial-averaged matrix of shape
        ``(n_clean_trials, n_neurons_in_area)``. Areas with zero neurons
        are silently omitted from the dict.

    Notes:
        Areas that the spec expects: ``{"V1", "AL", "LM", "RL"}``. The
        function does not assume which areas exist — it iterates over the
        unique values of ``brain_areas``, so a session with only V1 still
        works.
    """
    # Decode area labels if they came in as bytes.
    if brain_areas.dtype.kind in ("S", "O") and len(brain_areas) > 0:
        sample = brain_areas[0]
        if isinstance(sample, bytes):
            brain_areas_str = np.array([s.decode("utf-8") for s in brain_areas])
        else:
            brain_areas_str = brain_areas.astype(str)
    else:
        brain_areas_str = brain_areas.astype(str)

    n_clean = len(clean_trial_indices)

    # For each clean trial, slice the first n_frames timesteps and average.
    # We build a (n_clean_trials, n_neurons) matrix once, then split per area.
    averaged = np.empty(
        (n_clean, responses_preprocessed.shape[0]), dtype=np.float64
    )
    for out_idx, trial_idx in enumerate(clean_trial_indices):
        start = int(trial_boundaries[trial_idx])
        end = start + n_frames  # truncate
        window = responses_preprocessed[:, start:end]  # (n_neurons, n_frames)
        averaged[out_idx, :] = window.mean(axis=1)

    # Now split by area.
    per_area: dict[str, np.ndarray] = {}
    for area in np.unique(brain_areas_str):
        cols = np.where(brain_areas_str == area)[0]
        if len(cols) == 0:
            continue
        per_area[area] = averaged[:, cols]

    return per_area


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
