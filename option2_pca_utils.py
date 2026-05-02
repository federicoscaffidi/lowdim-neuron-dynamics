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
        Detrending uses an analytical closed-form OLS (slope =
        sum((t - t_mean) * (x - x_mean)) / sum((t - t_mean)^2)) computed
        in a fully vectorised form across all neurons at once, avoiding
        a Python loop. Z-scoring guards against zero-std neurons (none
        expected after EDA's silent-neuron check, but defensive: zero-std
        rows are returned as zeros).
    """
    _, n_timesteps = responses.shape

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


def fit_pca(
    X: np.ndarray,
    *,
    n_components: int = 10,
    random_state: int = 42,
) -> tuple[PCA, np.ndarray]:
    """Fit a sklearn PCA and return both the estimator and the projection.

    Args:
        X: shape ``(n_samples, n_features)`` — the trial-averaged matrix
            for one cortical area; rows are trials, columns are neurons.
        n_components: number of PCs to retain. Default 10 (the analysis
            visualises top 3 and reports variance explained on top 20 in
            the scree plot, but having 10 retained is a sensible compromise
            between flexibility and memory).
        random_state: seed for the (deterministic-with-default-solver) PCA;
            propagated to ``sklearn.decomposition.PCA``.

    Returns:
        Tuple ``(pca, X_pcs)``:

        - ``pca``: fitted ``sklearn.decomposition.PCA`` instance. Use
          ``pca.explained_variance_ratio_`` for the scree plot.
        - ``X_pcs``: shape ``(n_samples, n_components)``, the projection.

    Notes:
        ``whiten=False``: the EDA showed neuron correlations were low
        enough that explicit whitening would distort the variance ranking
        we want to interpret. Default solver (``"auto"``) picks ``"full"``
        for our matrix sizes, which is deterministic.
    """
    pca = PCA(n_components=n_components, whiten=False, random_state=random_state)
    X_pcs = pca.fit_transform(X)
    return pca, X_pcs


def silhouette_balanced(
    X_pcs: np.ndarray,
    labels: np.ndarray,
    *,
    n_replicates: int = 20,
    seed: int = 42,
) -> dict:
    """Class-balanced silhouette score, averaged over random subsamples.

    Silhouette score is biased by class imbalance: a class with many trials
    contributes many same-class neighbour distances, distorting the score.
    To remove this, we draw ``n_replicates`` balanced subsamples — each
    contains ``min(class_counts)`` trials per class — compute silhouette
    on each, and average.

    The function is generic over the shape of ``X_pcs``; the caller decides
    how many PC dimensions to pass in. The intended use in this analysis is
    top-3 PCs.

    Args:
        X_pcs: shape ``(n_samples, k)`` — coordinates in some space (PC,
            full feature, etc.).
        labels: shape ``(n_samples,)`` — class labels (strings or ints).
        n_replicates: number of balanced subsamples to average over.
        seed: master seed; per-replicate seeds derive deterministically.

    Returns:
        Dict with keys:

        - ``observed`` (float): mean silhouette across replicates.
        - ``per_replicate`` (np.ndarray of float): shape ``(n_replicates,)``,
          the silhouette computed on each subsample.
        - ``n_per_class`` (int): the per-class subsample size used.

    Notes:
        Uses ``sklearn.metrics.silhouette_score`` with the default
        Euclidean metric. If a class has fewer trials than required by
        ``n_per_class``, the function raises ``ValueError`` (caller bug,
        e.g. a session with no Trippy trials).
    """
    rng = np.random.default_rng(seed)
    label_arr = np.asarray(labels)
    classes, counts = np.unique(label_arr, return_counts=True)
    n_per_class = int(counts.min())
    if n_per_class < 2:
        raise ValueError(
            f"Cannot compute silhouette: minority class has {n_per_class} "
            f"trials (need at least 2)."
        )

    # Pre-compute per-class index pools.
    class_pools = {c: np.where(label_arr == c)[0] for c in classes}

    per_replicate = np.empty(n_replicates, dtype=np.float64)
    for r in range(n_replicates):
        # Deterministic per-replicate sub-seed.
        sub_rng = np.random.default_rng(rng.integers(0, 2**31 - 1))
        idx_chunks = [
            sub_rng.choice(pool, size=n_per_class, replace=False)
            for pool in class_pools.values()
        ]
        sample_idx = np.concatenate(idx_chunks)
        per_replicate[r] = silhouette_score(
            X_pcs[sample_idx], label_arr[sample_idx], metric="euclidean"
        )

    return {
        "observed": float(per_replicate.mean()),
        "per_replicate": per_replicate,
        "n_per_class": n_per_class,
    }


def silhouette_balanced_null(
    X_pcs: np.ndarray,
    labels: np.ndarray,
    *,
    n_replicates: int = 20,
    n_shuffles: int = 100,
    seed: int = 42,
) -> np.ndarray:
    """Null distribution of the balanced silhouette under label permutation.

    For each of ``n_shuffles`` random permutations of ``labels``, computes the
    balanced silhouette (averaged over ``n_replicates`` subsamples, identical
    scheme to :func:`silhouette_balanced`). Returns an array directly
    comparable to the observed value.

    Args:
        X_pcs: shape ``(n_samples, k)``.
        labels: shape ``(n_samples,)``.
        n_replicates: subsamples per shuffle.
        n_shuffles: number of label permutations.
        seed: master seed.

    Returns:
        ``(n_shuffles,)`` array of null silhouettes.

    Notes:
        Each shuffle is a full random permutation of ``labels`` (preserves
        class marginals on average across shuffles, exactly within each
        shuffle). The empirical p-value compares the observed silhouette to
        this distribution: ``p = (1 + (null >= observed).sum()) / (1 + n_shuffles)``
        (the ``+1`` pseudocount avoids ``p = 0``).
    """
    rng = np.random.default_rng(seed)
    label_arr = np.asarray(labels)

    null = np.empty(n_shuffles, dtype=np.float64)
    for s in range(n_shuffles):
        shuffled = rng.permutation(label_arr)
        # Use a sub-seed derived from the master RNG for the inner balance loop.
        sub_seed = int(rng.integers(0, 2**31 - 1))
        result = silhouette_balanced(
            X_pcs, shuffled, n_replicates=n_replicates, seed=sub_seed
        )
        null[s] = result["observed"]
    return null


def classify_cv(
    X: np.ndarray,
    labels: np.ndarray,
    *,
    n_folds: int = 5,
    seed: int = 42,
) -> dict:
    """Cross-validated multinomial logistic regression accuracy.

    Trains a multinomial logistic regression on the full feature matrix
    (no PCA), with stratified K-fold cross-validation. The full matrix is
    used (rather than the top-K PCs) so this metric is independent of
    PCA's variance-ranking — it answers "are stimuli linearly separable
    in neural state space?" without conditioning on top-variance directions.

    Args:
        X: shape ``(n_samples, n_features)`` — the trial-averaged matrix
            for one cortical area.
        labels: shape ``(n_samples,)`` — class labels.
        n_folds: number of stratified folds. Default 5.
        seed: random_state for the splitter and the classifier.

    Returns:
        Dict with keys:

        - ``observed`` (float): mean accuracy across folds.
        - ``per_fold`` (np.ndarray): shape ``(n_folds,)``.
        - ``chance`` (float): majority-class baseline accuracy (the
          accuracy of a "predict the most common class" classifier).
          Used as the reference for the "is the model better than chance?"
          comparison.

    Notes:
        The classifier is ``LogisticRegression(solver="lbfgs", max_iter=1000,
        C=1.0, random_state=seed)``. With ``solver="lbfgs"`` and multi-class
        data, sklearn defaults to multinomial logistic regression (the
        behavior of the deprecated ``multi_class="multinomial"`` argument).
        L2 regularisation at default strength is appropriate when the feature
        matrix has more columns than rows (e.g., V1 has 5485 features and
        only 453 samples).
    """
    label_arr = np.asarray(labels)
    classes, counts = np.unique(label_arr, return_counts=True)
    chance = float(counts.max() / counts.sum())

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    clf = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        C=1.0,
        random_state=seed,
    )
    scores = cross_val_score(clf, X, label_arr, cv=skf, scoring="accuracy")
    return {
        "observed": float(scores.mean()),
        "per_fold": scores,
        "chance": chance,
    }


def classify_cv_null(
    X: np.ndarray,
    labels: np.ndarray,
    *,
    n_folds: int = 5,
    n_shuffles: int = 100,
    seed: int = 42,
) -> np.ndarray:
    """Null distribution of CV accuracy under label permutation.

    For each of ``n_shuffles`` random permutations of ``labels``, computes
    the cross-validated logistic regression accuracy with the same scheme
    as :func:`classify_cv`. Returns an array directly comparable to the
    observed value.

    Args:
        X: shape ``(n_samples, n_features)``.
        labels: shape ``(n_samples,)``.
        n_folds: stratified folds per shuffle.
        n_shuffles: number of label permutations.
        seed: master seed.

    Returns:
        ``(n_shuffles,)`` array of null accuracies.

    Notes:
        With label permutation, ``StratifiedKFold`` still stratifies by
        the (shuffled) labels, so each shuffle is a balanced 5-fold CV
        on randomly-assigned labels. Expected null mean ≈ chance.
        Empirical p-value: ``p = (1 + (null >= observed).sum()) / (1 + n_shuffles)``.
    """
    rng = np.random.default_rng(seed)
    label_arr = np.asarray(labels)

    null = np.empty(n_shuffles, dtype=np.float64)
    for s in range(n_shuffles):
        shuffled = rng.permutation(label_arr)
        sub_seed = int(rng.integers(0, 2**31 - 1))
        result = classify_cv(X, shuffled, n_folds=n_folds, seed=sub_seed)
        null[s] = result["observed"]
    return null


def match_population_size(
    X: np.ndarray,
    n_match: int,
    *,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Random column subsample to a target neuron count.

    Used by Phase 6 of the notebook to confirm that cross-area differences
    in stimulus separability are not driven solely by population size:
    V1 has ~13× more neurons than AL, and PCA on a larger population has
    more variance to distribute, which can inflate downstream metrics.

    Args:
        X: shape ``(n_samples, n_neurons)``.
        n_match: number of columns to retain (e.g. AL's neuron count).
        seed: random seed for the subsample.

    Returns:
        Tuple ``(X_matched, col_indices)``:

        - ``X_matched``: shape ``(n_samples, n_match)``.
        - ``col_indices``: shape ``(n_match,)``, sorted indices into the
          original column axis (kept so the caller can record / log
          which neurons were sampled).

    Raises:
        ValueError if ``n_match > X.shape[1]``.
    """
    n_neurons = X.shape[1]
    if n_match > n_neurons:
        raise ValueError(
            f"Cannot match to {n_match} columns: X only has {n_neurons}."
        )
    rng = np.random.default_rng(seed)
    col_indices = np.sort(rng.choice(n_neurons, size=n_match, replace=False))
    return X[:, col_indices], col_indices


def run_area_pipeline(
    X: np.ndarray,
    labels: np.ndarray,
    area_name: str,
    *,
    n_components: int = 10,
    n_balance_replicates: int = 20,
    n_shuffles: int = 100,
    n_folds_cv: int = 5,
    seed: int = 42,
) -> dict:
    """Run the full per-area analysis pipeline in one call.

    Steps:

    1. Fit PCA, keep top ``n_components`` directions.
    2. Compute balanced silhouette on top-3 PCs + null distribution.
    3. Compute 5-fold CV multinomial logistic regression accuracy on the
       full matrix + null distribution.

    Args:
        X: shape ``(n_samples, n_neurons)`` — the trial-averaged matrix
            for the area.
        labels: shape ``(n_samples,)`` — stim class per trial.
        area_name: label used for logging / annotation only.
        n_components: PCs to retain.
        n_balance_replicates: passed to silhouette_balanced.
        n_shuffles: passed to both null functions.
        n_folds_cv: stratified CV folds.
        seed: master seed; sub-seeds derived deterministically.

    Returns:
        Dict with keys ``"area_name"``, ``"n_neurons"``, ``"pca"``,
        ``"X_pcs"``, ``"silhouette"`` (the ``silhouette_balanced`` dict
        plus ``"null"`` with the shuffle distribution and ``"empirical_p"``
        the empirical p-value), ``"classifier"`` (analogous structure for
        CV accuracy).

    Notes:
        The empirical p-value uses the standard pseudocount-1 formula:
        ``p = (1 + (null >= observed).sum()) / (1 + n_shuffles)``.
    """
    # Sub-seeds for each component, so all RNGs are deterministic from `seed`.
    sub_seeds = np.random.default_rng(seed).integers(0, 2**31 - 1, size=5)

    pca, X_pcs = fit_pca(
        X, n_components=n_components, random_state=int(sub_seeds[0])
    )

    sil = silhouette_balanced(
        X_pcs[:, :3], labels,
        n_replicates=n_balance_replicates, seed=int(sub_seeds[1]),
    )
    sil_null = silhouette_balanced_null(
        X_pcs[:, :3], labels,
        n_replicates=n_balance_replicates, n_shuffles=n_shuffles,
        seed=int(sub_seeds[2]),
    )
    sil_p = float((1 + (sil_null >= sil["observed"]).sum()) / (1 + n_shuffles))
    sil["null"] = sil_null
    sil["empirical_p"] = sil_p

    clf = classify_cv(
        X, labels, n_folds=n_folds_cv, seed=int(sub_seeds[3]),
    )
    clf_null = classify_cv_null(
        X, labels, n_folds=n_folds_cv, n_shuffles=n_shuffles,
        seed=int(sub_seeds[4]),
    )
    clf_p = float((1 + (clf_null >= clf["observed"]).sum()) / (1 + n_shuffles))
    clf["null"] = clf_null
    clf["empirical_p"] = clf_p

    return {
        "area_name": area_name,
        "n_neurons": int(X.shape[1]),
        "pca": pca,
        "X_pcs": X_pcs,
        "silhouette": sil,
        "classifier": clf,
    }


# =============================================================================
# (c) Plotting
# =============================================================================


# (plotters go here)


# =============================================================================
# (d) Cross-session aggregation
# =============================================================================


# (aggregation functions go here)
