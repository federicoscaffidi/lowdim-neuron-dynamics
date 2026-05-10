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

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_score


# Shared visual conventions for stim-class colouring across all plotters.
# Saturated jewel tones from neuro_palette.STIMULUS — designed for the
# presentation deck (works on both white and dark backgrounds).
STIM_COLORS = {
    "Clip":   "#60A5FA",    # sky blue
    "Monet2": "#F472B6",    # rose
    "Trippy": "#34D399",    # emerald
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
        - ``converged`` (bool): ``True`` if no fold raised a
          ``ConvergenceWarning``; ``False`` if any did.

    Notes:
        The classifier is ``LogisticRegression(solver="lbfgs", max_iter=5000,
        C=1.0, random_state=seed)``. With ``solver="lbfgs"`` and multi-class
        data, sklearn defaults to multinomial logistic regression (the
        behavior of the deprecated ``multi_class="multinomial"`` argument).
        L2 regularisation at default strength is appropriate when the feature
        matrix has more columns than rows (e.g., V1 has 5485 features and
        only 453 samples). ``max_iter`` is set to 5000 because the underdetermined
        p >> n case can need more iterations than the sklearn default. The
        returned dict includes ``converged`` (bool) — ``False`` if any fold
        raised ``sklearn.exceptions.ConvergenceWarning``; useful for downstream
        interpretation.
    """
    label_arr = np.asarray(labels)
    classes, counts = np.unique(label_arr, return_counts=True)
    chance = float(counts.max() / counts.sum())

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    clf = LogisticRegression(
        solver="lbfgs",
        max_iter=5000,
        C=1.0,
        random_state=seed,
    )
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", ConvergenceWarning)
        scores = cross_val_score(clf, X, label_arr, cv=skf, scoring="accuracy")
    converged = not any(
        issubclass(warning.category, ConvergenceWarning) for warning in captured
    )
    return {
        "observed": float(scores.mean()),
        "per_fold": scores,
        "chance": chance,
        "converged": converged,
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


def plot_pca_2d(
    X_pcs: np.ndarray,
    labels: np.ndarray,
    area_name: str,
    pca: PCA,
    *,
    ax: plt.Axes | None = None,
    show_legend: bool = True,
) -> plt.Axes:
    """2-D scatter of trials in the PC1–PC2 plane, coloured by stim class.

    Distinct markers per class (Clip=circle, Monet2=triangle, Trippy=square)
    so the minority classes (38 trials each in `7_5`) remain readable next
    to Clip's 377 trials. Class centroids are overlaid as crosses.

    Args:
        X_pcs: shape ``(n_trials, >=2)`` — needs at least PC1 and PC2.
        labels: shape ``(n_trials,)`` — stim class per trial.
        area_name: used in title.
        pca: fitted PCA, used to extract variance-explained for axis labels.
        ax: matplotlib axes to draw on; created if None.
        show_legend: if True (default), draw a per-panel legend. Set to
            False when composing many panels at the figure level (e.g. the
            2×2 cross-area grid in Task 6) and add a single shared legend
            at the figure level instead.

    Returns:
        The axes drawn on.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    label_arr = np.asarray(labels)
    for stim in STIM_ORDER:
        mask = label_arr == stim
        if not mask.any():
            continue
        # Larger markers + black edge for minority classes; the visual is
        # otherwise dominated by Clip.
        size = 25 if stim == "Clip" else 60
        ax.scatter(
            X_pcs[mask, 0], X_pcs[mask, 1],
            c=STIM_COLORS[stim], marker=STIM_MARKERS[stim],
            s=size, alpha=0.7, edgecolor="black" if stim != "Clip" else "none",
            linewidth=0.5, label=f"{stim} (n={mask.sum()})",
        )
        # Centroid as a large cross.
        cx = X_pcs[mask, 0].mean()
        cy = X_pcs[mask, 1].mean()
        ax.scatter(
            cx, cy, c=STIM_COLORS[stim], marker="x",
            s=200, linewidths=3, zorder=5,
        )

    pc1_var = 100 * pca.explained_variance_ratio_[0]
    pc2_var = 100 * pca.explained_variance_ratio_[1]
    ax.set_xlabel(f"PC1 ({pc1_var:.1f}% var.)")
    ax.set_ylabel(f"PC2 ({pc2_var:.1f}% var.)")
    ax.set_title(f"{area_name} — trials in PC1–PC2")
    if show_legend:
        ax.legend(loc="best", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    return ax


def plot_pca_3d_plotly(
    X_pcs: np.ndarray,
    labels: np.ndarray,
    area_name: str,
    pca: PCA,
):
    """Interactive 3-D scatter (PC1 / PC2 / PC3) for the oral presentation.

    Uses ``plotly.graph_objects`` directly so we can control marker shape
    and size per class explicitly (``plotly.express`` does not expose these
    cleanly when colour and symbol are both categorical).

    Args:
        X_pcs: shape ``(n_trials, >=3)``.
        labels: shape ``(n_trials,)``.
        area_name: used in title.
        pca: fitted PCA, used to extract variance-explained for axis labels.

    Returns:
        ``plotly.graph_objects.Figure``. Caller saves with ``fig.write_html(path)``.
    """
    import plotly.graph_objects as go

    label_arr = np.asarray(labels)
    fig = go.Figure()

    # Plotly-compatible marker symbols (circle / triangle-up / square equivalents).
    plotly_symbols = {"Clip": "circle", "Monet2": "diamond", "Trippy": "square"}

    for stim in STIM_ORDER:
        mask = label_arr == stim
        if not mask.any():
            continue
        size = 4 if stim == "Clip" else 7
        fig.add_trace(go.Scatter3d(
            x=X_pcs[mask, 0], y=X_pcs[mask, 1], z=X_pcs[mask, 2],
            mode="markers",
            marker=dict(
                size=size,
                color=STIM_COLORS[stim],
                symbol=plotly_symbols[stim],
                opacity=0.75,
                line=dict(width=0.5, color="black") if stim != "Clip" else dict(width=0),
            ),
            name=f"{stim} (n={int(mask.sum())})",
        ))

    pc_var = 100 * pca.explained_variance_ratio_[:3]
    fig.update_layout(
        title=f"{area_name} — trials in PC1–PC2–PC3",
        scene=dict(
            xaxis_title=f"PC1 ({pc_var[0]:.1f}% var.)",
            yaxis_title=f"PC2 ({pc_var[1]:.1f}% var.)",
            zaxis_title=f"PC3 ({pc_var[2]:.1f}% var.)",
        ),
        legend=dict(itemsizing="constant"),
        width=800, height=650,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def plot_scree(
    pca: PCA,
    area_name: str,
    *,
    n_show: int = 20,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Bar plot of per-PC variance ratios (elbow / scree plot).

    Shows how much variance each principal component captures individually.
    The "elbow" is the visual point where the bars stop dropping steeply —
    a common heuristic for choosing how many PCs to retain. For the
    cumulative trace, see :func:`plot_cumulative_variance`.

    Args:
        pca: fitted PCA. Must have at least ``min(n_show, n_components)`` PCs.
        area_name: used in title.
        n_show: number of PCs to display. If the PCA has fewer components
            than this, all are shown.
        ax: matplotlib axes; created if None.

    Returns:
        The axes drawn on.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    n_show = min(n_show, len(pca.explained_variance_ratio_))
    var_ratios = pca.explained_variance_ratio_[:n_show]
    indices = np.arange(1, n_show + 1)

    ax.bar(indices, var_ratios * 100,
           color="steelblue", alpha=0.85, edgecolor="black", linewidth=0.5)

    # Top-3 reference (the dimensions used by silhouette analysis).
    top3 = var_ratios[:3].sum() * 100 if n_show >= 3 else var_ratios.sum() * 100
    ax.axvline(3, color="grey", linestyle="--", alpha=0.5)
    # Use axes-fraction coordinates so the label is robust to varying
    # scree shapes (e.g. when PC1 dominates and would overlap the bar).
    ax.text(0.18, 0.95, f"top 3 = {top3:.1f}%",
            transform=ax.transAxes, color="grey", fontsize=9, va="top")

    ax.set_xlabel("Principal component")
    ax.set_ylabel("Variance explained (%)")
    ax.set_title(f"{area_name} — variance explained per PC (top {n_show})")
    ax.set_xticks(indices)
    ax.grid(True, alpha=0.3, axis="y")
    return ax


def plot_cumulative_variance(
    pca: PCA,
    area_name: str,
    *,
    n_show: int | None = None,
    thresholds: tuple[int, ...] = (50, 80, 90, 95),
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Cumulative variance explained as the number of PCs grows.

    Answers the question "can we go to a lower-dimensional representation?"
    by showing how many PCs are needed to capture standard fractions of
    total variance (50, 80, 90, 95% by default). The threshold-to-PC
    mapping is rendered as both an inset table (for direct reading) and
    as marked points on the curve (for visual reference).

    Args:
        pca: fitted PCA.
        area_name: used in title.
        n_show: number of PCs to display. ``None`` (default) shows all
            available PCs — needed to see the curve asymptote at 100%.
        thresholds: variance thresholds (in %) to mark on the plot.
        ax: matplotlib axes; created if None.

    Returns:
        The axes drawn on.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))

    available = len(pca.explained_variance_ratio_)
    if n_show is None:
        n_show = available
    n_show = min(n_show, available)
    var_ratios = pca.explained_variance_ratio_[:n_show]
    cumulative = np.cumsum(var_ratios) * 100
    indices = np.arange(1, n_show + 1)

    # Clean curve — no per-point markers (they get noisy at large n_show).
    ax.plot(indices, cumulative, "-",
            color="darkred", linewidth=2.2, zorder=3)

    # For each threshold: faint horizontal reference line + marker at the
    # crossing point. Annotation goes in the inset table below, not on the
    # curve, to keep the plot readable when n_show is large.
    threshold_to_pc: list[tuple[int, int | None]] = []
    for thr in thresholds:
        crosses = np.where(cumulative >= thr)[0]
        pc_idx = (crosses[0] + 1) if len(crosses) else None
        threshold_to_pc.append((thr, pc_idx))
        ax.axhline(thr, color="grey", linestyle=":", alpha=0.45,
                   linewidth=1, zorder=1)
        if pc_idx is not None:
            ax.plot(pc_idx, thr, "o",
                    color="darkred", markersize=9, zorder=5,
                    markeredgecolor="white", markeredgewidth=1.5)

    ax.set_xlabel("Principal component (#)")
    ax.set_ylabel("Cumulative variance explained (%)")
    ax.set_title(f"{area_name} — cumulative variance ({n_show} components)")

    # XTicks: anchor at 1, n_show, and the threshold-crossing PCs.
    # Then add round multiples of 100 only if they don't crowd a threshold tick.
    threshold_xticks = [pc for _, pc in threshold_to_pc if pc is not None]
    anchor_ticks = sorted({1, n_show, *threshold_xticks})
    min_gap = max(1, n_show // 25)
    final_ticks = list(anchor_ticks)
    if n_show >= 100:
        for candidate in range(100, n_show, 100):
            if min(abs(candidate - t) for t in final_ticks) >= min_gap:
                final_ticks.append(candidate)
    ax.set_xticks(sorted(final_ticks))

    ax.set_xlim(0, n_show * 1.02)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)

    # Inset table — directly answers "how many PCs for X% variance".
    table_lines = ["threshold → # PCs"]
    for thr, pc_idx in threshold_to_pc:
        if pc_idx is None:
            table_lines.append(f"  {thr:3d}%  → not reached")
        else:
            table_lines.append(f"  {thr:3d}%  → PC {pc_idx}")
    ax.text(
        0.02, 0.97, "\n".join(table_lines),
        transform=ax.transAxes,
        fontsize=9, family="monospace",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.45",
                  facecolor="white", edgecolor="grey", alpha=0.92),
    )
    return ax


def _plot_metric_with_null(
    observed: float,
    null_distribution: np.ndarray,
    area_name: str,
    metric_label: str,
    *,
    chance: float | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Internal: histogram of a null distribution with the observed value overlaid."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    ax.hist(null_distribution, bins=20, color="lightgrey",
            edgecolor="black", linewidth=0.5, label="null (shuffled labels)")
    ax.axvline(observed, color="darkred", linewidth=2, label=f"observed = {observed:.3f}")
    if chance is not None:
        ax.axvline(chance, color="grey", linestyle="--", linewidth=1.5,
                   label=f"chance = {chance:.3f}")

    n_shuffles = len(null_distribution)
    p = (1 + (null_distribution >= observed).sum()) / (1 + n_shuffles)
    ax.set_xlabel(metric_label)
    ax.set_ylabel("Count")
    ax.set_title(
        f"{area_name} — {metric_label} (empirical p = {p:.3f}, "
        f"n_shuffles = {n_shuffles})"
    )
    ax.legend(loc="best", fontsize=8)
    return ax


def plot_silhouette_with_null(
    observed: float,
    null_distribution: np.ndarray,
    area_name: str,
    *,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Histogram of the silhouette null with the observed value overlaid.

    Args:
        observed: observed balanced silhouette (mean over replicates).
        null_distribution: shape ``(n_shuffles,)`` from
            :func:`silhouette_balanced_null`.
        area_name: used in title.
        ax: matplotlib axes; created if None.

    Returns:
        The axes drawn on.
    """
    return _plot_metric_with_null(
        observed, null_distribution, area_name,
        metric_label="balanced silhouette (top-3 PCs)", ax=ax,
    )


def plot_classifier_with_null(
    observed: float,
    null_distribution: np.ndarray,
    area_name: str,
    chance: float,
    *,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Histogram of the classifier null with observed and chance overlaid.

    Args:
        observed: observed CV accuracy.
        null_distribution: shape ``(n_shuffles,)`` from :func:`classify_cv_null`.
        area_name: used in title.
        chance: majority-class baseline accuracy from :func:`classify_cv`.
        ax: matplotlib axes; created if None.

    Returns:
        The axes drawn on.
    """
    return _plot_metric_with_null(
        observed, null_distribution, area_name,
        metric_label="CV accuracy (5-fold logistic regression)",
        chance=chance, ax=ax,
    )


def plot_cross_area_grid(
    area_results: dict[str, dict],
    labels: np.ndarray,
    *,
    figsize: tuple[float, float] = (12, 10),
) -> plt.Figure:
    """2 × 2 grid of 2-D PC scatters, one panel per cortical area.

    All panels use the same colour and marker conventions; only the
    *pattern* of cluster separation is comparable across panels because
    each area's PCs are fit independently.

    Args:
        area_results: dict from area name to the dict returned by
            :func:`run_area_pipeline`.
        labels: shape ``(n_trials,)`` — same labels used in every PCA.
        figsize: figure size. Default (12, 10) gives ~6×5 per panel.

    Returns:
        The created Figure. Caller saves with ``fig.savefig(path)``.

    Notes:
        Areas not present in ``area_results`` produce empty panels.
        Default panel order: V1 (top-left), AL (top-right), LM (bottom-
        left), RL (bottom-right) — the canonical visual-cortex layout.
    """
    layout = [["V1", "AL"], ["LM", "RL"]]
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    for r in range(2):
        for c in range(2):
            area = layout[r][c]
            ax = axes[r][c]
            if area not in area_results:
                ax.set_visible(False)
                continue
            res = area_results[area]
            # show_legend=False so we can render a single shared figure-level
            # legend below (Task 6 modification — keeps panels uncluttered).
            plot_pca_2d(
                res["X_pcs"], labels, area_name=area, pca=res["pca"], ax=ax,
                show_legend=False,
            )

    # Collect handles+labels from the first panel that drew something, for a single shared legend.
    handles, panel_labels = None, None
    for r in range(2):
        for c in range(2):
            ax = axes[r][c]
            if ax.get_visible() and ax.has_data():
                h, l = ax.get_legend_handles_labels()
                if h:
                    handles, panel_labels = h, l
                    break
        if handles is not None:
            break

    if handles:
        fig.legend(handles, panel_labels, loc="lower center", ncol=len(handles),
                   bbox_to_anchor=(0.5, -0.02), fontsize=9, framealpha=0.9)

    fig.suptitle(
        "Per-area trial geometry (PC axes are area-specific; "
        "compare cluster separation, not coordinates)",
        fontsize=11, y=1.00,
    )
    fig.tight_layout()
    return fig


def plot_cross_area_metric_bars(
    area_results: dict[str, dict],
    metric: str,
    *,
    chance: float | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Per-area bar chart of observed metric with null distribution overlay.

    Args:
        area_results: dict from area name to the dict returned by
            :func:`run_area_pipeline`.
        metric: ``"silhouette"`` or ``"classifier"``.
        chance: optional reference line (e.g. majority-class baseline for
            classifier).
        ax: matplotlib axes; created if None.

    Returns:
        The axes drawn on.

    Notes:
        Each bar shows the observed value. Behind it, the area's null
        distribution is rendered as a thin grey vertical IQR range with a
        white median tick — so the reader sees significance at a glance.
        Empirical p-values are annotated above each bar.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))

    if metric not in ("silhouette", "classifier"):
        raise ValueError(f"metric must be 'silhouette' or 'classifier', got {metric!r}")

    metric_label = (
        "Balanced silhouette (top-3 PCs)" if metric == "silhouette"
        else "CV accuracy (5-fold logistic regression)"
    )

    areas = list(area_results.keys())
    observed = np.array([area_results[a][metric]["observed"] for a in areas])
    nulls = [area_results[a][metric]["null"] for a in areas]
    p_values = [area_results[a][metric]["empirical_p"] for a in areas]

    x = np.arange(len(areas))
    bars = ax.bar(x, observed, color="steelblue", alpha=0.85, edgecolor="black")

    # Overlay null IQR as a thin range behind/over each bar.
    for i, null in enumerate(nulls):
        q05 = np.quantile(null, 0.05)
        q50 = np.quantile(null, 0.50)
        q95 = np.quantile(null, 0.95)
        ax.vlines(x[i], q05, q95, color="darkgrey", linewidth=2, zorder=3)
        ax.scatter(x[i], q50, color="white", edgecolor="darkgrey",
                   s=30, zorder=4, marker="_", linewidths=2)

    # Empirical p-values above bars.
    y_max = ax.get_ylim()[1]
    for i, p in enumerate(p_values):
        sig = "*" if p < 0.05 else "ns"
        ax.text(x[i], observed[i] + 0.02 * y_max,
                f"p={p:.3f}\n{sig}",
                ha="center", va="bottom", fontsize=8)

    if chance is not None:
        ax.axhline(chance, color="grey", linestyle="--", linewidth=1.2,
                   label=f"chance = {chance:.3f}")
        ax.legend(loc="best", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(areas)
    ax.set_xlabel("Cortical area")
    ax.set_ylabel(metric_label)
    ax.set_title(f"Per-area {metric_label} with null IQR (5–95%)")
    ax.grid(True, alpha=0.3, axis="y")
    return ax


def plot_population_matched_comparison(
    full_results: dict[str, dict],
    matched_results: dict[str, list[dict]],
    metric: str,
    *,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Paired bars per area: observed metric for all-neurons vs matched.

    Args:
        full_results: ``area -> run_area_pipeline result`` (all neurons).
        matched_results: ``area -> list of run_area_pipeline results``,
            one per random subsample at the matched neuron count. The
            anchor area (e.g. AL) is run once and provided as a 1-element
            list; other areas have N entries (typically 20).
        metric: ``"silhouette"`` or ``"classifier"``.
        ax: matplotlib axes; created if None.

    Returns:
        The axes drawn on.

    Notes:
        Each "matched" bar shows the median over subsamples; vertical bar
        encodes IQR. Each "all-neurons" bar shows the single observed
        value. Two bars per area, side-by-side.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4.5))

    if metric not in ("silhouette", "classifier"):
        raise ValueError(f"metric must be 'silhouette' or 'classifier', got {metric!r}")

    areas = list(full_results.keys())
    full_obs = np.array([full_results[a][metric]["observed"] for a in areas])
    matched_obs = []
    matched_q25 = []
    matched_q75 = []
    for a in areas:
        vals = np.array([r[metric]["observed"] for r in matched_results[a]])
        matched_obs.append(np.median(vals))
        matched_q25.append(np.quantile(vals, 0.25))
        matched_q75.append(np.quantile(vals, 0.75))
    matched_obs = np.array(matched_obs)
    matched_q25 = np.array(matched_q25)
    matched_q75 = np.array(matched_q75)

    x = np.arange(len(areas))
    bar_width = 0.35
    ax.bar(x - bar_width / 2, full_obs, width=bar_width,
           color="steelblue", alpha=0.85, edgecolor="black", label="all neurons")
    ax.bar(x + bar_width / 2, matched_obs, width=bar_width,
           color="lightcoral", alpha=0.85, edgecolor="black",
           label="matched to AL n_neurons (median)")
    # IQR error bars on the matched group.
    matched_lower = matched_obs - matched_q25
    matched_upper = matched_q75 - matched_obs
    ax.errorbar(x + bar_width / 2, matched_obs,
                yerr=np.vstack([matched_lower, matched_upper]),
                fmt="none", ecolor="darkred", capsize=4, linewidth=1.5)

    metric_label = (
        "Balanced silhouette (top-3 PCs)" if metric == "silhouette"
        else "CV accuracy"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(areas)
    ax.set_xlabel("Cortical area")
    ax.set_ylabel(metric_label)
    ax.set_title(f"All-neurons vs population-matched — {metric_label}")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    return ax


# =============================================================================
# (d) Cross-session aggregation
# =============================================================================


def aggregate_cross_session(
    results_root: str | Path,
    *,
    analysis_filter: str = "all_neurons",
) -> pd.DataFrame:
    """Read per-session result CSVs and concatenate into a long-format frame.

    Walks the directory tree under ``results_root``, looking for files
    named ``{session}/silhouette_scores.csv``, and concatenates them
    with a ``session`` column added.

    Args:
        results_root: directory containing per-session subdirs (e.g.
            ``Path("results/option2")``).
        analysis_filter: which ``analysis`` rows to keep. Default
            ``"all_neurons"`` (the primary analysis). Other valid values
            include ``"equal_population"`` and ``"log_sensitivity"``.

    Returns:
        Long-format dataframe with columns ``session``, ``area``,
        ``n_neurons``, ``metric``, ``observed``, ``null_p05``,
        ``null_median``, ``null_p95``, ``empirical_p_value``, ``analysis``.
        Empty dataframe if no session CSVs are found.

    Notes:
        Sessions whose CSV is missing are silently skipped (the
        cross-session aggregation cell is a no-op until ≥1 session has
        been run).
    """
    results_root = Path(results_root)
    if not results_root.exists():
        return pd.DataFrame()

    frames = []
    for session_dir in sorted(results_root.iterdir()):
        if not session_dir.is_dir():
            continue
        csv_path = session_dir / "silhouette_scores.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        df["session"] = session_dir.name
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    if analysis_filter is not None:
        combined = combined[combined["analysis"] == analysis_filter].reset_index(drop=True)
    return combined


def plot_cross_session_metric(
    df: pd.DataFrame,
    metric: str,
    *,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Grouped bar plot: sessions on x-axis, areas as colour groups.

    Args:
        df: long-format dataframe from :func:`aggregate_cross_session`.
        metric: ``"silhouette"`` or ``"classifier"``.
        ax: matplotlib axes; created if None.

    Returns:
        The axes drawn on. If ``df`` is empty, returns the axes with a
        "No sessions found" annotation.

    Notes:
        Areas not present in a given session produce missing bars (skip,
        do not zero-fill).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4.5))

    if df.empty:
        ax.text(0.5, 0.5, "No per-session results found.\nRun this notebook "
                "with SESSION set to additional values to populate.",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=11, color="grey")
        ax.axis("off")
        return ax

    sub = df[df["metric"] == metric].copy()
    sessions = sorted(sub["session"].unique())
    areas = sorted(sub["area"].unique())

    x = np.arange(len(sessions))
    bar_width = 0.8 / max(len(areas), 1)
    palette = plt.get_cmap("tab10")

    for i, area in enumerate(areas):
        vals = []
        for s in sessions:
            row = sub[(sub["session"] == s) & (sub["area"] == area)]
            vals.append(row["observed"].iloc[0] if len(row) else np.nan)
        offsets = (i - (len(areas) - 1) / 2) * bar_width
        ax.bar(x + offsets, vals, width=bar_width,
               color=palette(i), edgecolor="black", linewidth=0.5,
               label=area)

    metric_label = "Balanced silhouette" if metric == "silhouette" else "CV accuracy"
    ax.set_xticks(x)
    ax.set_xticklabels(sessions)
    ax.set_xlabel("Session")
    ax.set_ylabel(metric_label)
    ax.set_title(f"Cross-session {metric_label} by area")
    ax.legend(loc="best", fontsize=8, title="Area")
    ax.grid(True, alpha=0.3, axis="y")
    return ax
