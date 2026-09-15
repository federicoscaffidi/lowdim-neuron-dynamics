"""option3_trajectories_utils: helper module for the time-resolved PCA trajectories notebook.

The module is organized in four sections:

(a) Trajectory construction: build_stim_trajectories, save_trajectories,
    load_trajectories.
(b) PCA fit and projection: fit_trajectory_pca.
(c) Distance metrics, bootstrap, null, and subsample pipelines:
    pairwise_trajectory_distance, bootstrap_distance_envelope,
    shuffle_null_max_distance, compute_onset_latency,
    crossvalidated_distance,
    subsample_population_run_pipeline, subsample_clip_trials_run_pipeline.
(d) Plotting: PSTH per area, 2-D / 3-D trajectory plots, distance time-course
    panels, cross-area headline, equal-population comparison, Clip-subsampling
    comparison.

Design constraints (see docs/specs/2026-05-04-option3-trajectories-design.md):
- Data functions are pure: no plotting, no file writes (except the explicit
  save_trajectories / load_trajectories).
- Plotting functions accept ``ax`` (or return a Figure for Plotly) so plots
  compose.
- Every public function has a NumPy-style docstring with Args, Returns, Notes.
- No magic constants: anything tunable is a parameter with a sensible default.
- All randomness is seeded via the ``seed`` (or ``random_state``) parameter;
  sub-seeds are derived deterministically from the master seed.
- Reuses option2_pca_utils.preprocess_responses (Stage A) and
  microns_eda.compute_clean_trial_indices.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# Reuse Option 2's stim conventions for visual consistency across the project.
from scripts.experiment1_utils import STIM_COLORS, STIM_MARKERS, STIM_ORDER


# =============================================================================
# (a) Trajectory construction
# =============================================================================


def build_stim_trajectories(
    responses_preprocessed: np.ndarray,
    trial_boundaries: np.ndarray,
    clean_trial_indices: np.ndarray,
    brain_areas: np.ndarray,
    labels: np.ndarray,
    *,
    n_frames: int = 75,
) -> dict[str, dict[str, np.ndarray]]:
    """Build per-area, per-stimulus trial-averaged trajectories.

    Three operations applied in order:

    1. For each clean trial, slice the first ``n_frames`` timesteps from the
       preprocessed neuron × time matrix.
    2. Group trials by stimulus class.
    3. Within each (area, stim) group, average across trials at each frame
       independently. Result: one (n_frames, n_neurons_in_area) matrix per
       (area, stim) pair, where row t is the mean activity at frame t across
       all trials of that stim in that area.

    Args:
        responses_preprocessed: shape ``(n_neurons, total_timesteps)``,
            output of ``option2_pca_utils.preprocess_responses``.
        trial_boundaries: shape ``(n_trials + 1,)``, cumulative timestep
            counts as returned by ``microns_eda.load_session_responses``.
        clean_trial_indices: shape ``(n_clean_trials,)``, indices of trials
            to keep, in original trial order.
        brain_areas: shape ``(n_neurons,)`` of bytes-or-string area labels.
            Decoded internally if bytes.
        labels: shape ``(n_clean_trials,)`` of stim-class labels per clean
            trial (e.g. ``"Clip"``, ``"Monet2"``, ``"Trippy"``).
        n_frames: number of leading frames to keep per trial.

    Returns:
        Nested dict ``{area: {stim: trajectory}}`` where each ``trajectory``
        has shape ``(n_frames, n_neurons_in_area)``. Areas with zero neurons
        and (area, stim) pairs with zero trials are silently omitted.

    Notes:
        Imbalanced trial counts (e.g. 7_5: 377 Clip / 38 Monet2 / 38 Trippy)
        produce trajectories with very different smoothness: the Clip
        trajectory is a much tighter mean estimate than the synthetics.
        The Phase 9 Clip-subsampling control addresses this directly.
    """
    # Decode area labels (matches the convention in option2_pca_utils).
    if brain_areas.dtype.kind in ("S", "O") and len(brain_areas) > 0:
        sample = brain_areas[0]
        if isinstance(sample, bytes):
            brain_areas_str = np.array([s.decode("utf-8") for s in brain_areas])
        else:
            brain_areas_str = brain_areas.astype(str)
    else:
        brain_areas_str = brain_areas.astype(str)

    label_arr = np.asarray(labels)
    if len(label_arr) != len(clean_trial_indices):
        raise ValueError(
            f"labels length {len(label_arr)} does not match "
            f"clean_trial_indices length {len(clean_trial_indices)}"
        )

    # Build a (n_clean_trials, n_frames, n_neurons) tensor by slicing.
    # Memory: 453 × 75 × 8194 × 8 bytes ≈ 2.2 GB for V1 alone if we kept all
    # neurons; instead we slice per area to bound memory.
    per_area: dict[str, dict[str, np.ndarray]] = {}
    for area in np.unique(brain_areas_str):
        cols = np.where(brain_areas_str == area)[0]
        if len(cols) == 0:
            continue
        n_neurons_a = len(cols)
        # Trial × time × neuron tensor for this area only.
        trials_t_n = np.empty((len(clean_trial_indices), n_frames, n_neurons_a),
                              dtype=np.float64)
        for out_idx, trial_idx in enumerate(clean_trial_indices):
            start = int(trial_boundaries[trial_idx])
            end = start + n_frames
            # Guard: a trial shorter than n_frames would silently read into
            # the next trial's frames.
            if end > int(trial_boundaries[trial_idx + 1]):
                raise ValueError(
                    f"trial {trial_idx} has "
                    f"{int(trial_boundaries[trial_idx + 1]) - start} frames, "
                    f"fewer than n_frames={n_frames}"
                )
            window = responses_preprocessed[cols, start:end]  # (n_a, n_frames)
            trials_t_n[out_idx] = window.T  # (n_frames, n_a)

        per_stim: dict[str, np.ndarray] = {}
        for stim in np.unique(label_arr):
            mask = label_arr == stim
            if not mask.any():
                continue
            # Average across trials at each frame.
            per_stim[stim] = trials_t_n[mask].mean(axis=0)  # (n_frames, n_a)
        per_area[area] = per_stim

    return per_area


def save_trajectories(
    trajectories_per_area: dict[str, dict[str, np.ndarray]],
    path: str | Path,
) -> None:
    """Persist per-area, per-stim trajectories to a single ``.npz`` file.

    Args:
        trajectories_per_area: output of :func:`build_stim_trajectories`.
        path: destination ``.npz`` path. Created if absent.

    Notes:
        Keys in the npz are flattened as ``{area}__{stim}``. A small
        metadata array under key ``__metadata__`` lists the (area, stim)
        pairs in the order they were saved, for round-trip verification.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    metadata = []
    for area in sorted(trajectories_per_area.keys()):
        for stim in sorted(trajectories_per_area[area].keys()):
            key = f"{area}__{stim}"
            arrays[key] = trajectories_per_area[area][stim]
            metadata.append([area, stim])
    arrays["__metadata__"] = np.array(metadata, dtype=object)
    np.savez(path, **arrays)


def load_trajectories(path: str | Path) -> dict[str, dict[str, np.ndarray]]:
    """Load per-area, per-stim trajectories from a ``.npz`` file.

    Args:
        path: path to a ``.npz`` produced by :func:`save_trajectories`.

    Returns:
        Same nested dict structure as :func:`build_stim_trajectories`.
    """
    with np.load(path, allow_pickle=True) as data:
        per_area: dict[str, dict[str, np.ndarray]] = {}
        for area, stim in data["__metadata__"]:
            key = f"{area}__{stim}"
            per_area.setdefault(area, {})[stim] = data[key]
    return per_area


# =============================================================================
# (b) PCA fit and projection
# =============================================================================


def fit_trajectory_pca(
    trajectories_per_area: dict[str, dict[str, np.ndarray]],
    *,
    n_components: int = 10,
    random_state: int = 42,
) -> dict[str, dict]:
    """Fit per-area PCA on stacked stimulus trajectories.

    For each area, vertically stacks the three stimulus trajectory matrices
    into a single ``(n_stim * n_frames, n_neurons_in_area)`` matrix and
    fits ``sklearn.decomposition.PCA``. The stacked input shares a common
    coordinate system within an area, so the projected trajectories
    (sliced back into per-stim blocks) can be compared geometrically.

    Args:
        trajectories_per_area: nested dict from :func:`build_stim_trajectories`.
        n_components: PCs to retain. Default 10.
        random_state: sklearn random_state.

    Returns:
        Per-area dict with keys:

        - ``"pca"``: fitted ``sklearn.decomposition.PCA``.
        - ``"stim_pcs"``: dict mapping stim → ``(n_frames, n_components)``
          projection.
        - ``"stacked_pcs"``: ``(n_stim * n_frames, n_components)`` projection
          of the stacked input (kept for downstream uses that need the full
          concatenated path; rarely used directly).

    Notes:
        PC1 of this stacked-matrix PCA is typically dominated by stimulus-
        common temporal dynamics (rise at onset, settle later); the
        stimulus-identity signal lives in PC2-PC3 in the visualisations.
        The PCA centers the stacked input, so trajectories are plotted
        relative to the grand mean across all stims and all frames.
    """
    out: dict[str, dict] = {}
    for area, per_stim in trajectories_per_area.items():
        stims_in_order = sorted(per_stim.keys())
        n_frames = per_stim[stims_in_order[0]].shape[0]
        # Stack stim trajectories: shape (n_stim * n_frames, n_neurons).
        stacked = np.vstack([per_stim[s] for s in stims_in_order])
        pca = PCA(
            n_components=n_components, whiten=False, random_state=random_state
        )
        stacked_pcs = pca.fit_transform(stacked)
        # Slice back into per-stim blocks.
        stim_pcs: dict[str, np.ndarray] = {}
        for i, s in enumerate(stims_in_order):
            stim_pcs[s] = stacked_pcs[i * n_frames : (i + 1) * n_frames]
        out[area] = {"pca": pca, "stim_pcs": stim_pcs, "stacked_pcs": stacked_pcs}
    return out


# =============================================================================
# (c) Distance metrics, bootstrap, null, and subsample pipelines
# =============================================================================


def pairwise_trajectory_distance(
    stim_trajectories: dict[str, np.ndarray],
    *,
    metric: str = "full",
    pca: PCA | None = None,
    n_pcs: int = 3,
) -> dict[frozenset[str], np.ndarray]:
    """Frame-by-frame Euclidean distance between each pair of stim trajectories.

    Args:
        stim_trajectories: dict mapping stim → ``(n_frames, n_features)``
            trajectory. ``n_features`` is ``n_neurons_in_area`` if
            ``metric="full"``; an arbitrary count if the trajectories have
            already been projected through PCA.
        metric: ``"full"`` to use the trajectories as given, or ``"top_pcs"``
            to project through ``pca`` and use the top ``n_pcs`` columns.
        pca: required if ``metric="top_pcs"``; ignored otherwise.
        n_pcs: number of PCs to use when ``metric="top_pcs"``. Default 3
            (matches the visualisation).

    Returns:
        Dict keyed by frozenset of stim-pair names (``frozenset({"Clip", "Monet2"})``
        etc.) with values ``(n_frames,)`` of pairwise Euclidean distances.

    Notes:
        For ``metric="top_pcs"``, the trajectories must have been built from
        the same population that ``pca`` was fit on (per-area). Mixing PCAs
        and trajectories from different areas will silently produce wrong
        distances.
    """
    if metric not in ("full", "top_pcs"):
        raise ValueError(f"metric must be 'full' or 'top_pcs', got {metric!r}")
    if metric == "top_pcs" and pca is None:
        raise ValueError("pca must be supplied when metric='top_pcs'")

    if metric == "top_pcs":
        # Re-project each trajectory through pca (n_features → n_components),
        # then keep only top n_pcs columns.
        projected = {
            stim: pca.transform(traj)[:, :n_pcs]
            for stim, traj in stim_trajectories.items()
        }
        traj_for_dist = projected
    else:
        traj_for_dist = stim_trajectories

    pairs: dict[frozenset[str], np.ndarray] = {}
    stims = sorted(traj_for_dist.keys())
    for a, b in combinations(stims, 2):
        diff = traj_for_dist[a] - traj_for_dist[b]
        pairs[frozenset({a, b})] = np.linalg.norm(diff, axis=1)
    return pairs


def _resample_per_class(
    trial_responses: np.ndarray,
    labels: np.ndarray,
    *,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Class-stratified bootstrap (sample with replacement within each class).

    Returns ``(resampled_responses, resampled_labels)`` with same shape as inputs.
    Each class is resampled to its own original size, preserving class
    marginals exactly across the bootstrap.
    """
    classes, counts = np.unique(labels, return_counts=True)
    out_idx = np.empty(len(labels), dtype=np.int64)
    write = 0
    for c, cnt in zip(classes, counts):
        pool = np.where(labels == c)[0]
        out_idx[write : write + cnt] = rng.choice(pool, size=cnt, replace=True)
        write += cnt
    return trial_responses[out_idx], labels[out_idx]


def _permute_labels(
    labels: np.ndarray,
    rng: np.random.Generator,
    *,
    groups: np.ndarray | None = None,
) -> np.ndarray:
    """Permute labels across trials, or across groups of trials.

    With ``groups=None`` this is ``rng.permutation(labels)``. With ``groups``
    (one id per trial, e.g. the clip hash), every trial of a group carries
    the same label and the permutation reassigns labels *between groups*, so
    repeated presentations of the same stimulus stay together. That is the
    exchangeable unit when trials are repeats of a smaller set of stimuli;
    permuting at trial level in that situation understates the variance of
    the class means and makes p-values optimistic.

    Class trial counts are preserved exactly in the trial-level case and
    only approximately in the group-level case (groups have unequal sizes).
    """
    if groups is None:
        return rng.permutation(labels)
    groups = np.asarray(groups)
    uniq, first_idx = np.unique(groups, return_index=True)
    group_labels = labels[first_idx]
    # Every trial in a group must share the label, otherwise the group is
    # not a valid exchangeable unit.
    for g, lab in zip(uniq, group_labels):
        members = labels[groups == g]
        if not np.all(members == lab):
            raise ValueError(f"group {g!r} carries more than one label")
    permuted = rng.permutation(group_labels)
    lookup = dict(zip(uniq, permuted))
    return np.array([lookup[g] for g in groups], dtype=labels.dtype)


def _bootstrap_trial_counts(
    n_trials: int,
    rng: np.random.Generator,
    *,
    group_ids: np.ndarray | None = None,
) -> np.ndarray:
    """Per-trial multiplicities for one bootstrap draw within a class.

    Trial-level: ``n_trials`` draws with replacement over trials. Group-level
    (``group_ids`` given, one id per trial): draws with replacement over the
    unique groups, and every trial of a drawn group is included once per
    draw (a cluster bootstrap).
    """
    if group_ids is None:
        idx = rng.integers(0, n_trials, size=n_trials)
        return np.bincount(idx, minlength=n_trials).astype(np.float64)
    uniq, inverse = np.unique(group_ids, return_inverse=True)
    drawn = rng.integers(0, len(uniq), size=len(uniq))
    group_counts = np.bincount(drawn, minlength=len(uniq)).astype(np.float64)
    return group_counts[inverse]


def _trajectories_from_trial_tensor(
    trials_t_n: np.ndarray,
    labels: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build per-stim trajectories from a (n_trials, n_frames, n_neurons) tensor."""
    out: dict[str, np.ndarray] = {}
    for stim in np.unique(labels):
        mask = labels == stim
        if not mask.any():
            continue
        out[stim] = trials_t_n[mask].mean(axis=0)
    return out


def bootstrap_distance_envelope(
    trial_tensor: np.ndarray,
    labels: np.ndarray,
    *,
    metric: str = "full",
    pca: PCA | None = None,
    n_pcs: int = 3,
    n_boot: int = 1000,
    seed: int = 42,
    groups: np.ndarray | None = None,
) -> dict[frozenset[str], dict[str, np.ndarray]]:
    """Bootstrap band on each pairwise trajectory distance (plotting only).

    For each of ``n_boot`` iterations, class-stratified resamples the trials,
    rebuilds per-stim trajectories from the resampled tensor, and recomputes
    pairwise distances. Returns the 2.5/97.5 percentile envelopes per pair
    per frame. **PCA is NOT refit**; it is fixed at the supplied basis to keep
    variability attributable to the trajectories, not the basis.

    **This band is not a confidence interval for the true distance and must
    not be compared against a shuffle null.** The statistic is a Euclidean
    norm of a difference of noisy means; its expectation is
    ``sqrt(d_true^2 + noise)`` with ``noise ~ N * (1/n_a + 1/n_b)`` in
    z-scored units. A bootstrap resample adds a *second* independent draw of
    that noise term, so the whole band sits above the observed curve by
    roughly the amount the observed curve sits above the truth. With no
    class difference at all, the 2.5th percentile of this band still clears
    the 95th percentile of the shuffle-null maximum on every frame. Use
    :func:`compute_onset_latency` (observed curve vs null) for inference and
    :func:`crossvalidated_distance` for a bias-corrected magnitude.

    Args:
        trial_tensor: ``(n_trials, n_frames, n_neurons_in_area)``, the per-area
            trial × time × neuron tensor (build once and pass in).
        labels: ``(n_trials,)`` stim labels.
        metric: ``"full"`` or ``"top_pcs"``; same convention as
            :func:`pairwise_trajectory_distance`.
        pca: required if ``metric="top_pcs"``.
        n_pcs: passed to distance computation.
        n_boot: number of bootstrap iterations.
        seed: master seed.
        groups: optional ``(n_trials,)`` group id per trial (e.g. clip
            hash). When given, resampling is done over groups within each
            class (cluster bootstrap) instead of over individual trials.

    Returns:
        Dict keyed by frozenset of stim-pair names; each value is a dict
        with keys ``"lower"`` (2.5th percentile, shape ``(n_frames,)``),
        ``"upper"`` (97.5th percentile), and ``"all"`` (full bootstrap
        distribution, shape ``(n_boot, n_frames)``).
    """
    rng = np.random.default_rng(seed)
    label_arr = np.asarray(labels)

    n_frames = trial_tensor.shape[1]
    # Discover which pairs we'll see by computing once on the original data.
    initial_traj = _trajectories_from_trial_tensor(trial_tensor, label_arr)
    initial_pairs = pairwise_trajectory_distance(
        initial_traj, metric=metric, pca=pca, n_pcs=n_pcs
    )
    pair_keys = list(initial_pairs.keys())

    # Performance optimisation: pre-compute per-stim sub-tensors once (one
    # ~1 GB allocation per stim for V1) and compute per-bootstrap means via
    # bincount + einsum, avoiding the ~1.5 GB tensor copy that the naïve
    # ``_resample_per_class`` produces at every iteration.
    group_arr = None if groups is None else np.asarray(groups)
    stim_subtensors: dict[str, np.ndarray] = {}
    stim_groups: dict[str, np.ndarray | None] = {}
    for stim in np.unique(label_arr):
        mask = label_arr == stim
        if mask.any():
            stim_subtensors[stim] = trial_tensor[mask]
            stim_groups[stim] = None if group_arr is None else group_arr[mask]

    accum = {pk: np.empty((n_boot, n_frames)) for pk in pair_keys}
    for b in range(n_boot):
        sub_seed = int(rng.integers(0, 2**31 - 1))
        sub_rng = np.random.default_rng(sub_seed)
        traj: dict[str, np.ndarray] = {}
        for stim, subtensor in stim_subtensors.items():
            n = subtensor.shape[0]
            counts = _bootstrap_trial_counts(
                n, sub_rng, group_ids=stim_groups[stim]
            )
            # Weighted mean = sum_i counts[i] * subtensor[i] / sum(counts)
            traj[stim] = np.einsum("i,ijk->jk", counts, subtensor) / counts.sum()
        pairs = pairwise_trajectory_distance(
            traj, metric=metric, pca=pca, n_pcs=n_pcs
        )
        for pk in pair_keys:
            accum[pk][b] = pairs[pk]

    out: dict[frozenset[str], dict[str, np.ndarray]] = {}
    for pk in pair_keys:
        out[pk] = {
            "lower": np.percentile(accum[pk], 2.5, axis=0),
            "upper": np.percentile(accum[pk], 97.5, axis=0),
            "all": accum[pk],
        }
    return out


def shuffle_null_max_distance(
    trial_tensor: np.ndarray,
    labels: np.ndarray,
    *,
    metric: str = "full",
    pca: PCA | None = None,
    n_pcs: int = 3,
    n_shuffles: int = 100,
    seed: int = 42,
    groups: np.ndarray | None = None,
) -> dict[frozenset[str], np.ndarray]:
    """Null distribution of max pairwise trajectory distance under label shuffle.

    For each of ``n_shuffles`` permutations of ``labels``, rebuilds per-stim
    trajectories on the same trial tensor, computes pairwise distances, and
    records the max distance reached at any frame. Returns one null
    distribution per pair. **PCA is NOT refit.**

    Args:
        trial_tensor: same as in :func:`bootstrap_distance_envelope`.
        labels: same.
        metric, pca, n_pcs: same.
        n_shuffles: number of label permutations.
        seed: master seed.
        groups: optional ``(n_trials,)`` group id per trial. When given,
            labels are permuted between groups, not between trials (see
            :func:`_permute_labels`). Use it whenever several trials are
            repeats of the same stimulus.

    Returns:
        Dict keyed by frozenset of stim-pair names; each value is a
        ``(n_shuffles,)`` array of null max distances.

    Notes:
        Empirical p-value per pair: ``p = (1 + (null >= observed_max).sum())
        / (1 + n_shuffles)`` (pseudocount-1, matches Option 2's convention).
    """
    rng = np.random.default_rng(seed)
    label_arr = np.asarray(labels)

    initial_traj = _trajectories_from_trial_tensor(trial_tensor, label_arr)
    initial_pairs = pairwise_trajectory_distance(
        initial_traj, metric=metric, pca=pca, n_pcs=n_pcs
    )
    pair_keys = list(initial_pairs.keys())

    nulls = {pk: np.empty(n_shuffles) for pk in pair_keys}
    for s in range(n_shuffles):
        shuffled = _permute_labels(label_arr, rng, groups=groups)
        traj = _trajectories_from_trial_tensor(trial_tensor, shuffled)
        pairs = pairwise_trajectory_distance(
            traj, metric=metric, pca=pca, n_pcs=n_pcs
        )
        for pk in pair_keys:
            nulls[pk][s] = float(pairs[pk].max())
    return nulls


def compute_onset_latency(
    observed_distance: np.ndarray,
    null_max_distribution: np.ndarray,
    *,
    alpha: float = 0.05,
) -> int | None:
    """First frame at which the observed distance exceeds the null of the max.

    The threshold is the ``(1 - alpha)`` quantile of the shuffle-null
    distribution of the *maximum over frames* (from
    :func:`shuffle_null_max_distance`). Testing every frame against the
    null of the max is the max-T (Westfall–Young) correction, so the
    family-wise error rate over the 75 frames is ``alpha``. A pair whose
    observed maximum is below the threshold (empirical p >= alpha)
    automatically gets no onset.

    Earlier versions compared the *bootstrap lower envelope* against this
    threshold. That criterion fires on pure noise (see the note in
    :func:`bootstrap_distance_envelope`) and produced onsets for pairs
    with p > 0.3; it is no longer used.

    Args:
        observed_distance: shape ``(n_frames,)``, the observed pairwise
            distance time course (from :func:`pairwise_trajectory_distance`).
        null_max_distribution: shape ``(n_shuffles,)``, the shuffle null of
            the max distance for the same pair and metric.
        alpha: significance level. Default 0.05.

    Returns:
        Onset frame index (0-based, integer), or ``None`` if the observed
        curve never exceeds the threshold.
    """
    threshold = float(np.percentile(null_max_distribution, 100 * (1 - alpha)))
    crosses = np.where(np.asarray(observed_distance) > threshold)[0]
    return int(crosses[0]) if len(crosses) else None


def crossvalidated_distance(
    trial_tensor: np.ndarray,
    labels: np.ndarray,
    *,
    metric: str = "full",
    pca: PCA | None = None,
    n_pcs: int = 3,
    n_splits: int = 20,
    seed: int = 42,
    groups: np.ndarray | None = None,
) -> dict[frozenset[str], np.ndarray]:
    """Bias-corrected (split-half) pairwise distance between class means.

    The plain distance ``||mean_a - mean_b||`` is inflated by sampling noise:
    its expectation is ``sqrt(d_true^2 + N * sigma^2 * (1/n_a + 1/n_b))``.
    In z-scored data with N neurons this floor is ``sqrt(N (1/n_a + 1/n_b))``
    and depends on the trial counts, so raw distances are not comparable
    between pairs with different ``n`` or between areas with different
    ``N``. This estimator removes the bias: trials of each class are split
    into two random halves, and the inner product
    ``<mean_a1 - mean_b1, mean_a2 - mean_b2>`` is unbiased for ``d_true^2``
    because the two halves carry independent noise. The result is averaged
    over ``n_splits`` random splits and reported as a signed square root
    (negative values mean the estimate of ``d_true^2`` is below zero, i.e.
    no separation).

    Args:
        trial_tensor: ``(n_trials, n_frames, n_neurons)``.
        labels: ``(n_trials,)``.
        metric, pca, n_pcs: as in :func:`pairwise_trajectory_distance`. For
            ``"top_pcs"`` both halves are projected through ``pca`` before
            the inner product.
        n_splits: number of random half-splits to average over.
        seed: master seed.
        groups: optional ``(n_trials,)`` group id per trial. When given, the
            split is done over groups so repeats of one stimulus never land
            in both halves (which would re-introduce correlated noise).

    Returns:
        Dict keyed by frozenset of stim-pair names; each value is
        ``(n_frames,)`` of signed bias-corrected distances.
    """
    if metric not in ("full", "top_pcs"):
        raise ValueError(f"metric must be 'full' or 'top_pcs', got {metric!r}")
    if metric == "top_pcs" and pca is None:
        raise ValueError("pca must be supplied when metric='top_pcs'")

    rng = np.random.default_rng(seed)
    label_arr = np.asarray(labels)
    group_arr = None if groups is None else np.asarray(groups)
    stims = sorted(np.unique(label_arr))
    n_frames = trial_tensor.shape[1]

    def _project(traj: np.ndarray) -> np.ndarray:
        if metric == "top_pcs":
            return pca.transform(traj)[:, :n_pcs]
        return traj

    # Per-class index pools (trial indices, and the unit to split over).
    pools: dict[str, np.ndarray] = {s: np.where(label_arr == s)[0] for s in stims}

    def _half_masks(idx: np.ndarray, sub_rng: np.random.Generator):
        """Return (idx_half1, idx_half2) splitting trials (or groups) in two."""
        if group_arr is None:
            perm = sub_rng.permutation(idx)
            half = len(perm) // 2
            return perm[:half], perm[half:]
        uniq = np.unique(group_arr[idx])
        perm_groups = sub_rng.permutation(uniq)
        half = len(perm_groups) // 2
        g1 = set(perm_groups[:half].tolist())
        in1 = np.array([group_arr[i] in g1 for i in idx])
        return idx[in1], idx[~in1]

    accum = {frozenset({a, b}): np.zeros(n_frames) for a, b in combinations(stims, 2)}
    for _ in range(n_splits):
        sub_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
        halves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for s in stims:
            i1, i2 = _half_masks(pools[s], sub_rng)
            if len(i1) == 0 or len(i2) == 0:
                raise ValueError(f"class {s!r} has too few units to split in half")
            halves[s] = (
                _project(trial_tensor[i1].mean(axis=0)),
                _project(trial_tensor[i2].mean(axis=0)),
            )
        for a, b in combinations(stims, 2):
            diff1 = halves[a][0] - halves[b][0]
            diff2 = halves[a][1] - halves[b][1]
            accum[frozenset({a, b})] += np.einsum("ij,ij->i", diff1, diff2)

    out: dict[frozenset[str], np.ndarray] = {}
    for pk, d2 in accum.items():
        d2 = d2 / n_splits
        out[pk] = np.sign(d2) * np.sqrt(np.abs(d2))
    return out


def subsample_population_run_pipeline(
    area_name: str,
    trial_tensor: np.ndarray,
    labels: np.ndarray,
    *,
    n_match: int,
    n_subsamples: int = 20,
    n_components: int = 10,
    seed: int = 42,
    groups: np.ndarray | None = None,
    n_cv_splits: int = 20,
) -> list[dict]:
    """Equal-population control: subsample neurons, recompute trajectories + distances.

    For each of ``n_subsamples`` random column subsamples to ``n_match``
    neurons, rebuilds per-stim trajectories on the subsampled tensor,
    refits a per-area PCA on the stacked subsampled trajectories, and
    computes pairwise distances (full-feature + top-3 PC).

    Args:
        area_name: used for logging.
        trial_tensor: ``(n_trials, n_frames, n_neurons_in_area)``.
        labels: ``(n_trials,)``.
        n_match: target neuron count (e.g. AL's count).
        n_subsamples: random subsamples to average over.
        n_components: PCs to retain in each refit PCA.
        seed: master seed.

    Returns:
        List of length ``n_subsamples``. Each entry is a dict with:

        - ``"pca"``: refit ``sklearn.decomposition.PCA``.
        - ``"trajectories"``: ``{stim: (n_frames, n_match)}``.
        - ``"distances_full"``: pair → ``(n_frames,)`` distances.
        - ``"distances_top3_pc"``: pair → ``(n_frames,)`` distances.
        - ``"distances_cv_full"``: pair → ``(n_frames,)`` bias-corrected
          distances from :func:`crossvalidated_distance` (``groups`` and
          ``n_cv_splits`` are forwarded). Raw distances carry a
          ``sqrt(N (1/n_a + 1/n_b))`` noise floor; use this key when
          comparing magnitudes across areas or pairs.
        - ``"col_indices"``: shape ``(n_match,)``, which neurons were sampled.

    Notes:
        Bootstrap envelopes and shuffle nulls are NOT recomputed per subsample
        (compute cost would 20×). The cross-area-ranking question is answered
        by comparing the median ± IQR of the bias-corrected distances across
        the 20 subsamples.
    """
    rng = np.random.default_rng(seed)
    n_neurons = trial_tensor.shape[2]
    if n_match > n_neurons:
        raise ValueError(
            f"n_match={n_match} > n_neurons={n_neurons} in area {area_name}"
        )

    label_arr = np.asarray(labels)
    out: list[dict] = []
    for s in range(n_subsamples):
        sub_seed = int(rng.integers(0, 2**31 - 1))
        sub_rng = np.random.default_rng(sub_seed)
        col_indices = np.sort(sub_rng.choice(n_neurons, size=n_match, replace=False))
        sub_tensor = trial_tensor[:, :, col_indices]
        traj = _trajectories_from_trial_tensor(sub_tensor, label_arr)
        # Refit PCA on the stacked subsampled trajectories.
        stims = sorted(traj.keys())
        stacked = np.vstack([traj[st] for st in stims])
        pca = PCA(n_components=n_components, whiten=False, random_state=sub_seed)
        pca.fit(stacked)
        out.append({
            "pca": pca,
            "trajectories": traj,
            "distances_full": pairwise_trajectory_distance(traj, metric="full"),
            "distances_top3_pc": pairwise_trajectory_distance(
                traj, metric="top_pcs", pca=pca, n_pcs=3
            ),
            "distances_cv_full": crossvalidated_distance(
                sub_tensor, label_arr, metric="full",
                n_splits=n_cv_splits, seed=sub_seed, groups=groups,
            ),
            "col_indices": col_indices,
        })
    return out


def subsample_clip_trials_run_pipeline(
    trial_tensor: np.ndarray,
    labels: np.ndarray,
    *,
    n_clip_target: int,
    n_subsamples: int = 20,
    n_components: int = 10,
    seed: int = 42,
    n_cv_splits: int = 20,
) -> list[dict]:
    """Clip-trial subsampling control: trim Clip trial count, recompute.

    For each of ``n_subsamples`` random subsamples of Clip trials down to
    ``n_clip_target``, rebuilds per-stim trajectories using the trimmed Clip
    trials plus all Monet2/Trippy trials, refits PCA, and computes pairwise
    distances.

    Args:
        trial_tensor: ``(n_trials, n_frames, n_neurons)``.
        labels: ``(n_trials,)``, must include at least one ``"Clip"`` trial.
        n_clip_target: target number of Clip trials (typically 38, matching
            the minority counts in session 7_5).
        n_subsamples: random subsamples to average over.
        n_components: PCs to retain in each refit PCA.
        seed: master seed.

    Returns:
        Same dict-per-iteration structure as
        :func:`subsample_population_run_pipeline` (minus ``col_indices``).

    Notes:
        Used to test whether Clip↔synthetic distances drop substantially
        when Clip's trial count is matched to the synthetics. A drop
        indicates part of the original separation was a sample-size artefact.
    """
    rng = np.random.default_rng(seed)
    label_arr = np.asarray(labels)
    clip_idx = np.where(label_arr == "Clip")[0]
    other_idx = np.where(label_arr != "Clip")[0]
    if len(clip_idx) < n_clip_target:
        raise ValueError(
            f"need {n_clip_target} Clip trials, have {len(clip_idx)}"
        )

    out: list[dict] = []
    for s in range(n_subsamples):
        sub_seed = int(rng.integers(0, 2**31 - 1))
        sub_rng = np.random.default_rng(sub_seed)
        kept_clip = sub_rng.choice(clip_idx, size=n_clip_target, replace=False)
        kept_idx = np.sort(np.concatenate([kept_clip, other_idx]))
        sub_tensor = trial_tensor[kept_idx]
        sub_labels = label_arr[kept_idx]
        traj = _trajectories_from_trial_tensor(sub_tensor, sub_labels)
        stims = sorted(traj.keys())
        stacked = np.vstack([traj[st] for st in stims])
        pca = PCA(n_components=n_components, whiten=False, random_state=sub_seed)
        pca.fit(stacked)
        out.append({
            "pca": pca,
            "trajectories": traj,
            "distances_full": pairwise_trajectory_distance(traj, metric="full"),
            "distances_top3_pc": pairwise_trajectory_distance(
                traj, metric="top_pcs", pca=pca, n_pcs=3
            ),
            "distances_cv_full": crossvalidated_distance(
                sub_tensor, sub_labels, metric="full",
                n_splits=n_cv_splits, seed=sub_seed,
            ),
        })
    return out


# =============================================================================
# (d) Plotting
# =============================================================================


def _frame_to_ms(frame: int | np.ndarray) -> int | np.ndarray:
    """Convert frame index to milliseconds at the MICrONS 7.5 Hz sampling rate."""
    return frame * 1000.0 / 7.5


def plot_psth_per_stim_area(
    trajectories_per_area: dict[str, dict[str, np.ndarray]],
    area_name: str,
    *,
    ax: plt.Axes | None = None,
    palette: dict[str, str] | None = None,
    stim_order: list[str] | None = None,
) -> plt.Axes:
    """Per-frame area-mean response, one curve per stim class.

    Sanity check before any geometric analysis. Three curves on shared
    axes; the rawest temporal view of stimulus differences.

    Args:
        trajectories_per_area: nested dict from
            :func:`build_stim_trajectories`.
        area_name: which area to plot.
        ax: matplotlib axes; created if None.
        palette: Optional dict ``{stim: hex_color}`` overriding
            ``STIM_COLORS``. Used when the same plotter is reused with
            different stim labels (e.g. Option 4's category palette).
            ``None`` keeps Option 3 behaviour.
        stim_order: Optional list of stim keys controlling iteration
            order, overriding ``STIM_ORDER``. ``None`` keeps Option 3
            behaviour.

    Returns:
        The axes drawn on.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    per_stim = trajectories_per_area[area_name]
    _palette = palette if palette is not None else STIM_COLORS
    _order = stim_order if stim_order is not None else STIM_ORDER
    for stim in _order:
        if stim not in per_stim:
            continue
        traj = per_stim[stim]  # (n_frames, n_neurons)
        area_mean_per_frame = traj.mean(axis=1)  # (n_frames,)
        n_frames = len(area_mean_per_frame)
        ax.plot(
            np.arange(n_frames), area_mean_per_frame,
            color=_palette[stim], linewidth=2, label=stim,
        )
    ax.set_xlabel("Frame")
    ax.set_ylabel("Area-mean activity (z-scored, post-detrend)")
    ax.set_title(f"{area_name}: per-stim PSTH (area mean across neurons)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    # Secondary axis with milliseconds.
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    n_frames = next(iter(per_stim.values())).shape[0]
    ms_ticks = np.linspace(0, n_frames - 1, 6)
    ax2.set_xticks(ms_ticks)
    ax2.set_xticklabels([f"{int(_frame_to_ms(t))}" for t in ms_ticks])
    ax2.set_xlabel("ms (post stim onset)")
    return ax


def plot_trajectory_2d(
    stim_pcs: dict[str, np.ndarray],
    area_name: str,
    pca: PCA,
    *,
    ax: plt.Axes | None = None,
    annotate_every: int = 15,
    palette: dict[str, str] | None = None,
    stim_order: list[str] | None = None,
) -> plt.Axes:
    """2-D PC1-PC2 trajectory plot, one curve per stim.

    Filled circle at frame 0, open circle at the last frame. Frame
    annotations every ``annotate_every`` frames.

    Args:
        stim_pcs: dict mapping stim → ``(n_frames, n_components)`` PC-space
            trajectory.
        area_name: used in title.
        pca: fitted PCA, used for variance-explained labels.
        ax: matplotlib axes; created if None.
        annotate_every: gap between time-annotation labels.
        palette: Optional dict ``{stim: hex_color}`` overriding
            ``STIM_COLORS``. Used when the same plotter is reused with
            different stim labels (e.g. Option 4's category palette).
            ``None`` keeps Option 3 behaviour.
        stim_order: Optional list of stim keys controlling iteration
            order, overriding ``STIM_ORDER``. ``None`` keeps Option 3
            behaviour.

    Returns:
        The axes drawn on.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5.5))
    _palette = palette if palette is not None else STIM_COLORS
    _order = stim_order if stim_order is not None else STIM_ORDER
    for stim in _order:
        if stim not in stim_pcs:
            continue
        traj = stim_pcs[stim]  # (n_frames, n_components)
        n_frames = traj.shape[0]
        ax.plot(
            traj[:, 0], traj[:, 1],
            color=_palette[stim], linewidth=2, alpha=0.85, label=stim,
        )
        # Frame 0 marker (filled).
        ax.scatter(traj[0, 0], traj[0, 1],
                   color=_palette[stim], s=80, zorder=4,
                   edgecolor="black", linewidth=1)
        # Last frame (open circle).
        ax.scatter(traj[-1, 0], traj[-1, 1],
                   facecolor="white", edgecolor=_palette[stim],
                   s=80, zorder=4, linewidth=2)
        # Light annotations every annotate_every frames.
        for f in range(annotate_every, n_frames - 1, annotate_every):
            ax.text(traj[f, 0], traj[f, 1], f"{f}",
                    color=_palette[stim], fontsize=7,
                    ha="center", va="center",
                    bbox=dict(boxstyle="circle,pad=0.15",
                              facecolor="white",
                              edgecolor=_palette[stim],
                              linewidth=0.6, alpha=0.85))

    pc1_var = 100 * pca.explained_variance_ratio_[0]
    pc2_var = 100 * pca.explained_variance_ratio_[1]
    ax.set_xlabel(f"PC1 ({pc1_var:.1f}% var.; likely temporal)")
    ax.set_ylabel(f"PC2 ({pc2_var:.1f}% var.)")
    ax.set_title(
        f"{area_name}: trajectories (filled = frame 0, open = last frame)"
    )
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    return ax


def plot_trajectory_3d_plotly(
    stim_pcs: dict[str, np.ndarray],
    area_name: str,
    pca: PCA,
    *,
    palette: dict[str, str] | None = None,
    stim_order: list[str] | None = None,
):
    """Interactive 3-D trajectory plot in PC1-PC2-PC3 space.

    Returns a Plotly Figure. Caller saves with ``fig.write_html(path)``.

    Args:
        palette: Optional dict ``{stim: hex_color}`` overriding
            ``STIM_COLORS``. Used when the same plotter is reused with
            different stim labels (e.g. Option 4's category palette).
            ``None`` keeps Option 3 behaviour.
        stim_order: Optional list of stim keys controlling iteration
            order, overriding ``STIM_ORDER``. ``None`` keeps Option 3
            behaviour.
    """
    import plotly.graph_objects as go
    fig = go.Figure()
    _palette = palette if palette is not None else STIM_COLORS
    _order = stim_order if stim_order is not None else STIM_ORDER
    for stim in _order:
        if stim not in stim_pcs:
            continue
        traj = stim_pcs[stim]
        n_frames = traj.shape[0]
        fig.add_trace(go.Scatter3d(
            x=traj[:, 0], y=traj[:, 1], z=traj[:, 2],
            mode="lines+markers",
            line=dict(color=_palette[stim], width=4),
            marker=dict(size=3, color=_palette[stim]),
            name=stim,
            text=[f"frame {f} ({int(_frame_to_ms(f))} ms)"
                  for f in range(n_frames)],
        ))
        # Frame 0 emphasised (filled).
        fig.add_trace(go.Scatter3d(
            x=[traj[0, 0]], y=[traj[0, 1]], z=[traj[0, 2]],
            mode="markers",
            marker=dict(size=8, color=_palette[stim],
                        line=dict(color="black", width=1)),
            name=f"{stim} t=0", showlegend=False,
        ))
    pc_var = 100 * pca.explained_variance_ratio_[:3]
    fig.update_layout(
        title=f"{area_name}: trajectories in PC1-PC2-PC3",
        scene=dict(
            xaxis_title=f"PC1 ({pc_var[0]:.1f}% var.) [likely temporal]",
            yaxis_title=f"PC2 ({pc_var[1]:.1f}% var.)",
            zaxis_title=f"PC3 ({pc_var[2]:.1f}% var.)",
        ),
        width=850, height=700,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def plot_pairwise_distance_time_course(
    distances: dict[frozenset[str], np.ndarray],
    envelopes: dict[frozenset[str], dict[str, np.ndarray]] | None,
    area_name: str,
    metric_label: str,
    *,
    null_p95: dict[frozenset[str], float] | None = None,
    ax: plt.Axes | None = None,
    pair_palette: dict[frozenset[str], str] | None = None,
    pair_labels: dict[frozenset[str], str] | None = None,
) -> plt.Axes:
    """Three pairwise-distance time courses on shared axes, with envelopes.

    Args:
        distances: pair-keyed dict of ``(n_frames,)`` observed distances.
        envelopes: optional pair-keyed dict with ``"lower"``/``"upper"``
            shape ``(n_frames,)``. If supplied, drawn as a shaded fill.
        area_name: used in title.
        metric_label: used in y-axis label (e.g. "Euclidean (full features)").
        null_p95: optional pair-keyed scalar, the 95th percentile of the
            shuffle null. Drawn as a horizontal dashed line per pair.
        ax: matplotlib axes; created if None.
        pair_palette: Optional dict
            ``{frozenset({a, b}): hex_color}`` overriding the hardcoded
            Clip/Monet2/Trippy pair colors. ``None`` keeps Option 3
            behaviour.
        pair_labels: Optional dict
            ``{frozenset({a, b}): label_str}`` overriding the hardcoded
            Clip↔Monet2 / Clip↔Trippy / Monet2↔Trippy labels. ``None``
            keeps Option 3 behaviour.

    Returns:
        The axes drawn on.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))

    pair_to_label = pair_labels if pair_labels is not None else {
        frozenset({"Clip", "Monet2"}): "Clip↔Monet2",
        frozenset({"Clip", "Trippy"}): "Clip↔Trippy",
        frozenset({"Monet2", "Trippy"}): "Monet2↔Trippy",
    }
    pair_to_color = pair_palette if pair_palette is not None else {
        frozenset({"Clip", "Monet2"}): "#0072B2",
        frozenset({"Clip", "Trippy"}): "#E69F00",
        frozenset({"Monet2", "Trippy"}): "#009E73",
    }
    for pair, dist in distances.items():
        n_frames = len(dist)
        x = np.arange(n_frames)
        color = pair_to_color.get(pair, "grey")
        label = pair_to_label.get(pair, "/".join(sorted(pair)))
        ax.plot(x, dist, color=color, linewidth=2, label=label)
        if envelopes and pair in envelopes:
            ax.fill_between(
                x, envelopes[pair]["lower"], envelopes[pair]["upper"],
                color=color, alpha=0.2,
            )
        if null_p95 and pair in null_p95:
            ax.axhline(null_p95[pair], color=color, linestyle="--",
                       alpha=0.6, linewidth=1)
    ax.set_xlabel("Frame")
    ax.set_ylabel(metric_label)
    ax.set_title(f"{area_name}: pairwise trajectory distance ({metric_label})")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    # Secondary ms axis.
    n_frames = len(next(iter(distances.values())))
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ms_ticks = np.linspace(0, n_frames - 1, 6)
    ax2.set_xticks(ms_ticks)
    ax2.set_xticklabels([f"{int(_frame_to_ms(t))}" for t in ms_ticks])
    ax2.set_xlabel("ms (post stim onset)")
    return ax


def plot_cross_area_monet2_trippy(
    per_area_distances: dict[str, np.ndarray],
    per_area_envelopes: dict[str, dict[str, np.ndarray]] | None,
    metric_label: str,
    *,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Headline figure: Monet2↔Trippy distance time course, all four areas.

    Args:
        per_area_distances: area → ``(n_frames,)`` Monet2↔Trippy distance.
        per_area_envelopes: area → dict with ``"lower"``/``"upper"``.
        metric_label: used in y-axis label.
        ax: matplotlib axes; created if None.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))
    palette = {"V1": "#0072B2", "AL": "#E69F00", "LM": "#009E73", "RL": "#CC79A7"}
    for area, dist in per_area_distances.items():
        x = np.arange(len(dist))
        color = palette.get(area, "grey")
        ax.plot(x, dist, color=color, linewidth=2.2, label=area)
        if per_area_envelopes and area in per_area_envelopes:
            ax.fill_between(
                x, per_area_envelopes[area]["lower"],
                per_area_envelopes[area]["upper"],
                color=color, alpha=0.2,
            )
    ax.set_xlabel("Frame")
    ax.set_ylabel(metric_label)
    ax.set_title(f"Monet2↔Trippy distance across cortical areas ({metric_label})")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    n_frames = len(next(iter(per_area_distances.values())))
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ms_ticks = np.linspace(0, n_frames - 1, 6)
    ax2.set_xticks(ms_ticks)
    ax2.set_xticklabels([f"{int(_frame_to_ms(t))}" for t in ms_ticks])
    ax2.set_xlabel("ms (post stim onset)")
    return ax


def plot_population_matched_distance_comparison(
    full_distance: np.ndarray,
    matched_distances: list[np.ndarray],
    area_name: str,
    pair_label: str,
    metric_label: str,
    *,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Full-population vs equal-population distance time courses for one area / pair.

    Args:
        full_distance: ``(n_frames,)`` distance from the all-neurons analysis.
        matched_distances: list of ``(n_frames,)`` distances, one per
            subsample (typically 20).
        area_name, pair_label, metric_label: used in title and labels.
        ax: matplotlib axes; created if None.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4.5))
    n_frames = len(full_distance)
    x = np.arange(n_frames)
    matched_arr = np.array(matched_distances)
    median = np.median(matched_arr, axis=0)
    q25 = np.quantile(matched_arr, 0.25, axis=0)
    q75 = np.quantile(matched_arr, 0.75, axis=0)
    ax.plot(x, full_distance, "-", color="steelblue", linewidth=2.2,
            label="all neurons")
    ax.plot(x, median, "-", color="lightcoral", linewidth=2,
            label="matched to AL (median)")
    ax.fill_between(x, q25, q75, color="lightcoral", alpha=0.25,
                    label="matched IQR (Q1-Q3)")
    ax.set_xlabel("Frame")
    ax.set_ylabel(metric_label)
    ax.set_title(f"{area_name}: {pair_label} (full vs. equal-population)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    return ax


def plot_clip_subsampling_comparison(
    full_distances: dict[frozenset[str], np.ndarray],
    subsampled_distances_list: list[dict[frozenset[str], np.ndarray]],
    metric_label: str,
    *,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """V1-only: full-Clip vs subsampled-Clip distance time courses.

    Two pairs shown on the same axes: Clip↔Monet2 and Clip↔Trippy
    (Monet2↔Trippy is unaffected by Clip-trial count and is omitted).

    Args:
        full_distances: pair → ``(n_frames,)`` from all-Clip analysis.
        subsampled_distances_list: list of pair → ``(n_frames,)`` dicts,
            one per Clip subsample.
        metric_label: used in y-axis label.
        ax: matplotlib axes; created if None.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))
    pairs_of_interest = [
        (frozenset({"Clip", "Monet2"}), "Clip↔Monet2", "#0072B2"),
        (frozenset({"Clip", "Trippy"}), "Clip↔Trippy", "#E69F00"),
    ]
    n_frames = len(next(iter(full_distances.values())))
    x = np.arange(n_frames)
    for pair, label, color in pairs_of_interest:
        if pair not in full_distances:
            continue
        ax.plot(x, full_distances[pair], "-", color=color, linewidth=2.2,
                label=f"{label} (all 377 Clip)")
        sub_arr = np.array([sd[pair] for sd in subsampled_distances_list])
        median = np.median(sub_arr, axis=0)
        q25 = np.quantile(sub_arr, 0.25, axis=0)
        q75 = np.quantile(sub_arr, 0.75, axis=0)
        ax.plot(x, median, "--", color=color, linewidth=2,
                label=f"{label} (Clip subsampled to 38)")
        ax.fill_between(x, q25, q75, color=color, alpha=0.18)
    ax.set_xlabel("Frame")
    ax.set_ylabel(metric_label)
    ax.set_title("V1: Clip-trial subsampling sanity check")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    return ax
