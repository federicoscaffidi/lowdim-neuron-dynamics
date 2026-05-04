"""option3_trajectories_utils — helper module for the time-resolved PCA trajectories notebook.

The module is organized in four sections:

(a) Trajectory construction — build_stim_trajectories, save_trajectories,
    load_trajectories.
(b) PCA fit and projection — fit_trajectory_pca.
(c) Distance metrics, bootstrap, null, and subsample pipelines —
    pairwise_trajectory_distance, bootstrap_distance_envelope,
    shuffle_null_max_distance, compute_onset_latency,
    subsample_population_run_pipeline, subsample_clip_trials_run_pipeline.
(d) Plotting — PSTH per area, 2-D / 3-D trajectory plots, distance time-course
    panels, cross-area headline, equal-population comparison, Clip-subsampling
    comparison.

Design constraints (see docs/specs/2026-05-04-option3-trajectories-design.md):
- Data functions are pure: no plotting, no file writes (except the explicit
  save_trajectories / load_trajectories).
- Plotting functions accept ``ax`` (or return a Figure for Plotly) so plots
  compose.
- Every public function has a NumPy-style docstring with Args, Returns, Notes.
- No magic constants — anything tunable is a parameter with a sensible default.
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
from option2_pca_utils import STIM_COLORS, STIM_MARKERS, STIM_ORDER


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
        produce trajectories with very different smoothness — the Clip
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
    Each class is resampled to its own original size — preserving class
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
) -> dict[frozenset[str], dict[str, np.ndarray]]:
    """95% bootstrap envelope on each pairwise trajectory distance.

    For each of ``n_boot`` iterations, class-stratified resamples the trials,
    rebuilds per-stim trajectories from the resampled tensor, and recomputes
    pairwise distances. Returns the 2.5/97.5 percentile envelopes per pair
    per frame. **PCA is NOT refit** — fixed at the supplied basis to keep
    variability attributable to the trajectories, not the basis.

    Args:
        trial_tensor: ``(n_trials, n_frames, n_neurons_in_area)`` — the per-area
            trial × time × neuron tensor (build once and pass in).
        labels: ``(n_trials,)`` stim labels.
        metric: ``"full"`` or ``"top_pcs"``; same convention as
            :func:`pairwise_trajectory_distance`.
        pca: required if ``metric="top_pcs"``.
        n_pcs: passed to distance computation.
        n_boot: number of bootstrap iterations.
        seed: master seed.

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

    accum = {pk: np.empty((n_boot, n_frames)) for pk in pair_keys}
    for b in range(n_boot):
        sub_seed = int(rng.integers(0, 2**31 - 1))
        sub_rng = np.random.default_rng(sub_seed)
        resampled_tensor, resampled_labels = _resample_per_class(
            trial_tensor, label_arr, rng=sub_rng
        )
        traj = _trajectories_from_trial_tensor(resampled_tensor, resampled_labels)
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
        shuffled = rng.permutation(label_arr)
        traj = _trajectories_from_trial_tensor(trial_tensor, shuffled)
        pairs = pairwise_trajectory_distance(
            traj, metric=metric, pca=pca, n_pcs=n_pcs
        )
        for pk in pair_keys:
            nulls[pk][s] = float(pairs[pk].max())
    return nulls


def compute_onset_latency(
    distance_lower_envelope: np.ndarray,
    null_distribution: np.ndarray,
    *,
    alpha: float = 0.05,
) -> int | None:
    """First frame at which bootstrap-lower exceeds null (1-alpha) percentile.

    Args:
        distance_lower_envelope: shape ``(n_frames,)``, the 2.5th-percentile
            bootstrap envelope of the observed pairwise distance.
        null_distribution: shape ``(n_shuffles,)``, the shuffle null max
            distances (used to compute the upper threshold).
        alpha: significance level. Default 0.05.

    Returns:
        Onset frame index (0-based, integer) at which the lower envelope
        first exceeds the ``(1 - alpha) * 100``-th percentile of the null.
        Returns ``None`` if no frame satisfies.
    """
    threshold = float(np.percentile(null_distribution, 100 * (1 - alpha)))
    crosses = np.where(distance_lower_envelope > threshold)[0]
    return int(crosses[0]) if len(crosses) else None


def subsample_population_run_pipeline(
    area_name: str,
    trial_tensor: np.ndarray,
    labels: np.ndarray,
    *,
    n_match: int,
    n_subsamples: int = 20,
    n_components: int = 10,
    seed: int = 42,
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
        - ``"col_indices"``: shape ``(n_match,)`` — which neurons were sampled.

    Notes:
        Bootstrap envelopes and shuffle nulls are NOT recomputed per subsample
        (compute cost would 20×). The cross-area-ranking question is answered
        by comparing the median ± IQR of the observed distances across the 20
        subsamples.
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
) -> list[dict]:
    """Clip-trial subsampling control: trim Clip trial count, recompute.

    For each of ``n_subsamples`` random subsamples of Clip trials down to
    ``n_clip_target``, rebuilds per-stim trajectories using the trimmed Clip
    trials plus all Monet2/Trippy trials, refits PCA, and computes pairwise
    distances.

    Args:
        trial_tensor: ``(n_trials, n_frames, n_neurons)``.
        labels: ``(n_trials,)`` — must include at least one ``"Clip"`` trial.
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
        })
    return out


# =============================================================================
# (d) Plotting
# =============================================================================


# (plotters go here)
