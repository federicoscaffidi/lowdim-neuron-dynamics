"""microns_eda: helper module for the MICrONS exploratory data analysis notebook.

The module is organized in four sections:

(a) Loading:     wrappers around MicronsFunctionalReader plus h5py fallback.
(b) Aggregation: per-session and cross-session summary statistics.
(c) Analysis:    deep-dive diagnostics (per-neuron stats, correlation, drift).
(d) Plotting:    thin matplotlib wrappers; all accept an optional `ax`.

Design constraints:
- Data functions are pure: no plotting, no file writes.
- Plotting functions accept `ax` so plots compose.
- Every public function has a NumPy-style docstring with Args, Returns, Notes.
- No magic constants: anything tunable is a parameter with a sensible default.
"""

from __future__ import annotations

import warnings
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


def open_dataset(datadir: str | Path) -> "MicronsFunctionalReader":
    """Construct a MicronsFunctionalReader pointed at the given data directory.

    Args:
        datadir: Directory containing ``microns.h5``. Relative paths are
            resolved against the current working directory (i.e. the location
            of the notebook).

    Returns:
        A configured ``microns_datacleaner.MicronsFunctionalReader`` instance.
        Reuse this object across loader calls; construction touches the H5
        file index and is not free.

    Notes:
        Imports ``microns_datacleaner`` lazily so this module can be imported
        in environments without it (e.g. for static type checking).
    """
    import microns_datacleaner as mic
    return mic.MicronsFunctionalReader(datadir=str(datadir))


@contextmanager
def open_h5(datadir: str | Path) -> Iterator[h5py.File]:
    """Open the MICrONS H5 read-only as a context manager.

    The file is expected at ``{datadir}/functional/microns_functional.h5``,
    the layout ``MicronsFunctionalReader`` looks for. If the file is
    actually somewhere else on disk (e.g. ``{datadir}/microns.h5`` from a
    direct download), create a symlink at the expected path; the
    notebook's setup cell does this automatically when needed.

    Used by loaders that need fields ``MicronsFunctionalReader`` does not
    expose: per-trial pupil, treadmill, stim_times, and the per-session
    ``meta`` group (coordinates, area_indices, brain_areas, unit_ids,
    condition_hashes).

    Args:
        datadir: Directory containing the ``functional/microns_functional.h5``
            file (matching the ``MicronsFunctionalReader`` convention).

    Yields:
        An open ``h5py.File`` in read-only mode.
    """
    path = Path(datadir) / "functional" / "microns_functional.h5"
    with h5py.File(path, "r") as f:
        yield f


def list_sessions(datadir: str | Path) -> list[str]:
    """Return the 14 session keys present in ``microns.h5``, sorted.

    Args:
        datadir: Directory containing ``microns.h5``.

    Returns:
        Sorted list of session keys (e.g. ``["4_7", "5_6", ..., "9_6"]``).

    Notes:
        Reads ``/sessions/`` directly via ``h5py``; the reader does not
        expose a session listing in a stable form.
    """
    with open_h5(datadir) as f:
        return sorted(f["sessions"].keys())


def get_session_meta(datadir: str | Path, session: str) -> dict:
    """Return per-session metadata for one session.

    Args:
        datadir: Directory containing ``microns.h5``.
        session: Session key, e.g. ``"7_5"``.

    Returns:
        Dictionary with keys:

        - ``n_neurons`` (int): total neurons in this session.
        - ``area_indices`` (dict[str, np.ndarray]): per-area arrays of
          neuron indices into the neuron axis.
        - ``coordinates`` (np.ndarray): shape ``(n_neurons, 3)``,
          neuron 3D positions in pial coordinates.
        - ``unit_ids`` (np.ndarray): shape ``(n_neurons,)``, the global
          unit ids.
        - ``condition_hashes`` (np.ndarray): shape ``(n_trials,)``, one
          condition hash per trial in trial order. May contain bytes.
        - ``n_trials`` (int): number of trials in this session.
        - ``brain_areas`` (np.ndarray): shape ``(n_neurons,)``, area label
          per neuron (bytes; decode if comparing to strings).

    Notes:
        Reads from ``/sessions/{session}/meta/`` directly via ``h5py``.
    """
    with open_h5(datadir) as f:
        meta = f["sessions"][session]["meta"]
        area_indices = {
            area: meta["area_indices"][area][...]
            for area in meta["area_indices"].keys()
        }
        coordinates = meta["coordinates"][...]
        unit_ids = meta["unit_ids"][...]
        condition_hashes = meta["condition_hashes"][...]
        brain_areas = meta["brain_areas"][...]
        n_neurons = int(coordinates.shape[0])
        n_trials = int(condition_hashes.shape[0])
    return {
        "n_neurons": n_neurons,
        "area_indices": area_indices,
        "coordinates": coordinates,
        "unit_ids": unit_ids,
        "condition_hashes": condition_hashes,
        "n_trials": n_trials,
        "brain_areas": brain_areas,
    }


def build_stim_type_map(reader, session: str) -> dict[str, str]:
    """Map every per-trial condition hash in a session to its stim type.

    Args:
        reader: A ``MicronsFunctionalReader`` (from ``open_dataset``).
        session: Session key, e.g. ``"7_5"``.

    Returns:
        Dictionary ``{condition_hash: stim_type}`` covering every trial
        in this session. ``stim_type`` is one of
        ``{"Clip", "Monet2", "Trippy", "Unknown"}``. ``"Unknown"`` is
        used for hashes the reader cannot classify.

    Notes:
        The library handles URL-encoding (``%2F`` ↔ ``/``) internally,
        so we work with the hash strings the reader returns directly.
    """
    hashes = reader.get_hashes_by_session(session)
    stim_map: dict[str, str] = {}
    for h in hashes:
        try:
            stim_type = reader.get_video_type(h)
        except Exception:
            stim_type = "Unknown"
        if stim_type not in {"Clip", "Monet2", "Trippy"}:
            stim_type = "Unknown"
        stim_map[h] = stim_type
    return stim_map


def load_trial(
    reader,
    datadir: str | Path,
    session: str,
    trial_idx: int,
) -> dict:
    """Load all per-trial data for one trial.

    Args:
        reader: A ``MicronsFunctionalReader``.
        datadir: Directory containing ``microns.h5``.
        session: Session key.
        trial_idx: Trial index (0-based) within the session.

    Returns:
        Dictionary with:

        - ``responses`` (np.ndarray): shape ``(n_neurons, n_frames)``.
        - ``pupil`` (np.ndarray): shape ``(4, n_frames)``. Channels are
          (center_x, center_y, diameter, fourth-channel-unverified). The
          notebook's preprocessing-diagnostics step verifies the diameter
          channel index empirically; document the choice when used.
        - ``treadmill`` (np.ndarray): shape ``(n_frames, 1)``, running
          speed.
        - ``stim_times`` (np.ndarray): shape ``(n_frames,)``, timestamp
          of each frame in the session clock.
        - ``stim_type`` (str): one of ``Clip / Monet2 / Trippy / Unknown``.
        - ``condition_hash`` (str): the trial's condition hash.

    Notes:
        ``responses``, ``pupil``, ``treadmill``, ``stim_times`` are read
        directly via ``h5py`` (library does not expose them as a single
        bundle). ``stim_type`` is resolved through the reader.
    """
    with open_h5(datadir) as f:
        trial = f["sessions"][session]["trials"][str(trial_idx)]
        responses = trial["responses"][...]
        pupil = trial["pupil"][...]
        treadmill = trial["treadmill"][...]
        stim_times = trial["stim_times"][...]
        condition_hash_raw = f["sessions"][session]["meta"]["condition_hashes"][trial_idx]

    if isinstance(condition_hash_raw, bytes):
        condition_hash = condition_hash_raw.decode("utf-8", errors="replace")
    else:
        condition_hash = str(condition_hash_raw)

    try:
        stim_type = reader.get_video_type(condition_hash)
        if stim_type not in {"Clip", "Monet2", "Trippy"}:
            stim_type = "Unknown"
    except Exception:
        stim_type = "Unknown"

    return {
        "responses": responses,
        "pupil": pupil,
        "treadmill": treadmill,
        "stim_times": stim_times,
        "stim_type": stim_type,
        "condition_hash": condition_hash,
    }


def load_session_responses(
    reader,
    datadir: str | Path,
    session: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load all responses for a session into one matrix, with trial boundaries.

    Args:
        reader: A ``MicronsFunctionalReader``.
        datadir: Directory containing ``microns.h5``.
        session: Session key.

    Returns:
        A tuple ``(responses, trial_boundaries, stim_types_per_trial)``.

        - ``responses``: shape ``(n_neurons, total_timesteps)``,
          horizontally concatenated across trials in trial order.
        - ``trial_boundaries``: shape ``(n_trials + 1,)``,
          cumulative timestep counts so trial ``i`` occupies columns
          ``trial_boundaries[i] : trial_boundaries[i+1]``.
        - ``stim_types_per_trial``: shape ``(n_trials,)``,
          dtype ``object``, stim type label per trial.

    Notes:
        Memory: a typical session is ~7,500 neurons × ~40,000 timesteps
        × 4 bytes ≈ 1.2 GB. Fits comfortably in laptop RAM.
    """
    meta = get_session_meta(datadir, session)
    n_trials = meta["n_trials"]
    n_neurons = meta["n_neurons"]
    condition_hashes = meta["condition_hashes"]

    # First pass: read each trial's response width to compute trial_boundaries.
    widths = np.empty(n_trials, dtype=np.int64)
    with open_h5(datadir) as f:
        trials_grp = f["sessions"][session]["trials"]
        for i in range(n_trials):
            widths[i] = trials_grp[str(i)]["responses"].shape[1]
    trial_boundaries = np.concatenate(([0], np.cumsum(widths)))
    total_timesteps = int(trial_boundaries[-1])

    responses = np.empty((n_neurons, total_timesteps), dtype=np.float32)
    stim_types_per_trial = np.empty(n_trials, dtype=object)

    with open_h5(datadir) as f:
        trials_grp = f["sessions"][session]["trials"]
        for i in range(n_trials):
            start = int(trial_boundaries[i])
            end = int(trial_boundaries[i + 1])
            responses[:, start:end] = trials_grp[str(i)]["responses"][...]

            h_raw = condition_hashes[i]
            h = h_raw.decode("utf-8", "replace") if isinstance(h_raw, bytes) else str(h_raw)
            try:
                stim_type = reader.get_video_type(h)
                if stim_type not in {"Clip", "Monet2", "Trippy"}:
                    stim_type = "Unknown"
            except Exception:
                stim_type = "Unknown"
            stim_types_per_trial[i] = stim_type

    return responses, trial_boundaries, stim_types_per_trial


# =============================================================================
# (b) Aggregation (cross-session)
# =============================================================================


def summarize_session(reader, datadir: str | Path, session: str) -> dict:
    """Compute one row of summary statistics for a session via streaming.

    Args:
        reader: A ``MicronsFunctionalReader``.
        datadir: Directory containing ``microns.h5``.
        session: Session key.

    Returns:
        Dictionary with:

        - ``session`` (str): the session key.
        - ``n_neurons`` (int): total neurons.
        - ``n_V1``, ``n_AL``, ``n_LM``, ``n_RL`` (int): neurons per area.
        - ``n_trials`` (int): trial count.
        - ``n_Clip``, ``n_Monet2``, ``n_Trippy``, ``n_Unknown`` (int):
          trial counts per stim class.
        - ``median_per_neuron_mean`` (float): median across neurons of
          per-neuron mean response.
        - ``median_per_neuron_var`` (float): median across neurons of
          per-neuron response variance.
        - ``mean_pupil_diameter`` (float): mean of the pupil diameter
          channel across all timesteps in the session.
        - ``mean_treadmill_speed`` (float): mean treadmill value across
          all timesteps in the session.
        - ``median_trial_duration`` (int): median trial length in
          timesteps.

    Notes:
        Streaming aggregation: holds at most one trial in memory at a
        time. Per-neuron mean and variance use Welford's online
        algorithm.

        The pupil tensor's diameter channel is taken to be index 2
        (channels: center_x, center_y, diameter, fourth-channel). This
        is the working assumption from h5_deep_inspection.txt; verify
        empirically in the deep-dive section 2e and update here if the
        index is different.
    """
    meta = get_session_meta(datadir, session)
    n_neurons = meta["n_neurons"]
    n_trials = meta["n_trials"]
    area_indices = meta["area_indices"]
    condition_hashes = meta["condition_hashes"]

    # Vectorized chunk-Welford: combine per-trial (count, mean, M2) into
    # running accumulators. Per-trial stats are computed with numpy
    # (no Python timestep loop).
    total_count = 0
    running_mean = np.zeros(n_neurons, dtype=np.float64)
    running_M2 = np.zeros(n_neurons, dtype=np.float64)

    pupil_sum = 0.0
    pupil_count = 0
    tread_sum = 0.0
    tread_count = 0

    stim_counts = {"Clip": 0, "Monet2": 0, "Trippy": 0, "Unknown": 0}
    durations = np.empty(n_trials, dtype=np.int64)

    PUPIL_DIAMETER_CHANNEL = 2

    with open_h5(datadir) as f:
        trials_grp = f["sessions"][session]["trials"]
        for i in range(n_trials):
            tr = trials_grp[str(i)]
            responses = tr["responses"][...].astype(np.float64, copy=False)
            pupil = tr["pupil"][...]
            treadmill = tr["treadmill"][...]
            n_t = responses.shape[1]
            durations[i] = n_t

            # Per-trial moments (vectorized over neurons).
            trial_mean = responses.mean(axis=1)
            trial_M2 = ((responses - trial_mean[:, None]) ** 2).sum(axis=1)

            # Chunk-Welford merge:
            # combined_n     = n1 + n2
            # delta          = mean2 - mean1
            # combined_mean  = mean1 + delta * n2 / n
            # combined_M2    = M2_1 + M2_2 + delta^2 * n1 * n2 / n
            new_count = total_count + n_t
            delta = trial_mean - running_mean
            if total_count == 0:
                running_mean = trial_mean.copy()
                running_M2 = trial_M2.copy()
            else:
                running_mean = running_mean + delta * (n_t / new_count)
                running_M2 = running_M2 + trial_M2 + (delta ** 2) * (total_count * n_t / new_count)
            total_count = new_count

            # Behavior: nan-aware. The eye tracker drops frames as NaN, and
            # treadmill occasionally has NaN too; ignore them so one bad frame
            # doesn't poison the whole session's mean.
            pupil_channel = pupil[PUPIL_DIAMETER_CHANNEL]
            pupil_sum += float(np.nansum(pupil_channel))
            pupil_count += int(np.isfinite(pupil_channel).sum())
            tread_sum += float(np.nansum(treadmill))
            tread_count += int(np.isfinite(treadmill).sum())

            h_raw = condition_hashes[i]
            h = h_raw.decode("utf-8", "replace") if isinstance(h_raw, bytes) else str(h_raw)
            try:
                stim_type = reader.get_video_type(h)
                if stim_type not in {"Clip", "Monet2", "Trippy"}:
                    stim_type = "Unknown"
            except Exception:
                stim_type = "Unknown"
            stim_counts[stim_type] += 1

    per_neuron_var = running_M2 / max(total_count - 1, 1)

    area_counts = {f"n_{a}": int(idx.shape[0]) for a, idx in area_indices.items()}

    return {
        "session": session,
        "n_neurons": int(n_neurons),
        **area_counts,
        "n_trials": int(n_trials),
        "n_Clip": stim_counts["Clip"],
        "n_Monet2": stim_counts["Monet2"],
        "n_Trippy": stim_counts["Trippy"],
        "n_Unknown": stim_counts["Unknown"],
        "median_per_neuron_mean": float(np.median(running_mean)),
        "median_per_neuron_var": float(np.median(per_neuron_var)),
        "mean_pupil_diameter": pupil_sum / max(pupil_count, 1),
        "mean_treadmill_speed": tread_sum / max(tread_count, 1),
        "median_trial_duration": int(np.median(durations)),
    }


def summarize_all_sessions(reader, datadir: str | Path) -> pd.DataFrame:
    """Apply ``summarize_session`` across every session in ``microns.h5``.

    Args:
        reader: A ``MicronsFunctionalReader``.
        datadir: Directory containing ``microns.h5``.

    Returns:
        DataFrame with one row per session, indexed by session key in
        sorted order. Columns are the keys returned by
        ``summarize_session``.

    Notes:
        This is the slow part of Part 1; expect minutes, not seconds.
        Each session is fully streamed once.
    """
    sessions = list_sessions(datadir)
    rows = [summarize_session(reader, datadir, s) for s in sessions]
    df = pd.DataFrame(rows).set_index("session")
    return df


def flag_outlier_sessions(summary_df: pd.DataFrame, z_thresh: float = 2.0) -> pd.DataFrame:
    """Annotate the summary DataFrame with z-scores and an outlier flag.

    Args:
        summary_df: Output of ``summarize_all_sessions``.
        z_thresh: Absolute z-score above which a session is flagged.

    Returns:
        Copy of ``summary_df`` with added columns:

        - ``z_n_neurons``, ``z_median_per_neuron_mean``,
          ``z_mean_pupil_diameter``, ``z_mean_treadmill_speed``,
          ``z_n_Unknown`` (float): per-metric z-scores.
        - ``warning`` (str): a comma-separated list of metrics on which
          the session is an outlier, or empty string if none.
        - ``is_outlier`` (bool): True iff ``warning != ""``.

    Notes:
        Z-scores are computed using sample mean and std across the 14
        sessions. With 14 sessions a |z|>2 threshold is roughly the
        outer ~5% of a normal-ish distribution; treat it as a flag, not
        a verdict.

        Robustness: a metric whose column is constant across all
        sessions (std = 0, e.g. ``n_Unknown`` when the library
        classifies every trial cleanly) yields a z-score of 0, so no
        flag is raised, since "every session is identical" is not an
        outlier signal. A metric that is NaN everywhere (e.g. broken
        sensor) yields NaN z-scores, which never trip the |z| > 2
        check, so the column is silently ignored.
    """
    metrics = [
        "n_neurons",
        "median_per_neuron_mean",
        "mean_pupil_diameter",
        "mean_treadmill_speed",
        "n_Unknown",
    ]
    out = summary_df.copy()
    warnings_per_session: list[list[str]] = [[] for _ in range(len(out))]
    for m in metrics:
        col = out[m].astype(float)
        std = col.std(ddof=0)
        if not np.isfinite(std) or std == 0:
            z = pd.Series(np.zeros(len(col), dtype=float), index=col.index)
        else:
            z = (col - col.mean()) / std
        out[f"z_{m}"] = z
        for i, val in enumerate(z.abs() > z_thresh):
            if bool(val):
                warnings_per_session[i].append(m)
    out["warning"] = [", ".join(w) for w in warnings_per_session]
    out["is_outlier"] = out["warning"] != ""
    return out


# =============================================================================
# (c) Analysis (deep-dive diagnostics)
# =============================================================================


def compute_neuron_stats(
    responses: np.ndarray,
    sparsity_quantile: float = 0.05,
) -> pd.DataFrame:
    """Per-neuron summary statistics for the deep-dive session.

    Args:
        responses: shape ``(n_neurons, total_timesteps)``.
        sparsity_quantile: quantile that defines the silence threshold
            *per neuron*. ``sparsity`` is the fraction of timesteps the
            neuron exceeds its own quantile-of-self. Default 0.05.

    Returns:
        DataFrame indexed by neuron id (int), with columns:

        - ``mean`` (float): per-neuron mean response.
        - ``var`` (float): per-neuron variance.
        - ``std`` (float): sqrt of var.
        - ``sparsity`` (float): see Args.
        - ``is_silent`` (bool): True if ``var == 0`` exactly, indicating
          a neuron that never responds in this session.
    """
    means = responses.mean(axis=1)
    variances = responses.var(axis=1)
    thresholds = np.quantile(responses, sparsity_quantile, axis=1)
    sparsity = (responses > thresholds[:, None]).mean(axis=1)
    return pd.DataFrame({
        "mean": means,
        "var": variances,
        "std": np.sqrt(variances),
        "sparsity": sparsity,
        "is_silent": variances == 0,
    })


def compute_correlation_matrix(
    responses: np.ndarray,
    n_subsample: int = 2000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Sub-sampled neuron × neuron correlation matrix.

    Args:
        responses: shape ``(n_neurons, total_timesteps)``.
        n_subsample: number of neurons to sample without replacement.
            Capped at ``n_neurons``.
        seed: RNG seed for the sample.

    Returns:
        Tuple ``(corr, sampled_indices)``.

        - ``corr``: shape ``(k, k)`` where ``k = min(n_subsample,
          n_neurons)``. Pearson correlation across timesteps.
        - ``sampled_indices``: shape ``(k,)``, the neuron indices chosen.

    Notes:
        Why sub-sample? See the `CORRELATION_SUBSAMPLE_N` note in
        Part 0 of the notebook.
    """
    n = responses.shape[0]
    k = min(n_subsample, n)
    rng = np.random.default_rng(seed)
    sampled_indices = np.sort(rng.choice(n, size=k, replace=False))
    sub = responses[sampled_indices]
    corr = np.corrcoef(sub)
    return corr, sampled_indices


def compute_drift(
    responses: np.ndarray,
    trial_boundaries: np.ndarray,
) -> pd.DataFrame:
    """Mean population activity per trial vs trial index, with a linear fit.

    Args:
        responses: shape ``(n_neurons, total_timesteps)``.
        trial_boundaries: shape ``(n_trials + 1,)``.

    Returns:
        DataFrame indexed by trial number, with columns:

        - ``mean_activity`` (float): mean of all (neuron, time) values
          in the trial.
        - ``trend`` (float): the value of a linear OLS fit at this trial
          index. ``slope`` is identical for every row and stored in
          ``df.attrs["slope"]``.

    Notes:
        Use to spot photobleaching (monotonic decrease) or sudden
        recording artifacts (jumps).
    """
    n_trials = len(trial_boundaries) - 1
    mean_activity = np.empty(n_trials)
    for i in range(n_trials):
        s, e = int(trial_boundaries[i]), int(trial_boundaries[i + 1])
        mean_activity[i] = float(responses[:, s:e].mean())

    x = np.arange(n_trials)
    slope, intercept = np.polyfit(x, mean_activity, 1)
    trend = slope * x + intercept

    df = pd.DataFrame({"mean_activity": mean_activity, "trend": trend})
    df.attrs["slope"] = float(slope)
    df.attrs["intercept"] = float(intercept)
    return df


def compute_behavior_alignment(
    responses: np.ndarray,
    trial_boundaries: np.ndarray,
    pupil_per_trial: list[np.ndarray],
    treadmill_per_trial: list[np.ndarray],
    pupil_diameter_channel: int = 2,
) -> pd.DataFrame:
    """Per-trial population mean and behavior summaries, with correlations.

    Args:
        responses: shape ``(n_neurons, total_timesteps)``.
        trial_boundaries: shape ``(n_trials + 1,)``.
        pupil_per_trial: list of length ``n_trials``, each a
            ``(4, n_frames_in_trial)`` array.
        treadmill_per_trial: list of length ``n_trials``, each a
            ``(n_frames_in_trial, 1)`` array.
        pupil_diameter_channel: which channel of pupil to treat as
            diameter. Default 2; verify in the deep-dive 2e.

    Returns:
        DataFrame with columns ``mean_response``, ``mean_pupil``,
        ``mean_treadmill``, indexed by trial. Plus a small attached
        ``attrs`` dict: ``{"corr_response_pupil": float,
        "corr_response_treadmill": float}``, the Pearson correlations
        across trials.

    Notes:
        NaN-aware: pupil and treadmill frames may be NaN (sensor
        dropouts). Per-trial means use ``np.nanmean`` and yield NaN if
        all frames in a trial are NaN. Cross-trial correlations are
        computed via ``pd.Series.corr`` which pairwise-deletes NaN
        rows, consistent with the nan-aware aggregation already used
        in ``summarize_session``.
    """
    n_trials = len(trial_boundaries) - 1
    mean_response = np.empty(n_trials)
    mean_pupil = np.empty(n_trials)
    mean_treadmill = np.empty(n_trials)

    def _safe_nanmean(x: np.ndarray) -> float:
        arr = np.asarray(x)
        if arr.size == 0 or np.all(np.isnan(arr)):
            return float("nan")
        return float(np.nanmean(arr))

    for i in range(n_trials):
        s, e = int(trial_boundaries[i]), int(trial_boundaries[i + 1])
        mean_response[i] = float(responses[:, s:e].mean())
        mean_pupil[i] = _safe_nanmean(pupil_per_trial[i][pupil_diameter_channel])
        mean_treadmill[i] = _safe_nanmean(treadmill_per_trial[i])

    df = pd.DataFrame({
        "mean_response": mean_response,
        "mean_pupil": mean_pupil,
        "mean_treadmill": mean_treadmill,
    })
    df.attrs["corr_response_pupil"] = float(df["mean_response"].corr(df["mean_pupil"]))
    df.attrs["corr_response_treadmill"] = float(df["mean_response"].corr(df["mean_treadmill"]))
    return df


def compute_clean_trial_indices(
    per_trial_tread_means: np.ndarray,
    *,
    threshold: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Identify trials surviving the running-outlier filter.

    Trials with mean treadmill speed exceeding ``threshold`` (in absolute
    value) are flagged as running outliers and dropped. The default
    threshold of 1.0 was set in the EDA after the per-trial behavior–response
    correlation revealed that the treadmill confound (Pearson r ≈ −0.39)
    was driven by a small minority of running trials; dropping them lowered
    |r| to ~0.22, and was adopted as the project-wide cleaning rule.

    Args:
        per_trial_tread_means: shape ``(n_trials,)``, the per-trial mean
            treadmill speed (typically computed with ``np.nanmean``).
        threshold: trials with ``|per_trial_tread_means[i]| > threshold``
            are dropped.

    Returns:
        Tuple ``(clean_trial_indices, running_mask)``.

        - ``clean_trial_indices`` (np.ndarray of int): indices of trials to
          KEEP (i.e. trials that did NOT exceed the threshold), in original
          trial order. Length = ``n_trials - running_mask.sum()``.
        - ``running_mask`` (np.ndarray of bool): shape ``(n_trials,)``;
          ``True`` for trials that exceeded the threshold (i.e. dropped
          trials). Useful for diagnostics, e.g. checking which stim classes
          the dropped trials belonged to.

    Notes:
        Both outputs are derived from a single boolean comparison; the
        function is provided as a single point of truth so the EDA and
        downstream analyses (PCA, CEBRA) all use the same rule.
    """
    per_trial_tread_means = np.asarray(per_trial_tread_means, dtype=np.float64)
    n_nan = int(np.isnan(per_trial_tread_means).sum())
    if n_nan:
        # NaN > threshold is False, so such trials would be kept silently.
        warnings.warn(
            f"{n_nan} trial(s) have NaN mean treadmill speed and are kept "
            f"as clean; they could not be screened for running.",
            stacklevel=2,
        )
    running_mask = np.abs(per_trial_tread_means) > threshold
    keep_mask = ~running_mask
    clean_trial_indices = np.where(keep_mask)[0]
    return clean_trial_indices, running_mask


# =============================================================================
# (d) Plotting
# =============================================================================


def plot_cross_session_overview(
    summary_df: pd.DataFrame,
    axes: tuple | None = None,
) -> tuple:
    """Bar plots: neurons by area (stacked) and trials by stim type per session.

    Args:
        summary_df: Output of ``summarize_all_sessions``.
        axes: Optional pair of matplotlib axes ``(ax_neurons, ax_stim)``.
            If None, a new ``(2, 1)`` figure is created.

    Returns:
        Tuple ``(fig, (ax_neurons, ax_stim))``.

    Notes:
        Useful for visually spotting sessions whose neuron count or stim
        distribution differs structurally from the rest.
    """
    if axes is None:
        fig, (ax_neurons, ax_stim) = plt.subplots(2, 1, figsize=(12, 8))
    else:
        ax_neurons, ax_stim = axes
        fig = ax_neurons.figure

    area_cols = [c for c in ["n_V1", "n_AL", "n_LM", "n_RL"] if c in summary_df.columns]
    summary_df[area_cols].plot.bar(stacked=True, ax=ax_neurons)
    ax_neurons.set_ylabel("Neuron count")
    ax_neurons.set_title("Neurons per session, stacked by area")
    ax_neurons.tick_params(axis="x", rotation=45)

    stim_cols = ["n_Clip", "n_Monet2", "n_Trippy", "n_Unknown"]
    summary_df[stim_cols].plot.bar(ax=ax_stim)
    ax_stim.set_ylabel("Trial count")
    ax_stim.set_title("Trials per stim class, per session")
    ax_stim.tick_params(axis="x", rotation=45)

    fig.tight_layout()
    return fig, (ax_neurons, ax_stim)


def plot_response_magnitude_by_session(
    reader,
    datadir: str | Path,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Boxplot of per-neuron mean response across sessions.

    Args:
        reader: A ``MicronsFunctionalReader``.
        datadir: Directory containing ``microns.h5``.
        ax: Optional matplotlib axes. If None, a new figure is created.

    Returns:
        The axes on which the boxplot was drawn.

    Notes:
        For each session, computes per-neuron mean response (averaging
        every neuron over every timestep in the session) and draws one
        box per session. Outliers indicate possible imaging-quality
        issues (calibration drift, photobleaching).

        This is a heavy operation (streams every session). Cache the
        result if you call it more than once.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))

    sessions = list_sessions(datadir)
    per_session_means: list[np.ndarray] = []
    for s in sessions:
        meta = get_session_meta(datadir, s)
        n_neurons = meta["n_neurons"]
        sums = np.zeros(n_neurons, dtype=np.float64)
        count = 0
        with open_h5(datadir) as f:
            trials_grp = f["sessions"][s]["trials"]
            for i in range(meta["n_trials"]):
                r = trials_grp[str(i)]["responses"][...]
                sums += r.sum(axis=1)
                count += r.shape[1]
        per_session_means.append(sums / max(count, 1))

    ax.boxplot(per_session_means, tick_labels=sessions, showfliers=False)
    ax.set_ylabel("Per-neuron mean response")
    ax.set_xlabel("Session")
    ax.set_title("Per-neuron mean response distribution by session")
    ax.tick_params(axis="x", rotation=45)
    return ax


def plot_behavior_by_session(
    summary_df: pd.DataFrame,
    axes: tuple | None = None,
) -> tuple:
    """Per-session bar plot of mean pupil diameter and mean treadmill speed.

    Args:
        summary_df: Output of ``summarize_all_sessions``.
        axes: Optional pair ``(ax_pupil, ax_tread)``. If None, a
            ``(2, 1)`` figure is created.

    Returns:
        Tuple ``(fig, (ax_pupil, ax_tread))``.

    Notes:
        Pupil mean is taken from the diameter channel only (see
        ``summarize_session`` notes). Treadmill mean uses the full
        signed value; large positive values indicate forward running.
    """
    if axes is None:
        fig, (ax_pupil, ax_tread) = plt.subplots(2, 1, figsize=(12, 6))
    else:
        ax_pupil, ax_tread = axes
        fig = ax_pupil.figure

    summary_df["mean_pupil_diameter"].plot.bar(ax=ax_pupil, color="purple")
    ax_pupil.set_ylabel("Mean pupil diameter")
    ax_pupil.set_title("Mean pupil diameter per session")
    ax_pupil.tick_params(axis="x", rotation=45)

    summary_df["mean_treadmill_speed"].plot.bar(ax=ax_tread, color="orange")
    ax_tread.set_ylabel("Mean treadmill speed")
    ax_tread.set_title("Mean treadmill speed per session")
    ax_tread.tick_params(axis="x", rotation=45)

    fig.tight_layout()
    return fig, (ax_pupil, ax_tread)


def plot_outlier_table(summary_df_with_flags: pd.DataFrame) -> pd.DataFrame:
    """Return a styled view of the outlier flag table.

    Args:
        summary_df_with_flags: Output of ``flag_outlier_sessions``.

    Returns:
        A DataFrame containing the outlier-relevant columns
        (``warning``, ``is_outlier``, all ``z_*`` columns, plus
        ``n_neurons`` and ``n_Unknown``). Render in a notebook by
        returning the result from a cell.

    Notes:
        Kept simple; Jupyter renders DataFrames natively. We don't
        build a styled HTML object because that complicates downstream
        copy/paste into reports.
    """
    z_cols = [c for c in summary_df_with_flags.columns if c.startswith("z_")]
    cols = ["n_neurons", "n_Unknown", *z_cols, "warning", "is_outlier"]
    return summary_df_with_flags[cols].copy()


def plot_neuron_coords_3d(meta: dict, ax: plt.Axes | None = None) -> plt.Axes:
    """3D scatter of neuron coordinates colored by brain area.

    Args:
        meta: Output of ``get_session_meta`` for one session.
        ax: Optional 3D axes. If None, a new figure with a 3D subplot
            is created.

    Returns:
        The 3D axes used for plotting.

    Notes:
        Coordinates are pial; we invert the z-axis so cortical depth
        increases downward, matching the professor's tutorial style.
    """
    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(projection="3d")
        ax.zaxis.set_inverted(True)

    coords = meta["coordinates"]
    colors = {"V1": "tab:blue", "AL": "tab:orange", "LM": "tab:green", "RL": "tab:red"}
    for area, idx in meta["area_indices"].items():
        ax.scatter(
            coords[idx, 0], coords[idx, 2], coords[idx, 1],
            s=3, alpha=0.4, color=colors.get(area, "gray"), label=area,
        )
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_zlabel("y (depth)")
    ax.legend(markerscale=3)
    ax.set_title("Neuron positions colored by area")
    return ax


def plot_neuron_coords_3d_interactive(
    meta: dict,
    marker_size: float = 2.0,
    opacity: float = 0.5,
):
    """Interactive 3D scatter of neuron coordinates colored by area (plotly).

    Counterpart to ``plot_neuron_coords_3d`` for use in Jupyter, where
    rotation and zoom help inspect the cortical layout. The matplotlib
    version remains the right choice for static figures (papers, PDFs);
    use this one when reading the notebook interactively.

    Args:
        meta: Output of ``get_session_meta`` for one session. Must
            contain ``coordinates`` (shape ``(n_neurons, 3)``) and
            ``area_indices`` (dict mapping area name to neuron-index
            array).
        marker_size: Scatter marker size in plotly screen units.
        opacity: Marker opacity in [0, 1]. Lower values help reveal
            interior structure when the cloud is dense.

    Returns:
        A ``plotly.graph_objects.Figure``. Display in a notebook with
        ``fig.show()``.

    Notes:
        Axis convention matches the matplotlib version:
        x = ``coordinates[:, 0]``, y = ``coordinates[:, 2]``,
        z = ``coordinates[:, 1]`` (depth, axis reversed so deeper
        neurons sit lower in the plot).

        ``plotly`` is imported lazily so the module can be imported in
        environments without it.
    """
    import plotly.graph_objects as go

    coords = meta["coordinates"]
    colors = {"V1": "#1f77b4", "AL": "#ff7f0e", "LM": "#2ca02c", "RL": "#d62728"}
    traces = []
    for area, idx in meta["area_indices"].items():
        traces.append(
            go.Scatter3d(
                x=coords[idx, 0],
                y=coords[idx, 2],
                z=coords[idx, 1],
                mode="markers",
                marker=dict(
                    size=marker_size,
                    color=colors.get(area, "gray"),
                    opacity=opacity,
                ),
                name=area,
            )
        )
    fig = go.Figure(data=traces)
    fig.update_layout(
        title="Neuron positions colored by area",
        scene=dict(
            xaxis_title="x",
            yaxis_title="z",
            zaxis_title="y (depth)",
            zaxis=dict(autorange="reversed"),
        ),
        legend=dict(title="Area"),
        margin=dict(l=0, r=0, t=40, b=0),
        height=600,
    )
    return fig


def plot_trial_timeline(
    stim_types_per_trial: np.ndarray,
    trial_durations: np.ndarray | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Colored strip showing the stim type for each trial in trial order.

    Args:
        stim_types_per_trial: shape ``(n_trials,)``, one of
            ``"Clip" / "Monet2" / "Trippy" / "Unknown"`` per trial.
        trial_durations: shape ``(n_trials,)``, timesteps per trial.
            If None, all trials are drawn equal width.
        ax: Optional matplotlib axes. If None, a new figure is created.

    Returns:
        The axes on which the strip was drawn.

    Notes:
        Use to spot blocked vs interleaved experimental designs.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 1.4))

    colors = {"Clip": "#1f77b4", "Monet2": "#ff7f0e", "Trippy": "#2ca02c", "Unknown": "#bbbbbb"}
    n = len(stim_types_per_trial)
    if trial_durations is None:
        trial_durations = np.ones(n)
    starts = np.concatenate(([0], np.cumsum(trial_durations[:-1])))
    for s, d, t in zip(starts, trial_durations, stim_types_per_trial):
        ax.barh(0, width=d, left=s, height=1.0, color=colors.get(t, "#bbbbbb"))

    ax.set_yticks([])
    ax.set_xlabel("Cumulative timesteps" if not np.allclose(trial_durations, 1) else "Trial index")
    ax.set_title("Trial timeline (colored by stim type)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()]
    ax.legend(handles, list(colors.keys()), loc="upper right", ncol=4, fontsize=8)
    ax.set_xlim(0, starts[-1] + trial_durations[-1])
    return ax


def get_video_clip(reader, condition_hash: str) -> np.ndarray | None:
    """Frames of one video as a ``(frames, width, height)`` array, or ``None``.

    ``MicronsFunctionalReader.get_video_data`` returned a ``(clip, stim_type)``
    tuple in older releases and returns a dict (keys ``clip``, ``stim_type``,
    ``fps``, ...) from 0.2.1.x, but still returns ``(None, None)`` when the
    hash is not in ``/videos/``. This wraps both.
    """
    out = reader.get_video_data(condition_hash)
    if isinstance(out, dict):
        return out.get("clip")
    if isinstance(out, tuple):
        return out[0]
    return None


def plot_stim_examples(reader, stim_map: dict[str, str]) -> plt.Figure:
    """Show one mid-clip frame from each of Clip / Monet2 / Trippy.

    Args:
        reader: A ``MicronsFunctionalReader``.
        stim_map: Output of ``build_stim_type_map`` for any session
            (used to find one example hash per stim class).

    Returns:
        The figure with three subplots, one per stim class.

    Notes:
        Picks the first hash encountered for each class. The frame
        shown is the middle frame of the clip.

        ``MicronsFunctionalReader.get_video_data`` returns a 2-tuple
        ``(clip, stim_type)`` where ``clip`` is a ``(frames, width,
        height)`` array, or ``(None, None)`` if the hash is missing
        from ``/videos/``.
    """
    examples: dict[str, str] = {}
    for h, t in stim_map.items():
        if t in {"Clip", "Monet2", "Trippy"} and t not in examples:
            examples[t] = h
        if len(examples) == 3:
            break

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, stim_type in zip(axes, ["Clip", "Monet2", "Trippy"]):
        if stim_type in examples:
            clip = get_video_clip(reader, examples[stim_type])
            if clip is not None:
                mid = clip.shape[0] // 2
                ax.imshow(clip[mid], cmap="gray")
            else:
                ax.text(0.5, 0.5, "video missing", ha="center", va="center")
        ax.set_title(f"{stim_type} (mid-frame)")
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    return fig


def plot_response_heatmap(
    responses: np.ndarray,
    trial_idx: int,
    trial_boundaries: np.ndarray,
    sort_neurons_by_mean: bool = True,
    vmin: float | None = 0.0,
    vmax: float | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Heatmap of one trial's response matrix, neurons × time.

    Args:
        responses: shape ``(n_neurons, total_timesteps)`` from
            ``load_session_responses``.
        trial_idx: 0-based trial index within the session.
        trial_boundaries: shape ``(n_trials + 1,)``, cumulative
            timestep counts.
        sort_neurons_by_mean: If True, sort neurons by their mean
            response within this trial (descending). Default: True.
        vmin: Lower bound of the color scale. Defaults to 0
            (responses are typically non-negative).
        vmax: Upper bound of the color scale. If None, defaults to
            the 99th percentile of the *whole-session* responses
            matrix (not the trial) so a few bright neurons don't
            wash out the structure visible in the rest. Pass an
            explicit value when comparing multiple trials so all
            heatmaps share a scale.
        ax: Optional matplotlib axes.

    Returns:
        The axes on which the heatmap was drawn.

    Notes:
        Uses ``imshow`` with ``aspect="auto"`` so the heatmap fits
        the trial's frame count without distortion.

        Why percentile-clipping vmax? Per-neuron variance spans
        several orders of magnitude in MICrONS: a tiny minority of
        neurons hit values in the hundreds, while the bulk live in
        ~0–10. Linear-scaling the colormap against the maximum
        compresses the typical range to nearly black, hiding the
        structure that matters. Clipping at the 99th percentile
        recovers visible detail.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    start = int(trial_boundaries[trial_idx])
    end = int(trial_boundaries[trial_idx + 1])
    trial = responses[:, start:end]

    if sort_neurons_by_mean:
        order = np.argsort(-trial.mean(axis=1))
        trial = trial[order]

    if vmax is None:
        vmax = float(np.percentile(responses, 99))

    im = ax.imshow(
        trial, aspect="auto", cmap="magma", interpolation="nearest",
        vmin=vmin, vmax=vmax,
    )
    ax.set_xlabel("Frame within trial")
    ax.set_ylabel("Neuron (sorted)" if sort_neurons_by_mean else "Neuron")
    plt.colorbar(im, ax=ax, label="Response")
    return ax


def plot_psth_by_stim(
    responses: np.ndarray,
    trial_boundaries: np.ndarray,
    stim_types_per_trial: np.ndarray,
    target_length: int = 75,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Trial-averaged mean population response per stim class.

    Args:
        responses: shape ``(n_neurons, total_timesteps)``.
        trial_boundaries: shape ``(n_trials + 1,)``.
        stim_types_per_trial: shape ``(n_trials,)``.
        target_length: timesteps per trial used for averaging. Trials
            longer than this are truncated to ``target_length``; trials
            shorter than this are skipped (so a 75-vs-113 mix doesn't
            silently corrupt the PSTH). Default: 75 (the Clip length).
        ax: Optional matplotlib axes.

    Returns:
        The axes on which the PSTHs were drawn.

    Notes:
        Mean is taken across trials and across neurons, leaving a
        single trace per stim class as a function of within-trial
        timestep.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    durations = np.diff(trial_boundaries)
    n_trials = len(stim_types_per_trial)

    classes = ["Clip", "Monet2", "Trippy"]
    for cls in classes:
        traces = []
        for i in range(n_trials):
            if stim_types_per_trial[i] != cls:
                continue
            if durations[i] < target_length:
                continue
            start = int(trial_boundaries[i])
            trial = responses[:, start:start + target_length]
            traces.append(trial.mean(axis=0))
        if not traces:
            continue
        mean_trace = np.mean(traces, axis=0)
        ax.plot(mean_trace, label=f"{cls} (n={len(traces)})")

    ax.set_xlabel("Frame within trial")
    ax.set_ylabel("Mean population response")
    ax.set_title(f"PSTH per stim class (first {target_length} frames)")
    ax.legend()
    return ax


def plot_behavior_traces(
    trial: dict,
    pupil_diameter_channel: int = 2,
    vmin: float | None = 0.0,
    vmax: float | None = None,
    axes: tuple | None = None,
) -> tuple:
    """Pupil and treadmill traces aligned to a single trial's response heatmap.

    Args:
        trial: A single trial dict from ``load_trial``.
        pupil_diameter_channel: pupil channel treated as diameter.
        vmin: Lower bound of the heatmap color scale. Default 0
            (responses are typically non-negative).
        vmax: Upper bound of the heatmap color scale. If None,
            defaults to the 99th percentile of *this trial's*
            responses, so a few bright neurons don't compress the
            rest into near-black. Pass an explicit value (e.g. the
            99th percentile of the whole-session matrix) to share
            the scale across multiple plots.
        axes: Optional triple ``(ax_resp, ax_pupil, ax_tread)``. If
            None, a stacked ``(3, 1)`` figure is created.

    Returns:
        Tuple ``(fig, (ax_resp, ax_pupil, ax_tread))``.

    Notes:
        See ``plot_response_heatmap`` for the rationale behind
        percentile-clipping vmax.
    """
    if axes is None:
        fig, (ax_resp, ax_pupil, ax_tread) = plt.subplots(
            3, 1, figsize=(10, 8),
            gridspec_kw={"height_ratios": [3, 1, 1]},
            sharex=True,
        )
    else:
        ax_resp, ax_pupil, ax_tread = axes
        fig = ax_resp.figure

    responses = trial["responses"]
    order = np.argsort(-responses.mean(axis=1))
    if vmax is None:
        vmax = float(np.percentile(responses, 99))
    im = ax_resp.imshow(
        responses[order], aspect="auto", cmap="magma", interpolation="nearest",
        vmin=vmin, vmax=vmax,
    )
    ax_resp.set_ylabel("Neuron (sorted)")
    plt.colorbar(im, ax=ax_resp, label="Response")

    ax_pupil.plot(trial["pupil"][pupil_diameter_channel], color="purple")
    ax_pupil.set_ylabel("Pupil diameter")

    ax_tread.plot(trial["treadmill"][:, 0], color="orange")
    ax_tread.set_ylabel("Treadmill speed")
    ax_tread.set_xlabel("Frame")

    fig.suptitle(f"Trial {trial.get('condition_hash', '?')[:10]}… ({trial.get('stim_type', '?')})")
    fig.tight_layout()
    return fig, (ax_resp, ax_pupil, ax_tread)


def plot_neuron_stat_distributions(
    stats_df: pd.DataFrame,
    axes: tuple | None = None,
) -> tuple:
    """Histograms of per-neuron mean, variance, sparsity, plus log–log mean-vs-var.

    Args:
        stats_df: Output of ``compute_neuron_stats``.
        axes: Optional 4-tuple of axes ``(ax_mean, ax_var, ax_sparsity,
            ax_meanvar)``. If None, a 2×2 figure is created.

    Returns:
        Tuple ``(fig, (ax_mean, ax_var, ax_sparsity, ax_meanvar))``.
    """
    if axes is None:
        fig, ax_arr = plt.subplots(2, 2, figsize=(12, 9))
        axes_flat = ax_arr.ravel()
    else:
        axes_flat = axes
        fig = axes_flat[0].figure

    ax_mean, ax_var, ax_sparsity, ax_meanvar = axes_flat

    ax_mean.hist(stats_df["mean"], bins=80, color="steelblue", edgecolor="none")
    ax_mean.set_title("Per-neuron mean")
    ax_mean.set_xlabel("mean response")
    ax_mean.set_ylabel("Neuron count")

    ax_var.hist(stats_df["var"], bins=80, color="firebrick", edgecolor="none", log=True)
    ax_var.set_title("Per-neuron variance (log y)")
    ax_var.set_xlabel("variance")
    ax_var.set_ylabel("Neuron count")

    ax_sparsity.hist(stats_df["sparsity"], bins=40, color="seagreen", edgecolor="none")
    ax_sparsity.set_title("Per-neuron sparsity")
    ax_sparsity.set_xlabel("Fraction of timesteps above 5th percentile")
    ax_sparsity.set_ylabel("Neuron count")

    valid = (stats_df["mean"] > 0) & (stats_df["var"] > 0)
    ax_meanvar.scatter(
        stats_df.loc[valid, "mean"],
        stats_df.loc[valid, "var"],
        s=4, alpha=0.4, color="black",
    )
    ax_meanvar.set_xscale("log")
    ax_meanvar.set_yscale("log")
    ax_meanvar.set_xlabel("Mean (log)")
    ax_meanvar.set_ylabel("Variance (log)")
    ax_meanvar.set_title("Mean vs variance per neuron (slope ≈ 1 → Poisson; ≈ 2 → multiplicative)")

    fig.tight_layout()
    return fig, axes_flat


def plot_correlation_heatmap(
    corr: np.ndarray,
    area_per_neuron: np.ndarray | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Neuron × neuron correlation heatmap, optionally sorted by area.

    Args:
        corr: shape ``(k, k)``.
        area_per_neuron: shape ``(k,)``, area string per row of ``corr``.
            If provided, neurons are reordered so all V1 are contiguous,
            then AL, LM, RL, and dashed lines are drawn at area
            boundaries.
        ax: Optional matplotlib axes.

    Returns:
        The axes the heatmap was drawn on.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    if area_per_neuron is not None:
        order_keys = ["V1", "AL", "LM", "RL"]
        order = np.concatenate([
            np.where(np.array(area_per_neuron) == k)[0] for k in order_keys
        ])
        corr_sorted = corr[order][:, order]
        boundaries = np.cumsum([np.sum(np.array(area_per_neuron) == k) for k in order_keys[:-1]])
    else:
        corr_sorted = corr
        boundaries = []

    vmax = np.percentile(np.abs(corr_sorted), 99)
    im = ax.imshow(corr_sorted, cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Pearson r")
    for b in boundaries:
        ax.axhline(b, color="black", linewidth=0.5, linestyle="--")
        ax.axvline(b, color="black", linewidth=0.5, linestyle="--")
    ax.set_title(f"Neuron×neuron correlation ({corr_sorted.shape[0]} neurons)")
    ax.set_xticks([]); ax.set_yticks([])
    return ax


def plot_drift(drift_df: pd.DataFrame, ax: plt.Axes | None = None) -> plt.Axes:
    """Mean population activity per trial across the session, with linear fit.

    Args:
        drift_df: Output of ``compute_drift``.
        ax: Optional matplotlib axes.

    Returns:
        The axes the plot was drawn on.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(drift_df.index, drift_df["mean_activity"], color="steelblue", label="per-trial mean")
    ax.plot(drift_df.index, drift_df["trend"], color="black", linestyle="--",
            label=f"linear fit (slope={drift_df.attrs.get('slope', 0):+.2e}/trial)")
    ax.set_xlabel("Trial index")
    ax.set_ylabel("Mean population activity")
    ax.set_title("Drift over trials")
    ax.legend()
    return ax
