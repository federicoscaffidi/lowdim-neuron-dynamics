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


def open_dataset(datadir: str | Path) -> "MicronsFunctionalReader":
    """Construct a MicronsFunctionalReader pointed at the given data directory.

    Args:
        datadir: Directory containing ``microns.h5``. Relative paths are
            resolved against the current working directory (i.e. the location
            of the notebook).

    Returns:
        A configured ``microns_datacleaner.MicronsFunctionalReader`` instance.
        Reuse this object across loader calls — construction touches the H5
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

    The file is expected at ``{datadir}/functional/microns_functional.h5``
    — the layout ``MicronsFunctionalReader`` looks for. If the file is
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
        This is the slow part of Part 1 — expect minutes, not seconds.
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
        classifies every trial cleanly) yields a z-score of 0 — no
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


# (analyzers go here)


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

    ax.boxplot(per_session_means, labels=sessions, showfliers=False)
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
        Kept simple — Jupyter renders DataFrames natively. We don't
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
            clip, _ = reader.get_video_data(examples[stim_type])
            if clip is not None:
                mid = clip.shape[0] // 2
                ax.imshow(clip[mid], cmap="gray")
            else:
                ax.text(0.5, 0.5, "video missing", ha="center", va="center")
        ax.set_title(f"{stim_type} (mid-frame)")
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    return fig
