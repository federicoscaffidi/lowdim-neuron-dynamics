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
    """Open ``{datadir}/microns.h5`` read-only as a context manager.

    Used by loaders that need fields ``MicronsFunctionalReader`` does not
    expose: per-trial pupil, treadmill, stim_times, and the per-session
    ``meta`` group (coordinates, area_indices, brain_areas, unit_ids,
    condition_hashes).

    Args:
        datadir: Directory containing ``microns.h5``.

    Yields:
        An open ``h5py.File`` in read-only mode.
    """
    path = Path(datadir) / "microns.h5"
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

            pupil_sum += float(pupil[PUPIL_DIAMETER_CHANNEL].sum())
            pupil_count += int(pupil.shape[1])
            tread_sum += float(treadmill.sum())
            tread_count += int(treadmill.size)

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
        z = (col - col.mean()) / col.std(ddof=0)
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


# (plotters go here)
