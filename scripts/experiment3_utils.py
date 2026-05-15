"""option4_clip_utils: helpers for the Clip-category trajectory analysis (Option 4).

Sections:

(a) Category labels and inventory: resolve condition_hash → short_movie_name,
    build per-trial labels, and compile the unique-clip inventory DataFrame.
(b) Part-1 catalog plotting: balance table, trial-count histogram, category
    exemplars, frame-strip preview, and the big-picture 240-panel grid.
(c) Pipeline glue for Part 2: restrict-to-Clip helper and the cross-area
    headline plotter (1×3 panels, all three pairs symmetrically).

Design constraints (see docs/specs/2026-05-07-option4-clip-categories-design.md):
- Pure data functions: no plotting, no file writes.
- Plotting functions accept `ax` (or return Figure for multi-panel layouts).
- NumPy-style docstrings + type hints on every public function.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Palette (distinct from option2_pca_utils.STIM_COLORS so figures from
# Option 4 stay visually distinguishable from Option 2 / Option 3 in the
# final report).
# ---------------------------------------------------------------------------

CATEGORY_COLORS: dict[str, str] = {
    "Cinematic": "#F87171",  # coral red, feature films
    "sports1m":  "#22D3EE",  # cyan,      amateur sports
    "Rendered":  "#A855F7",  # violet,    synthetic 3-D
}
CATEGORY_MARKERS: dict[str, str] = {"Cinematic": "o", "sports1m": "s", "Rendered": "^"}
CATEGORY_ORDER: list[str] = ["Cinematic", "sports1m", "Rendered"]

# Pair palette derived from category palette: mid-tone of the two category
# colours, picked manually to read cleanly on a white background.
_PAIR_COLORS: dict[frozenset[str], str] = {
    frozenset({"Cinematic", "sports1m"}): "#7B7CB8",  # dusty indigo (red+cyan)
    frozenset({"Cinematic", "Rendered"}): "#C75CB1",  # magenta      (red+violet)
    frozenset({"sports1m",  "Rendered"}): "#6CB7CC",  # teal-blue    (cyan+violet)
}
_PAIR_LABELS: dict[frozenset[str], str] = {
    frozenset({"Cinematic", "sports1m"}): "Cinematic↔sports1m",
    frozenset({"Cinematic", "Rendered"}): "Cinematic↔Rendered",
    frozenset({"sports1m",  "Rendered"}): "sports1m↔Rendered",
}


def category_pair_palette() -> dict[frozenset[str], str]:
    """Return the pair-keyed palette for the three Clip-category pairs.

    Returns:
        Dict from ``frozenset({a, b})`` to a hex colour string. Keys cover
        all three pairs across ``CATEGORY_ORDER``.
    """
    return dict(_PAIR_COLORS)


def category_pair_labels() -> dict[frozenset[str], str]:
    """Return human-readable labels for the three Clip-category pairs.

    Returns:
        Dict from ``frozenset({a, b})`` to a label like ``"Cinematic↔sports1m"``.
    """
    return dict(_PAIR_LABELS)


# =============================================================================
# (a) Category labels and inventory
# =============================================================================


_HASH_TO_CATEGORY_CACHE: dict[str, dict[str, str]] = {}


def build_hash_to_category(h5_path: str | Path) -> dict[str, str]:
    """Resolve every Clip movie's condition hash to its short_movie_name category.

    Reads ``/videos/`` once and returns a dictionary mapping the
    ``original_hash`` attribute (the un-encoded hash, matching what
    ``meta/condition_hashes`` decodes to) to the ``short_movie_name``
    attribute (one of ``"Cinematic"``, ``"sports1m"``, ``"Rendered"``).

    Args:
        h5_path: Path to ``microns_functional.h5``.

    Returns:
        Dict mapping ``original_hash → short_movie_name`` for every video
        whose ``type == "Clip"``. Hashes from non-Clip videos (Monet2,
        Trippy) are not present.

    Notes:
        Cached at module level keyed by ``str(h5_path)``. The H5 read is
        cheap (~2 s) but the function may be called twice: once for the
        Part-1 catalog, once for the Part-2 trajectory labels.
    """
    key = str(Path(h5_path).resolve())
    if key in _HASH_TO_CATEGORY_CACHE:
        return _HASH_TO_CATEGORY_CACHE[key]

    out: dict[str, str] = {}
    with h5py.File(h5_path, "r") as f:
        videos = f["videos"]
        for k in videos.keys():
            attrs = videos[k].attrs
            if attrs.get("type", "") != "Clip":
                continue
            if "short_movie_name" not in attrs:
                continue
            original_hash = attrs["original_hash"]
            if isinstance(original_hash, bytes):
                original_hash = original_hash.decode("utf-8", "replace")
            short = attrs["short_movie_name"]
            if isinstance(short, bytes):
                short = short.decode("utf-8", "replace")
            out[original_hash] = short
    _HASH_TO_CATEGORY_CACHE[key] = out
    return out


def category_labels_for_session(
    reader,
    datadir: str | Path,
    session: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-trial stim type and (for Clip trials only) short_movie_name category.

    Args:
        reader: A ``MicronsFunctionalReader`` (from ``microns_eda.open_dataset``).
        datadir: Directory containing the H5 (passed to ``microns_eda.open_h5``).
        session: Session key, e.g. ``"7_5"``.

    Returns:
        Tuple ``(stim_per_trial, category_per_trial)``, both of shape
        ``(n_trials,)`` with ``dtype=object``.

        - ``stim_per_trial``: one of ``"Clip"``, ``"Monet2"``, ``"Trippy"``,
          ``"Unknown"`` per trial. Mirrors what
          ``microns_eda.build_stim_type_map`` returns.
        - ``category_per_trial``: the ``short_movie_name`` for trials whose
          ``stim_per_trial == "Clip"``; the empty string ``""`` otherwise.

    Notes:
        Decoded internally; callers do not deal with bytes.
    """
    import scripts.microns_eda as microns_eda

    h5_path = Path(datadir) / "functional" / "microns_functional.h5"
    hash_to_category = build_hash_to_category(h5_path)

    meta = microns_eda.get_session_meta(datadir, session)
    raw_hashes = meta["condition_hashes"]
    decoded: list[str] = []
    for h in raw_hashes:
        decoded.append(h.decode("utf-8", "replace") if isinstance(h, bytes) else str(h))

    # Build hash → stim_type once over unique hashes (avoids reader call per trial).
    unique_hashes = set(decoded)
    hash_to_stim: dict[str, str] = {}
    for h in unique_hashes:
        try:
            t = reader.get_video_type(h)
        except Exception:
            t = "Unknown"
        hash_to_stim[h] = t if t in {"Clip", "Monet2", "Trippy"} else "Unknown"

    n = len(decoded)
    stim_per_trial = np.empty(n, dtype=object)
    category_per_trial = np.empty(n, dtype=object)
    for i, h in enumerate(decoded):
        stim = hash_to_stim[h]
        stim_per_trial[i] = stim
        category_per_trial[i] = hash_to_category.get(h, "") if stim == "Clip" else ""
    return stim_per_trial, category_per_trial


def restrict_to_clip(
    clean_trial_indices: np.ndarray,
    stim_per_trial: np.ndarray,
    category_per_trial: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Filter the clean-trial mask further to keep only Clip trials.

    Args:
        clean_trial_indices: Output of
            ``microns_eda.compute_clean_trial_indices``, an int array of
            trial indices that survive the running-outlier QC.
        stim_per_trial: Shape ``(n_trials_total,)``, output of
            :func:`category_labels_for_session` (full-session, pre-QC).
        category_per_trial: Shape ``(n_trials_total,)``, output of
            :func:`category_labels_for_session`.

    Returns:
        Tuple ``(clip_trial_indices, clip_category_labels)``.

        - ``clip_trial_indices``: subset of ``clean_trial_indices`` whose
          ``stim_per_trial == "Clip"``, in original trial order.
        - ``clip_category_labels``: shape
          ``(len(clip_trial_indices),)``, one of
          ``"Cinematic" / "sports1m" / "Rendered"`` per trial.
    """
    keep_mask = np.array(
        [stim_per_trial[i] == "Clip" for i in clean_trial_indices], dtype=bool
    )
    clip_trial_indices = clean_trial_indices[keep_mask]
    clip_category_labels = np.array(
        [category_per_trial[i] for i in clip_trial_indices], dtype=object
    )
    return clip_trial_indices, clip_category_labels


def _read_clip_movie_names(h5_path: str | Path, hashes: set[str]) -> dict[str, str]:
    """Internal helper: read /videos/{hash}.attrs['movie_name'] for the given hashes.

    Args:
        h5_path: Path to ``microns_functional.h5``.
        hashes: Set of original_hash strings to look up.

    Returns:
        Dict from ``original_hash → movie_name``. Hashes not found in
        ``/videos/`` are omitted.
    """
    out: dict[str, str] = {}
    with h5py.File(h5_path, "r") as f:
        videos = f["videos"]
        for k in videos.keys():
            attrs = videos[k].attrs
            if attrs.get("type", "") != "Clip":
                continue
            oh = attrs.get("original_hash", "")
            if isinstance(oh, bytes):
                oh = oh.decode("utf-8", "replace")
            if oh not in hashes:
                continue
            mn = attrs.get("movie_name", "")
            if isinstance(mn, bytes):
                mn = mn.decode("utf-8", "replace")
            out[oh] = str(mn)
    return out


def build_clip_inventory(
    reader,
    hashes_per_trial: np.ndarray,
    hash_to_category: dict[str, str],
    h5_path: str | Path,
) -> pd.DataFrame:
    """One-row-per-unique-Clip-hash inventory DataFrame.

    Args:
        reader: A ``MicronsFunctionalReader`` (used for clip-frame access).
        hashes_per_trial: Shape ``(n_trials,)``, decoded condition hashes
            for the trials of interest (typically the post-QC Clip subset).
        hash_to_category: Output of :func:`build_hash_to_category`.
        h5_path: Path to ``microns_functional.h5``, used to read
            ``movie_name`` attrs alongside the category dict.

    Returns:
        DataFrame with columns:

        - ``hash`` (str): original (non-URL-encoded) condition hash.
        - ``category`` (str): one of ``"Cinematic" / "sports1m" / "Rendered"``.
        - ``movie_name`` (str): full title from the video attrs (empty
          string if attr is absent).
        - ``n_trials`` (int): how many times this hash appears in
          ``hashes_per_trial``.
        - ``n_frames_clip`` (int): number of frames in the underlying
          video, from ``reader.get_video_data``. ``None`` if the video
          payload is missing.

        Sorted by ``category`` (Cinematic, sports1m, Rendered) and within
        each category by ``n_trials`` descending.

    Notes:
        Calls ``reader.get_video_data`` once per unique hash. For session
        ``7_5`` that's ~240 calls; total runtime ~10 s. ``movie_name`` is
        read directly from the H5 in a single pass, avoiding any
        dependency on optional ``reader`` accessor methods.
    """
    counts = Counter(hashes_per_trial)
    needed = {h for h in counts.keys() if hash_to_category.get(h) in CATEGORY_ORDER}
    hash_to_movie_name = _read_clip_movie_names(h5_path, needed)

    rows: list[dict] = []
    for h, c in counts.items():
        category = hash_to_category.get(h, "")
        if category not in CATEGORY_ORDER:
            continue
        clip, _stim = reader.get_video_data(h)
        n_frames = int(clip.shape[0]) if clip is not None else None
        rows.append({
            "hash": h,
            "category": category,
            "movie_name": hash_to_movie_name.get(h, ""),
            "n_trials": int(c),
            "n_frames_clip": n_frames,
        })
    df = pd.DataFrame(rows)
    cat_order = pd.CategoricalDtype(CATEGORY_ORDER, ordered=True)
    df["category"] = df["category"].astype(cat_order)
    df = df.sort_values(["category", "n_trials"], ascending=[True, False]).reset_index(drop=True)
    df["category"] = df["category"].astype(str)
    return df


# =============================================================================
# (b) Part-1 catalog plotting
# =============================================================================


def plot_trial_count_histogram(
    inventory_df: pd.DataFrame,
    *,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Stacked histogram of n_trials across unique clips, colored by category.

    Args:
        inventory_df: Output of :func:`build_clip_inventory`.
        ax: matplotlib axes; created if None.

    Returns:
        The axes drawn on.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 3.8))

    max_t = int(inventory_df["n_trials"].max())
    bins = np.arange(0, max_t + 2) - 0.5
    bottom = np.zeros(len(bins) - 1)
    for cat in CATEGORY_ORDER:
        sub = inventory_df.loc[inventory_df["category"] == cat, "n_trials"].values
        counts, _ = np.histogram(sub, bins=bins)
        ax.bar(
            (bins[:-1] + bins[1:]) / 2, counts, width=1.0, bottom=bottom,
            color=CATEGORY_COLORS[cat], edgecolor="white", linewidth=0.6,
            label=cat,
        )
        bottom = bottom + counts

    ax.set_xlabel("Trials per unique Clip movie")
    ax.set_ylabel("Number of unique clips")
    ax.set_title("Replay structure of the Clip class (stacked by category)")
    ax.legend(loc="best", fontsize=9, title="category")
    ax.grid(True, alpha=0.3, axis="y")
    return ax


def plot_category_exemplars(
    reader,
    inventory_df: pd.DataFrame,
    *,
    n_per_category: int = 8,
):
    """3-row × n_per_category-col mid-frame thumbnails: exemplars per category.

    Args:
        reader: A ``MicronsFunctionalReader``.
        inventory_df: Output of :func:`build_clip_inventory`.
        n_per_category: Columns per row (default 8).

    Returns:
        ``matplotlib.figure.Figure``; caller saves and shows it.
    """
    fig, axes = plt.subplots(
        len(CATEGORY_ORDER), n_per_category,
        figsize=(2.3 * n_per_category, 2.5 * len(CATEGORY_ORDER)),
        squeeze=False,
    )
    for i, cat in enumerate(CATEGORY_ORDER):
        sub = inventory_df.loc[inventory_df["category"] == cat].head(n_per_category)
        for j in range(n_per_category):
            ax = axes[i, j]
            ax.set_xticks([]); ax.set_yticks([])
            if j >= len(sub):
                ax.axis("off")
                continue
            row = sub.iloc[j]
            clip, _ = reader.get_video_data(row["hash"])
            if clip is None:
                ax.text(0.5, 0.5, "missing", ha="center", va="center",
                        transform=ax.transAxes, color="firebrick", fontsize=8)
            else:
                mid = clip.shape[0] // 2
                ax.imshow(clip[mid], cmap="gray")
            title_parts = [f"{row['hash'][:8]}…"]
            mn = row.get("movie_name") or ""
            if mn:
                short_mn = mn[:30] + ("…" if len(mn) > 30 else "")
                title_parts.append(short_mn)
            title_parts.append(f"{int(row['n_trials'])} trial{'s' if row['n_trials'] != 1 else ''}")
            ax.set_title("\n".join(title_parts), fontsize=7,
                         color=CATEGORY_COLORS[cat])
        # Row label on the leftmost.
        axes[i, 0].set_ylabel(cat, color=CATEGORY_COLORS[cat],
                              fontsize=12, fontweight="bold",
                              rotation=0, labelpad=40, ha="right", va="center")

    fig.suptitle(f"Clip-category exemplars (top {n_per_category} most-replayed per category)",
                 fontsize=12, y=1.005)
    fig.tight_layout()
    return fig


def plot_frame_strip_per_category(
    reader,
    inventory_df: pd.DataFrame,
    *,
    n_frames: int = 6,
):
    """One example clip per category × n_frames evenly-spaced frames.

    Args:
        reader: A ``MicronsFunctionalReader``.
        inventory_df: Output of :func:`build_clip_inventory`.
        n_frames: Frames sampled per clip (default 6).

    Returns:
        ``matplotlib.figure.Figure``.

    Notes:
        Picks the most-replayed clip per category as the example.
    """
    fig, axes = plt.subplots(
        len(CATEGORY_ORDER), n_frames,
        figsize=(1.9 * n_frames, 2.0 * len(CATEGORY_ORDER)),
        squeeze=False,
    )
    for i, cat in enumerate(CATEGORY_ORDER):
        sub = inventory_df.loc[inventory_df["category"] == cat]
        if len(sub) == 0:
            for j in range(n_frames):
                axes[i, j].axis("off")
            continue
        row = sub.iloc[0]
        clip, _ = reader.get_video_data(row["hash"])
        if clip is None:
            for j in range(n_frames):
                ax = axes[i, j]
                ax.text(0.5, 0.5, "missing", ha="center", va="center",
                        transform=ax.transAxes, color="firebrick", fontsize=8)
                ax.set_xticks([]); ax.set_yticks([])
            continue
        n_total = clip.shape[0]
        sample_idx = np.linspace(0, n_total - 1, n_frames).astype(int)
        for j, fi in enumerate(sample_idx):
            ax = axes[i, j]
            ax.imshow(clip[fi], cmap="gray")
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(f"frame {fi}/{n_total - 1}", fontsize=8)
            if j == 0:
                title = f"{cat}\n{row['hash'][:8]}…"
                ax.set_ylabel(title, color=CATEGORY_COLORS[cat],
                              fontsize=10, fontweight="bold",
                              rotation=0, labelpad=40, ha="right", va="center")

    fig.suptitle("Frame-strip preview (one clip per category)", fontsize=12, y=1.005)
    fig.tight_layout()
    return fig


def plot_full_thumbnail_grid(
    reader,
    inventory_df: pd.DataFrame,
    *,
    ncols: int = 16,
):
    """The big-picture 240-panel grid: one mid-frame per unique Clip.

    Sorted by category (Cinematic block, sports1m block, Rendered block).
    Panel titles colored by category for fast visual chunking.

    Args:
        reader: A ``MicronsFunctionalReader``.
        inventory_df: Output of :func:`build_clip_inventory`. **Must already
            be sorted by category** (this is the natural sort order from
            ``build_clip_inventory``).
        ncols: Columns in the grid (default 16).

    Returns:
        ``matplotlib.figure.Figure``, or ``None`` if the inventory has more
        than 400 rows (skipped silently to avoid pathological figures on
        other sessions).

    Notes:
        Caller saves this at modest dpi (recommended: 100) to keep the
        file size manageable.
    """
    n = len(inventory_df)
    if n > 400:
        return None
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.0 * ncols, 2.2 * nrows))
    axes_flat = np.atleast_1d(axes).ravel()
    for i, row in enumerate(inventory_df.itertuples(index=False)):
        ax = axes_flat[i]
        ax.set_xticks([]); ax.set_yticks([])
        clip, _ = reader.get_video_data(row.hash)
        if clip is None:
            ax.text(0.5, 0.5, "missing", ha="center", va="center",
                    transform=ax.transAxes, color="firebrick", fontsize=7)
        else:
            mid = clip.shape[0] // 2
            ax.imshow(clip[mid], cmap="gray")
        ax.set_title(f"{row.hash[:6]}…", fontsize=6,
                     color=CATEGORY_COLORS.get(row.category, "black"))
    for j in range(n, len(axes_flat)):
        axes_flat[j].axis("off")
    fig.suptitle(f"All {n} unique Clip movies in session (sorted by category)",
                 fontsize=12, y=1.002)
    fig.tight_layout()
    return fig


# =============================================================================
# (c) Part-2 pipeline glue
# =============================================================================


def plot_cross_area_three_pairs(
    per_area_results: dict[str, dict],
    metric_label: str,
    *,
    metric_key: str = "distances_full",
    envelope_key: str = "envelopes_full",
):
    """Cross-area headline: 1×3 panels, one per category pair.

    Each panel shows that pair's distance time course across all four
    cortical areas (V1, AL, LM, RL) with bootstrap envelopes shaded.

    Args:
        per_area_results: ``{area: {distances_full, distances_top3, envelopes_full,
            envelopes_top3, ...}}`` as built by the Part-2 notebook code (mirrors
            the structure used in Option 3's Part 6+).
        metric_label: y-axis label, e.g. ``"Euclidean (full features)"``.
        metric_key: Key into ``per_area_results[area]`` for the observed
            distances (default ``"distances_full"``; alternative
            ``"distances_top3"`` for the top-3 PC metric).
        envelope_key: Key into ``per_area_results[area]`` for the bootstrap
            envelopes (default ``"envelopes_full"``).

    Returns:
        ``matplotlib.figure.Figure`` with 1×3 axes: one panel per pair
        in the order ``Cinematic↔sports1m``, ``Cinematic↔Rendered``,
        ``sports1m↔Rendered``.
    """
    pair_palette = category_pair_palette()
    pair_labels = category_pair_labels()
    area_palette = {"V1": "#0072B2", "AL": "#E69F00", "LM": "#009E73", "RL": "#CC79A7"}
    pairs_in_order = [
        frozenset({"Cinematic", "sports1m"}),
        frozenset({"Cinematic", "Rendered"}),
        frozenset({"sports1m", "Rendered"}),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for j, pair in enumerate(pairs_in_order):
        ax = axes[j]
        for area, res in per_area_results.items():
            d = res[metric_key].get(pair)
            if d is None:
                continue
            x = np.arange(len(d))
            ax.plot(x, d, color=area_palette.get(area, "grey"),
                    linewidth=2, label=area)
            env = res[envelope_key].get(pair)
            if env is not None:
                ax.fill_between(
                    x, env["lower"], env["upper"],
                    color=area_palette.get(area, "grey"), alpha=0.18,
                )
        ax.set_xlabel("Frame")
        if j == 0:
            ax.set_ylabel(metric_label)
        ax.set_title(pair_labels[pair], color=pair_palette[pair])
        ax.legend(loc="best", fontsize=8, title="area")
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"Cross-area Clip-category pairwise distance ({metric_label})",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    return fig
