"""neuro_palette: palette + matplotlib style for the MICrONS presentation.

White-background variant: keeps the saturated jewel-tone STIMULUS / SUBCATEGORY
palettes, but the plot chrome (figure / axes / grid / text colours) is tuned
for white slides and printable handouts rather than the dark deck.

Usage in a notebook:

    from neuro_palette import (
        STIMULUS, SUBCATEGORY, NEUTRAL, HIGHLIGHT, THRESHOLD,
        apply_style, color_for,
    )
    apply_style()
    plt.plot(t, traj_clip,   color=STIMULUS["Clip"])
    plt.plot(t, traj_monet2, color=STIMULUS["Monet2"])
    plt.plot(t, traj_trippy, color=STIMULUS["Trippy"])
"""
from __future__ import annotations

import matplotlib as mpl

# ---------------------------------------------------------------------------
# Palettes
# ---------------------------------------------------------------------------

# Stimulus class palette (Option 2, Option 3, EDA stim exemplars).
STIMULUS: dict[str, str] = {
    "Clip":   "#60A5FA",  # sky blue
    "Monet2": "#F472B6",  # rose
    "Trippy": "#34D399",  # emerald
}

# Within-Clip subcategory palette (Option 4).
SUBCATEGORY: dict[str, str] = {
    "Cinematic": "#F87171",  # coral red
    "sports1m":  "#22D3EE",  # cyan
    "Rendered":  "#A855F7",  # violet
}

# Generic accents (designed against white backgrounds).
NEUTRAL: str   = "#5F6B7A"  # slate grey for non-highlighted bars
HIGHLIGHT: str = "#E69F00"  # amber for the bar/element you want to call out
THRESHOLD: str = "#B91C1C"  # red for dashed threshold lines + above-threshold bars


# ---------------------------------------------------------------------------
# Style hook
# ---------------------------------------------------------------------------

def apply_style() -> None:
    """Set matplotlib defaults for the deck. Call once at the top of a notebook.

    White figure / axes background, slate axis chrome, light grid, top + right
    spines hidden. Designed so that any subsequent ``plt.subplots`` inherits
    these without further configuration.
    """
    mpl.rcParams.update({
        "figure.facecolor":   "white",
        "axes.facecolor":     "white",
        "savefig.facecolor":  "white",
        "axes.edgecolor":     "#1F2937",
        "axes.labelcolor":    "#1F2937",
        "xtick.color":        "#1F2937",
        "ytick.color":        "#1F2937",
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.grid":          True,
        "axes.axisbelow":     True,
        "grid.color":         "#E5E7EB",
        "grid.linewidth":     0.8,
        "font.family":        "DejaVu Sans",
        "font.size":          12,
        "axes.labelsize":     13,
        "axes.titlesize":     14,
        "axes.titleweight":   "bold",
        "axes.titlecolor":    "#1F2937",
        "figure.titlesize":   14,
        "figure.titleweight": "bold",
        "xtick.labelsize":    11,
        "ytick.labelsize":    11,
        "legend.frameon":     False,
        "legend.fontsize":    11,
        "text.color":         "#1F2937",
    })


def color_for(label: str) -> str:
    """Return the canonical colour for a stim class or sub-category label.

    Args:
        label: One of the keys in :data:`STIMULUS` or :data:`SUBCATEGORY`.

    Returns:
        Hex colour string, e.g. ``"#60A5FA"``.

    Raises:
        KeyError: If the label is not in either palette.
    """
    if label in STIMULUS:
        return STIMULUS[label]
    if label in SUBCATEGORY:
        return SUBCATEGORY[label]
    raise KeyError(f"No colour mapped for {label!r}")
