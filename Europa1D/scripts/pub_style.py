"""
Publication-quality figure style for Europa ice shell convection paper.

Provides consistent rcParams, colour palettes, and helper functions
for journal-ready figures (targeting Icarus / JGR-Planets column widths).

Usage:
    from pub_style import apply_style, PAL, figsize_single, figsize_double
    apply_style()
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ── Column widths (inches) ──────────────────────────────────────────────────
# Icarus / JGR-Planets / Nature Geoscience typical widths
SINGLE_COL = 3.50   # ~89 mm
DOUBLE_COL = 7.20   # ~183 mm
FULL_PAGE_HEIGHT = 9.50  # ~241 mm


def figsize_single(aspect=0.75):
    """Single-column figure size."""
    return (SINGLE_COL, SINGLE_COL * aspect)


def figsize_double(aspect=0.45):
    """Double-column figure size."""
    return (DOUBLE_COL, DOUBLE_COL * aspect)


def figsize_double_tall(aspect=0.65):
    """Double-column tall figure (e.g. 4-panel grids)."""
    return (DOUBLE_COL, DOUBLE_COL * aspect)


# ── Colour palette (colorblind-friendly, Wong 2011) ────────────────────────
class PAL:
    """Colorblind-safe palette based on Wong (2011) Nature Methods."""
    BLUE    = "#0072B2"
    ORANGE  = "#E69F00"
    GREEN   = "#009E73"
    RED     = "#D55E00"
    PURPLE  = "#CC79A7"
    CYAN    = "#56B4E9"
    YELLOW  = "#F0E442"
    BLACK   = "#000000"

    # Semantic aliases for this project
    COND    = BLUE       # conductive lid
    CONV    = RED        # convective sublayer
    TOTAL   = BLACK      # total shell
    GLOBAL  = BLUE
    EQUATOR = ORANGE
    POLE    = GREEN

    # Thickness bin colours (ordered light→dark, distinguishable)
    BIN_COLOURS = [GREEN, ORANGE, PURPLE, RED, CYAN]

    @staticmethod
    def alpha(hex_colour, a=0.20):
        """Return RGBA tuple from hex + alpha."""
        rgb = mpl.colors.to_rgb(hex_colour)
        return (*rgb, a)


# ── Thickness bins (shared across diagnostic figures) ──────────────────────
THICKNESS_BINS = [
    ("< 15 km",   0,  15, PAL.BIN_COLOURS[0]),
    ("15-30 km",  15, 30, PAL.BIN_COLOURS[1]),
    ("30-50 km",  30, 50, PAL.BIN_COLOURS[2]),
    ("50-80 km",  50, 80, PAL.BIN_COLOURS[3]),
    ("> 80 km",   80, 999, PAL.BIN_COLOURS[4]),
]


# ── rcParams ───────────────────────────────────────────────────────────────
def apply_style():
    """Set matplotlib rcParams for publication figures."""
    mpl.rcParams.update({
        # Font
        "font.family":       "sans-serif",
        "font.sans-serif":   ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size":         8,
        "mathtext.fontset":  "stixsans",

        # Axes
        "axes.labelsize":    9,
        "axes.titlesize":    9,
        "axes.titleweight":  "bold",
        "axes.linewidth":    0.6,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.labelpad":     3,
        "axes.titlepad":     6,

        # Ticks
        "xtick.labelsize":   7.5,
        "ytick.labelsize":   7.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.minor.width": 0.3,
        "ytick.minor.width": 0.3,
        "xtick.major.size":  3,
        "ytick.major.size":  3,
        "xtick.minor.size":  1.5,
        "ytick.minor.size":  1.5,
        "xtick.direction":   "in",
        "ytick.direction":   "in",

        # Lines
        "lines.linewidth":   1.2,
        "lines.markersize":  4,

        # Legend
        "legend.fontsize":   7,
        "legend.frameon":    False,
        "legend.handlelength": 1.5,
        "legend.handletextpad": 0.4,
        "legend.borderaxespad": 0.3,

        # Figure
        "figure.dpi":        150,
        "savefig.dpi":       300,
        "savefig.bbox":      "tight",
        "savefig.pad_inches": 0.03,

        # Grid
        "axes.grid":         False,

        # Histogram
        "hist.bins":         50,

        # Patch
        "patch.linewidth":   0.5,
    })


def label_panel(ax, letter, x=-0.12, y=1.06):
    """Add bold panel label (a), (b), ... to an axes."""
    ax.text(x, y, f"({letter})", transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="bottom", ha="left")


def save_fig(fig, name, figures_dir, formats=("png", "pdf")):
    """Save figure in multiple formats."""
    import os
    os.makedirs(figures_dir, exist_ok=True)
    for fmt in formats:
        path = os.path.join(figures_dir, f"{name}.{fmt}")
        fig.savefig(path, dpi=300 if fmt == "png" else None,
                    transparent=(fmt == "pdf"))
    plt.close(fig)
    print(f"  Saved: {name}.{{{', '.join(formats)}}}")


def add_minor_gridlines(ax, axis="both", alpha=0.15):
    """Add subtle minor gridlines."""
    ax.grid(True, which="major", lw=0.3, alpha=0.25, color="0.5")
    ax.grid(True, which="minor", lw=0.15, alpha=alpha, color="0.7")
    ax.minorticks_on()
