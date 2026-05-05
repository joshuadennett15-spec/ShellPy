"""
Poster-ready Juno MWR comparison: D_cond violin per equatorial mode.

Single figure showing each ocean heat transport scenario's conductive lid
distribution against the Juno constraint window. Designed for readability
at poster scale (1-2 m viewing distance).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)

import numpy as np
import matplotlib.pyplot as plt
from pub_style import apply_style, PAL, save_fig, DOUBLE_COL

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures', 'pub')

apply_style()

# Override font sizes for poster legibility
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
})

# Juno MWR constraints
JUNO_PURE = (29.0, 10.0)    # pure water ice
JUNO_SALTY = (24.0, 10.0)   # low salinity

# Scenarios ordered from most depleted to most enhanced
MODES = [
    ("Depleted strong\n(0.55\u00d7)",  "eq_depleted_strong_andrade.npz", PAL.PURPLE),
    ("Depleted\n(0.67\u00d7)",         "eq_depleted_andrade.npz",        PAL.CYAN),
    ("Baseline\n(1.0\u00d7)",          "eq_baseline_andrade.npz",        PAL.GREEN),
    ("Moderate\n(1.2\u00d7)",          "eq_moderate_andrade.npz",        PAL.ORANGE),
    ("Strong\n(1.5\u00d7)",            "eq_strong_andrade.npz",          PAL.BLUE),
]


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(DOUBLE_COL, DOUBLE_COL * 0.50))

    # Juno constraint band (1-sigma)
    juno_lo = min(JUNO_PURE[0] - JUNO_PURE[1], JUNO_SALTY[0] - JUNO_SALTY[1])
    juno_hi = max(JUNO_PURE[0] + JUNO_PURE[1], JUNO_SALTY[0] + JUNO_SALTY[1])
    ax.axvspan(juno_lo, juno_hi, color="#E8E8E8", zorder=0,
               label=f"Juno MWR 1\u03c3 ({juno_lo:.0f}\u2013{juno_hi:.0f} km)")

    # Juno central estimates
    ax.axvline(JUNO_PURE[0], color="0.45", ls="--", lw=1.2, zorder=1)
    ax.axvline(JUNO_SALTY[0], color="0.45", ls=":", lw=1.2, zorder=1)
    # Place labels just inside top edge, clear of violins
    ax.annotate("Pure water", xy=(JUNO_PURE[0], 1), xycoords=("data", "axes fraction"),
                xytext=(3, -6), textcoords="offset points",
                fontsize=8.5, color="0.35", va="top", ha="left")
    ax.annotate("Low salinity", xy=(JUNO_SALTY[0], 1), xycoords=("data", "axes fraction"),
                xytext=(-3, -6), textcoords="offset points",
                fontsize=8.5, color="0.35", va="top", ha="right")

    # Violins
    violin_data = []
    colours = []
    labels = []

    for i, (label, filename, col) in enumerate(MODES):
        path = os.path.join(RESULTS_DIR, filename)
        data = np.load(path)
        Dc = data["D_cond_km"]
        violin_data.append(Dc)
        colours.append(col)
        labels.append(label)

    positions = list(range(len(MODES)))

    parts = ax.violinplot(violin_data, positions=positions, vert=False,
                          showmedians=False, showextrema=False)

    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(colours[i])
        body.set_edgecolor(colours[i])
        body.set_alpha(0.55)
        body.set_linewidth(0.8)

    # Overlay median + IQR as box-like markers
    for i, Dc in enumerate(violin_data):
        med = np.median(Dc)
        q25, q75 = np.percentile(Dc, [25, 75])
        ax.plot([q25, q75], [i, i], color=colours[i], lw=3.5,
                solid_capstyle="round", zorder=3)
        ax.plot(med, i, "o", color="white", ms=6, zorder=4,
                markeredgecolor=colours[i], markeredgewidth=1.5)
        # Median annotation
        ax.text(med, i + 0.28, f"{med:.0f} km",
                ha="center", va="bottom", fontsize=8.5,
                fontweight="bold", color=colours[i])

    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel(r"Equatorial conductive lid thickness $D_{\rm cond}$ (km)")
    ax.set_xlim(0, 65)
    ax.invert_yaxis()

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_title("Equatorial $D_{\\rm cond}$ vs. Juno MWR constraint",
                 fontweight="bold", pad=10)

    fig.tight_layout()
    save_fig(fig, "fig_poster_juno_dcond", FIGURES_DIR)


if __name__ == "__main__":
    main()
