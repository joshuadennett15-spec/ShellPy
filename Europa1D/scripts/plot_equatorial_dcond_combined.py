"""
Combined equatorial D_cond distributions across all five transport proxies.

This figure is used in the thesis results chapter to show how weakly the
conductive lid responds to the equatorial transport-proxy sweep compared with
the much larger response seen in total shell thickness and D_conv.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from pub_style import PAL, add_minor_gridlines, apply_style, save_fig


RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures", "pub")

MODES = [
    ("Depleted strong", "eq_depleted_strong_andrade.npz", PAL.PURPLE),
    ("Depleted", "eq_depleted_andrade.npz", PAL.CYAN),
    ("Baseline", "eq_baseline_andrade.npz", PAL.GREEN),
    ("Moderate", "eq_moderate_andrade.npz", PAL.ORANGE),
    ("Strong", "eq_strong_andrade.npz", PAL.RED),
]

JUNO_DCOND = 29.0
JUNO_SIGMA = 10.0
X_MAX = 60.0
X_PTS = 600


def _kde(values, x_grid):
    if len(values) < 10 or np.std(values) < 1e-10:
        return None
    return gaussian_kde(values)(x_grid)


def main():
    apply_style()

    x_grid = np.linspace(0.0, X_MAX, X_PTS)
    fig, ax = plt.subplots(figsize=(4.2, 2.7))

    ax.axvspan(
        JUNO_DCOND - JUNO_SIGMA,
        JUNO_DCOND + JUNO_SIGMA,
        color="0.55",
        alpha=0.15,
        label=r"Juno $29 \pm 10$ km",
    )
    ax.axvline(JUNO_DCOND, color="0.35", lw=0.8, ls="--")

    for label, filename, color in MODES:
        path = os.path.join(RESULTS_DIR, filename)
        if not os.path.exists(path):
            print(f"skip (missing): {filename}")
            continue

        d_cond = np.load(path)["D_cond_km"]
        pdf = _kde(d_cond, x_grid)
        if pdf is None:
            continue

        median = float(np.median(d_cond))
        ax.plot(
            x_grid,
            pdf,
            color=color,
            lw=1.3,
            label=rf"{label}, med $= {median:.1f}$ km",
        )
        ax.fill_between(x_grid, 0, pdf, color=color, alpha=0.08)

        print(f"{filename}: median = {median:.2f} km (N={len(d_cond)})")

    ax.set_xlabel(r"$D_{\mathrm{cond}}$ (km)")
    ax.set_ylabel("Probability density")
    ax.set_xlim(0, X_MAX)
    ax.set_ylim(bottom=0)
    add_minor_gridlines(ax, axis="y")
    ax.legend(loc="upper right", fontsize=6.2, ncol=2, handlelength=1.3)

    fig.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    save_fig(fig, "fig_eq_dcond_combined", FIGURES_DIR)


if __name__ == "__main__":
    main()
