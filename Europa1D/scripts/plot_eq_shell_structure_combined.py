"""
Combined equatorial shell structure across all five transport proxies.

Two-panel figure: (a) D_cond distributions, (b) H_total distributions.
Directly visualises the thesis argument that D_cond barely varies while
H_total (and hence D_conv) varies dramatically across ocean-transport
scenarios. Replaces the single-panel D_cond-only figure.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from pub_style import PAL, DOUBLE_COL, add_minor_gridlines, apply_style, save_fig


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
X_PTS = 600


def _kde(values, x_grid):
    if len(values) < 10 or np.std(values) < 1e-10:
        return None
    return gaussian_kde(values)(x_grid)


def _label_panel(ax, letter):
    ax.text(
        -0.02, 1.02, f"({letter})",
        transform=ax.transAxes, fontsize=9, fontweight="bold",
        va="bottom", ha="right",
    )


def main():
    apply_style()

    fig, (ax_dc, ax_ht) = plt.subplots(
        1, 2,
        figsize=(DOUBLE_COL, DOUBLE_COL * 0.35),
    )

    x_dc = np.linspace(0.0, 45.0, X_PTS)
    x_ht = np.linspace(0.0, 100.0, X_PTS)

    # Juno constraint on D_cond panel
    ax_dc.axvspan(
        JUNO_DCOND - JUNO_SIGMA,
        JUNO_DCOND + JUNO_SIGMA,
        color="0.55", alpha=0.15,
        label=r"Juno $29 \pm 10$ km",
    )
    ax_dc.axvline(JUNO_DCOND, color="0.35", lw=0.8, ls="--")

    for label, filename, color in MODES:
        path = os.path.join(RESULTS_DIR, filename)
        if not os.path.exists(path):
            print(f"skip (missing): {filename}")
            continue

        data = np.load(path)
        d_cond = data["D_cond_km"]
        h_total = data["thicknesses_km"]

        dc_med = float(np.median(d_cond))
        ht_med = float(np.median(h_total))

        # D_cond panel
        pdf_dc = _kde(d_cond, x_dc)
        if pdf_dc is not None:
            ax_dc.plot(x_dc, pdf_dc, color=color, lw=1.3,
                       label=rf"{label}, med $= {dc_med:.1f}$")
            ax_dc.fill_between(x_dc, 0, pdf_dc, color=color, alpha=0.08)

        # H_total panel
        pdf_ht = _kde(h_total, x_ht)
        if pdf_ht is not None:
            ax_ht.plot(x_ht, pdf_ht, color=color, lw=1.3,
                       label=rf"{label}, med $= {ht_med:.1f}$")
            ax_ht.fill_between(x_ht, 0, pdf_ht, color=color, alpha=0.08)

        print(f"{filename}: D_cond={dc_med:.1f}, H_total={ht_med:.1f} km (N={len(d_cond)})")

    # D_cond panel formatting
    ax_dc.set_xlabel(r"$D_{\mathrm{cond}}$ (km)")
    ax_dc.set_ylabel("Probability density")
    ax_dc.set_xlim(0, 45)
    ax_dc.set_ylim(bottom=0)
    add_minor_gridlines(ax_dc, axis="y")
    ax_dc.legend(loc="upper right", fontsize=5.5, handlelength=1.3)
    _label_panel(ax_dc, "a")

    # H_total panel formatting
    ax_ht.set_xlabel(r"$H_{\mathrm{total}}$ (km)")
    ax_ht.set_ylabel("Probability density")
    ax_ht.set_xlim(0, 100)
    ax_ht.set_ylim(bottom=0)
    add_minor_gridlines(ax_ht, axis="y")
    ax_ht.legend(loc="upper right", fontsize=5.5, handlelength=1.3)
    _label_panel(ax_ht, "b")

    fig.tight_layout(w_pad=2.0)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    save_fig(fig, "fig_eq_shell_structure_combined", FIGURES_DIR)


if __name__ == "__main__":
    main()
