"""
Fig 3: Ice shell regime decomposition.

(a) Total thickness PDF split by conductive vs convective subpopulations.
(b) Internal structure (D_cond, D_conv) conditioned on the convective CBE.

Usage:
    python plot_regime_decomposition.py
    python plot_regime_decomposition.py --filepath results/mc_15000_optionA_v2_andrade.npz
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from pub_style import (
    apply_style, PAL,
    figsize_double, label_panel, save_fig, add_minor_gridlines,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures", "pub")

apply_style()

RA_CRIT = 1000.0


def main(filepath):
    data = np.load(filepath)
    H = data["thicknesses_km"]
    D_cond = data["D_cond_km"]
    D_conv = data["D_conv_km"]
    Ra = data["Ra_values"] if "Ra_values" in data else np.zeros(len(H))

    conv_mask = Ra >= RA_CRIT
    H_conv = H[conv_mask]
    H_cond = H[~conv_mask]
    frac_conv = len(H_conv) / len(H)
    frac_cond = 1.0 - frac_conv

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize_double(0.45))

    # ── (a) Regime-split thickness PDF ────────────────────────────────
    x_grid = np.linspace(0, np.percentile(H, 99.5) + 5, 400)

    # Full distribution (dashed reference)
    kde_all = gaussian_kde(H)
    ax1.plot(x_grid, kde_all(x_grid), color="0.5", lw=0.8, ls="--", alpha=0.5)

    # Conductive
    kde_cond = gaussian_kde(H_cond)
    pdf_cond = kde_cond(x_grid) * frac_cond
    ax1.fill_between(x_grid, 0, pdf_cond, color=PAL.COND, alpha=0.25)
    ax1.plot(x_grid, pdf_cond, color=PAL.COND, lw=1.5,
             label=f"Conductive ({frac_cond:.0%})")
    cbe_cond = float(x_grid[np.argmax(pdf_cond)])
    ax1.axvline(cbe_cond, color=PAL.COND, ls=":", lw=0.8, alpha=0.6)
    ax1.text(cbe_cond + 1.5, pdf_cond.max() * 0.88,
             f"{cbe_cond:.0f} km", fontsize=7, color=PAL.COND, ha="left")

    # Convective
    kde_conv = gaussian_kde(H_conv)
    pdf_conv = kde_conv(x_grid) * frac_conv
    ax1.fill_between(x_grid, 0, pdf_conv, color=PAL.CONV, alpha=0.20)
    ax1.plot(x_grid, pdf_conv, color=PAL.CONV, lw=1.5,
             label=f"Convective ({frac_conv:.0%})")
    cbe_conv = float(x_grid[np.argmax(pdf_conv)])
    ax1.axvline(cbe_conv, color=PAL.CONV, ls=":", lw=0.8, alpha=0.6)
    ax1.text(cbe_conv + 1.5, pdf_conv.max() * 0.88,
             f"{cbe_conv:.0f} km", fontsize=7, color=PAL.CONV, ha="left")

    ax1.set_xlabel("Ice shell thickness (km)")
    ax1.set_ylabel("Probability density")
    ax1.set_xlim(0, np.percentile(H, 99.5) + 5)
    ax1.set_ylim(bottom=0)
    ax1.legend(loc="upper right", fontsize=7, framealpha=0.9)
    add_minor_gridlines(ax1, axis="y")
    label_panel(ax1, "a")

    # ── (b) Convective structure at convective CBE ────────────────────
    D_cond_conv = D_cond[conv_mask]
    D_conv_conv = D_conv[conv_mask]

    width = 5.0
    near_cbe = np.abs(H_conv - cbe_conv) <= width
    if near_cbe.sum() < 50:
        near_cbe = np.abs(H_conv - cbe_conv) <= width * 2

    cond_near = D_cond_conv[near_cbe]
    conv_near = D_conv_conv[near_cbe]

    x_max = max(cond_near.max(), conv_near[conv_near > 0.5].max()) + 3
    x_layer = np.linspace(0, x_max, 300)

    # Lid
    kde_c = gaussian_kde(cond_near)
    pdf_c = kde_c(x_layer)
    ax2.fill_between(x_layer, 0, pdf_c, color=PAL.COND, alpha=0.20)
    ax2.plot(x_layer, pdf_c, color=PAL.COND, lw=1.5,
             label=r"$D_{\rm cond}$ (lid)")
    mode_c = float(x_layer[np.argmax(pdf_c)])
    p16_c = float(np.percentile(cond_near, 15.87))
    p84_c = float(np.percentile(cond_near, 84.13))
    ax2.annotate(
        f"{mode_c:.0f} (+{p84_c - mode_c:.0f} / \u2212{mode_c - p16_c:.0f}) km",
        xy=(mode_c, pdf_c.max()),
        xytext=(mode_c + 10, pdf_c.max() * 0.78),
        fontsize=7, color=PAL.COND,
        arrowprops=dict(arrowstyle="-", color=PAL.COND, lw=0.5))

    # Sublayer
    conv_pos = conv_near[conv_near > 0.5]
    kde_v = gaussian_kde(conv_pos)
    pdf_v = kde_v(x_layer)
    ax2.fill_between(x_layer, 0, pdf_v, color=PAL.CONV, alpha=0.20)
    ax2.plot(x_layer, pdf_v, color=PAL.CONV, lw=1.5,
             label=r"$D_{\rm conv}$ (sublayer)")
    mode_v = float(x_layer[np.argmax(pdf_v)])
    p16_v = float(np.percentile(conv_pos, 15.87))
    p84_v = float(np.percentile(conv_pos, 84.13))
    ax2.annotate(
        f"{mode_v:.0f} (+{p84_v - mode_v:.0f} / \u2212{mode_v - p16_v:.0f}) km",
        xy=(mode_v, pdf_v.max()),
        xytext=(mode_v + 8, pdf_v.max() * 0.78),
        fontsize=7, color=PAL.CONV,
        arrowprops=dict(arrowstyle="-", color=PAL.CONV, lw=0.5))

    ax2.set_xlabel("Layer thickness (km)")
    ax2.set_ylabel("Probability density")
    ax2.set_xlim(0, 70)
    ax2.set_ylim(bottom=0)
    ax2.legend(loc="upper right", fontsize=7, framealpha=0.9)
    add_minor_gridlines(ax2, axis="y")
    label_panel(ax2, "b")

    ax2.text(0.03, 0.95,
        f"Convective branch\n"
        f"H = {cbe_conv:.0f} \u00b1 {width:.0f} km (N = {near_cbe.sum():,})",
        transform=ax2.transAxes, fontsize=6.5, va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", lw=0.4))

    fig.suptitle(
        f"Ice shell regime decomposition (N = {len(H):,})",
        fontsize=9, y=1.02)
    fig.tight_layout(w_pad=2.5)
    save_fig(fig, "fig3_howell_4b", FIGURES_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", default="")
    args = parser.parse_args()

    if args.filepath:
        fp = args.filepath
    else:
        fp = os.path.join(RESULTS_DIR, "mc_15000_optionA_v2_andrade.npz")

    if not os.path.exists(fp):
        print(f"Not found: {fp}")
        sys.exit(1)

    print(f"Loading: {fp}")
    main(fp)
