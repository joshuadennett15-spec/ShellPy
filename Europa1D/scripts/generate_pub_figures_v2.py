#!/usr/bin/env python3
"""
Generate publication-quality fig1 + fig2 from 1D Monte Carlo results.

Reads NPZ from the 1D Monte Carlo engine and produces:
  fig1: (a) H distribution + KDE, (b) D_cond distribution + KDE
  fig2: (a) D_cond/D_conv overlapping PDFs, (b) stacked bar structure

Usage:
    python generate_pub_figures_v2.py
    python generate_pub_figures_v2.py --filepath results/mc_15000_howell.npz
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from pub_style import (
    apply_style, PAL,
    figsize_double,
    label_panel, save_fig, add_minor_gridlines,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures", "pub")

apply_style()


# ═══════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════

def _load_mc(filepath):
    """Load 1D MC NPZ, return (H, D_cond, D_conv, lid_frac, Ra, n)."""
    data = np.load(filepath)
    H = data["thicknesses_km"]
    D_cond = data["D_cond_km"]
    D_conv = data["D_conv_km"]
    lid_frac = data["lid_fractions"]
    Ra = data["Ra_values"] if "Ra_values" in data else np.zeros(len(H))
    return H, D_cond, D_conv, lid_frac, Ra, len(H)


def _kde_smooth(values, n_pts=300):
    """Gaussian KDE on values, evaluated on a data-driven grid."""
    if len(values) < 10:
        return None, None
    kde = gaussian_kde(values)
    lo = max(0, np.percentile(values, 0.5) - 2)
    hi = np.percentile(values, 99.5) + 2
    x_grid = np.linspace(lo, hi, n_pts)
    return x_grid, kde(x_grid)


def _binned_means(H, D_cond, D_conv, n_bins=30):
    """Compute binned mean layer thicknesses by total H."""
    h_max = np.percentile(H, 98)
    bin_edges = np.linspace(max(H.min(), 0), h_max, n_bins)
    bc = (bin_edges[:-1] + bin_edges[1:]) / 2
    dig = np.digitize(H, bin_edges)
    mc = np.array([D_cond[dig == i].mean() if np.sum(dig == i) > 5
                   else np.nan for i in range(1, len(bin_edges))])
    mv = np.array([D_conv[dig == i].mean() if np.sum(dig == i) > 5
                   else np.nan for i in range(1, len(bin_edges))])
    mH = np.array([H[dig == i].mean() if np.sum(dig == i) > 5
                   else np.nan for i in range(1, len(bin_edges))])
    return bin_edges, bc, mc, mv, mH


# ═══════════════════════════════════════════════════════════════════════════
# Figure 1: Monte Carlo thickness distributions
# ═══════════════════════════════════════════════════════════════════════════

def fig1_mc_distributions(H, D_cond, n):
    """Two-panel: (a) total thickness PDF, (b) conductive lid PDF."""
    print("Figure 1: Monte Carlo distributions")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize_double(0.40))

    # ── (a) Total thickness ──────────────────────────────────────────────
    bins_h = np.linspace(0, np.percentile(H, 99.5) + 2, 60)
    ax1.hist(H, bins=bins_h, density=True, color=PAL.COND, alpha=0.25,
             edgecolor=PAL.COND, linewidth=0.3)
    x_h, pdf_h = _kde_smooth(H)
    if pdf_h is not None:
        ax1.plot(x_h, pdf_h, color=PAL.COND, lw=1.5)

        cbe = float(x_h[np.argmax(pdf_h)])
        median_h = float(np.median(H))
        p16 = float(np.percentile(H, 15.87))
        p84 = float(np.percentile(H, 84.13))

        ax1.axvline(cbe, color=PAL.BLACK, ls="--", lw=0.8, alpha=0.7)
        ax1.axvspan(p16, p84, color=PAL.COND, alpha=0.06,
                    label=rf"$1\sigma$ [{p16:.0f}, {p84:.0f}] km")

        ax1.text(cbe + 1.5, pdf_h.max() * 0.93,
                 f"CBE = {cbe:.1f} km", fontsize=6.5, ha="left", color="0.2")
        ax1.text(0.97, 0.92, f"Median = {median_h:.1f} km",
                 transform=ax1.transAxes, fontsize=6.5, ha="right", va="top",
                 color="0.3")

    ax1.set_xlabel("Ice shell thickness (km)")
    ax1.set_ylabel("Probability density")
    ax1.set_xlim(0, np.percentile(H, 99.5) + 5)
    ax1.set_ylim(bottom=0)
    add_minor_gridlines(ax1, axis="y")
    label_panel(ax1, "a")

    # ── (b) Conductive lid thickness ─────────────────────────────────────
    bins_d = np.linspace(0, np.percentile(D_cond, 99.5) + 2, 50)
    ax2.hist(D_cond, bins=bins_d, density=True, color=PAL.COND, alpha=0.25,
             edgecolor=PAL.COND, linewidth=0.3)
    x_d, pdf_d = _kde_smooth(D_cond)
    if pdf_d is not None:
        ax2.plot(x_d, pdf_d, color=PAL.COND, lw=1.5)

        cbe_d = float(x_d[np.argmax(pdf_d)])
        ax2.axvline(cbe_d, color=PAL.BLACK, ls="--", lw=0.8, alpha=0.7)
        ax2.annotate(f"Mode = {cbe_d:.1f} km",
                     xy=(cbe_d, pdf_d.max() * 0.97),
                     xytext=(cbe_d + 8, pdf_d.max() * 0.85),
                     fontsize=7, ha="left",
                     arrowprops={"arrowstyle": "-", "lw": 0.5, "color": "0.4"})

    ax2.set_xlabel(r"Conductive lid thickness $D_\mathrm{cond}$ (km)")
    ax2.set_ylabel("Probability density")
    ax2.set_xlim(0, np.percentile(D_cond, 99.5) + 5)
    ax2.set_ylim(bottom=0)
    add_minor_gridlines(ax2, axis="y")
    label_panel(ax2, "b")

    fig.suptitle(
        f"Monte Carlo ice shell thickness (N = {n:,})",
        fontsize=9, y=1.02,
    )
    fig.tight_layout(w_pad=2.5)
    save_fig(fig, "fig1_mc_distributions", FIGURES_DIR)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 2: Shell structure (layer PDFs + stacked bar)
# ═══════════════════════════════════════════════════════════════════════════

def fig2_shell_structure(H, D_cond, D_conv, n):
    """Two-panel: (a) D_cond/D_conv PDFs, (b) stacked mean structure vs H."""
    print("Figure 2: Shell structure")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize_double(0.42))

    # ── (a) Layer distributions ──────────────────────────────────────────
    bins = np.linspace(0, np.percentile(H, 99), 55)

    ax1.hist(D_cond, bins=bins, density=True, color=PAL.COND, alpha=0.30,
             edgecolor=PAL.COND, linewidth=0.3,
             label=r"$D_\mathrm{cond}$ (lid)")

    D_conv_active = D_conv[D_conv > 0.5]
    if len(D_conv_active) > 0:
        frac_conv = len(D_conv_active) / len(D_conv)
        ax1.hist(D_conv_active, bins=bins, density=True, color=PAL.CONV, alpha=0.30,
                 edgecolor=PAL.CONV, linewidth=0.3,
                 weights=np.full(len(D_conv_active), frac_conv),
                 label=rf"$D_\mathrm{{conv}}$ ({frac_conv:.0%} convecting)")

    x_c, pdf_c = _kde_smooth(D_cond)
    if pdf_c is not None:
        ax1.plot(x_c, pdf_c, color=PAL.COND, lw=1.2, alpha=0.8)
    if len(D_conv_active) > 20:
        x_v, pdf_v = _kde_smooth(D_conv_active)
        if pdf_v is not None:
            ax1.plot(x_v, pdf_v * frac_conv, color=PAL.CONV, lw=1.2, alpha=0.8)

    ax1.set_xlabel("Layer thickness (km)")
    ax1.set_ylabel("Probability density")
    ax1.legend(loc="upper right", fontsize=6.5)
    ax1.set_xlim(0, np.percentile(H, 99))
    ax1.set_ylim(bottom=0)
    label_panel(ax1, "a")

    # ── (b) Stacked bar: mean structure by thickness bin ─────────────────
    be, bc, mc_cond, mc_conv, mH = _binned_means(H, D_cond, D_conv)
    ok = ~np.isnan(mc_cond) & ~np.isnan(mc_conv) & ~np.isnan(mH)
    w = be[1] - be[0]

    ax2.bar(bc[ok], mc_cond[ok], width=w * 0.92, color=PAL.COND, alpha=0.75,
            label=r"$D_\mathrm{cond}$")
    ax2.bar(bc[ok], mc_conv[ok], width=w * 0.92, bottom=mc_cond[ok],
            color=PAL.CONV, alpha=0.75, label=r"$D_\mathrm{conv}$")
    ax2.plot(bc[ok], mH[ok], "k-", lw=1.2, label=r"$H_\mathrm{total}$")

    lim = max(np.nanmax(mH[ok]), bc[ok].max()) * 1.05
    ax2.plot([0, lim], [0, lim], color="0.6", ls="--", lw=0.5, zorder=0)

    ax2.set_xlabel("Total ice shell thickness bin (km)")
    ax2.set_ylabel("Mean layer thickness (km)")
    ax2.legend(loc="upper left", fontsize=6.5)
    label_panel(ax2, "b")

    fig.suptitle(f"Ice shell structure (N = {n:,})", fontsize=9, y=1.02)
    fig.tight_layout(w_pad=2.5)
    save_fig(fig, "fig2_shell_structure", FIGURES_DIR)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", default="", help="Path to MC NPZ file.")
    args = parser.parse_args()

    if args.filepath:
        filepath = args.filepath
    else:
        default = os.path.join(RESULTS_DIR, "mc_15000_howell.npz")
        if os.path.exists(default):
            filepath = default
        else:
            # Fall back to largest file
            import glob
            candidates = glob.glob(os.path.join(RESULTS_DIR, "*.npz"))
            filepath = max(candidates, key=os.path.getsize) if candidates else None

    if not filepath or not os.path.exists(filepath):
        print(f"No results found. Run the MC first.")
        sys.exit(1)

    print(f"Loading: {filepath}")
    H, D_cond, D_conv, lid_frac, Ra, n = _load_mc(filepath)
    print(f"  N = {n:,} valid samples")

    fig1_mc_distributions(H, D_cond, n)
    fig2_shell_structure(H, D_cond, D_conv, n)
