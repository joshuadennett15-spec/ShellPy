"""
Per-mode equatorial MC figures:
  (a) Total thickness PDF with regime split
  (b) D_cond PDF
  (c) Shell structure (D_cond/D_conv stacked bar)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from pub_style import (
    apply_style, PAL, figsize_double_tall,
    label_panel, save_fig, add_minor_gridlines,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures', 'pub')

apply_style()

RA_CRIT = 1000.0

SHOW_JUNO_OVERLAY = False

JUNO_MODELS = [
    (29.0, 10.0, 3.0, "Pure water"),
    (24.0, 10.0, 3.0, "Low salinity"),
]

MODES = [
    ("eq_depleted_strong_andrade.npz", "Depleted strong (0.55x)", "depleted_strong"),
    ("eq_depleted_andrade.npz",        "Depleted (0.67x)",        "depleted"),
    ("eq_baseline_andrade.npz",        "Baseline (1.0x)",         "baseline"),
    ("eq_moderate_andrade.npz",        "Moderate (1.2x)",         "moderate"),
    ("eq_strong_andrade.npz",          "Strong (1.5x)",           "strong"),
]


def _kde(values, n_pts=300):
    """Gaussian KDE with guard against small or degenerate samples."""
    if len(values) < 10:
        return None, None
    if np.std(values) < 1e-10:
        return None, None
    kde = gaussian_kde(values)
    lo = max(0, np.percentile(values, 0.5) - 2)
    hi = np.percentile(values, 99.5) + 2
    x = np.linspace(lo, hi, n_pts)
    return x, kde(x)


def plot_mode(filepath, title, tag):
    """Three-panel figure for one equatorial mode."""
    data = np.load(filepath)
    H = data['thicknesses_km']
    D_cond = data['D_cond_km']
    D_conv = data['D_conv_km']
    Ra = data['Ra_values'] if 'Ra_values' in data else np.zeros(len(H))
    n = len(H)

    conv_mask = Ra >= RA_CRIT
    frac_conv = conv_mask.mean()

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10.0, 3.2))

    # (a) Total thickness with regime split
    x_grid = np.linspace(0, np.percentile(H, 99.5) + 5, 400)

    H_cond = H[~conv_mask]
    H_conv = H[conv_mask]
    frac_cond = 1.0 - frac_conv

    x_c, pdf_c = _kde(H_cond)
    if pdf_c is not None:
        pdf_c_grid = np.interp(x_grid, x_c, pdf_c, left=0, right=0) * frac_cond
        ax1.fill_between(x_grid, 0, pdf_c_grid, color=PAL.COND, alpha=0.25)
        ax1.plot(x_grid, pdf_c_grid, color=PAL.COND, lw=1.2,
                 label=f"Cond. ({frac_cond:.0%})")

    x_v, pdf_v = _kde(H_conv)
    if pdf_v is not None:
        pdf_v_grid = np.interp(x_grid, x_v, pdf_v, left=0, right=0) * frac_conv
        ax1.fill_between(x_grid, 0, pdf_v_grid, color=PAL.CONV, alpha=0.20)
        ax1.plot(x_grid, pdf_v_grid, color=PAL.CONV, lw=1.2,
                 label=f"Conv. ({frac_conv:.0%})")

    ax1.set_xlabel("Ice shell thickness (km)")
    ax1.set_ylabel("Probability density")
    ax1.set_xlim(0, np.percentile(H, 99.5) + 5)
    ax1.set_ylim(bottom=0)
    ax1.legend(fontsize=6)
    label_panel(ax1, "a")

    # (b) D_cond distribution
    x_d, pdf_d = _kde(D_cond)
    if pdf_d is not None:
        ax2.fill_between(x_d, 0, pdf_d, color=PAL.COND, alpha=0.25)
        ax2.plot(x_d, pdf_d, color=PAL.COND, lw=1.5, label=r"$D_{\rm cond}$")

    if SHOW_JUNO_OVERLAY:
        x_juno = np.linspace(0, 70, 300)
        for D_obs, sigma_obs, sigma_model, jlabel in JUNO_MODELS:
            sigma_tot = np.sqrt(sigma_obs**2 + sigma_model**2)
            juno_pdf = np.exp(-0.5 * ((x_juno - D_obs) / sigma_tot)**2)
            juno_pdf /= (sigma_tot * np.sqrt(2 * np.pi))
            ls = "--" if "Pure" in jlabel else ":"
            ax2.plot(x_juno, juno_pdf, color=PAL.BLACK, lw=1.0, ls=ls,
                     alpha=0.7, label=f"Juno {jlabel}")

    dc_med = float(np.median(D_cond))
    dc_16, dc_84 = np.percentile(D_cond, [15.87, 84.13])
    ax2.axvspan(dc_16, dc_84, color=PAL.COND, alpha=0.10, zorder=0)
    ax2.axvline(dc_med, color=PAL.COND, lw=1.0, ls="--", alpha=0.9)

    ax2.set_xlabel(r"$D_{\rm cond}$ (km)")
    ax2.set_ylabel("Probability density")
    ax2.set_xlim(0, 60)
    ax2.set_ylim(bottom=0)
    ax2.legend(fontsize=5.5, loc="upper right")
    label_panel(ax2, "b")

    # (c) Shell structure stacked bar
    h_max = np.percentile(H, 98)
    bin_edges = np.linspace(max(H.min(), 0), h_max, 25)
    bc = (bin_edges[:-1] + bin_edges[1:]) / 2
    dig = np.digitize(H, bin_edges)
    w = bin_edges[1] - bin_edges[0]

    mc = np.array([D_cond[dig == i].mean() if np.sum(dig == i) > 5
                   else np.nan for i in range(1, len(bin_edges))])
    mv = np.array([D_conv[dig == i].mean() if np.sum(dig == i) > 5
                   else np.nan for i in range(1, len(bin_edges))])
    mH = np.array([H[dig == i].mean() if np.sum(dig == i) > 5
                   else np.nan for i in range(1, len(bin_edges))])

    ok = ~np.isnan(mc) & ~np.isnan(mv)
    ax3.bar(bc[ok], mc[ok], width=w * 0.9, color=PAL.COND, alpha=0.75,
            label=r"$D_{\rm cond}$")
    ax3.bar(bc[ok], mv[ok], width=w * 0.9, bottom=mc[ok],
            color=PAL.CONV, alpha=0.75, label=r"$D_{\rm conv}$")
    ax3.plot(bc[ok], mH[ok], "k-", lw=1.0, label=r"$H_{\rm total}$")

    ax3.set_xlabel("Total thickness bin (km)")
    ax3.set_ylabel("Mean layer thickness (km)")
    ax3.legend(fontsize=6, loc="upper left")
    label_panel(ax3, "c")

    fig.suptitle(f"Equatorial proxy: {title} (N = {n:,})", fontsize=9, y=1.02)
    fig.tight_layout(w_pad=1.5)
    save_fig(fig, f"fig_eq_{tag}", FIGURES_DIR)


def main():
    for filename, title, tag in MODES:
        filepath = os.path.join(RESULTS_DIR, filename)
        if os.path.exists(filepath):
            print(f"\nPlotting: {title}")
            plot_mode(filepath, title, tag)
        else:
            print(f"Skipping {title}: {filepath} not found")


if __name__ == "__main__":
    main()
