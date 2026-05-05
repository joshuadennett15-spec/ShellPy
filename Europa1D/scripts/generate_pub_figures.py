#!/usr/bin/env python3
"""
Generate all publication-quality figures for Europa ice shell convection paper.

Reads existing Monte Carlo results from results/ and produces polished
figures in figures/pub/.

Usage:
    python generate_pub_figures.py          # all figures
    python generate_pub_figures.py fig1     # specific figure
    python generate_pub_figures.py fig1 fig3 fig5   # multiple
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator, LogLocator, AutoMinorLocator
from scipy.stats import gaussian_kde

from pub_style import (
    apply_style, PAL, THICKNESS_BINS,
    figsize_single, figsize_double, figsize_double_tall,
    label_panel, save_fig, add_minor_gridlines,
    SINGLE_COL, DOUBLE_COL,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures", "pub")

apply_style()

RA_CRIT = 1.0e3


# ═══════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════

def _load_mc(filename):
    """Load MC results, return (H, D_cond, D_conv, lid_frac, Ra, data)."""
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        print(f"  WARNING: {filename} not found, skipping")
        return None
    data = np.load(path)
    H = data["thicknesses_km"]
    D_cond = data["D_cond_km"]
    D_conv = data["D_conv_km"]
    lid_frac = data["lid_fractions"]
    Ra = data["Ra_values"] if "Ra_values" in data else np.zeros(len(H))
    return H, D_cond, D_conv, lid_frac, Ra, data


def _load_diag(filename):
    """Load diagnostic profile results."""
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        print(f"  WARNING: {filename} not found, skipping")
        return None
    return np.load(path)


def _kde_smooth(values, x_grid=None, n_pts=300):
    """Gaussian KDE on values, evaluated on x_grid."""
    if len(values) < 10:
        return None, None
    kde = gaussian_kde(values)
    if x_grid is None:
        lo = max(0, np.percentile(values, 0.5) - 2)
        hi = np.percentile(values, 99.5) + 2
        x_grid = np.linspace(lo, hi, n_pts)
    return x_grid, kde(x_grid)


def _binned_means(H, D_cond, D_conv, n_bins=30):
    """Binned mean layer thicknesses."""
    h_max = np.percentile(H, 98)
    bin_edges = np.linspace(H.min(), h_max, n_bins)
    bc = (bin_edges[:-1] + bin_edges[1:]) / 2
    dig = np.digitize(H, bin_edges)
    mc = np.array([D_cond[dig == i].mean() if np.sum(dig == i) > 5
                   else np.nan for i in range(1, len(bin_edges))])
    mv = np.array([D_conv[dig == i].mean() if np.sum(dig == i) > 5
                   else np.nan for i in range(1, len(bin_edges))])
    mH = np.array([H[dig == i].mean() if np.sum(dig == i) > 5
                   else np.nan for i in range(1, len(bin_edges))])
    return bin_edges, bc, mc, mv, mH


def _envelope(profiles, H_arr, mask, n_interp=100):
    """Percentile envelope on normalised depth grid."""
    zn = np.linspace(0, 1, n_interp)
    nx = profiles.shape[1]
    z_norm = np.linspace(0, 1, nx)
    stack = []
    for i in np.where(mask)[0]:
        stack.append(np.interp(zn, z_norm, profiles[i]))
    if not stack:
        return zn, None, None, None
    stack = np.array(stack)
    return (zn, np.median(stack, axis=0),
            np.percentile(stack, 10, axis=0),
            np.percentile(stack, 90, axis=0))


# ═══════════════════════════════════════════════════════════════════════════
# Figure 1: Monte Carlo thickness distributions
# ═══════════════════════════════════════════════════════════════════════════

def fig1_mc_distributions():
    """Two-panel: (a) total thickness PDF, (b) conductive lid PDF."""
    print("Figure 1: Monte Carlo distributions")

    # Audited 2026 baseline (Andrade, pure-ice, direct q_basal)
    result = _load_mc("mc_15000_optionA_v2_andrade.npz")
    label_suffix = ""
    if result is None:
        result = _load_mc("global_updated_ptidal.npz")
    if result is None:
        result = _load_mc("monte_carlo_results.npz")
    if result is None:
        return
    H, D_cond, D_conv, lid_frac, Ra, data = result
    n = len(H)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize_double(0.40))

    # ── (a) Total thickness ──
    bins_h = np.linspace(0, np.percentile(H, 99.5) + 2, 60)
    ax1.hist(H, bins=bins_h, density=True, color=PAL.BLUE, alpha=0.25,
             edgecolor=PAL.BLUE, linewidth=0.3)
    x_h, pdf_h = _kde_smooth(H)
    ax1.plot(x_h, pdf_h, color=PAL.BLUE, lw=1.5, label="KDE")

    # CBE
    cbe = float(x_h[np.argmax(pdf_h)])
    median_h = float(np.median(H))
    p16 = float(np.percentile(H, 15.87))
    p84 = float(np.percentile(H, 84.13))

    ax1.axvline(cbe, color=PAL.BLACK, ls="--", lw=0.8, alpha=0.7)
    ax1.axvspan(p16, p84, color=PAL.BLUE, alpha=0.06,
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

    # ── (b) Conductive lid thickness ──
    bins_d = np.linspace(0, np.percentile(D_cond, 99.5) + 2, 50)
    ax2.hist(D_cond, bins=bins_d, density=True, color=PAL.BLUE, alpha=0.25,
             edgecolor=PAL.BLUE, linewidth=0.3)
    x_d, pdf_d = _kde_smooth(D_cond)
    ax2.plot(x_d, pdf_d, color=PAL.BLUE, lw=1.5, label="KDE")

    cbe_d = float(x_d[np.argmax(pdf_d)])
    ax2.axvline(cbe_d, color=PAL.BLACK, ls="--", lw=0.8, alpha=0.7)
    ax2.annotate(f"Mode = {cbe_d:.1f} km",
                 xy=(cbe_d, pdf_d.max() * 0.97), xytext=(cbe_d + 8, pdf_d.max() * 0.85),
                 fontsize=7, ha="left",
                 arrowprops=dict(arrowstyle="-", lw=0.5, color="0.4"))

    ax2.set_xlabel(r"Conductive lid thickness $D_\mathrm{cond}$ (km)")
    ax2.set_ylabel("Probability density")
    ax2.set_xlim(0, np.percentile(D_cond, 99.5) + 5)
    ax2.set_ylim(bottom=0)
    add_minor_gridlines(ax2, axis="y")
    label_panel(ax2, "b")

    fig.suptitle(f"Monte Carlo ice shell thickness (N = {n:,}{label_suffix})",
                 fontsize=9, y=1.02)
    fig.tight_layout(w_pad=2.5)
    save_fig(fig, "fig1_mc_distributions", FIGURES_DIR)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 2: Shell structure (layer PDFs + stacked bar)
# ═══════════════════════════════════════════════════════════════════════════

def fig2_shell_structure():
    """Two-panel: (a) D_cond/D_conv PDFs, (b) stacked mean structure vs H."""
    print("Figure 2: Shell structure")

    result = _load_mc("mc_15000_optionA_v2_andrade.npz")
    if result is None:
        result = _load_mc("global_updated_ptidal.npz")
    if result is None:
        return
    H, D_cond, D_conv, lid_frac, Ra, data = result
    n = len(H)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize_double(0.42))

    # ── (a) Layer distributions ──
    bins = np.linspace(0, np.percentile(H, 99), 55)
    ax1.hist(D_cond, bins=bins, density=True, color=PAL.COND, alpha=0.30,
             edgecolor=PAL.COND, linewidth=0.3, label=r"$D_\mathrm{cond}$ (lid)")

    # Only show D_conv for convecting samples
    D_conv_active = D_conv[D_conv > 0.5]
    if len(D_conv_active) > 0:
        frac_conv = len(D_conv_active) / len(D_conv)
        ax1.hist(D_conv_active, bins=bins, density=True, color=PAL.CONV, alpha=0.30,
                 edgecolor=PAL.CONV, linewidth=0.3,
                 weights=np.full(len(D_conv_active), frac_conv),
                 label=rf"$D_\mathrm{{conv}}$ ({frac_conv:.0%} convecting)")

    # Add KDE lines
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

    # ── (b) Stacked bar: mean structure by thickness bin ──
    be, bc, mc, mv, mH = _binned_means(H, D_cond, D_conv)
    ok = ~np.isnan(mc) & ~np.isnan(mv) & ~np.isnan(mH)
    w = be[1] - be[0]

    ax2.bar(bc[ok], mc[ok], width=w * 0.92, color=PAL.COND, alpha=0.75,
            label=r"$D_\mathrm{cond}$")
    ax2.bar(bc[ok], mv[ok], width=w * 0.92, bottom=mc[ok], color=PAL.CONV,
            alpha=0.75, label=r"$D_\mathrm{conv}$")
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
# Figure 3: Budget-constrained regional comparison
# ═══════════════════════════════════════════════════════════════════════════

def _load_and_merge(eq_file, pole_file):
    """Load equator + pole results, merge into a single combined dataset."""
    eq = _load_mc(eq_file)
    pole = _load_mc(pole_file)
    if eq is None and pole is None:
        return None
    arrays = []
    for r in (eq, pole):
        if r is not None:
            arrays.append(r)
    H = np.concatenate([a[0] for a in arrays])
    Dc = np.concatenate([a[1] for a in arrays])
    Dv = np.concatenate([a[2] for a in arrays])
    lf = np.concatenate([a[3] for a in arrays])
    Ra = np.concatenate([a[4] for a in arrays])
    return H, Dc, Dv, lf, Ra


def fig3_regional_comparison():
    """
    Four-panel equatorial ocean heat-transport comparison.

    Scenarios (audited 2026 priors, Andrade rheology, N=15,000):
      Global baseline        — audited global, no latitude forcing
      Eq Depleted strong     — equatorial 0.55x (Lemasquerier 2023, q*=0.91)
      Eq Depleted            — equatorial 0.67x (Lemasquerier 2023, 2:1 pole/eq)
      Eq Baseline            — equatorial surface (1.0x uniform)
      Eq Moderate            — equatorial + 1.2x (Soderlund 2014 proxy)
      Eq Strong              — equatorial + 1.5x enhancement
    """
    print("Figure 3: Equatorial ocean heat-transport comparison (audited 2026)")

    SCENARIOS = [
        ("Global\n(Audited baseline)",
         "mc_15000_optionA_v2_andrade.npz", PAL.BLACK),
        ("Eq Depleted strong\n(0.55\u00d7 Lemasquerier)",
         "eq_depleted_strong_andrade.npz", PAL.PURPLE),
        ("Eq Depleted\n(0.67\u00d7 Lemasquerier)",
         "eq_depleted_andrade.npz", PAL.CYAN),
        ("Eq Baseline\n(1.0\u00d7 uniform)",
         "eq_baseline_andrade.npz", PAL.GREEN),
        ("Eq Moderate\n(1.2\u00d7 Soderlund)",
         "eq_moderate_andrade.npz", PAL.ORANGE),
        ("Eq Strong\n(1.5\u00d7 enhancement)",
         "eq_strong_andrade.npz", PAL.BLUE),
    ]

    loaded = {}
    for label, filename, col in SCENARIOS:
        result = _load_mc(filename)
        if result is not None:
            loaded[label] = (result[:5], col)

    if len(loaded) < 2:
        print(f"  Only {len(loaded)} scenarios loaded, need >= 2. Skipping fig3.")
        return

    SUPTITLE = ("Equatorial ocean heat-transport comparison "
                "(audited 2026 priors, Andrade, N = 15,000)")

    def _plot_ab(fig_or_axes=None, panel_a_label="a", panel_b_label="b"):
        """Panels (a) thickness PDFs and (b) box-whisker."""
        if fig_or_axes is None:
            fig_ab = plt.figure(figsize=(DOUBLE_COL, DOUBLE_COL * 0.42))
            gs_ab = gridspec.GridSpec(1, 2, wspace=0.35)
            ax_a = fig_ab.add_subplot(gs_ab[0, 0])
            ax_b = fig_ab.add_subplot(gs_ab[0, 1])
        else:
            ax_a, ax_b = fig_or_axes
            fig_ab = None

        for label in loaded:
            (H, Dc, Dv, lf, Ra), col = loaded[label]
            x, pdf = _kde_smooth(H)
            if pdf is None:
                continue
            short = label.split("\n")[0]
            ax_a.plot(x, pdf, color=col, lw=1.4,
                      label=f"{short} (N={len(H):,})")
            ax_a.fill_between(x, 0, pdf, color=col, alpha=0.08)
        ax_a.set_xlabel("Total ice shell thickness (km)")
        ax_a.set_ylabel("Probability density")
        ax_a.legend(fontsize=5.5, loc="upper right")
        ax_a.set_xlim(left=0)
        ax_a.set_ylim(bottom=0)
        label_panel(ax_a, panel_a_label)

        positions = []
        box_data = []
        tick_labels = []
        pos = 1
        col_cycle = [PAL.COND, PAL.CONV, "0.5"]
        group_info = []

        for label in loaded:
            (H, Dc, Dv, lf, Ra), col = loaded[label]
            Dv_active = Dv[Dv > 0.5] if np.sum(Dv > 0.5) > 0 else Dv
            box_data.extend([Dc, Dv_active, H])
            positions.extend([pos, pos + 1, pos + 2])
            tick_labels.extend(["$D_c$", "$D_v$", "$H$"])
            group_info.append((pos + 1, label.split("\n")[0], col))
            pos += 4

        bp = ax_b.boxplot(box_data, positions=positions, widths=0.60,
                          patch_artist=True, showfliers=False,
                          medianprops=dict(color="k", lw=0.8),
                          whiskerprops=dict(lw=0.5),
                          capprops=dict(lw=0.5))
        for i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(col_cycle[i % 3])
            patch.set_alpha(0.55)

        ax_b.set_xticks(positions)
        ax_b.set_xticklabels(tick_labels, fontsize=5.5)
        ax_b.set_ylabel("Thickness (km)")

        for cx, short_label, col in group_info:
            ax_b.annotate(short_label, xy=(cx, 0), xytext=(cx, -0.13),
                          xycoords=("data", "axes fraction"),
                          textcoords=("data", "axes fraction"),
                          ha="center", va="top", fontsize=5.5,
                          fontweight="bold", color=col)
        label_panel(ax_b, panel_b_label)
        return fig_ab

    def _plot_cd(fig_or_axes=None, panel_c_label="c", panel_d_label="d"):
        """Panels (c) mean layer vs H and (d) lid fraction."""
        if fig_or_axes is None:
            fig_cd = plt.figure(figsize=(DOUBLE_COL, DOUBLE_COL * 0.42))
            gs_cd = gridspec.GridSpec(1, 2, wspace=0.35)
            ax_c = fig_cd.add_subplot(gs_cd[0, 0])
            ax_d = fig_cd.add_subplot(gs_cd[0, 1])
        else:
            ax_c, ax_d = fig_or_axes
            fig_cd = None

        for label in loaded:
            (H, Dc, Dv, lf, Ra), col = loaded[label]
            short = label.split("\n")[0]
            _, bc, mc, mv, _ = _binned_means(H, Dc, Dv)
            ok = ~np.isnan(mc)
            ax_c.plot(bc[ok], mc[ok], "-", color=col, lw=1.2, marker="o", ms=2,
                      label=rf"{short} $D_\mathrm{{cond}}$")
            ok = ~np.isnan(mv)
            ax_c.plot(bc[ok], mv[ok], "--", color=col, lw=0.9, marker="s", ms=1.5,
                      alpha=0.65, label=rf"{short} $D_\mathrm{{conv}}$")
        ax_c.plot([0, 100], [0, 100], color="0.6", ls=":", lw=0.4, zorder=0)
        ax_c.set_xlabel("Total ice shell thickness (km)")
        ax_c.set_ylabel("Mean layer thickness (km)")
        ax_c.legend(fontsize=4.5, ncol=2, loc="upper left",
                    columnspacing=0.8, handletextpad=0.3)
        label_panel(ax_c, panel_c_label)

        for label in loaded:
            (H, Dc, Dv, lf, Ra), col = loaded[label]
            short = label.split("\n")[0]
            x, pdf = _kde_smooth(lf, x_grid=np.linspace(0, 1, 200))
            if pdf is not None:
                ax_d.plot(x, pdf, color=col, lw=1.4,
                          label=f"{short} (med={np.median(lf):.0%})")
                ax_d.fill_between(x, 0, pdf, color=col, alpha=0.08)
        ax_d.set_xlabel(r"Lid fraction ($D_\mathrm{cond} / H_\mathrm{total}$)")
        ax_d.set_ylabel("Probability density")
        ax_d.legend(fontsize=5.5)
        ax_d.set_xlim(0, 1)
        ax_d.set_ylim(bottom=0)
        label_panel(ax_d, panel_d_label)
        return fig_cd

    # ── Split figures ──
    fig_ab = _plot_ab()
    fig_ab.suptitle(SUPTITLE, fontsize=9, y=1.03)
    save_fig(fig_ab, "fig3_regional_comparison_ab", FIGURES_DIR)

    fig_cd = _plot_cd()
    fig_cd.suptitle(SUPTITLE, fontsize=9, y=1.03)
    save_fig(fig_cd, "fig3_regional_comparison_cd", FIGURES_DIR)

    # ── Combined 2x2 (keep for thesis) ──
    fig = plt.figure(figsize=figsize_double_tall(0.78))
    gs = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.35)
    _plot_ab(fig_or_axes=(fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])))
    _plot_cd(fig_or_axes=(fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])))
    fig.suptitle(SUPTITLE, fontsize=9, y=1.01)
    save_fig(fig, "fig3_regional_comparison", FIGURES_DIR)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 4: Howell Figure 4b reconstruction (conductive vs convective split)
# ═══════════════════════════════════════════════════════════════════════════

def fig4_howell_4b():
    """Conductive vs convective layer split at CBE thickness."""
    print("Figure 4: Howell Fig 4b reconstruction")

    result = _load_mc("mc_15000_optionA_v2_andrade.npz")
    if result is None:
        result = _load_mc("global_updated_ptidal.npz")
    if result is None:
        result = _load_mc("monte_carlo_results.npz")
    if result is None:
        return
    H, D_cond, D_conv, lid_frac, Ra, data = result

    # Conditional analysis: samples near the mode
    x_h, pdf_h = _kde_smooth(H)
    cbe = float(x_h[np.argmax(pdf_h)])

    # Select samples within 1 bin-width of CBE
    width = 3.0  # km window
    mask = np.abs(H - cbe) <= width
    if mask.sum() < 30:
        mask = np.abs(H - cbe) <= width * 2

    cond = D_cond[mask]
    conv = D_conv[mask]

    fig, ax = plt.subplots(figsize=figsize_single(0.80))

    # KDE for each layer
    x_grid = np.linspace(0, max(cond.max(), conv.max()) + 5, 300)

    if len(cond) > 5:
        _, pdf_c = _kde_smooth(cond, x_grid)
        ax.fill_between(x_grid, 0, pdf_c, color=PAL.COND, alpha=0.20)
        ax.plot(x_grid, pdf_c, color=PAL.COND, lw=1.5, label="Conductive lid")

        cond_mode = float(x_grid[np.argmax(pdf_c)])
        p16_c = float(np.percentile(cond, 15.87))
        p84_c = float(np.percentile(cond, 84.13))
        ax.annotate(
            f"{cond_mode:.1f}"
            rf"$^{{+{p84_c - cond_mode:.1f}}}_{{-{cond_mode - p16_c:.1f}}}$ km",
            xy=(cond_mode, pdf_c.max()),
            xytext=(cond_mode + 8, pdf_c.max() * 0.85),
            fontsize=7, color=PAL.COND,
            arrowprops=dict(arrowstyle="-", color=PAL.COND, lw=0.5))

    if len(conv[conv > 0.5]) > 5:
        conv_pos = conv[conv > 0.5]
        _, pdf_v = _kde_smooth(conv_pos, x_grid)
        ax.fill_between(x_grid, 0, pdf_v, color=PAL.CONV, alpha=0.20)
        ax.plot(x_grid, pdf_v, color=PAL.CONV, lw=1.5, label="Convective sublayer")

        conv_mode = float(x_grid[np.argmax(pdf_v)])
        p16_v = float(np.percentile(conv_pos, 15.87))
        p84_v = float(np.percentile(conv_pos, 84.13))
        ax.annotate(
            f"{conv_mode:.1f}"
            rf"$^{{+{p84_v - conv_mode:.1f}}}_{{-{conv_mode - p16_v:.1f}}}$ km",
            xy=(conv_mode, pdf_v.max()),
            xytext=(conv_mode + 8, pdf_v.max() * 0.85),
            fontsize=7, color=PAL.CONV,
            arrowprops=dict(arrowstyle="-", color=PAL.CONV, lw=0.5))

    ax.set_xlabel("Layer thickness (km)")
    ax.set_ylabel("Probability density")
    ax.set_xlim(0, 60)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", fontsize=7)

    # Annotate total CBE
    ax.text(0.03, 0.95, f"CBE total = {cbe:.1f} km\n(N = {mask.sum()} samples)",
            transform=ax.transAxes, fontsize=6.5, va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", lw=0.4))

    fig.tight_layout()
    save_fig(fig, "fig4_howell_4b", FIGURES_DIR)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 5: Diagnostic heating profiles
# ═══════════════════════════════════════════════════════════════════════════

def fig5_diagnostic_profiles():
    """Three-panel: (a) T(z/H), (b) q_tidal(T), (c) q_tidal(z/H)."""
    print("Figure 5: Diagnostic heating profiles")

    d = _load_diag("diagnostic_profiles.npz")
    if d is None:
        return

    H = d["H_km"]
    T_pr = d["T_profiles"].copy()
    q_pr = d["q_profiles"] * 1e6  # W/m^3 -> uW/m^3
    Nu = d["Nu"]
    D_cond = d["D_cond_km"]
    z_pr = d["z_profiles"]
    n = len(H)

    # Enforce isothermal convective cores for visualization
    if "T_c" in d:
        T_c = d["T_c"]
        for i in range(n):
            if Nu[i] > 1.0 and D_cond[i] < H[i]:
                is_conv = z_pr[i] > D_cond[i]
                if np.any(is_conv):
                    T_base = T_pr[i, -1]
                    T_pr[i, is_conv] = T_c[i]
                    T_pr[i, -1] = T_base

    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, 2.8))

    # ── (a) T(z/H) ──
    ax = axes[0]
    for label, lo, hi, col in THICKNESS_BINS:
        mask = (H >= lo) & (H < hi)
        zn, med, p10, p90 = _envelope(T_pr, H, mask)
        if med is None:
            continue
        ax.plot(med, zn, color=col, lw=1.3,
                label=f"{label} (N={mask.sum()})")
        ax.fill_betweenx(zn, p10, p90, color=col, alpha=0.12)
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Normalised depth  $z / H$")
    ax.invert_yaxis()
    ax.legend(fontsize=5.5, loc="lower left")
    label_panel(ax, "a")

    # ── (b) q_tidal(T) ──
    ax = axes[1]
    T_common = np.linspace(100, 273, 200)
    for label, lo, hi, col in THICKNESS_BINS:
        mask = (H >= lo) & (H < hi)
        stack = []
        for i in np.where(mask)[0]:
            order = np.argsort(T_pr[i])
            stack.append(np.interp(T_common, T_pr[i][order], q_pr[i][order],
                                   left=0, right=0))
        if not stack:
            continue
        stack = np.array(stack)
        med = np.median(stack, axis=0)
        p10 = np.percentile(stack, 10, axis=0)
        p90 = np.percentile(stack, 90, axis=0)
        ax.plot(T_common, med, color=col, lw=1.3, label=f"{label}")
        ax.fill_between(T_common, p10, p90, color=col, alpha=0.10)
    ax.set_yscale("log")
    ax.set_ylim(1e-4, np.percentile(q_pr.max(axis=1), 99) * 2)
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel(r"Tidal heating $\dot{q}$ ($\mu$W m$^{-3}$)")
    ax.axvline(255, color="0.4", ls=":", lw=0.6, alpha=0.6)
    ax.text(256, ax.get_ylim()[1] * 0.5, r"$T_\mathrm{opt}$",
            fontsize=6, color="0.4", va="center")
    ax.legend(fontsize=5.5, loc="upper left")
    label_panel(ax, "b")

    # ── (c) q_tidal(z/H) ──
    ax = axes[2]
    for label, lo, hi, col in THICKNESS_BINS:
        mask = (H >= lo) & (H < hi)
        zn, med, p10, p90 = _envelope(q_pr, H, mask)
        if med is None:
            continue
        ax.plot(med, zn, color=col, lw=1.3, label=f"{label}")
        ax.fill_betweenx(zn, np.maximum(p10, 1e-5), p90, color=col, alpha=0.10)
    ax.set_xscale("log")
    ax.set_xlim(1e-4, np.percentile(q_pr.max(axis=1), 99) * 2)
    ax.set_xlabel(r"Tidal heating $\dot{q}$ ($\mu$W m$^{-3}$)")
    ax.set_ylabel("Normalised depth  $z / H$")
    ax.invert_yaxis()
    ax.legend(fontsize=5.5, loc="upper right")
    label_panel(ax, "c")

    fig.suptitle(f"Diagnostic heating profiles (N = {n:,})",
                 fontsize=9, y=1.03)
    fig.tight_layout(w_pad=1.8)
    save_fig(fig, "fig5_diagnostic_profiles", FIGURES_DIR)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 6: Viscosity diagnostics
# ═══════════════════════════════════════════════════════════════════════════

def fig6_viscosity():
    """Six-panel viscosity diagnostic."""
    print("Figure 6: Viscosity diagnostics")

    d = _load_diag("diagnostic_profiles.npz")
    if d is None:
        return

    H = d["H_km"]
    T_pr = d["T_profiles"]
    eta_pr = d["eta_profiles"]
    D_cond = d["D_cond_km"]
    d_grain = d["d_grain"]
    Nu = d["Nu"]
    n = len(H)
    nx = T_pr.shape[1]

    mu_shear = 3.3e9
    omega = 2.047e-5
    eta_opt = mu_shear / omega

    fig, axes = plt.subplots(2, 3, figsize=(DOUBLE_COL, DOUBLE_COL * 0.62))

    # ── (a) eta(z/H) ──
    ax = axes[0, 0]
    log_eta = np.log10(eta_pr)
    for label, lo, hi, col in THICKNESS_BINS:
        mask = (H >= lo) & (H < hi)
        zn, med, p10, p90 = _envelope(log_eta, H, mask)
        if med is None:
            continue
        ax.plot(10**med, zn, color=col, lw=1.2,
                label=f"{label} (N={mask.sum()})")
        ax.fill_betweenx(zn, 10**p10, 10**p90, color=col, alpha=0.10)
    ax.axvline(eta_opt, color="0.3", ls=":", lw=0.7)
    ax.text(eta_opt * 1.5, 0.03, r"$\eta_\mathrm{opt}$",
            fontsize=6, color="0.3")
    ax.set_xscale("log")
    ax.set_xlabel(r"Viscosity $\eta$ (Pa s)")
    ax.set_ylabel("Normalised depth $z / H$")
    ax.invert_yaxis()
    ax.legend(fontsize=5, loc="upper left")
    label_panel(ax, "a")

    # ── (b) eta(T) ──
    ax = axes[0, 1]
    T_common = np.linspace(50, 273, 300)
    for label, lo, hi, col in THICKNESS_BINS:
        mask = (H >= lo) & (H < hi)
        stack = []
        for i in np.where(mask)[0]:
            order = np.argsort(T_pr[i])
            eta_log = np.log10(eta_pr[i][order])
            stack.append(np.interp(T_common, T_pr[i][order], eta_log,
                                   left=eta_log[0], right=eta_log[-1]))
        if not stack:
            continue
        stack = np.array(stack)
        med = np.median(stack, axis=0)
        p10 = np.percentile(stack, 10, axis=0)
        p90 = np.percentile(stack, 90, axis=0)
        ax.plot(T_common, 10**med, color=col, lw=1.2, label=label)
        ax.fill_between(T_common, 10**p10, 10**p90, color=col, alpha=0.08)
    ax.axhline(eta_opt, color="0.3", ls=":", lw=0.7)
    ax.text(55, eta_opt * 2, r"$\eta_\mathrm{opt}$", fontsize=6, color="0.3")
    ax.set_yscale("log")
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel(r"Viscosity $\eta$ (Pa s)")
    ax.legend(fontsize=5, loc="upper right")
    label_panel(ax, "b")

    # ── (c) Basal viscosity vs thickness ──
    ax = axes[0, 2]
    eta_base = eta_pr[:, -1]
    for label, lo, hi, col in THICKNESS_BINS:
        mask = (H >= lo) & (H < hi)
        ax.scatter(H[mask], eta_base[mask], c=col, s=3, alpha=0.35,
                   edgecolors="none", label=label, rasterized=True)
    ax.axhline(eta_opt, color="0.3", ls=":", lw=0.7)
    ax.set_yscale("log")
    ax.set_xlabel("Ice shell thickness (km)")
    ax.set_ylabel(r"Basal viscosity $\eta_\mathrm{base}$ (Pa s)")
    ax.legend(fontsize=5, loc="upper left", markerscale=2)
    label_panel(ax, "c")

    # ── (d) Active zone fraction ──
    ax = axes[1, 0]
    frac_active = np.array([np.mean(eta_pr[i] <= eta_opt) for i in range(n)])
    for label, lo, hi, col in THICKNESS_BINS:
        mask = (H >= lo) & (H < hi)
        ax.scatter(H[mask], frac_active[mask] * 100, c=col, s=3, alpha=0.35,
                   edgecolors="none", label=label, rasterized=True)
    # Running median
    sort_idx = np.argsort(H)
    H_s = H[sort_idx]
    fa_s = (frac_active * 100)[sort_idx]
    win = max(n // 30, 20)
    med_H = np.convolve(H_s, np.ones(win) / win, mode="valid")
    nmed = len(med_H)
    med_fa = np.array([np.median(fa_s[max(0, i - win // 2):i + win // 2])
                       for i in range(win // 2, win // 2 + nmed)])
    ax.plot(med_H, med_fa, "k-", lw=1.2, label="Running median")
    ax.set_xlabel("Ice shell thickness (km)")
    ax.set_ylabel(r"Shell fraction $\eta \leq \eta_\mathrm{opt}$ (%)")
    ax.legend(fontsize=5, loc="upper right", markerscale=2)
    label_panel(ax, "d")

    # ── (e) Mid-shell eta vs d_grain ──
    ax = axes[1, 1]
    eta_mid = eta_pr[:, nx // 2]
    sc = ax.scatter(d_grain * 1e3, eta_mid, c=H, s=3, alpha=0.4,
                    cmap="viridis", edgecolors="none", rasterized=True,
                    vmin=np.percentile(H, 2), vmax=np.percentile(H, 98))
    cb = fig.colorbar(sc, ax=ax, fraction=0.05, pad=0.02, aspect=20)
    cb.set_label("$H$ (km)", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    ax.axhline(eta_opt, color="0.3", ls=":", lw=0.7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Grain size $d$ (mm)")
    ax.set_ylabel(r"Mid-shell $\eta_\mathrm{mid}$ (Pa s)")
    label_panel(ax, "e")

    # ── (f) Viscosity dynamic range ──
    ax = axes[1, 2]
    eta_range = np.log10(eta_pr.max(axis=1)) - np.log10(eta_pr.min(axis=1))
    for label, lo, hi, col in THICKNESS_BINS:
        mask = (H >= lo) & (H < hi)
        ax.scatter(H[mask], eta_range[mask], c=col, s=3, alpha=0.35,
                   edgecolors="none", label=label, rasterized=True)
    er_s = eta_range[sort_idx]
    med_er = np.array([np.median(er_s[max(0, i - win // 2):i + win // 2])
                       for i in range(win // 2, win // 2 + nmed)])
    ax.plot(med_H, med_er, "k-", lw=1.2, label="Running median")
    ax.set_xlabel("Ice shell thickness (km)")
    ax.set_ylabel("Viscosity range (orders of mag.)")
    ax.legend(fontsize=5, loc="upper right", markerscale=2)
    label_panel(ax, "f")

    fig.suptitle(f"Viscosity diagnostics (N = {n:,})", fontsize=9, y=1.01)
    fig.tight_layout(w_pad=1.2, h_pad=1.5)
    save_fig(fig, "fig6_viscosity", FIGURES_DIR)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 7: Sensitivity analysis (streamlined 4-panel)
# ═══════════════════════════════════════════════════════════════════════════

PARAM_LABELS = {
    "d_grain":    r"$d_\mathrm{grain}$",
    "epsilon_0":  r"$\varepsilon_0$",
    "T_surf":     r"$T_\mathrm{surf}$",
    "D_H2O":      r"$D_\mathrm{H_2O}$",
    "mu_ice":     r"$\mu_\mathrm{ice}$",
    "Q_v":        r"$Q_v$",
    "Q_b":        r"$Q_b$",
    "H_rad":      r"$H_\mathrm{rad}$",
    "P_tidal":    r"$P_\mathrm{tidal}$",
    "f_porosity": r"$f_\mathrm{por}$",
    "f_salt":     r"$f_\mathrm{salt}$",
    "T_phi":      r"$T_\phi$",
    "B_k":        r"$B_k$",
    "D0v":        r"$D_{0v}$",
    "D0b":        r"$D_{0b}$",
    "d_del":      r"$\delta_\mathrm{gb}$",
    # Sobol factor names (include unit suffixes)
    "q_basal_target_mW_m2": r"$q_\mathrm{basal}$",
    "d_grain_mm":           r"$d_\mathrm{grain}$",
    "T_surf_K":             r"$T_\mathrm{surf}$",
    "D_H2O_km":             r"$D_\mathrm{H_2O}$",
    "mu_ice_GPa":           r"$\mu_\mathrm{ice}$",
    "Q_v_kJ_mol":           r"$Q_v$",
    "Q_b_kJ_mol":           r"$Q_b$",
    "H_rad_pW_kg":          r"$H_\mathrm{rad}$",
}


def fig7_sensitivity():
    """Streamlined 4-panel: Spearman, PRCC, RF importance, composite."""
    print("Figure 7: Sensitivity analysis")

    import pandas as pd

    # Strategy 1: legacy composite CSV
    csv_path = os.path.join(RESULTS_DIR, "sensitivity_indices.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(RESULTS_DIR, "sobol_indices.csv")

    # Strategy 2: Sobol CSV from the analysis pipeline
    sobol_csv = None
    sobol_dir = os.path.join(RESULTS_DIR, "sobol")
    if os.path.isdir(sobol_dir):
        for sub in sorted(os.listdir(sobol_dir)):
            candidate = os.path.join(sobol_dir, sub, f"{sub}_indices.csv")
            if os.path.exists(candidate):
                sobol_csv = candidate
                break  # use first (global audited)

    if os.path.exists(csv_path):
        _fig7_legacy(csv_path)
    elif sobol_csv is not None:
        _fig7_sobol(sobol_csv)
    else:
        print("  No sensitivity CSV found, skipping")


def _fig7_legacy(csv_path):
    """Fig 7 from legacy composite sensitivity CSV."""
    import pandas as pd
    df = pd.read_csv(csv_path)

    has_composite = "composite" in df.columns
    has_spearman = "spearman_rho" in df.columns
    has_prcc = "PRCC" in df.columns
    has_rf = "RF_importance" in df.columns

    if not has_composite:
        print("  CSV doesn't have composite column, skipping")
        return

    df_sorted = df.sort_values("composite", ascending=True)
    labels = [PARAM_LABELS.get(n, n) for n in df_sorted["parameter"]]
    y_pos = np.arange(len(labels))

    fig, axes = plt.subplots(2, 2, figsize=figsize_double_tall(0.85))

    if has_spearman:
        ax = axes[0, 0]
        vals = df_sorted["spearman_rho"].values
        colors = [PAL.RED if v < 0 else PAL.BLUE for v in vals]
        ax.barh(y_pos, vals, color=colors, height=0.7, edgecolor="none")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=6.5)
        ax.set_xlabel(r"Spearman $\rho$")
        ax.axvline(0, color="k", lw=0.4)
        label_panel(ax, "a")

    if has_prcc:
        ax = axes[0, 1]
        vals = df_sorted["PRCC"].values
        colors = [PAL.RED if v < 0 else PAL.BLUE for v in vals]
        ax.barh(y_pos, vals, color=colors, height=0.7, edgecolor="none")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=6.5)
        ax.set_xlabel("PRCC")
        ax.axvline(0, color="k", lw=0.4)
        label_panel(ax, "b")

    if has_rf:
        ax = axes[1, 0]
        vals = df_sorted["RF_importance"].values
        errs = df_sorted.get("RF_importance_std", np.zeros(len(vals)))
        if isinstance(errs, float):
            errs = np.zeros(len(vals))
        else:
            errs = errs.values
        ax.barh(y_pos, vals, color=PAL.GREEN, height=0.7, edgecolor="none",
                xerr=errs, capsize=1.5, ecolor="0.4", error_kw={"lw": 0.5})
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=6.5)
        ax.set_xlabel(r"Permutation importance ($\Delta R^2$)")
        label_panel(ax, "c")

    ax = axes[1, 1]
    vals = df_sorted["composite"].values
    bar_colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(vals)))
    ax.barh(y_pos, vals, color=bar_colors, height=0.7, edgecolor="none")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=6.5)
    ax.set_xlabel("Composite score (normalised)")
    label_panel(ax, "d")

    fig.suptitle("Global sensitivity analysis", fontsize=9, y=1.00)
    fig.tight_layout(w_pad=1.5, h_pad=1.5)
    save_fig(fig, "fig7_sensitivity", FIGURES_DIR)


def _fig7_sobol(csv_path):
    """Fig 7 from Sobol variance-decomposition indices (S1, ST)."""
    import pandas as pd
    df = pd.read_csv(csv_path)

    # Filter to thickness output at largest available sample size
    df_t = df[df["output"] == "thickness_km"].copy()
    if df_t.empty:
        print("  No thickness_km rows in Sobol CSV, skipping")
        return
    max_n = df_t["sample_size"].max()
    df_main = df_t[(df_t["sample_size"] == max_n) & (df_t["index_type"] == "main")]

    # Drop NaN rows and sort by total-order index
    df_main = df_main.dropna(subset=["ST"])
    df_main = df_main.sort_values("ST", ascending=True)

    # Map factor names to LaTeX labels
    factor_to_label = {}
    for f in df_main["factor"]:
        if f in PARAM_LABELS:
            factor_to_label[f] = PARAM_LABELS[f]
        else:
            # Strip units suffix and try base name
            for key in PARAM_LABELS:
                if f.startswith(key) or key in f:
                    factor_to_label[f] = PARAM_LABELS[key]
                    break
        if f not in factor_to_label:
            factor_to_label[f] = f.replace("_", " ")

    labels = [factor_to_label[f] for f in df_main["factor"]]
    y_pos = np.arange(len(labels))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize_double(0.50))

    # ── (a) First-order S1 ──
    s1 = df_main["S1"].values
    s1_ci = df_main["S1_conf"].values
    colors_s1 = [PAL.BLUE if v >= 0.01 else "0.6" for v in s1]
    ax1.barh(y_pos, s1, xerr=s1_ci, height=0.7, color=colors_s1,
             edgecolor="none", capsize=2, ecolor="0.4",
             error_kw={"lw": 0.5})
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels, fontsize=6.5)
    ax1.set_xlabel(r"First-order index $S_1$")
    ax1.set_xlim(left=-0.05)
    ax1.axvline(0, color="k", lw=0.3)
    label_panel(ax1, "a")

    # ── (b) Total-order ST ──
    st = df_main["ST"].values
    st_ci = df_main["ST_conf"].values
    colors_st = [PAL.RED if v >= 0.01 else "0.6" for v in st]
    ax2.barh(y_pos, st, xerr=st_ci, height=0.7, color=colors_st,
             edgecolor="none", capsize=2, ecolor="0.4",
             error_kw={"lw": 0.5})
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels, fontsize=6.5)
    ax2.set_xlabel(r"Total-order index $S_T$")
    ax2.set_xlim(left=-0.05)
    ax2.axvline(0, color="k", lw=0.3)
    label_panel(ax2, "b")

    fig.suptitle(
        f"Sobol sensitivity indices — ice shell thickness (N = {max_n})",
        fontsize=9, y=1.02)
    fig.tight_layout(w_pad=2.0)
    save_fig(fig, "fig7_sensitivity_sobol", FIGURES_DIR)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 8: SHAP beeswarm
# ═══════════════════════════════════════════════════════════════════════════

def _shap_beeswarm(ax, shap_values, X, labels_all, idx, rng,
                    n_show_feats, xlabel, add_colorbar=False):
    """Draw a SHAP beeswarm on *ax*. Helper for fig8_shap."""
    order = np.argsort(np.abs(shap_values).mean(axis=0))[::-1]
    for rank in range(n_show_feats):
        feat_i = order[rank]
        sv = shap_values[idx, feat_i]
        fv = X[idx, feat_i]
        fmin, fmax = fv.min(), fv.max()
        fv_norm = (fv - fmin) / (fmax - fmin + 1e-30)
        jitter = rng.normal(0, 0.12, size=len(idx))
        y_val = (n_show_feats - 1 - rank) + jitter
        ax.scatter(sv, y_val, c=fv_norm, cmap="coolwarm", s=2, alpha=0.4,
                   edgecolors="none", rasterized=True)

    ax.set_yticks(range(n_show_feats))
    ax.set_yticklabels(list(reversed([labels_all[order[i]]
                                      for i in range(n_show_feats)])),
                       fontsize=7)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.axvline(0, color="k", lw=0.4)
    ax.tick_params(axis="x", labelsize=7)

    if add_colorbar:
        sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(0, 1))
        sm.set_array([])
        cb = ax.figure.colorbar(sm, ax=ax, fraction=0.02, pad=0.02, aspect=20)
        cb.set_ticks([0, 1])
        cb.set_ticklabels(["Low", "High"])
        cb.set_label("Feature value", fontsize=6)
        cb.ax.tick_params(labelsize=5)


def fig8_shap():
    """Side-by-side SHAP beeswarms: (a) H_total, (b) D_cond."""
    print("Figure 8: SHAP analysis (H_total vs D_cond)")

    mc_path = os.path.join(RESULTS_DIR, "mc_15000_optionA_v2_andrade.npz")
    if not os.path.exists(mc_path):
        mc_path = os.path.join(RESULTS_DIR, "monte_carlo_results.npz")
    if not os.path.exists(mc_path):
        print("  Need MC results .npz with param_ arrays, skipping")
        return

    try:
        from sklearn.ensemble import RandomForestRegressor
        import shap
    except ImportError:
        print("  Need sklearn + shap, skipping")
        return

    data = np.load(mc_path)
    Y_htotal = data["thicknesses_km"]
    Y_dcond = data["D_cond_km"]

    param_names = []
    param_arrays = []
    for key in sorted(data.keys()):
        if key.startswith("param_"):
            name = key[6:]
            arr = data[key]
            if arr.shape[0] == Y_htotal.shape[0]:
                param_names.append(name)
                param_arrays.append(arr)
    if not param_names:
        print("  No param arrays in MC results, skipping")
        return

    X_full = np.column_stack(param_arrays)
    labels_all = [PARAM_LABELS.get(n, n) for n in param_names]

    # Subsample BEFORE RF/SHAP to keep runtime reasonable
    rng = np.random.default_rng(0)
    n_sub = min(3000, X_full.shape[0])
    idx = rng.choice(X_full.shape[0], n_sub, replace=False)
    X = X_full[idx]
    Y_htotal = Y_htotal[idx]
    Y_dcond = Y_dcond[idx]
    n_show_feats = min(8, len(param_names))
    plot_idx = np.arange(min(2000, n_sub))  # for beeswarm scatter

    # Train two RFs and compute SHAP for each target
    shap_dict = {}
    for label, Y in [("H_total", Y_htotal), ("D_cond", Y_dcond)]:
        print(f"  Training RF for {label} ...")
        rf = RandomForestRegressor(n_estimators=200, max_depth=None,
                                   min_samples_leaf=5, random_state=42,
                                   n_jobs=-1)
        rf.fit(X, Y)
        explainer = shap.TreeExplainer(rf)
        shap_dict[label] = explainer.shap_values(X)

    # ── Side-by-side figure ──
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(DOUBLE_COL, DOUBLE_COL * 0.48),
                                      sharey=False)
    fig.subplots_adjust(wspace=0.45, left=0.08, right=0.93, top=0.92, bottom=0.08)

    _shap_beeswarm(ax_a, shap_dict["H_total"], X, labels_all, plot_idx, rng,
                   n_show_feats,
                   xlabel="SHAP value (impact on $H_\\mathrm{total}$, km)")
    label_panel(ax_a, "a")
    ax_a.set_title("$H_\\mathrm{total}$", fontsize=9, pad=4)

    _shap_beeswarm(ax_b, shap_dict["D_cond"], X, labels_all, plot_idx, rng,
                   n_show_feats,
                   xlabel="SHAP value (impact on $D_\\mathrm{cond}$, km)",
                   add_colorbar=True)
    label_panel(ax_b, "b")
    ax_b.set_title("$D_\\mathrm{cond}$", fontsize=9, pad=4)

    save_fig(fig, "fig8_shap", FIGURES_DIR)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 9: Conditional sensitivity (KS + regime PRCC)
# ═══════════════════════════════════════════════════════════════════════════

def fig9_conditional():
    """Four-panel: KS stat, regime PRCC, top-2 CDFs."""
    print("Figure 9: Conditional sensitivity")

    csv_path = os.path.join(RESULTS_DIR, "sensitivity_conditional.csv")
    mc_path = os.path.join(RESULTS_DIR, "monte_carlo_results.npz")

    if not os.path.exists(csv_path) or not os.path.exists(mc_path):
        print("  Need sensitivity_conditional.csv + MC results, skipping")
        return

    import pandas as pd
    df_cond = pd.read_csv(csv_path)

    data = np.load(mc_path)
    Y = data["thicknesses_km"]
    param_names = []
    param_arrays = []
    for key in sorted(data.keys()):
        if key.startswith("param_"):
            name = key[6:]
            arr = data[key]
            if arr.shape[0] == Y.shape[0]:
                param_names.append(name)
                param_arrays.append(arr)
    X = np.column_stack(param_arrays)

    q_lo = np.percentile(Y, 20)
    q_hi = np.percentile(Y, 80)
    thin_mask = Y <= q_lo
    thick_mask = Y >= q_hi

    df_sorted = df_cond.sort_values("KS_stat", ascending=True)
    labels = [PARAM_LABELS.get(n, n) for n in df_sorted["parameter"]]
    y_pos = np.arange(len(labels))

    fig = plt.figure(figsize=figsize_double_tall(0.72))
    gs = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.35)

    # ── (a) KS statistic ──
    ax = fig.add_subplot(gs[0, 0])
    sig_colors = [PAL.RED if p < 0.01 else PAL.BLUE for p in df_sorted["KS_p"]]
    ax.barh(y_pos, df_sorted["KS_stat"].values, color=sig_colors,
            height=0.7, edgecolor="none")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=6.5)
    ax.set_xlabel("KS statistic")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=PAL.RED, label="$p < 0.01$"),
                       Patch(color=PAL.BLUE, label="$p \\geq 0.01$")],
              fontsize=6, loc="lower right")
    label_panel(ax, "a")

    # ── (b) Regime PRCC ──
    ax = fig.add_subplot(gs[0, 1])
    w = 0.35
    ax.barh(y_pos - w / 2, df_sorted["PRCC_thin"].values,
            height=w, color=PAL.CYAN, label=f"Thin (<{q_lo:.0f} km)",
            edgecolor="none")
    ax.barh(y_pos + w / 2, df_sorted["PRCC_thick"].values,
            height=w, color=PAL.RED, label=f"Thick (>{q_hi:.0f} km)",
            edgecolor="none")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=6.5)
    ax.set_xlabel("PRCC")
    ax.axvline(0, color="k", lw=0.4)
    ax.legend(fontsize=6, loc="lower right")
    label_panel(ax, "b")

    # ── (c, d) Top-2 CDF comparisons ──
    top2_ks = df_cond.sort_values("KS_stat", ascending=False).head(2)
    for panel_i, (_, row) in enumerate(top2_ks.iterrows()):
        ax = fig.add_subplot(gs[1, panel_i])
        pname = row["parameter"]
        j = param_names.index(pname)
        label = PARAM_LABELS.get(pname, pname)

        xs_thin = np.sort(X[thin_mask, j])
        ys_thin = np.arange(1, len(xs_thin) + 1) / len(xs_thin)
        xs_thick = np.sort(X[thick_mask, j])
        ys_thick = np.arange(1, len(xs_thick) + 1) / len(xs_thick)

        ax.step(xs_thin, ys_thin, color=PAL.CYAN, lw=1.2,
                label=f"Thin (<{q_lo:.0f} km)")
        ax.step(xs_thick, ys_thick, color=PAL.RED, lw=1.2,
                label=f"Thick (>{q_hi:.0f} km)")
        ax.set_xlabel(label)
        ax.set_ylabel("CDF")
        ax.set_title(f"KS = {row['KS_stat']:.3f}", fontsize=8)
        ax.legend(fontsize=6)
        label_panel(ax, chr(99 + panel_i))

    fig.suptitle("Conditional sensitivity: thin vs thick regimes",
                 fontsize=9, y=1.00)
    save_fig(fig, "fig9_conditional", FIGURES_DIR)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 10: Rheology comparison (Maxwell vs Andrade)
# ═══════════════════════════════════════════════════════════════════════════

def fig10_rheology_comparison():
    """Single-panel: overlaid thickness PDFs for Maxwell vs Andrade."""
    print("Figure 10: Rheology comparison")

    maxwell_path = os.path.join(RESULTS_DIR, "mc_maxwell_1000.npz")
    andrade_path = os.path.join(RESULTS_DIR, "mc_andrade_1000.npz")

    if not os.path.exists(maxwell_path) or not os.path.exists(andrade_path):
        print("  Need mc_maxwell_1000.npz + mc_andrade_1000.npz, skipping")
        return

    d_m = np.load(maxwell_path)
    d_a = np.load(andrade_path)
    h_m = d_m["thicknesses_km"]
    h_a = d_a["thicknesses_km"]

    fig, ax = plt.subplots(figsize=figsize_single(0.72))

    bins = np.linspace(0, 100, 55)

    # Histograms
    ax.hist(h_m, bins=bins, density=True, alpha=0.20, color=PAL.BLUE,
            edgecolor=PAL.BLUE, linewidth=0.3)
    ax.hist(h_a, bins=bins, density=True, alpha=0.20, color=PAL.ORANGE,
            edgecolor=PAL.ORANGE, linewidth=0.3)

    # KDE
    x_m, pdf_m = _kde_smooth(h_m)
    x_a, pdf_a = _kde_smooth(h_a)
    cbe_m = float(x_m[np.argmax(pdf_m)]) if pdf_m is not None else np.nan
    cbe_a = float(x_a[np.argmax(pdf_a)]) if pdf_a is not None else np.nan

    if pdf_m is not None:
        ax.plot(x_m, pdf_m, color=PAL.BLUE, lw=1.5,
                label=f"Maxwell (CBE = {cbe_m:.1f} km)")
    if pdf_a is not None:
        ax.plot(x_a, pdf_a, color=PAL.ORANGE, lw=1.5,
                label=f"Andrade (CBE = {cbe_a:.1f} km)")

    ax.set_xlabel("Ice shell thickness (km)")
    ax.set_ylabel("Probability density")
    ax.set_xlim(0, 100)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=7, loc="upper right")
    add_minor_gridlines(ax, axis="y")

    fig.tight_layout()
    save_fig(fig, "fig10_rheology", FIGURES_DIR)


# ═══════════════════════════════════════════════════════════════════════════
# Main dispatcher
# ═══════════════════════════════════════════════════════════════════════════

FIGURES = {
    "fig1": fig1_mc_distributions,
    "fig2": fig2_shell_structure,
    "fig3": fig3_regional_comparison,
    "fig4": fig4_howell_4b,
    "fig5": fig5_diagnostic_profiles,
    "fig6": fig6_viscosity,
    "fig7": fig7_sensitivity,
    "fig8": fig8_shap,
    "fig9": fig9_conditional,
    "fig10": fig10_rheology_comparison,
}


def main():
    args = sys.argv[1:]
    if not args:
        targets = list(FIGURES.keys())
    else:
        targets = [a for a in args if a in FIGURES]
        unknown = [a for a in args if a not in FIGURES]
        if unknown:
            print(f"Unknown figures: {unknown}")
            print(f"Available: {list(FIGURES.keys())}")
            return

    os.makedirs(FIGURES_DIR, exist_ok=True)
    print(f"Generating {len(targets)} publication figures -> {FIGURES_DIR}/\n")

    for name in targets:
        try:
            FIGURES[name]()
        except Exception as e:
            print(f"  ERROR in {name}: {e}")
            import traceback
            traceback.print_exc()
        print()

    print("Done.")


if __name__ == "__main__":
    main()
