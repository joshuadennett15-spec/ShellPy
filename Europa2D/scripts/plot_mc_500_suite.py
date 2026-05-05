"""
Publication figure suite for the 500-sample latitude-aware Monte Carlo runs.

Generates per-scenario and combined comparison figures in a new folder:
  Europa2D/figures/mc_500/

Figures produced:
  1. Per-scenario shell structure (H_total + D_cond + D_conv cross-section)
  2. Combined 4-scenario thickness comparison (2x2 panels)
  3. Combined 4-scenario convection diagnostics (conv%, Nu, Ra)
  4. D_cond violin plot (equatorial vs polar split violins)
  5. Summary table printed to console
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.join(_SCRIPT_DIR, "..")
sys.path.insert(0, os.path.join(_PROJECT_DIR, "src"))
sys.path.insert(0, os.path.join(_PROJECT_DIR, "..", "Europa1D", "src"))
sys.path.insert(0, _SCRIPT_DIR)

from pub_style import (apply_style, PAL, label_panel, save_fig,
                        add_minor_gridlines, DOUBLE_COL, SINGLE_COL)
from profile_diagnostics import band_mean_samples

RESULTS_DIR = os.path.join(_PROJECT_DIR, "results")
FIGURES_DIR = os.path.join(_PROJECT_DIR, "figures", "mc_500")
N_ITER = 500

# Juno induction constraint (Khurana et al. 2024)
JUNO_DCOND_KM = 29.0
JUNO_DCOND_ERR = 10.0
JUNO_LAT_DEG = 35.0

RA_CRIT = 1000.0

# Latitude bands for area-weighted summaries
BAND_EQ = (0.0, 10.0)
BAND_MID = (30.0, 50.0)
BAND_POLAR = (80.0, 90.0)

# Wider bands for violin plot
VIOLIN_EQ = (0.0, 30.0)
VIOLIN_POLAR = (60.0, 90.0)

SCENARIOS = [
    ("uniform_transport",             "Uniform transport",      PAL.BLACK),
    ("soderlund2014_equator",         "Equator-enhanced",       "#B8860B"),
    ("lemasquerier2023_polar",        "Polar-enhanced",         PAL.BLUE),
    ("lemasquerier2023_polar_strong", "Strong polar-enhanced",  PAL.RED),
]

CITATIONS = {
    "uniform_transport": "Ashkenazy & Tziperman (2021)",
    "soderlund2014_equator": "Soderlund et al. (2014)",
    "lemasquerier2023_polar": "Lemasquerier et al. (2023)",
    "lemasquerier2023_polar_strong": "Lemasquerier et al. (2023)",
}


def _load(scenario_key):
    path = os.path.join(RESULTS_DIR, f"mc_2d_{scenario_key}_{N_ITER}.npz")
    return dict(np.load(path, allow_pickle=True))


def _interp_at(lat, arr, target_lat):
    """Interpolate a per-latitude array at a single target latitude for each sample."""
    return np.array([np.interp(target_lat, lat, arr[i]) for i in range(arr.shape[0])])


# ── Figure 1: Per-scenario individual shell structure ─────────────────────

def plot_individual_scenarios():
    """One figure per scenario: detailed shell cross-section with diagnostics."""
    apply_style()

    for key, title, sc_color in SCENARIOS:
        d = _load(key)
        lat = d["latitudes_deg"]
        H = d["H_profiles"]
        Dc = d["D_cond_profiles"]
        Dv = d["D_conv_profiles"]
        Nu = d["Nu_profiles"]
        Ra = d["Ra_profiles"]
        n_valid = int(d["n_valid"])

        fig, axes = plt.subplots(
            2, 2,
            figsize=(DOUBLE_COL, DOUBLE_COL * 0.72),
            gridspec_kw={"height_ratios": [1.6, 1]},
        )
        fig.subplots_adjust(hspace=0.38, wspace=0.35)

        # ── (a) Shell structure ──
        ax = axes[0, 0]
        H_med = np.median(H, axis=0)
        H_p16, H_p84 = np.percentile(H, 16, axis=0), np.percentile(H, 84, axis=0)
        H_p05, H_p95 = np.percentile(H, 5, axis=0), np.percentile(H, 95, axis=0)
        Dc_med = np.median(Dc, axis=0)
        Dc_p16, Dc_p84 = np.percentile(Dc, 16, axis=0), np.percentile(Dc, 84, axis=0)

        ax.fill_between(lat, H_p05, H_p95, color="0.82", alpha=0.35, lw=0)
        ax.fill_between(lat, H_p16, H_p84, color="0.68", alpha=0.30, lw=0)
        ax.fill_between(lat, 0, Dc_med, color=PAL.CYAN, alpha=0.22, lw=0)
        ax.fill_between(lat, Dc_med, H_med, color=PAL.ORANGE, alpha=0.22, lw=0)
        ax.fill_between(lat, Dc_p16, Dc_p84, color=PAL.CYAN, alpha=0.12, lw=0)
        ax.plot(lat, H_med, color=PAL.BLACK, lw=1.5, zorder=3)
        ax.plot(lat, Dc_med, color=PAL.BLUE, lw=1.2, zorder=3)

        # Band-mean markers
        for band, marker, ms in [(BAND_EQ, "o", 4), (BAND_MID, "s", 3.5), (BAND_POLAR, "^", 4)]:
            bc = (band[0] + band[1]) / 2
            h_b = band_mean_samples(lat, H, band)
            dc_b = band_mean_samples(lat, Dc, band)
            ax.plot(bc, np.median(h_b), marker=marker, ms=ms,
                    color=PAL.BLACK, mec=PAL.BLACK, mfc="white", mew=0.8, zorder=4)
            ax.plot(bc, np.median(dc_b), marker=marker, ms=ms,
                    color=PAL.BLUE, mec=PAL.BLUE, mfc="white", mew=0.8, zorder=4)

        # Juno constraint
        ax.errorbar(JUNO_LAT_DEG, JUNO_DCOND_KM, yerr=JUNO_DCOND_ERR,
                     fmt="D", ms=4, color=PAL.RED, ecolor=PAL.RED,
                     elinewidth=0.8, capsize=2.5, capthick=0.8, zorder=6)

        ax.set_xlim(0, 90)
        ax.set_ylim(0, 70)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(15))
        ax.yaxis.set_major_locator(mticker.MultipleLocator(10))
        ax.set_ylabel("Thickness (km)")
        ax.set_title("Shell structure", fontsize=8, fontweight="bold", loc="left")
        add_minor_gridlines(ax)
        label_panel(ax, "a")

        # Summary table inset
        dc_at_juno = _interp_at(lat, Dc, JUNO_LAT_DEG)
        h_eq = band_mean_samples(lat, H, BAND_EQ)
        h_mid = band_mean_samples(lat, H, BAND_MID)
        h_po = band_mean_samples(lat, H, BAND_POLAR)
        dc_eq = band_mean_samples(lat, Dc, BAND_EQ)
        dc_mid = band_mean_samples(lat, Dc, BAND_MID)
        dc_po = band_mean_samples(lat, Dc, BAND_POLAR)
        table = (
            f"         eq    mid   pole\n"
            f"H    {np.median(h_eq):5.1f} {np.median(h_mid):5.1f} {np.median(h_po):5.1f}\n"
            f"Dc   {np.median(dc_eq):5.1f} {np.median(dc_mid):5.1f} {np.median(dc_po):5.1f}\n"
            f"Juno Dc(35)={np.median(dc_at_juno):.1f} km\n"
            f"N={n_valid}"
        )
        ax.text(0.98, 0.97, table, transform=ax.transAxes, fontsize=5,
                va="top", ha="right", fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.92,
                          ec="0.75", lw=0.3))

        # ── (b) D_cond profile detail ──
        ax = axes[0, 1]
        Dv_med = np.median(Dv, axis=0)
        Dv_p16, Dv_p84 = np.percentile(Dv, 16, axis=0), np.percentile(Dv, 84, axis=0)

        ax.fill_between(lat, Dc_p16, Dc_p84, color=PAL.CYAN, alpha=0.20, lw=0)
        ax.plot(lat, Dc_med, color=PAL.BLUE, lw=1.5, label=r"$D_{\rm cond}$")
        ax.fill_between(lat, Dv_p16, Dv_p84, color=PAL.ORANGE, alpha=0.20, lw=0)
        ax.plot(lat, Dv_med, color=PAL.ORANGE, lw=1.2, label=r"$D_{\rm conv}$")

        ax.errorbar(JUNO_LAT_DEG, JUNO_DCOND_KM, yerr=JUNO_DCOND_ERR,
                     fmt="D", ms=4, color=PAL.RED, ecolor=PAL.RED,
                     elinewidth=0.8, capsize=2.5, capthick=0.8, zorder=6)

        ax.set_xlim(0, 90)
        ax.set_ylim(0, None)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(15))
        ax.set_ylabel("Layer thickness (km)")
        ax.set_title(r"$D_{\rm cond}$ and $D_{\rm conv}$", fontsize=8,
                      fontweight="bold", loc="left")
        ax.legend(fontsize=6, loc="upper right")
        add_minor_gridlines(ax)
        label_panel(ax, "b")

        # ── (c) Convecting fraction ──
        ax = axes[1, 0]
        conv_frac = np.mean(Nu > 1.1, axis=0) * 100
        ax.plot(lat, conv_frac, color=sc_color, lw=1.5)
        ax.axhline(50, color="0.6", lw=0.5, ls=":", zorder=0)
        ax.set_xlim(0, 90)
        ax.set_ylim(0, 100)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(15))
        ax.yaxis.set_major_locator(mticker.MultipleLocator(20))
        ax.set_xlabel(r"Latitude ($\degree$)")
        ax.set_ylabel("Convecting samples (%)")
        ax.set_title("Convecting fraction", fontsize=8, fontweight="bold", loc="left")
        add_minor_gridlines(ax)
        label_panel(ax, "c")

        # ── (d) Conditional Ra ──
        ax = axes[1, 1]
        n_lat = len(lat)
        ra_med = np.full(n_lat, np.nan)
        ra_lo = np.full(n_lat, np.nan)
        ra_hi = np.full(n_lat, np.nan)
        for j in range(n_lat):
            mask = Ra[:, j] > RA_CRIT
            if np.sum(mask) > 5:
                ra_med[j] = np.median(Ra[mask, j])
                ra_lo[j] = np.percentile(Ra[mask, j], 25)
                ra_hi[j] = np.percentile(Ra[mask, j], 75)
        ax.fill_between(lat, ra_lo, ra_hi, color=sc_color, alpha=0.15, lw=0)
        ax.plot(lat, ra_med, color=sc_color, lw=1.5)
        ax.axhline(RA_CRIT, color="0.6", lw=0.5, ls=":", zorder=0,
                   label=r"$Ra_{\rm crit}$")
        ax.set_yscale("log")
        ax.set_xlim(0, 90)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(15))
        ax.set_xlabel(r"Latitude ($\degree$)")
        ax.set_ylabel("Ra | convecting")
        ax.set_title("Rayleigh number (convecting)", fontsize=8,
                      fontweight="bold", loc="left")
        add_minor_gridlines(ax)
        label_panel(ax, "d")

        cite = CITATIONS[key]
        fig.suptitle(f"{title}  ({cite})\n{N_ITER} MC samples",
                     fontsize=9, fontweight="bold", y=1.02)

        save_fig(fig, f"scenario_{key}", FIGURES_DIR)


# ── Figure 2: Combined 4-scenario thickness (2x2) ────────────────────────

def plot_combined_thickness():
    """2x2 panels, one per scenario — shell cross-section with envelopes."""
    apply_style()

    fig, axes = plt.subplots(
        2, 2,
        figsize=(DOUBLE_COL, DOUBLE_COL * 0.78),
        sharex=True, sharey=True,
    )
    fig.subplots_adjust(hspace=0.30, wspace=0.12)

    for idx, (key, title, sc_color) in enumerate(SCENARIOS):
        ax = axes.flat[idx]
        d = _load(key)
        lat = d["latitudes_deg"]
        H = d["H_profiles"]
        Dc = d["D_cond_profiles"]
        n_valid = int(d["n_valid"])

        H_med = np.median(H, axis=0)
        H_p16, H_p84 = np.percentile(H, 16, axis=0), np.percentile(H, 84, axis=0)
        H_p05, H_p95 = np.percentile(H, 5, axis=0), np.percentile(H, 95, axis=0)
        Dc_med = np.median(Dc, axis=0)
        Dc_p16, Dc_p84 = np.percentile(Dc, 16, axis=0), np.percentile(Dc, 84, axis=0)

        ax.fill_between(lat, H_p05, H_p95, color="0.82", alpha=0.35, lw=0)
        ax.fill_between(lat, H_p16, H_p84, color="0.68", alpha=0.30, lw=0)
        ax.fill_between(lat, 0, Dc_med, color=PAL.CYAN, alpha=0.22, lw=0)
        ax.fill_between(lat, Dc_med, H_med, color=PAL.ORANGE, alpha=0.22, lw=0)
        ax.fill_between(lat, Dc_p16, Dc_p84, color=PAL.CYAN, alpha=0.12, lw=0)
        ax.plot(lat, H_med, color=PAL.BLACK, lw=1.5, zorder=3)
        ax.plot(lat, Dc_med, color=PAL.BLUE, lw=1.2, zorder=3)

        # Band-mean markers
        for band, marker, ms in [(BAND_EQ, "o", 4), (BAND_MID, "s", 3.5), (BAND_POLAR, "^", 4)]:
            bc = (band[0] + band[1]) / 2
            h_b = band_mean_samples(lat, H, band)
            dc_b = band_mean_samples(lat, Dc, band)
            ax.plot(bc, np.median(h_b), marker=marker, ms=ms,
                    color=PAL.BLACK, mec=PAL.BLACK, mfc="white", mew=0.8, zorder=4)
            ax.plot(bc, np.median(dc_b), marker=marker, ms=ms,
                    color=PAL.BLUE, mec=PAL.BLUE, mfc="white", mew=0.8, zorder=4)

        add_minor_gridlines(ax)
        ax.set_xlim(0, 90)
        ax.set_ylim(0, 70)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(15))
        ax.yaxis.set_major_locator(mticker.MultipleLocator(10))

        cite = CITATIONS[key]
        ax.set_title(title, fontsize=8, fontweight="bold", loc="left")
        ax.text(0.01, 0.91, cite, transform=ax.transAxes, fontsize=5.5,
                fontstyle="italic", color="0.45", va="top")
        label_panel(ax, chr(97 + idx))

        # Band-mean table inset
        h_eq = band_mean_samples(lat, H, BAND_EQ)
        h_mid = band_mean_samples(lat, H, BAND_MID)
        h_po = band_mean_samples(lat, H, BAND_POLAR)
        dc_eq = band_mean_samples(lat, Dc, BAND_EQ)
        dc_mid = band_mean_samples(lat, Dc, BAND_MID)
        dc_po = band_mean_samples(lat, Dc, BAND_POLAR)
        table = (
            f"         eq    mid   pole\n"
            f"H    {np.median(h_eq):5.1f} {np.median(h_mid):5.1f} {np.median(h_po):5.1f}\n"
            f"Dc   {np.median(dc_eq):5.1f} {np.median(dc_mid):5.1f} {np.median(dc_po):5.1f}\n"
            f"N={n_valid}"
        )
        ax.text(0.98, 0.97, table, transform=ax.transAxes, fontsize=5,
                va="top", ha="right", fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.92,
                          ec="0.75", lw=0.3))

    for ax in axes[1, :]:
        ax.set_xlabel(r"Latitude ($\degree$)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Thickness (km)")

    legend_elements = [
        Line2D([0], [0], color=PAL.BLACK, lw=1.5, label=r"$H_{\rm total}$ median"),
        Line2D([0], [0], color=PAL.BLUE, lw=1.2, label=r"$D_{\rm cond}$ median"),
        Patch(fc=PAL.CYAN, alpha=0.30, ec="none", label=r"$D_{\rm cond}$ (lid)"),
        Patch(fc=PAL.ORANGE, alpha=0.30, ec="none", label=r"$D_{\rm conv}$ (sublayer)"),
        Patch(fc="0.75", alpha=0.35, ec="none", label=r"$H_{\rm total}$ 1$\sigma$/2$\sigma$"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3, fontsize=6,
               bbox_to_anchor=(0.5, -0.03), columnspacing=1.5, handletextpad=0.5)

    save_fig(fig, "combined_4scenario_thickness", FIGURES_DIR)


# ── Figure 3: Combined 4-scenario diagnostics ────────────────────────────

def plot_combined_diagnostics():
    """3-panel: convecting fraction, conditional Nu, conditional Ra."""
    apply_style()

    fig, axes = plt.subplots(
        1, 3,
        figsize=(DOUBLE_COL, DOUBLE_COL * 0.36),
        sharex=True,
    )
    fig.subplots_adjust(wspace=0.40)
    ax_frac, ax_nu, ax_ra = axes

    for key, title, color in SCENARIOS:
        d = _load(key)
        lat = d["latitudes_deg"]
        Nu = d["Nu_profiles"]
        Ra = d["Ra_profiles"]
        n_lat = len(lat)

        # (a) Convecting fraction
        conv_frac = np.mean(Nu > 1.1, axis=0)
        ax_frac.plot(lat, conv_frac * 100, color=color, lw=1.3, label=title)

        # (b) Conditional Nu — median + IQR
        nu_med = np.full(n_lat, np.nan)
        nu_lo = np.full(n_lat, np.nan)
        nu_hi = np.full(n_lat, np.nan)
        for j in range(n_lat):
            mask = Nu[:, j] > 1.1
            if np.sum(mask) > 5:
                nu_med[j] = np.median(Nu[mask, j])
                nu_lo[j] = np.percentile(Nu[mask, j], 25)
                nu_hi[j] = np.percentile(Nu[mask, j], 75)
        ax_nu.fill_between(lat, nu_lo, nu_hi, color=color, alpha=0.10, lw=0)
        ax_nu.plot(lat, nu_med, color=color, lw=1.3)

        # (c) Conditional Ra — median + IQR
        ra_med = np.full(n_lat, np.nan)
        ra_lo = np.full(n_lat, np.nan)
        ra_hi = np.full(n_lat, np.nan)
        for j in range(n_lat):
            mask = Ra[:, j] > RA_CRIT
            if np.sum(mask) > 5:
                ra_med[j] = np.median(Ra[mask, j])
                ra_lo[j] = np.percentile(Ra[mask, j], 25)
                ra_hi[j] = np.percentile(Ra[mask, j], 75)
        ax_ra.fill_between(lat, ra_lo, ra_hi, color=color, alpha=0.10, lw=0)
        ax_ra.plot(lat, ra_med, color=color, lw=1.3)

    ax_frac.set_ylabel("Convecting samples (%)")
    ax_frac.set_ylim(0, 100)
    ax_frac.yaxis.set_major_locator(mticker.MultipleLocator(20))
    ax_frac.axhline(50, color="0.6", lw=0.5, ls=":", zorder=0)
    label_panel(ax_frac, "a")
    add_minor_gridlines(ax_frac)
    ax_frac.legend(fontsize=5.5, loc="lower left")

    ax_nu.set_ylabel("Nu | convecting")
    ax_nu.set_ylim(1, None)
    label_panel(ax_nu, "b")
    add_minor_gridlines(ax_nu)

    ax_ra.set_ylabel("Ra | convecting")
    ax_ra.set_yscale("log")
    ax_ra.axhline(RA_CRIT, color="0.6", lw=0.5, ls=":", zorder=0,
                  label=r"$Ra_{\rm crit}$")
    label_panel(ax_ra, "c")
    add_minor_gridlines(ax_ra)

    for ax in axes:
        ax.set_xlim(0, 90)
        ax.set_xlabel(r"Latitude ($\degree$)")
        ax.xaxis.set_major_locator(mticker.MultipleLocator(15))

    save_fig(fig, "combined_4scenario_diagnostics", FIGURES_DIR)


# ── Figure 4: D_cond violin plot ──────────────────────────────────────────

def _draw_half_violin(ax, datasets, positions, side, color, alpha=0.6):
    """Draw split violins with strip overlay and summary statistics."""
    parts = ax.violinplot(
        datasets, positions=positions, widths=0.7,
        showmeans=False, showmedians=False, showextrema=False,
    )
    for body in parts["bodies"]:
        verts = body.get_paths()[0].vertices
        m = np.mean(verts[:, 0])
        if side == "left":
            verts[:, 0] = np.clip(verts[:, 0], -np.inf, m)
        else:
            verts[:, 0] = np.clip(verts[:, 0], m, np.inf)
        body.set_facecolor(color)
        body.set_edgecolor("k")
        body.set_linewidth(0.5)
        body.set_alpha(alpha)

    rng = np.random.default_rng(42)
    for i, d in enumerate(datasets):
        med = np.median(d)
        q25, q75 = np.percentile(d, [25, 75])
        x = positions[i]
        dx = 0.18

        jitter = rng.uniform(0.02, dx * 0.95, len(d))
        if side == "left":
            ax.scatter(x - jitter, d, s=8, color=color, alpha=0.40,
                       edgecolors="k", linewidths=0.12, zorder=3, rasterized=True)
            ax.hlines(med, x - dx, x, color="k", lw=1.5, zorder=6)
            ax.hlines([q25, q75], x - dx * 0.6, x, color="k", lw=0.6,
                      ls="--", zorder=6)
            ax.text(x - dx - 0.04, med, f"{med:.0f}",
                    ha="right", va="center", fontsize=6.5, fontweight="bold", zorder=7)
        else:
            ax.scatter(x + jitter, d, s=8, color=color, alpha=0.40,
                       edgecolors="k", linewidths=0.12, zorder=3, rasterized=True)
            ax.hlines(med, x, x + dx, color="k", lw=1.5, zorder=6)
            ax.hlines([q25, q75], x, x + dx * 0.6, color="k", lw=0.6,
                      ls="--", zorder=6)
            ax.text(x + dx + 0.04, med, f"{med:.0f}",
                    ha="left", va="center", fontsize=6.5, fontweight="bold", zorder=7)


MAX_PHYSICAL_D_COND = 150.0


def plot_violin_dcond():
    """Split violin: equatorial vs polar D_cond across all 4 scenarios."""
    apply_style()

    fig, ax = plt.subplots(figsize=(DOUBLE_COL, DOUBLE_COL * 0.50))

    positions = np.arange(len(SCENARIOS))
    eq_data, polar_data = [], []

    for key, _, _ in SCENARIOS:
        d = _load(key)
        lat = d["latitudes_deg"]
        Dc = d["D_cond_profiles"]
        eq = band_mean_samples(lat, Dc, VIOLIN_EQ)
        polar = band_mean_samples(lat, Dc, VIOLIN_POLAR)
        mask = (eq < MAX_PHYSICAL_D_COND) & (polar < MAX_PHYSICAL_D_COND)
        eq_data.append(eq[mask])
        polar_data.append(polar[mask])

    _draw_half_violin(ax, eq_data, positions, "left", PAL.CYAN)
    _draw_half_violin(ax, polar_data, positions, "right", PAL.ORANGE)

    # Delta-D brackets
    for i in range(len(SCENARIOS)):
        eq_med = np.median(eq_data[i])
        polar_med = np.median(polar_data[i])
        x_bracket = positions[i] + 0.30
        ax.annotate("", xy=(x_bracket, polar_med), xytext=(x_bracket, eq_med),
                     arrowprops=dict(arrowstyle="<->", color="0.4", lw=0.8))
        dH = polar_med - eq_med
        ax.text(x_bracket + 0.05, (eq_med + polar_med) / 2,
                f"{dH:+.1f} km", fontsize=6, color="0.3", va="center", ha="left")

    ax.set_xticks(positions)
    ax.set_xticklabels([title for _, title, _ in SCENARIOS], fontsize=7)
    ax.set_ylabel(r"$D_{\rm cond}$ (km)", fontsize=9)
    ax.set_ylim(0, None)
    ax.set_title(
        f"Conductive lid thickness by ocean transport scenario\n"
        f"{N_ITER} MC samples, area-weighted band means",
        fontsize=9, fontweight="bold",
    )

    eq_patch = mpatches.Patch(facecolor=PAL.CYAN, alpha=0.6, edgecolor="k",
                               lw=0.5, label=r"Equatorial (0--30$\degree$)")
    polar_patch = mpatches.Patch(facecolor=PAL.ORANGE, alpha=0.6, edgecolor="k",
                                  lw=0.5, label=r"Polar (60--90$\degree$)")
    ax.legend(handles=[eq_patch, polar_patch], loc="upper left", fontsize=7)

    save_fig(fig, "violin_dcond_4scenario", FIGURES_DIR)


# ── Figure 5: H_total violin plot ─────────────────────────────────────────

def plot_violin_htotal():
    """Split violin: equatorial vs polar H_total across all 4 scenarios."""
    apply_style()

    fig, ax = plt.subplots(figsize=(DOUBLE_COL, DOUBLE_COL * 0.50))

    positions = np.arange(len(SCENARIOS))
    eq_data, polar_data = [], []

    for key, _, _ in SCENARIOS:
        d = _load(key)
        lat = d["latitudes_deg"]
        H = d["H_profiles"]
        eq = band_mean_samples(lat, H, VIOLIN_EQ)
        polar = band_mean_samples(lat, H, VIOLIN_POLAR)
        mask = (eq < 200.0) & (polar < 200.0)
        eq_data.append(eq[mask])
        polar_data.append(polar[mask])

    _draw_half_violin(ax, eq_data, positions, "left", PAL.CYAN)
    _draw_half_violin(ax, polar_data, positions, "right", PAL.ORANGE)

    # Delta-H brackets
    for i in range(len(SCENARIOS)):
        eq_med = np.median(eq_data[i])
        polar_med = np.median(polar_data[i])
        x_bracket = positions[i] + 0.30
        ax.annotate("", xy=(x_bracket, polar_med), xytext=(x_bracket, eq_med),
                     arrowprops=dict(arrowstyle="<->", color="0.4", lw=0.8))
        dH = polar_med - eq_med
        ax.text(x_bracket + 0.05, (eq_med + polar_med) / 2,
                f"{dH:+.1f} km", fontsize=6, color="0.3", va="center", ha="left")

    ax.set_xticks(positions)
    ax.set_xticklabels([title for _, title, _ in SCENARIOS], fontsize=7)
    ax.set_ylabel(r"$H_{\rm total}$ (km)", fontsize=9)
    ax.set_ylim(0, None)
    ax.set_title(
        f"Total ice shell thickness by ocean transport scenario\n"
        f"{N_ITER} MC samples, area-weighted band means",
        fontsize=9, fontweight="bold",
    )

    eq_patch = mpatches.Patch(facecolor=PAL.CYAN, alpha=0.6, edgecolor="k",
                               lw=0.5, label=r"Equatorial (0--30$\degree$)")
    polar_patch = mpatches.Patch(facecolor=PAL.ORANGE, alpha=0.6, edgecolor="k",
                                  lw=0.5, label=r"Polar (60--90$\degree$)")
    ax.legend(handles=[eq_patch, polar_patch], loc="upper left", fontsize=7)

    save_fig(fig, "violin_htotal_4scenario", FIGURES_DIR)


# ── Figure 6: Temperature profiles (T_c and Ti) ──────────────────────────

def plot_temperature_profiles():
    """2x2: T_c and Ti median + 1-sigma envelope per scenario."""
    apply_style()

    fig, axes = plt.subplots(
        2, 2,
        figsize=(DOUBLE_COL, DOUBLE_COL * 0.78),
        sharex=True, sharey=True,
    )
    fig.subplots_adjust(hspace=0.30, wspace=0.18)

    for idx, (key, title, sc_color) in enumerate(SCENARIOS):
        ax = axes.flat[idx]
        d = _load(key)
        lat = d["latitudes_deg"]
        Tc = d["T_c_profiles"]
        Ti = d["Ti_profiles"]

        Tc_med = np.median(Tc, axis=0)
        Tc_p16, Tc_p84 = np.percentile(Tc, 16, axis=0), np.percentile(Tc, 84, axis=0)
        Ti_med = np.median(Ti, axis=0)
        Ti_p16, Ti_p84 = np.percentile(Ti, 16, axis=0), np.percentile(Ti, 84, axis=0)

        ax.fill_between(lat, Tc_p16, Tc_p84, color=PAL.RED, alpha=0.15, lw=0)
        ax.plot(lat, Tc_med, color=PAL.RED, lw=1.5, label=r"$T_c$ (convecting)")
        ax.fill_between(lat, Ti_p16, Ti_p84, color=PAL.BLUE, alpha=0.15, lw=0)
        ax.plot(lat, Ti_med, color=PAL.BLUE, lw=1.2, label=r"$T_i$ (interior)")

        # Melting point reference
        ax.axhline(273.15, color="0.5", lw=0.5, ls=":", zorder=0)
        ax.text(88, 273.15 + 1, "273 K", fontsize=5, color="0.5", ha="right", va="bottom")

        add_minor_gridlines(ax)
        ax.set_xlim(0, 90)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(15))

        cite = CITATIONS[key]
        ax.set_title(title, fontsize=8, fontweight="bold", loc="left")
        ax.text(0.01, 0.07, cite, transform=ax.transAxes, fontsize=5.5,
                fontstyle="italic", color="0.45", va="bottom")
        label_panel(ax, chr(97 + idx))

        if idx == 0:
            ax.legend(fontsize=6, loc="upper right")

    for ax in axes[1, :]:
        ax.set_xlabel(r"Latitude ($\degree$)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Temperature (K)")

    save_fig(fig, "temperature_profiles_4scenario", FIGURES_DIR)


# ── Figure 7: Lid fraction vs latitude ────────────────────────────────────

def plot_lid_fraction():
    """Overlay: lid fraction (D_cond/H_total) median + envelope for all scenarios."""
    apply_style()

    fig, ax = plt.subplots(figsize=(DOUBLE_COL, DOUBLE_COL * 0.40))

    for key, title, color in SCENARIOS:
        d = _load(key)
        lat = d["latitudes_deg"]
        lf = d["lid_fraction_profiles"]

        lf_med = np.median(lf, axis=0)
        lf_p16, lf_p84 = np.percentile(lf, 16, axis=0), np.percentile(lf, 84, axis=0)

        ax.fill_between(lat, lf_p16 * 100, lf_p84 * 100, color=color, alpha=0.10, lw=0)
        ax.plot(lat, lf_med * 100, color=color, lw=1.5, label=title)

    ax.axhline(100, color="0.6", lw=0.5, ls=":", zorder=0)
    ax.text(88, 101, "Fully conductive", fontsize=5.5, color="0.5", ha="right", va="bottom")
    ax.axhline(50, color="0.6", lw=0.5, ls=":", zorder=0)

    ax.set_xlim(0, 90)
    ax.set_ylim(0, 110)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(15))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(20))
    ax.set_xlabel(r"Latitude ($\degree$)")
    ax.set_ylabel(r"Lid fraction $D_{\rm cond}/H_{\rm total}$ (%)")
    ax.set_title(f"Conductive lid fraction vs latitude ({N_ITER} MC)",
                  fontsize=9, fontweight="bold")
    ax.legend(fontsize=6, loc="lower right")
    add_minor_gridlines(ax)
    label_panel(ax, "a", x=-0.08)

    save_fig(fig, "lid_fraction_4scenario", FIGURES_DIR)


# ── Figure 8: Juno importance-reweighted posteriors ───────────────────────

SIGMA_EFF = np.sqrt(JUNO_DCOND_ERR**2 + 3.0**2)  # obs + model uncertainty


def _gaussian_lk(dc):
    """Gaussian likelihood for Juno D_cond constraint."""
    return np.exp(-0.5 * ((dc - JUNO_DCOND_KM) / SIGMA_EFF)**2)


def _weighted_median_profile(arr, weights):
    """Weighted median along latitude axis."""
    n_lat = arr.shape[1]
    med = np.zeros(n_lat)
    for j in range(n_lat):
        col = arr[:, j]
        idx = np.argsort(col)
        cw = np.cumsum(weights[idx])
        cw /= cw[-1]
        med[j] = col[idx[np.searchsorted(cw, 0.50)]]
    return med


def _weighted_conv_frac(Nu, weights):
    """Importance-weighted convecting fraction."""
    conv = (Nu > 1.1).astype(float)
    return np.sum(weights[:, None] * conv, axis=0)


def plot_juno_reweighted():
    """4-row x 3-col: prior vs Juno-reweighted for H_total, D_cond, Conv%."""
    apply_style()

    fig, axes = plt.subplots(
        4, 3,
        figsize=(DOUBLE_COL, DOUBLE_COL * 1.05),
        sharex=True,
    )
    fig.subplots_adjust(hspace=0.35, wspace=0.35)

    C_PRIOR = "0.55"
    C_JUNO = PAL.BLUE

    col_titles = [
        r"$H_{\rm total}$ (km)",
        r"$D_{\rm cond}$ (km)",
        "Convecting (%)",
    ]
    panel_idx = 0

    for row, (key, title, _) in enumerate(SCENARIOS):
        d = _load(key)
        lat = d["latitudes_deg"]
        H = d["H_profiles"]
        Dc = d["D_cond_profiles"]
        Nu = d["Nu_profiles"]

        # Juno importance weights
        dc35 = _interp_at(lat, Dc, JUNO_LAT_DEG)
        lk = _gaussian_lk(dc35)
        w = lk / lk.sum()
        n_eff = 1.0 / np.sum(w**2)

        # Col 0: H_total
        ax = axes[row, 0]
        ax.plot(lat, np.median(H, axis=0), color=C_PRIOR, lw=1.0, ls="--")
        ax.plot(lat, _weighted_median_profile(H, w), color=C_JUNO, lw=1.5)
        ax.set_ylim(15, 65)
        if row == 0:
            ax.set_title(col_titles[0], fontsize=7, fontweight="bold")
        ax.text(-0.38, 0.5, title, transform=ax.transAxes, fontsize=6.5,
                fontweight="bold", va="center", ha="center", rotation=90)
        ax.text(0.97, 0.05, f"$N_{{\\rm eff}}$={n_eff:.0f}", transform=ax.transAxes,
                fontsize=5.5, ha="right", va="bottom", color="0.4")
        add_minor_gridlines(ax)
        label_panel(ax, chr(97 + panel_idx))
        panel_idx += 1

        # Col 1: D_cond
        ax = axes[row, 1]
        ax.plot(lat, np.median(Dc, axis=0), color=C_PRIOR, lw=1.0, ls="--")
        ax.plot(lat, _weighted_median_profile(Dc, w), color=C_JUNO, lw=1.5)
        ax.errorbar(JUNO_LAT_DEG, JUNO_DCOND_KM, yerr=JUNO_DCOND_ERR,
                     fmt="D", ms=3, color=PAL.ORANGE, ecolor=PAL.ORANGE,
                     elinewidth=0.6, capsize=1.5, capthick=0.6, zorder=5)
        ax.set_ylim(5, 55)
        if row == 0:
            ax.set_title(col_titles[1], fontsize=7, fontweight="bold")
        add_minor_gridlines(ax)
        label_panel(ax, chr(97 + panel_idx))
        panel_idx += 1

        # Col 2: Conv fraction
        ax = axes[row, 2]
        ax.plot(lat, np.mean(Nu > 1.1, axis=0) * 100, color=C_PRIOR, lw=1.0, ls="--")
        ax.plot(lat, _weighted_conv_frac(Nu, w) * 100, color=C_JUNO, lw=1.5)
        ax.set_ylim(0, 100)
        ax.axhline(50, color="0.7", lw=0.4, ls=":", zorder=0)
        if row == 0:
            ax.set_title(col_titles[2], fontsize=7, fontweight="bold")
        add_minor_gridlines(ax)
        label_panel(ax, chr(97 + panel_idx))
        panel_idx += 1

    for ax in axes[3, :]:
        ax.set_xlabel(r"Latitude ($\degree$)", fontsize=7)
        ax.set_xlim(0, 90)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(30))

    legend_elements = [
        Line2D([0], [0], color=C_PRIOR, lw=1.0, ls="--", label="Prior (flat)"),
        Line2D([0], [0], color=C_JUNO, lw=1.5, label="Juno-reweighted"),
        Line2D([0], [0], marker="D", ms=3, ls="none", color=PAL.ORANGE,
               label=f"Juno $D_{{\\rm cond}}$({JUNO_LAT_DEG:.0f}$\\degree$)"),
    ]
    fig.legend(legend_elements, [e.get_label() for e in legend_elements],
               loc="lower center", ncol=3, fontsize=7,
               bbox_to_anchor=(0.5, -0.01), columnspacing=2.0, handletextpad=0.5)

    save_fig(fig, "juno_reweighted_4scenario", FIGURES_DIR)


# ── Figure 9: Bayesian shrinkage ──────────────────────────────────────────

def plot_bayesian_shrinkage():
    """Bar chart: shift in median D_cond and H_total at key latitudes after Juno reweighting."""
    apply_style()

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, DOUBLE_COL * 0.40))
    fig.subplots_adjust(wspace=0.35)

    lats_of_interest = [5.0, 35.0, 85.0]
    lat_labels = [r"5$\degree$", r"35$\degree$", r"85$\degree$"]
    n_lats = len(lats_of_interest)

    bar_width = 0.18
    scenario_labels = [title for _, title, _ in SCENARIOS]

    for col_idx, (field, ylabel, panel_label) in enumerate([
        ("D_cond_profiles", r"$\Delta D_{\rm cond}$ (km)", "a"),
        ("H_profiles", r"$\Delta H_{\rm total}$ (km)", "b"),
    ]):
        ax = axes[col_idx]
        x_base = np.arange(n_lats)

        for s_idx, (key, title, color) in enumerate(SCENARIOS):
            d = _load(key)
            lat = d["latitudes_deg"]
            arr = d[field]
            Dc = d["D_cond_profiles"]

            dc35 = _interp_at(lat, Dc, JUNO_LAT_DEG)
            lk = _gaussian_lk(dc35)
            w = lk / lk.sum()

            shifts = []
            for target in lats_of_interest:
                vals = _interp_at(lat, arr, target)
                prior_med = np.median(vals)
                idx = np.argsort(vals)
                cw = np.cumsum(w[idx])
                cw /= cw[-1]
                post_med = vals[idx[np.searchsorted(cw, 0.50)]]
                shifts.append(post_med - prior_med)

            x_pos = x_base + (s_idx - 1.5) * bar_width
            bars = ax.bar(x_pos, shifts, bar_width * 0.9, color=color, alpha=0.7,
                          edgecolor="k", linewidth=0.3, label=title if col_idx == 0 else None)

            for bar, s in zip(bars, shifts):
                if abs(s) > 0.3:
                    ax.text(bar.get_x() + bar.get_width() / 2, s,
                            f"{s:+.1f}", ha="center",
                            va="bottom" if s > 0 else "top",
                            fontsize=4.5, fontweight="bold")

        ax.set_xticks(x_base)
        ax.set_xticklabels(lat_labels)
        ax.set_xlabel(r"Latitude")
        ax.set_ylabel(ylabel)
        ax.axhline(0, color="k", lw=0.5)
        add_minor_gridlines(ax)
        label_panel(ax, panel_label, x=-0.12)

    axes[0].legend(fontsize=5.5, loc="best")
    fig.suptitle(f"Juno reweighting shrinkage ({N_ITER} MC)", fontsize=9, fontweight="bold")

    save_fig(fig, "bayesian_shrinkage_4scenario", FIGURES_DIR)


# ── Thesis folder: collect key figures ────────────────────────────────────

THESIS_DIR = os.path.join(_PROJECT_DIR, "figures", "thesis")

THESIS_FIGURES = [
    # (source_name, thesis_name) — renamed for clarity
    ("combined_4scenario_thickness",    "fig_2d_shell_structure"),
    ("combined_4scenario_diagnostics",  "fig_2d_convection_diagnostics"),
    ("violin_dcond_4scenario",          "fig_2d_violin_dcond"),
    ("violin_htotal_4scenario",         "fig_2d_violin_htotal"),
    ("temperature_profiles_4scenario",  "fig_2d_temperature_profiles"),
    ("lid_fraction_4scenario",          "fig_2d_lid_fraction"),
    ("juno_reweighted_4scenario",       "fig_2d_juno_reweighted"),
    ("bayesian_shrinkage_4scenario",    "fig_2d_bayesian_shrinkage"),
    ("scenario_uniform_transport",      "fig_2d_scenario_uniform"),
    ("scenario_soderlund2014_equator",  "fig_2d_scenario_equator_enhanced"),
    ("scenario_lemasquerier2023_polar", "fig_2d_scenario_polar_enhanced"),
    ("scenario_lemasquerier2023_polar_strong", "fig_2d_scenario_polar_strong"),
]


def collect_thesis_figures():
    """Copy key figures to thesis folder with clean names."""
    import shutil
    os.makedirs(THESIS_DIR, exist_ok=True)

    copied = 0
    for src_name, dst_name in THESIS_FIGURES:
        for fmt in ("png", "pdf"):
            src = os.path.join(FIGURES_DIR, f"{src_name}.{fmt}")
            dst = os.path.join(THESIS_DIR, f"{dst_name}.{fmt}")
            if os.path.exists(src):
                shutil.copy2(src, dst)
                copied += 1

    print(f"  Copied {copied} files to {THESIS_DIR}")


# ── Summary table ─────────────────────────────────────────────────────────

def print_summary_table():
    """Print a formatted band-mean summary table for all scenarios."""
    print(f"\n{'-'*92}")
    print(f"  Band-mean summary (N={N_ITER} MC, area-weighted cos(phi))")
    print(f"{'-'*92}")
    header = (
        f"  {'Scenario':<26s}"
        f"  {'H_eq':>6s} {'H_mid':>6s} {'H_pole':>6s}"
        f"  {'Dc_eq':>6s} {'Dc_mid':>6s} {'Dc_pole':>6s}"
        f"  {'Dc(35)':>7s}"
        f"  {'Conv%eq':>7s} {'Conv%po':>7s}"
    )
    print(header)
    print(f"  {'':-<26s}  {'':->6s} {'':->6s} {'':->6s}  {'':->6s} {'':->6s} {'':->6s}  {'':->7s}  {'':->7s} {'':->7s}")

    for key, title, _color in SCENARIOS:
        d = _load(key)
        lat = d["latitudes_deg"]
        H = d["H_profiles"]
        Dc = d["D_cond_profiles"]
        Nu = d["Nu_profiles"]

        h_eq = np.median(band_mean_samples(lat, H, BAND_EQ))
        h_mid = np.median(band_mean_samples(lat, H, BAND_MID))
        h_po = np.median(band_mean_samples(lat, H, BAND_POLAR))
        dc_eq = np.median(band_mean_samples(lat, Dc, BAND_EQ))
        dc_mid = np.median(band_mean_samples(lat, Dc, BAND_MID))
        dc_po = np.median(band_mean_samples(lat, Dc, BAND_POLAR))
        dc_35 = np.median(_interp_at(lat, Dc, JUNO_LAT_DEG))
        conv_eq = np.mean(Nu[:, 0] > 1.1) * 100
        conv_po = np.mean(Nu[:, -1] > 1.1) * 100

        print(
            f"  {title:<26s}"
            f"  {h_eq:6.1f} {h_mid:6.1f} {h_po:6.1f}"
            f"  {dc_eq:6.1f} {dc_mid:6.1f} {dc_po:6.1f}"
            f"  {dc_35:7.1f}"
            f"  {conv_eq:6.0f}% {conv_po:6.0f}%"
        )

    print(f"\n  Juno constraint: D_cond({JUNO_LAT_DEG:.0f} deg) = {JUNO_DCOND_KM} +/- {JUNO_DCOND_ERR} km")
    print(f"  Bands: eq={BAND_EQ}, mid={BAND_MID}, polar={BAND_POLAR}")
    print(f"{'-'*92}\n")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    print(f"Generating {N_ITER} MC figure suite -> {FIGURES_DIR}")

    print_summary_table()

    print("\n[1/9] Per-scenario shell structure...")
    plot_individual_scenarios()

    print("[2/9] Combined 4-scenario thickness...")
    plot_combined_thickness()

    print("[3/9] Combined 4-scenario diagnostics...")
    plot_combined_diagnostics()

    print("[4/9] D_cond violin plot...")
    plot_violin_dcond()

    print("[5/9] H_total violin plot...")
    plot_violin_htotal()

    print("[6/9] Temperature profiles (T_c, Ti)...")
    plot_temperature_profiles()

    print("[7/9] Lid fraction vs latitude...")
    plot_lid_fraction()

    print("[8/9] Juno importance-reweighted posteriors...")
    plot_juno_reweighted()

    print("[9/9] Bayesian shrinkage...")
    plot_bayesian_shrinkage()

    print("\nCollecting thesis figures...")
    collect_thesis_figures()

    print(f"\nDone. All figures saved to {FIGURES_DIR}")
    print(f"Thesis figures copied to {THESIS_DIR}")


if __name__ == "__main__":
    main()
