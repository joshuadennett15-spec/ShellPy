#!/usr/bin/env python3
"""
Publication-quality violin comparison of conductive lid thickness (D_cond)
across ocean transport scenarios and latitude endmembers.

DATA PROVENANCE
===============
Panel (a) — Unconstrained endmember 1D proxies:
    Equator and pole draws from audited 2026 priors with Andrade rheology.
    Files: endmember_{scenario}_{eq,pole}_andrade.npz
    These are forward MC samples with NO observational conditioning.

Panel (b) — Mid-latitude (35 deg) Juno-conditioned posteriors:
    Iterative Bayesian refit against Juno MWR D_cond = 29 +/- 10 km.
    Files: midlat_juno/midlat35_{scenario}_constrained.npz
    These are MC draws from priors tightened via importance reweighting
    against the Juno constraint.  The Juno band in panel (b) is the
    conditioning target, NOT an independent validation reference.

If true mid-latitude endmember NPZs exist (endmember_*_mid_andrade.npz),
they are used instead and the figure collapses to a single panel.

IMPORTANT: The mid-latitude products are Juno-conditioned posteriors,
not raw unconstrained endmember draws.  This mixed provenance is made
explicit through separate panels, distinct titling, and legend labels.

At 35 deg the ocean-transport q_tidal_multiplier is ~1.0 for all three
patterns, so the constrained mid-latitude distributions are expected to
be nearly identical across scenarios.  This is a real physical result,
not an artefact.

Usage:
    python plot_endmember_latitude_dcond.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde
from pub_style import (
    apply_style,
    PAL,
    label_panel,
    save_fig,
    add_minor_gridlines,
)

apply_style()

# ── Paths ────────────────────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
MIDLAT_DIR = os.path.join(RESULTS_DIR, "midlat_juno")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures", "pub")

# ── Juno MWR reference ───────────────────────────────────────────────────────
JUNO_D_OBS = 29.0
JUNO_SIGMA = 10.0

# ── Latitude colours (colorblind-safe, user spec) ────────────────────────────
CLR_EQ = PAL.CYAN  # #56B4E9  — equator
CLR_MID = PAL.GREEN  # #009E73  — mid-latitude
CLR_POLE = PAL.ORANGE  # #E69F00  — pole

# ── Display parameters ───────────────────────────────────────────────────────
VIOLIN_WIDTH = 0.38
JITTER_HALF = 0.11
POINT_ALPHA = 0.13
POINT_SIZE = 1.2
N_DISPLAY = 500  # max jittered points per violin
KDE_BW = 0.18  # KDE bandwidth for violin shape
SEED_DISPLAY = 42

# ── Scenario definitions ─────────────────────────────────────────────────────
SCENARIOS = ["Uniform", "Soderlund", "Lemasquerier"]
SCENARIO_KEYS = ["uniform", "soderlund", "lemasquerier"]

# ── Group layout ─────────────────────────────────────────────────────────────
GROUP_CENTERS = [0.0, 1.8, 3.6]
LAT_OFFSET = 0.27  # +/- offset for eq/pole within a group


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════


def load_dcond(path):
    """Load D_cond_km from an NPZ file; return array or None."""
    if not os.path.exists(path):
        print(f"  MISSING: {os.path.relpath(path)}")
        return None
    data = np.load(path)
    arr = data["D_cond_km"]
    valid = arr[np.isfinite(arr)]
    print(
        f"  Loaded : {os.path.basename(path):45s}  "
        f"n={len(valid):5d}  median={np.median(valid):5.1f} km"
    )
    return valid


def _kde_violin_coords(data, bw=KDE_BW, n_grid=300):
    """Return (y_grid, density) for a filled violin shape."""
    lo = max(0, data.min() - 3)
    hi = data.max() + 3
    y_grid = np.linspace(lo, hi, n_grid)
    kde = gaussian_kde(data, bw_method=bw)
    density = kde(y_grid)
    return y_grid, density


def draw_violin(ax, data, pos, colour, width=VIOLIN_WIDTH, alpha_fill=0.25):
    """Render a single filled violin centred on *pos*."""
    if len(data) < 30:
        return
    y, dens = _kde_violin_coords(data)
    dens_norm = dens / dens.max() * (width / 2)
    ax.fill_betweenx(
        y,
        pos - dens_norm,
        pos + dens_norm,
        facecolor=colour,
        alpha=alpha_fill,
        edgecolor=colour,
        linewidth=0.55,
        zorder=2,
    )


def draw_points(ax, data, pos, colour, rng):
    """Overlay subsampled jittered raw points."""
    n = len(data)
    subset = data if n <= N_DISPLAY else data[rng.choice(n, N_DISPLAY, replace=False)]
    x_jit = rng.uniform(-JITTER_HALF, JITTER_HALF, size=len(subset))
    ax.scatter(
        pos + x_jit,
        subset,
        s=POINT_SIZE,
        c=colour,
        alpha=POINT_ALPHA,
        edgecolors="none",
        rasterized=True,
        zorder=3,
    )


def draw_stats(ax, data, pos, colour, annotate_median=True):
    """Draw median bar, IQR bar, 16th-84th whisker, and optional label."""
    med = float(np.median(data))
    q25 = float(np.percentile(data, 25))
    q75 = float(np.percentile(data, 75))
    p16 = float(np.percentile(data, 15.87))
    p84 = float(np.percentile(data, 84.13))

    hw = VIOLIN_WIDTH * 0.32

    # 16-84 whisker
    ax.plot([pos, pos], [p16, p84], color=colour, lw=0.7, alpha=0.45, zorder=4)
    # IQR bar
    ax.plot([pos, pos], [q25, q75], color=colour, lw=1.6, alpha=0.70, zorder=5)
    # Median
    ax.plot(
        [pos - hw, pos + hw],
        [med, med],
        color=colour,
        lw=2.2,
        solid_capstyle="round",
        zorder=6,
    )

    if annotate_median:
        ax.annotate(
            f"{med:.0f}",
            xy=(pos + hw + 0.02, med),
            xytext=(3, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=5.5,
            color=colour,
            fontweight="bold",
            zorder=7,
        )
    return med


def add_juno_band(ax, is_conditioning_target=False):
    """Shade the Juno MWR D_cond reference region."""
    lo = JUNO_D_OBS - JUNO_SIGMA
    hi = JUNO_D_OBS + JUNO_SIGMA
    shade_alpha = 0.18
    ax.axhspan(lo, hi, color="0.72", alpha=shade_alpha, zorder=0)
    ax.axhline(JUNO_D_OBS, color="0.45", ls="--", lw=0.8, zorder=1)


def _juno_label_text(is_target):
    if is_target:
        return f"Juno conditioning\ntarget {JUNO_D_OBS} \u00b1 {JUNO_SIGMA:.0f} km"
    return f"Juno MWR ref.\n{JUNO_D_OBS} \u00b1 {JUNO_SIGMA:.0f} km"


def add_group_separators(ax, centers):
    """Vertical dotted lines between scenario groups."""
    for i in range(len(centers) - 1):
        mid = (centers[i] + centers[i + 1]) / 2
        ax.axvline(mid, color="0.87", ls=":", lw=0.5, zorder=0)


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════


def main():
    print("=" * 60)
    print("  D_cond latitude violin figure")
    print("=" * 60)

    rng = np.random.default_rng(SEED_DISPLAY)

    # ── Load unconstrained endmember data ────────────────────────────────
    print("\nUnconstrained endmember files:")
    eq_data = {}
    pole_data = {}
    for key in SCENARIO_KEYS:
        eq_data[key] = load_dcond(
            os.path.join(RESULTS_DIR, f"endmember_{key}_eq_andrade.npz")
        )
        pole_data[key] = load_dcond(
            os.path.join(RESULTS_DIR, f"endmember_{key}_pole_andrade.npz")
        )

    # ── Try true mid endmember files first, fall back to Juno-constrained ─
    print("\nMid-latitude files:")
    mid_data = {}
    mid_source = "endmember"
    for key in SCENARIO_KEYS:
        mid_data[key] = load_dcond(
            os.path.join(RESULTS_DIR, f"endmember_{key}_mid_andrade.npz")
        )

    if not any(v is not None for v in mid_data.values()):
        print("  -> No mid endmember files.  Trying midlat_juno constrained:")
        mid_source = "juno_constrained"
        for key in SCENARIO_KEYS:
            mid_data[key] = load_dcond(
                os.path.join(MIDLAT_DIR, f"midlat35_{key}_constrained.npz")
            )

    has_mid = any(v is not None for v in mid_data.values())
    two_panel = mid_source == "juno_constrained" and has_mid

    print(f"\nMid-lat source : {mid_source}")
    print(f"Layout         : {'two-panel (a+b)' if two_panel else 'single panel'}")

    # ── Figure scaffold ──────────────────────────────────────────────────
    if two_panel:
        fig = plt.figure(figsize=(7.20, 5.40))
        gs = gridspec.GridSpec(
            2, 1, height_ratios=[1.8, 1.0], hspace=0.38, figure=fig
        )
        ax_a = fig.add_subplot(gs[0])
        ax_b = fig.add_subplot(gs[1])
    else:
        fig, ax_a = plt.subplots(1, 1, figsize=(7.20, 4.0))
        ax_b = None

    x_lo = GROUP_CENTERS[0] - 0.70
    x_hi = GROUP_CENTERS[-1] + 0.70
    y_hi = 72

    # ══════════════════════════════════════════════════════════════════════
    # Panel (a): Unconstrained Eq + Pole
    # ══════════════════════════════════════════════════════════════════════
    medians_eq = []
    medians_pole = []

    for i, key in enumerate(SCENARIO_KEYS):
        cx = GROUP_CENTERS[i]

        # Equator (left of centre)
        if eq_data[key] is not None:
            pos = cx - LAT_OFFSET
            draw_violin(ax_a, eq_data[key], pos, CLR_EQ)
            draw_points(ax_a, eq_data[key], pos, CLR_EQ, rng)
            m = draw_stats(ax_a, eq_data[key], pos, CLR_EQ)
            medians_eq.append(m)
        else:
            medians_eq.append(None)

        # Pole (right of centre)
        if pole_data[key] is not None:
            pos = cx + LAT_OFFSET
            draw_violin(ax_a, pole_data[key], pos, CLR_POLE)
            draw_points(ax_a, pole_data[key], pos, CLR_POLE, rng)
            m = draw_stats(ax_a, pole_data[key], pos, CLR_POLE)
            medians_pole.append(m)
        else:
            medians_pole.append(None)

    # Delta annotations (Pole - Eq)
    for i, cx in enumerate(GROUP_CENTERS):
        meq = medians_eq[i]
        mpo = medians_pole[i]
        if meq is not None and mpo is not None:
            delta = mpo - meq
            sign = "+" if delta >= 0 else "\u2212"
            y_ann = max(
                np.percentile(pole_data[SCENARIO_KEYS[i]], 90),
                np.percentile(eq_data[SCENARIO_KEYS[i]], 90),
            ) + 3
            ax_a.text(
                cx,
                min(y_ann, y_hi - 4),
                f"$\\Delta$={sign}{abs(delta):.0f}",
                ha="center",
                va="bottom",
                fontsize=5.5,
                color="0.40",
                fontstyle="italic",
            )

    add_juno_band(ax_a, is_conditioning_target=False)
    add_group_separators(ax_a, GROUP_CENTERS)

    ax_a.set_xticks(GROUP_CENTERS)
    ax_a.set_xticklabels(SCENARIOS, fontsize=8)
    ax_a.set_ylabel(r"Conductive lid thickness $D_\mathrm{cond}$ (km)")
    ax_a.set_xlim(x_lo, x_hi)
    ax_a.set_ylim(0, y_hi)
    add_minor_gridlines(ax_a, axis="y")

    # Juno text label
    ax_a.text(
        x_hi - 0.05,
        JUNO_D_OBS + 0.8,
        _juno_label_text(False),
        ha="right",
        va="bottom",
        fontsize=5.5,
        color="0.45",
        fontstyle="italic",
    )

    # Legend
    handles_a = [
        Patch(facecolor=CLR_EQ, alpha=0.30, edgecolor=CLR_EQ, linewidth=0.6,
              label="Equator (110 K, unconstrained)"),
        Patch(facecolor=CLR_POLE, alpha=0.30, edgecolor=CLR_POLE, linewidth=0.6,
              label="Pole (50 K, unconstrained)"),
    ]
    ax_a.legend(
        handles=handles_a,
        loc="upper left",
        fontsize=6.5,
        handlelength=1.2,
        borderpad=0.4,
    )

    label_panel(ax_a, "a")
    ax_a.set_title("Unconstrained endmember 1D proxies", fontsize=9, pad=8)

    # ══════════════════════════════════════════════════════════════════════
    # Panel (b): Juno-conditioned mid-latitude (35 deg)
    # ══════════════════════════════════════════════════════════════════════
    if two_panel and ax_b is not None:
        for i, key in enumerate(SCENARIO_KEYS):
            cx = GROUP_CENTERS[i]
            if mid_data[key] is not None:
                draw_violin(ax_b, mid_data[key], cx, CLR_MID, alpha_fill=0.28)
                draw_points(ax_b, mid_data[key], cx, CLR_MID, rng)
                draw_stats(ax_b, mid_data[key], cx, CLR_MID)

        add_juno_band(ax_b, is_conditioning_target=True)
        add_group_separators(ax_b, GROUP_CENTERS)

        ax_b.set_xticks(GROUP_CENTERS)
        ax_b.set_xticklabels(SCENARIOS, fontsize=8)
        ax_b.set_ylabel(r"$D_\mathrm{cond}$ (km)")
        ax_b.set_xlim(x_lo, x_hi)
        ax_b.set_ylim(0, 58)
        add_minor_gridlines(ax_b, axis="y")

        # Juno target label
        ax_b.text(
            x_hi - 0.05,
            JUNO_D_OBS + 0.8,
            _juno_label_text(True),
            ha="right",
            va="bottom",
            fontsize=5.5,
            color="0.45",
            fontstyle="italic",
        )

        handles_b = [
            Patch(
                facecolor=CLR_MID,
                alpha=0.30,
                edgecolor=CLR_MID,
                linewidth=0.6,
                label=r"Mid-lat 35° posterior ($D_\mathrm{cond}$"
                f" = {JUNO_D_OBS} \u00b1 {JUNO_SIGMA:.0f} km target)",
            ),
        ]
        ax_b.legend(
            handles=handles_b,
            loc="upper left",
            fontsize=6.5,
            handlelength=1.2,
            borderpad=0.4,
        )

        label_panel(ax_b, "b")
        ax_b.set_title(
            r"Mid-latitude 35° posteriors conditioned on Juno MWR"
            f" ({JUNO_D_OBS} \u00b1 {JUNO_SIGMA:.0f} km)",
            fontsize=8,
            pad=8,
        )

    # ── Save ─────────────────────────────────────────────────────────────
    save_fig(fig, "fig_endmember_latitude_dcond", FIGURES_DIR)

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  Figure saved to figures/pub/fig_endmember_latitude_dcond.{{png,pdf}}")
    print(f"  Mid-latitude source: {mid_source}")
    if mid_source == "juno_constrained":
        print(
            "  NOTE: Mid-lat distributions are Juno-conditioned posteriors.\n"
            "        The Juno band in panel (b) is the conditioning target,\n"
            "        not an independent validation reference."
        )
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
