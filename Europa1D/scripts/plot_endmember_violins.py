#!/usr/bin/env python3
"""
Violin plots for the endmember proxy suite.

Generates three publication-quality figures:
  fig_violin_endmember_thickness: Total ice shell thickness by scenario
  fig_violin_endmember_structure: D_cond and D_conv side-by-side
  fig_violin_endmember_regimes:   Regime-split (conv/cond) per scenario

Usage:
    python plot_endmember_violins.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.patches import Patch
from pub_style import (
    apply_style, PAL,
    figsize_double, figsize_double_tall,
    label_panel, save_fig, add_minor_gridlines,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures", "pub")

apply_style()

# ── Scenario definitions ─────────────────────────────────────────────────────
SCENARIOS = [
    ("Uniform\nEq",    "endmember_uniform_eq_andrade",           PAL.GREEN),
    ("Uniform\nMid",   "endmember_uniform_mid_andrade",          PAL.GREEN),
    ("Uniform\nPole",  "endmember_uniform_pole_andrade",         PAL.GREEN),
    ("Soderlund\nEq",    "endmember_soderlund_eq_andrade",       PAL.ORANGE),
    ("Soderlund\nMid",   "endmember_soderlund_mid_andrade",      PAL.ORANGE),
    ("Soderlund\nPole",  "endmember_soderlund_pole_andrade",     PAL.ORANGE),
    ("Lemasq.\nEq",    "endmember_lemasquerier_eq_andrade",      PAL.BLUE),
    ("Lemasq.\nMid",   "endmember_lemasquerier_mid_andrade",     PAL.BLUE),
    ("Lemasq.\nPole",  "endmember_lemasquerier_pole_andrade",    PAL.BLUE),
]

RA_CRIT = 1000.0


def _load(name):
    """Load endmember NPZ, return dict of arrays."""
    path = os.path.join(RESULTS_DIR, f"{name}.npz")
    data = np.load(path)
    H = data["thicknesses_km"]
    D_cond = data["D_cond_km"]
    D_conv = data["D_conv_km"]
    Ra = data["Ra_values"] if "Ra_values" in data else np.zeros(len(H))
    return {"H": H, "D_cond": D_cond, "D_conv": D_conv, "Ra": Ra}


# ═════════════════════════════════════════════════════════════════════════════
# Figure A: Total thickness violins
# ═════════════════════════════════════════════════════════════════════════════

def fig_thickness_violins():
    """Violin plot of total ice shell thickness for each endmember."""
    print("Violin plot: Total thickness")

    fig, ax = plt.subplots(1, 1, figsize=figsize_double(0.50))

    all_data = []
    labels = []
    colours = []
    for label, name, colour in SCENARIOS:
        d = _load(name)
        all_data.append(d["H"])
        labels.append(label)
        colours.append(colour)

    positions = np.arange(len(all_data))
    parts = ax.violinplot(
        all_data, positions=positions, showmedians=True,
        showextrema=False, widths=0.75,
    )

    # Colour each violin by its scenario group
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colours[i])
        pc.set_edgecolor(colours[i])
        pc.set_alpha(0.4)
        pc.set_linewidth(0.5)
    parts["cmedians"].set_edgecolor(PAL.BLACK)
    parts["cmedians"].set_linewidth(1.0)

    # Overlay CBE diamond + 1-sigma whisker
    for i, (H, colour) in enumerate(zip(all_data, colours)):
        kde = gaussian_kde(H)
        x_grid = np.linspace(H.min(), H.max(), 300)
        cbe = float(x_grid[np.argmax(kde(x_grid))])
        p16 = float(np.percentile(H, 15.87))
        p84 = float(np.percentile(H, 84.13))

        ax.scatter(i, cbe, marker="D", s=18, color=colour,
                   edgecolors="k", linewidths=0.4, zorder=5)
        ax.plot([i, i], [p16, p84], color=colour, lw=1.5, alpha=0.7, zorder=4)

    # Formatting
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=6.5)
    ax.set_ylabel("Ice shell thickness (km)")
    ax.set_ylim(0, 120)
    add_minor_gridlines(ax, axis="y")

    # Group separators
    for sep in [2.5, 5.5]:
        ax.axvline(sep, color="0.8", ls=":", lw=0.5, zorder=0)

    # Group labels at top
    for x, lbl in [(1.0, "Uniform"), (4.0, "Soderlund"), (7.0, "Lemasquerier")]:
        ax.text(x, 117, lbl, ha="center", va="top", fontsize=7.5,
                fontstyle="italic", color="0.4")

    fig.suptitle("Endmember ice shell thickness distributions", fontsize=9, y=1.01)
    fig.tight_layout()
    save_fig(fig, "fig_violin_endmember_thickness", FIGURES_DIR)


# ═════════════════════════════════════════════════════════════════════════════
# Figure B: D_cond + D_conv paired violins
# ═════════════════════════════════════════════════════════════════════════════

def fig_structure_violins():
    """Paired violin plot: D_cond (left half) and D_conv (right half)."""
    print("Violin plot: Shell structure (D_cond / D_conv)")

    fig, ax = plt.subplots(1, 1, figsize=figsize_double(0.55))

    positions = np.arange(len(SCENARIOS))

    for i, (label, name, colour) in enumerate(SCENARIOS):
        d = _load(name)
        D_cond = d["D_cond"]
        D_conv = d["D_conv"]
        D_conv_active = D_conv[D_conv > 0.5]

        # Left half-violin: D_cond
        vp_cond = ax.violinplot(
            [D_cond], positions=[i], showmedians=True,
            showextrema=False, widths=0.8,
        )
        for pc in vp_cond["bodies"]:
            m = np.mean(pc.get_paths()[0].vertices[:, 0])
            pc.get_paths()[0].vertices[:, 0] = np.clip(
                pc.get_paths()[0].vertices[:, 0], -np.inf, m
            )
            pc.set_facecolor(PAL.COND)
            pc.set_edgecolor(PAL.COND)
            pc.set_alpha(0.4)
            pc.set_linewidth(0.5)
        vp_cond["cmedians"].set_edgecolor(PAL.COND)
        vp_cond["cmedians"].set_linewidth(0.8)

        # Right half-violin: D_conv (active only)
        if len(D_conv_active) > 10:
            vp_conv = ax.violinplot(
                [D_conv_active], positions=[i], showmedians=True,
                showextrema=False, widths=0.8,
            )
            for pc in vp_conv["bodies"]:
                m = np.mean(pc.get_paths()[0].vertices[:, 0])
                pc.get_paths()[0].vertices[:, 0] = np.clip(
                    pc.get_paths()[0].vertices[:, 0], m, np.inf
                )
                pc.set_facecolor(PAL.CONV)
                pc.set_edgecolor(PAL.CONV)
                pc.set_alpha(0.4)
                pc.set_linewidth(0.5)
            vp_conv["cmedians"].set_edgecolor(PAL.CONV)
            vp_conv["cmedians"].set_linewidth(0.8)

    # Formatting
    labels = [s[0] for s in SCENARIOS]
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=6.5)
    ax.set_ylabel("Layer thickness (km)")
    ax.set_ylim(0, 100)
    add_minor_gridlines(ax, axis="y")

    # Group separators
    for sep in [2.5, 5.5]:
        ax.axvline(sep, color="0.8", ls=":", lw=0.5, zorder=0)

    # Legend
    legend_elements = [
        Patch(facecolor=PAL.COND, alpha=0.4, edgecolor=PAL.COND,
              label=r"$D_\mathrm{cond}$ (lid)"),
        Patch(facecolor=PAL.CONV, alpha=0.4, edgecolor=PAL.CONV,
              label=r"$D_\mathrm{conv}$ (sublayer)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=7)

    fig.suptitle("Endmember shell structure: conductive lid vs convective sublayer",
                 fontsize=9, y=1.01)
    fig.tight_layout()
    save_fig(fig, "fig_violin_endmember_structure", FIGURES_DIR)


# ═════════════════════════════════════════════════════════════════════════════
# Figure C: Convective fraction bar + violin combo
# ═════════════════════════════════════════════════════════════════════════════

def fig_regime_violins():
    """Violins split by convective vs conductive regime per scenario."""
    print("Violin plot: Regime-split thickness")

    fig, axes = plt.subplots(3, 3, figsize=figsize_double_tall(1.05),
                             sharey=True)

    for i, (label, name, colour) in enumerate(SCENARIOS):
        row, col = divmod(i, 3)
        ax = axes[row, col]
        d = _load(name)
        H = d["H"]
        Ra = d["Ra"]

        conv_mask = Ra >= RA_CRIT
        H_conv = H[conv_mask]
        H_cond = H[~conv_mask]

        data = []
        tick_labels = []
        tick_colours = []

        if len(H_cond) > 10:
            data.append(H_cond)
            frac = len(H_cond) / len(H) * 100
            tick_labels.append(f"Cond\n({frac:.0f}%)")
            tick_colours.append(PAL.COND)
        if len(H_conv) > 10:
            data.append(H_conv)
            frac = len(H_conv) / len(H) * 100
            tick_labels.append(f"Conv\n({frac:.0f}%)")
            tick_colours.append(PAL.CONV)

        if data:
            vp = ax.violinplot(
                data, positions=range(len(data)),
                showmedians=True, showextrema=False, widths=0.7,
            )
            for j, pc in enumerate(vp["bodies"]):
                pc.set_facecolor(tick_colours[j])
                pc.set_edgecolor(tick_colours[j])
                pc.set_alpha(0.45)
                pc.set_linewidth(0.5)
            vp["cmedians"].set_edgecolor(PAL.BLACK)
            vp["cmedians"].set_linewidth(0.8)

            ax.set_xticks(range(len(data)))
            ax.set_xticklabels(tick_labels, fontsize=6)

        clean_label = label.replace("\n", " ")
        ax.set_title(clean_label, fontsize=8)
        ax.set_ylim(0, 110)
        if col == 0:
            ax.set_ylabel("Ice shell thickness (km)")
        add_minor_gridlines(ax, axis="y")
        label_panel(ax, chr(ord("a") + i))

    fig.suptitle("Regime-split thickness distributions", fontsize=9, y=1.01)
    fig.tight_layout(h_pad=2.0, w_pad=1.5)
    save_fig(fig, "fig_violin_endmember_regimes", FIGURES_DIR)


if __name__ == "__main__":
    fig_thickness_violins()
    fig_structure_violins()
    fig_regime_violins()
