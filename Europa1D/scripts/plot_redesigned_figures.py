#!/usr/bin/env python3
"""
Generate three redesigned publication figures for the Europa ice shell thesis.

Figures:
  1. fig_sobol_lollipop.pdf     -- Sobol sensitivity lollipop ranking (3-panel)
  2. fig_bayesian_2panel.pdf    -- Prior/posterior D_cond & H_total with Juno reweighting
  3. fig_shrinkage_dumbbells.pdf -- Parameter shrinkage dumbbells from Bayesian update

Usage:
    python plot_redesigned_figures.py
"""
from __future__ import annotations

import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

# ── Resolve project paths ──────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, os.pardir))
EUROPA_1D = os.path.join(PROJECT_ROOT, "Europa1D")
FIGURES_DIR = os.path.join(EUROPA_1D, "figures", "pub")

# Add Europa2D/scripts to path for pub_style import
sys.path.insert(0, os.path.join(PROJECT_ROOT, "Europa2D", "scripts"))
from pub_style import (
    DOUBLE_COL,
    SINGLE_COL,
    PAL,
    add_minor_gridlines,
    apply_style,
    label_panel,
    save_fig,
)

apply_style()

# ── Nice parameter labels ──────────────────────────────────────────────────
PARAM_LABELS = {
    "q_basal_target_mW_m2": r"$q_\mathrm{basal}$",
    "d_grain_mm": r"$d_\mathrm{grain}$",
    "Q_v_kJ_mol": r"$Q_v$",
    "epsilon_0": r"$\varepsilon_0$",
    "T_surf_K": r"$T_\mathrm{surf}$",
    "mu_ice_GPa": r"$\mu_\mathrm{ice}$",
    "D_H2O_km": r"$D_\mathrm{H_2O}$",
    "Q_b_kJ_mol": r"$Q_b$",
    "H_rad_pW_kg": r"$H_\mathrm{rad}$",
    "f_porosity": r"$f_\mathrm{porosity}$",
}

OUTPUT_TITLES = {
    "thickness_km": r"$H_\mathrm{total}$",
    "D_cond_km": r"$D_\mathrm{cond}$",
    "D_conv_km": r"$D_\mathrm{conv}$",
    "lid_fraction": "Lid fraction",
    "convective_flag": "Convective fraction",
    "Nu": r"Nusselt number $\mathrm{Nu}$",
}


# ═════════════════════════════════════════════════════════════════════════════
# Figure 1 -- Sobol lollipop ranking
# ═════════════════════════════════════════════════════════════════════════════
def fig_sobol_lollipop() -> None:
    """Dot-and-whisker (lollipop) Sobol sensitivity for 3 outputs."""
    csv_path = os.path.join(
        EUROPA_DJ,
        "results", "sobol",
        "global_audited_sobol_params_N512",
        "global_audited_sobol_params_N512_indices.csv",
    )
    df = pd.read_csv(csv_path)

    # Filter: N=512, main indices only
    mask = (df["sample_size"] == 512) & (df["index_type"] == "main")
    df_main = df[mask].copy()

    outputs = ["thickness_km", "D_conv_km", "lid_fraction", "convective_flag"]
    fig, axes = plt.subplots(
        1, 4,
        figsize=(DOUBLE_COL, DOUBLE_COL * 0.32),
        sharey=False,
    )

    for ax, output, letter in zip(axes, outputs, "abcd"):
        sub = df_main[df_main["output"] == output].copy()

        # Filter to ST > 0.02
        sub = sub[sub["ST"] > 0.02].copy()

        # Sort by ST descending (bottom = highest for horizontal lollipop)
        sub = sub.sort_values("ST", ascending=True).reset_index(drop=True)

        y_pos = np.arange(len(sub))
        labels = [PARAM_LABELS.get(f, f) for f in sub["factor"]]

        # Lollipop: thin lines from 0 to S1
        ax.hlines(y_pos, 0, sub["S1"].values, color=PAL.BLUE, linewidth=0.9)
        # S1 filled dots
        ax.plot(
            sub["S1"].values, y_pos, "o",
            color=PAL.BLUE, markersize=5, zorder=5,
            label=r"First-order ($S_1$)",
        )
        # ST open circles
        ax.plot(
            sub["ST"].values, y_pos, "o",
            markerfacecolor="white", markeredgecolor=PAL.RED,
            markeredgewidth=1.2, markersize=5.5, zorder=5,
            label=r"Total-order ($S_T$)",
        )
        # Confidence whiskers for S1
        ax.errorbar(
            sub["S1"].values, y_pos,
            xerr=sub["S1_conf"].values,
            fmt="none", ecolor=PAL.BLUE, elinewidth=0.6, capsize=2, capthick=0.5,
        )
        # Confidence whiskers for ST
        ax.errorbar(
            sub["ST"].values, y_pos,
            xerr=sub["ST_conf"].values,
            fmt="none", ecolor=PAL.RED, elinewidth=0.6, capsize=2, capthick=0.5,
        )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.set_xlim(-0.02, 1.0)
        ax.set_xlabel("Sensitivity index")
        ax.set_title(OUTPUT_TITLES[output])
        ax.axvline(0, color="0.6", linewidth=0.4, zorder=0)
        label_panel(ax, letter)

    # Legend on rightmost panel
    axes[-1].legend(loc="lower right", fontsize=6.5)

    fig.tight_layout(w_pad=1.8)
    save_fig(fig, "fig_sobol_lollipop", FIGURES_DIR)


# ═════════════════════════════════════════════════════════════════════════════
# Bayesian helpers (shared by Fig 2 and Fig 3)
# ═════════════════════════════════════════════════════════════════════════════
def _load_andrade_and_weights():
    """Load the 5000-sample Andrade archive and compute Juno importance weights."""
    npz_path = os.path.join(
        EUROPA_DJ, "results", "mc_15000_optionA_v2_andrade.npz",
    )
    data = np.load(npz_path, allow_pickle=True)

    d_cond = data["D_cond_km"]
    h_total = data["thicknesses_km"]

    # Juno importance reweighting
    d_cond_obs = 29.0  # km
    sigma_meas = 10.0  # km  (measurement)
    sigma_model = 3.0  # km  (model spread)
    sigma_eff = np.sqrt(sigma_meas**2 + sigma_model**2)

    log_w = -0.5 * ((d_cond - d_cond_obs) / sigma_eff) ** 2
    log_w -= log_w.max()  # numerical stability
    weights = np.exp(log_w)
    weights /= weights.sum()

    return data, d_cond, h_total, weights, sigma_eff


def _weighted_quantile(values, weights, q):
    """Compute weighted quantile (linear interpolation)."""
    idx = np.argsort(values)
    sv = values[idx]
    sw = weights[idx]
    cum = np.cumsum(sw)
    cum /= cum[-1]
    return np.interp(q, cum, sv)


def _weighted_std(values, weights):
    """Weighted standard deviation."""
    mean = np.average(values, weights=weights)
    var = np.average((values - mean) ** 2, weights=weights)
    return np.sqrt(var)


# ═════════════════════════════════════════════════════════════════════════════
# Figure 2 -- Bayesian predictive 2-panel
# ═════════════════════════════════════════════════════════════════════════════
def fig_bayesian_2panel() -> None:
    """Publication-quality Bayesian update figure (2-panel).

    (a) D_cond prior KDE, Juno likelihood, posterior KDE (the textbook
        prior x likelihood -> posterior picture).
    (b) H_total downstream prior vs posterior KDE.
    Inset diagnostics in panel (a) report N_eff and D_KL.
    """
    _data, d_cond, h_total, weights, sigma_eff = _load_andrade_and_weights()

    n = int(d_cond.size)
    n_eff = float(1.0 / np.sum(weights**2))
    n_eff_frac = n_eff / n
    log_term = np.where(weights > 0, np.log(weights * n + 1e-300), 0.0)
    kl_nats = float(np.sum(weights * log_term))

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2,
        figsize=(DOUBLE_COL, DOUBLE_COL * 0.42),
        gridspec_kw=dict(wspace=0.22),
    )

    # ── Panel (a): D_cond Bayesian update ──────────────────────────────
    x_d = np.linspace(0.0, 40.0, 600)
    kde_prior_d = gaussian_kde(d_cond, bw_method=0.18)
    kde_post_d = gaussian_kde(d_cond, weights=weights, bw_method=0.18)
    p_prior_d = kde_prior_d(x_d)
    p_post_d = kde_post_d(x_d)
    lik_pdf = np.exp(-0.5 * ((x_d - 29.0) / sigma_eff) ** 2) / (
        sigma_eff * np.sqrt(2 * np.pi)
    )

    ax_a.axvspan(29.0 - 10.0, 29.0 + 10.0, color=PAL.ORANGE, alpha=0.10, lw=0)
    ax_a.fill_between(x_d, p_prior_d, color="0.55", alpha=0.30, lw=0,
                      label="Prior")
    ax_a.plot(x_d, p_prior_d, color="0.30", lw=0.9)
    ax_a.plot(x_d, lik_pdf, color=PAL.ORANGE, lw=1.4, ls="--",
              label="Juno likelihood")
    ax_a.fill_between(x_d, p_post_d, color=PAL.BLUE, alpha=0.32, lw=0,
                      label="Posterior")
    ax_a.plot(x_d, p_post_d, color=PAL.BLUE, lw=1.4)

    prior_med_d = float(np.median(d_cond))
    post_med_d = float(_weighted_quantile(d_cond, weights, 0.5))
    y_top = max(p_prior_d.max(), p_post_d.max(), lik_pdf.max())
    ax_a.plot([prior_med_d], [0], marker="^", color="0.30",
              markersize=5, clip_on=False, zorder=6)
    ax_a.plot([post_med_d], [0], marker="^", color=PAL.BLUE,
              markersize=5, clip_on=False, zorder=6)
    y_arrow_a = y_top * 0.30
    ax_a.annotate(
        "", xy=(post_med_d, y_arrow_a),
        xytext=(prior_med_d, y_arrow_a),
        arrowprops=dict(arrowstyle="->", color="0.20", lw=1.0),
    )
    ax_a.text(
        0.5 * (prior_med_d + post_med_d), y_arrow_a + y_top * 0.04,
        fr"$+{post_med_d - prior_med_d:.1f}$ km",
        ha="center", va="bottom", fontsize=6.5, color="0.15",
    )

    ax_a.set_xlabel(r"$D_\mathrm{cond}$ (km)")
    ax_a.set_ylabel("Probability density")
    ax_a.set_xlim(0.0, 40.0)
    ax_a.set_ylim(bottom=0)
    ax_a.legend(loc="upper right", fontsize=6.5, frameon=False,
                handlelength=1.6, labelspacing=0.35)
    add_minor_gridlines(ax_a, axis="y")

    diag_text = (
        f"$N$ = {n:,}\n"
        f"$N_\\mathrm{{eff}}$ = {n_eff:,.0f} ({100*n_eff_frac:.1f}\\%)\n"
        f"$D_\\mathrm{{KL}}$ = {kl_nats:.3f} nats"
    )
    ax_a.text(
        0.025, 0.97, diag_text, transform=ax_a.transAxes,
        fontsize=6.3, va="top", ha="left",
        bbox=dict(facecolor="white", edgecolor="0.75",
                  alpha=0.95, boxstyle="round,pad=0.32"),
    )
    label_panel(ax_a, "a")

    # ── Panel (b): H_total downstream ──────────────────────────────────
    x_h = np.linspace(0.0, 80.0, 600)
    kde_prior_h = gaussian_kde(h_total, bw_method=0.18)
    kde_post_h = gaussian_kde(h_total, weights=weights, bw_method=0.18)
    p_prior_h = kde_prior_h(x_h)
    p_post_h = kde_post_h(x_h)

    ax_b.fill_between(x_h, p_prior_h, color="0.55", alpha=0.30, lw=0,
                      label="Prior")
    ax_b.plot(x_h, p_prior_h, color="0.30", lw=0.9)
    ax_b.fill_between(x_h, p_post_h, color=PAL.BLUE, alpha=0.32, lw=0,
                      label="Posterior")
    ax_b.plot(x_h, p_post_h, color=PAL.BLUE, lw=1.4)

    prior_med_h = float(np.median(h_total))
    post_med_h = float(_weighted_quantile(h_total, weights, 0.5))
    y_top_h = max(p_prior_h.max(), p_post_h.max())
    ax_b.plot([prior_med_h], [0], marker="^", color="0.30",
              markersize=5, clip_on=False, zorder=6)
    ax_b.plot([post_med_h], [0], marker="^", color=PAL.BLUE,
              markersize=5, clip_on=False, zorder=6)
    y_arrow_b = y_top_h * 0.30
    ax_b.annotate(
        "", xy=(post_med_h, y_arrow_b),
        xytext=(prior_med_h, y_arrow_b),
        arrowprops=dict(arrowstyle="->", color="0.20", lw=1.0),
    )
    ax_b.text(
        0.5 * (prior_med_h + post_med_h), y_arrow_b + y_top_h * 0.04,
        fr"$+{post_med_h - prior_med_h:.1f}$ km",
        ha="center", va="bottom", fontsize=6.5, color="0.15",
    )

    ax_b.set_xlabel(r"$H_\mathrm{total}$ (km)")
    ax_b.set_ylabel("Probability density")
    ax_b.set_xlim(0.0, 80.0)
    ax_b.set_ylim(bottom=0)
    ax_b.legend(loc="upper right", fontsize=6.5, frameon=False,
                handlelength=1.6, labelspacing=0.35)
    add_minor_gridlines(ax_b, axis="y")
    label_panel(ax_b, "b")

    fig.tight_layout(w_pad=1.5)
    save_fig(fig, "fig_bayesian_2panel", FIGURES_DIR)


# ═════════════════════════════════════════════════════════════════════════════
# Figure 3 -- Bayesian shrinkage dumbbells
# ═════════════════════════════════════════════════════════════════════════════
def fig_shrinkage_dumbbells() -> None:
    """Dumbbell chart showing parameter variance shrinkage from Juno update."""
    data, _d_cond, _h_total, weights, _ = _load_andrade_and_weights()

    # Parameters to analyse (npz key -> display label)
    params = [
        ("param_d_grain", r"$d_\mathrm{grain}$"),
        ("param_Q_v", r"$Q_v$"),
        ("param_P_tidal", r"$q_\mathrm{basal}$"),
        ("param_epsilon_0", r"$\varepsilon_0$"),
        ("param_T_surf", r"$T_\mathrm{surf}$"),
    ]

    shrinkages = []
    labels = []
    for key, lbl in params:
        vals = data[key]
        prior_std = np.std(vals)
        post_std = _weighted_std(vals, weights)
        shrink = (1.0 - post_std / prior_std) * 100.0
        shrinkages.append(shrink)
        labels.append(lbl)

    shrinkages = np.array(shrinkages)

    # Sort by shrinkage descending
    order = np.argsort(shrinkages)[::-1]
    shrinkages = shrinkages[order]
    labels = [labels[i] for i in order]

    y_pos = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL * 0.8))

    for i, (s, lbl) in enumerate(zip(shrinkages, labels)):
        colour = PAL.BLUE if s >= 0 else PAL.RED
        # Connecting line from 0 to shrinkage
        ax.plot([0, s], [i, i], color=colour, linewidth=1.5, zorder=3)
        # Dot at 0 (prior reference)
        ax.plot(0, i, "o", color="0.5", markersize=5, zorder=4)
        # Dot at shrinkage value
        ax.plot(s, i, "o", color=colour, markersize=6, zorder=4)

    # Reference lines
    ax.axvline(0, color="0.4", linewidth=0.6, zorder=1)
    ax.axvline(10, color="0.5", linewidth=0.6, linestyle=":", zorder=1)
    ax.axvline(30, color="0.5", linewidth=0.6, linestyle="--", zorder=1)

    # Threshold annotations
    ax.text(10, len(labels) - 0.3, "negligible", fontsize=5.5,
            color="0.45", ha="center", va="bottom")
    ax.text(30, len(labels) - 0.3, "weak", fontsize=5.5,
            color="0.45", ha="center", va="bottom")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Marginal shrinkage (%)")
    ax.set_title("Juno-reweighted parameter shrinkage")

    # Extend x-axis a bit past max
    x_lo = min(-5, shrinkages.min() - 5)
    x_hi = max(40, shrinkages.max() + 5)
    ax.set_xlim(x_lo, x_hi)

    fig.tight_layout()
    save_fig(fig, "fig_shrinkage_dumbbells", FIGURES_DIR)


# ═════════════════════════════════════════════════════════════════════════════
# Figure 1b -- Sobol lollipop for convection-state outputs
# ═════════════════════════════════════════════════════════════════════════════
def fig_sobol_convection_lollipop() -> None:
    """Lollipop Sobol sensitivity for convective fraction and Nu."""
    csv_path = os.path.join(
        EUROPA_DJ,
        "results", "sobol",
        "global_audited_sobol_params_N512",
        "global_audited_sobol_params_N512_indices.csv",
    )
    df = pd.read_csv(csv_path)
    mask = (df["sample_size"] == 512) & (df["index_type"] == "main")
    df_main = df[mask].copy()

    outputs = ["convective_flag", "Nu"]
    fig, axes = plt.subplots(
        1, 2,
        figsize=(DOUBLE_COL * 0.65, DOUBLE_COL * 0.34),
        sharey=False,
    )

    for ax, output, letter in zip(axes, outputs, "ab"):
        sub = df_main[df_main["output"] == output].copy()
        sub = sub[sub["ST"] > 0.02].copy()
        sub = sub.sort_values("ST", ascending=True).reset_index(drop=True)

        y_pos = np.arange(len(sub))
        labels = [PARAM_LABELS.get(f, f) for f in sub["factor"]]

        ax.hlines(y_pos, 0, sub["S1"].values, color=PAL.BLUE, linewidth=0.9)
        ax.plot(
            sub["S1"].values, y_pos, "o",
            color=PAL.BLUE, markersize=5, zorder=5,
            label=r"First-order ($S_1$)",
        )
        ax.plot(
            sub["ST"].values, y_pos, "o",
            markerfacecolor="white", markeredgecolor=PAL.RED,
            markeredgewidth=1.2, markersize=5.5, zorder=5,
            label=r"Total-order ($S_T$)",
        )
        ax.errorbar(
            sub["S1"].values, y_pos,
            xerr=sub["S1_conf"].values,
            fmt="none", ecolor=PAL.BLUE, elinewidth=0.6, capsize=2, capthick=0.5,
        )
        ax.errorbar(
            sub["ST"].values, y_pos,
            xerr=sub["ST_conf"].values,
            fmt="none", ecolor=PAL.RED, elinewidth=0.6, capsize=2, capthick=0.5,
        )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.set_xlim(-0.02, 1.0)
        ax.set_xlabel("Sensitivity index")
        ax.set_title(OUTPUT_TITLES[output])
        ax.axvline(0, color="0.6", linewidth=0.4, zorder=0)
        label_panel(ax, letter)

    axes[-1].legend(loc="lower right", fontsize=6.5)
    fig.tight_layout(w_pad=1.8)
    save_fig(fig, "fig_sobol_convection_lollipop", FIGURES_DIR)


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════
def main() -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    print(f"Output directory: {FIGURES_DIR}\n")

    print("Figure 1: Sobol lollipop ranking ...")
    fig_sobol_lollipop()

    print("Figure 1b: Sobol convection-state lollipop ...")
    fig_sobol_convection_lollipop()

    print("Figure 2: Bayesian predictive 2-panel ...")
    fig_bayesian_2panel()

    print("Figure 3: Bayesian shrinkage dumbbells ...")
    fig_shrinkage_dumbbells()

    print("\nDone -- all figures saved.")


if __name__ == "__main__":
    main()
