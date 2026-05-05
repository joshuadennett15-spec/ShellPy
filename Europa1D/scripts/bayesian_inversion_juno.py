#!/usr/bin/env python3
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
"""
Bayesian reanalysis of Europa ice shell using Juno MWR constraint.

Implements importance reweighting (likelihood weighting) of prior MC draws
against the Juno conductive shell thickness observation (Levin et al., 2026).

Observation models:
  A: D_cond ~ N(29, sigma_total)  [pure water, mainline]
  B: D_cond ~ N(24, sigma_total)  [low-salinity sensitivity]

Usage:
    python bayesian_inversion_juno.py
    python bayesian_inversion_juno.py --filepath results/mc_15000_optionA_andrade.npz
    python bayesian_inversion_juno.py --sigma-model 3.0
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)

import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

from pub_style import (
    apply_style, PAL, figsize_double, figsize_double_tall,
    label_panel, save_fig, add_minor_gridlines, DOUBLE_COL,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures', 'pub')

apply_style()


# ═══════════════════════════════════════════════════════════════════════════
# Core Bayesian machinery
# ═══════════════════════════════════════════════════════════════════════════

def compute_log_weights(D_cond, D_obs, sigma_obs, sigma_model):
    """
    Gaussian log-likelihood weights for importance reweighting.

    w_i ∝ p(D_obs | D_cond_i) = N(D_cond_i; D_obs, sigma_total)
    """
    sigma_total = np.sqrt(sigma_obs**2 + sigma_model**2)
    return -0.5 * ((D_cond - D_obs) / sigma_total) ** 2


def normalize_weights(log_w):
    """Stabilize and normalize log-weights to proper weights."""
    log_w_stable = log_w - np.max(log_w)
    w = np.exp(log_w_stable)
    return w / w.sum()


def effective_sample_size(w):
    """Kish's ESS: n_eff = 1 / sum(w_i^2) for normalized weights."""
    return 1.0 / np.sum(w**2)


def weighted_percentile(values, weights, percentile):
    """Compute weighted percentile."""
    idx = np.argsort(values)
    sorted_v = values[idx]
    sorted_w = weights[idx]
    cum_w = np.cumsum(sorted_w)
    return float(np.interp(percentile / 100.0, cum_w, sorted_v))


def weighted_kde(values, weights, n_pts=300):
    """Weighted KDE using resampling."""
    n_resample = min(10000, len(values))
    idx = np.random.choice(len(values), size=n_resample, p=weights, replace=True)
    resampled = values[idx]
    if len(np.unique(resampled)) < 5:
        return None, None
    kde = gaussian_kde(resampled)
    lo = max(0, np.percentile(resampled, 1) - 2)
    hi = np.percentile(resampled, 99) + 2
    x = np.linspace(lo, hi, n_pts)
    return x, kde(x)


def posterior_summary(values, weights, label):
    """Print weighted posterior summary."""
    med = weighted_percentile(values, weights, 50)
    lo = weighted_percentile(values, weights, 15.87)
    hi = weighted_percentile(values, weights, 84.13)
    lo90 = weighted_percentile(values, weights, 5)
    hi90 = weighted_percentile(values, weights, 95)
    print(f"  {label:<20s}: median={med:.2f}, 1\u03c3=[{lo:.2f}, {hi:.2f}], 90%=[{lo90:.2f}, {hi90:.2f}]")
    return med, lo, hi


# ═══════════════════════════════════════════════════════════════════════════
# Figures
# ═══════════════════════════════════════════════════════════════════════════

def fig_prior_vs_posterior_params(data, weights, model_label, ess):
    """Prior vs posterior KDEs for key parameters."""
    print(f"\nFigure: Prior vs posterior ({model_label})")

    # Back-calculate q_basal: q_rad + P_tidal/A
    A_surf = 3.063e13
    D_H2O = data['param_D_H2O']
    H_rad = data['param_H_rad']
    R_rock = 1.561e6 - D_H2O
    M_rock = (4.0 / 3.0) * np.pi * (R_rock ** 3) * 3500.0
    q_rad = (H_rad * M_rock) / A_surf
    q_tidal = data['param_P_tidal'] / A_surf
    q_basal_wm2 = (q_rad + q_tidal) * 1e3  # mW/m^2

    params = [
        (q_basal_wm2, r'$q_\mathrm{basal}$ (mW/m$^2$)', (5, 35)),
        (data['param_d_grain'] * 1e3, r'$d_\mathrm{grain}$ (mm)', (0, 5)),
        (data['param_epsilon_0'] * 1e5, r'$\varepsilon_0$ ($\times 10^{-5}$)', (0, 4)),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, DOUBLE_COL * 0.35))

    for ax, (raw, xlabel, xlim) in zip(axes, params):
        # Prior KDE
        kde_prior = gaussian_kde(raw)
        x = np.linspace(xlim[0], xlim[1], 300)
        ax.fill_between(x, kde_prior(x), alpha=0.15, color=PAL.BLUE)
        ax.plot(x, kde_prior(x), color=PAL.BLUE, lw=1.0, label="Prior")

        # Posterior KDE
        xp, pdf_p = weighted_kde(raw, weights)
        if pdf_p is not None:
            ax.fill_between(xp, pdf_p, alpha=0.20, color=PAL.RED)
            ax.plot(xp, pdf_p, color=PAL.RED, lw=1.2, label="Posterior")

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
        ax.set_xlim(xlim)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=6)
        add_minor_gridlines(ax, axis="y")

    label_panel(axes[0], "a")
    label_panel(axes[1], "b")
    label_panel(axes[2], "c")

    fig.suptitle(
        f"Juno MWR posterior update \u2014 {model_label} (ESS = {ess:.0f})",
        fontsize=9, y=1.02,
    )
    fig.tight_layout(w_pad=2.0)
    tag = model_label.lower().replace(" ", "_").replace("\u00b1", "pm").replace(":", "")
    save_fig(fig, f"fig_bayesian_params_{tag}", FIGURES_DIR)


def fig_posterior_predictive(D_cond, H_total, weights, D_obs, sigma_obs, sigma_model, model_label, ess):
    """Posterior predictive check: does the posterior reproduce the Juno obs?"""
    print(f"\nFigure: Posterior predictive ({model_label})")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize_double(0.42))

    bins = np.linspace(0, 80, 60)

    # (a) D_cond: prior, posterior, Juno constraint
    ax1.hist(D_cond, bins=bins, density=True, color=PAL.BLUE, alpha=0.15,
             edgecolor=PAL.BLUE, linewidth=0.3, label="Prior")

    n_resample = min(50000, len(D_cond))
    idx = np.random.choice(len(D_cond), size=n_resample, p=weights, replace=True)
    D_post = D_cond[idx]
    ax1.hist(D_post, bins=bins, density=True, color=PAL.RED, alpha=0.25,
             edgecolor=PAL.RED, linewidth=0.3, label="Posterior")

    sigma_total = np.sqrt(sigma_obs**2 + sigma_model**2)
    x_juno = np.linspace(0, 70, 300)
    juno_pdf = np.exp(-0.5 * ((x_juno - D_obs) / sigma_total)**2) / (sigma_total * np.sqrt(2 * np.pi))
    ax1.plot(x_juno, juno_pdf, color=PAL.BLACK, lw=1.2, ls="--",
             label=f"Juno N({D_obs:.0f}, {sigma_total:.0f})")
    ax1.axvline(D_obs, color=PAL.BLACK, lw=0.6, ls=":", alpha=0.5)

    ax1.set_xlabel(r"Conductive lid $D_\mathrm{cond}$ (km)")
    ax1.set_ylabel("Density")
    ax1.set_xlim(0, 70)
    ax1.set_ylim(bottom=0)
    ax1.legend(fontsize=6, loc="upper right")
    label_panel(ax1, "a")

    # (b) H_total: prior vs posterior
    ax2.hist(H_total, bins=bins, density=True, color=PAL.BLUE, alpha=0.15,
             edgecolor=PAL.BLUE, linewidth=0.3, label="Prior")
    H_post = H_total[idx]
    ax2.hist(H_post, bins=bins, density=True, color=PAL.RED, alpha=0.25,
             edgecolor=PAL.RED, linewidth=0.3, label="Posterior")
    ax2.axvline(D_obs, color=PAL.BLACK, lw=0.6, ls=":", alpha=0.5,
                label=f"Juno {D_obs:.0f} km ref")

    ax2.set_xlabel(r"Total shell $H_\mathrm{total}$ (km)")
    ax2.set_ylabel("Density")
    ax2.set_xlim(0, 70)
    ax2.set_ylim(bottom=0)
    ax2.legend(fontsize=6, loc="upper right")
    label_panel(ax2, "b")

    fig.suptitle(
        f"Posterior predictive \u2014 {model_label} (ESS = {ess:.0f})",
        fontsize=9, y=1.02,
    )
    fig.tight_layout(w_pad=2.5)
    tag = model_label.lower().replace(" ", "_").replace("\u00b1", "pm").replace(":", "")
    save_fig(fig, f"fig_bayesian_predictive_{tag}", FIGURES_DIR)


def fig_corner_plot(data, weights, model_label, ess):
    """Corner plot for the three key parameters."""
    print(f"\nFigure: Corner plot ({model_label})")

    params = [
        ('D_cond_km', r'$D_\mathrm{cond}$ (km)', lambda x: x),
        ('param_d_grain', r'$d_\mathrm{grain}$ (mm)', lambda x: x * 1e3),
        ('param_epsilon_0', r'$\varepsilon_0$ ($\times 10^{-5}$)', lambda x: x * 1e5),
    ]

    n = len(params)
    fig, axes = plt.subplots(n, n, figsize=(5.5, 5.5))

    n_resample = min(20000, len(weights))
    idx_post = np.random.choice(len(weights), size=n_resample, p=weights, replace=True)

    for i, (ki, li, ci) in enumerate(params):
        vi_prior = ci(data[ki])
        vi_post = ci(data[ki][idx_post])

        for j, (kj, lj, cj) in enumerate(params):
            ax = axes[i, j]

            if j > i:
                ax.set_visible(False)
                continue

            vj_prior = cj(data[kj])
            vj_post = cj(data[kj][idx_post])

            if i == j:
                kde_pr = gaussian_kde(vi_prior)
                xg = np.linspace(np.percentile(vi_prior, 1), np.percentile(vi_prior, 99), 200)
                ax.fill_between(xg, kde_pr(xg), alpha=0.15, color=PAL.BLUE)
                ax.plot(xg, kde_pr(xg), color=PAL.BLUE, lw=0.8)
                if len(np.unique(vi_post)) > 5:
                    kde_po = gaussian_kde(vi_post)
                    ax.fill_between(xg, kde_po(xg), alpha=0.25, color=PAL.RED)
                    ax.plot(xg, kde_po(xg), color=PAL.RED, lw=1.0)
                ax.set_yticks([])
            else:
                ax.scatter(vj_prior, vi_prior, s=0.3, alpha=0.05, color=PAL.BLUE, rasterized=True)
                ax.scatter(vj_post, vi_post, s=0.5, alpha=0.15, color=PAL.RED, rasterized=True)

            if i == n - 1:
                ax.set_xlabel(lj, fontsize=7)
            else:
                ax.set_xticklabels([])
            if j == 0 and i != 0:
                ax.set_ylabel(li, fontsize=7)
            elif j == 0 and i == 0:
                ax.set_ylabel("Density", fontsize=7)

    fig.suptitle(
        f"Corner plot \u2014 {model_label} (ESS = {ess:.0f})",
        fontsize=9, y=1.01,
    )
    fig.tight_layout()
    tag = model_label.lower().replace(" ", "_").replace("\u00b1", "pm").replace(":", "")
    save_fig(fig, f"fig_bayesian_corner_{tag}", FIGURES_DIR)


# ═══════════════════════════════════════════════════════════════════════════
# Main workflow
# ═══════════════════════════════════════════════════════════════════════════

def run_model(data, D_obs, sigma_obs, sigma_model, model_label):
    """Run one observation model: weights, summaries, figures."""
    D_cond = data['D_cond_km']
    H_total = data['thicknesses_km']

    print(f"\n{'=' * 60}")
    print(f"OBSERVATION MODEL: {model_label}")
    print(f"  D_obs = {D_obs} km, sigma_obs = {sigma_obs} km, sigma_model = {sigma_model} km")
    sigma_total = np.sqrt(sigma_obs**2 + sigma_model**2)
    print(f"  sigma_total = {sigma_total:.1f} km")
    print(f"{'=' * 60}")

    # Stage 0: prior predictive check
    prior_overlap = np.mean((D_cond > D_obs - 2 * sigma_total) & (D_cond < D_obs + 2 * sigma_total))
    print(f"\nStage 0: Prior predictive overlap with Juno \u00b12\u03c3: {prior_overlap:.1%}")
    if prior_overlap < 0.05:
        print("  WARNING: Very low prior overlap \u2014 inversion may be unstable.")

    # Stage 1: compute weights
    log_w = compute_log_weights(D_cond, D_obs, sigma_obs, sigma_model)
    w = normalize_weights(log_w)
    ess = effective_sample_size(w)
    print(f"\nStage 1: Importance reweighting")
    print(f"  ESS = {ess:.0f} / {len(w)} ({100 * ess / len(w):.1f}%)")

    # Posterior summaries
    print(f"\nPosterior summaries:")
    posterior_summary(D_cond, w, "D_cond (km)")
    posterior_summary(H_total, w, "H_total (km)")
    posterior_summary(data['D_conv_km'], w, "D_conv (km)")
    posterior_summary(data['lid_fractions'], w, "Lid fraction")

    # Key parameters
    print(f"\nKey parameter posteriors:")
    # Reconstruct q_basal
    A_surf = 3.063e13
    D_H2O = data['param_D_H2O']
    H_rad = data['param_H_rad']
    R_rock = 1.561e6 - D_H2O
    M_rock = (4.0 / 3.0) * np.pi * (R_rock ** 3) * 3500.0
    q_rad = (H_rad * M_rock) / A_surf
    q_tidal = data['param_P_tidal'] / A_surf
    q_basal_mw = (q_rad + q_tidal) * 1e3
    posterior_summary(q_basal_mw, w, "q_basal (mW/m\u00b2)")
    posterior_summary(data['param_d_grain'] * 1e3, w, "d_grain (mm)")
    posterior_summary(data['param_epsilon_0'] * 1e5, w, "\u03b5\u2080 (\u00d710\u207b\u2075)")
    posterior_summary(data['param_T_surf'], w, "T_surf (K)")

    # Convective fraction in posterior
    conv_mask = data['D_conv_km'] > 0.5
    post_conv_frac = np.sum(w[conv_mask])
    print(f"\n  Posterior convective fraction: {post_conv_frac:.1%} (prior: {conv_mask.mean():.1%})")

    # Stage 2: figures
    fig_prior_vs_posterior_params(data, w, model_label, ess)
    fig_posterior_predictive(D_cond, H_total, w, D_obs, sigma_obs, sigma_model, model_label, ess)
    fig_corner_plot(data, w, model_label, ess)

    return w, ess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", default="")
    parser.add_argument("--sigma-model", type=float, default=3.0,
                        help="Model discrepancy term (km). Default 3.0.")
    args = parser.parse_args()

    if args.filepath:
        filepath = args.filepath
    else:
        filepath = os.path.join(RESULTS_DIR, "mc_15000_optionA_andrade.npz")

    if not os.path.exists(filepath):
        print(f"Not found: {filepath}")
        sys.exit(1)

    print(f"Loading: {filepath}")
    data = np.load(filepath)
    n = len(data['thicknesses_km'])
    print(f"  N = {n:,} samples")

    np.random.seed(42)

    # Model A: pure water (mainline)
    w_a, ess_a = run_model(data, D_obs=29.0, sigma_obs=10.0,
                           sigma_model=args.sigma_model,
                           model_label="Model A 29\u00b110 km")

    # Model B: low-salinity sensitivity
    w_b, ess_b = run_model(data, D_obs=24.0, sigma_obs=10.0,
                           sigma_model=args.sigma_model,
                           model_label="Model B 24\u00b110 km")

    print(f"\n{'=' * 60}")
    print("DONE \u2014 all figures saved to figures/pub/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
