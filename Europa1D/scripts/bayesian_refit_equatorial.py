#!/usr/bin/env python3
"""
Bayesian posterior refit of equatorial-proxy Monte Carlo draws against
the Juno MWR conductive-shell constraint (Levin et al., 2026).

This is NOT a visual overlay of Juno curves on prior KDEs.
This script computes proper importance-reweighted posteriors for the
most informative physical parameters, with PSIS diagnostics and
prior-to-posterior shrinkage metrics.

Method:
  - Importance reweighting of archived MC draws
  - Gaussian likelihood: D_obs ~ N(D_cond_model, sigma_total)
    with sigma_total^2 = sigma_obs^2 + sigma_model^2
  - PSIS (Pareto-smoothed importance sampling) stability diagnostics
  - Prior-to-posterior shrinkage for target parameters

Observation models:
  A (pure water):    D_cond = 29 +/- 10 km
  B (low salinity):  D_cond = 24 +/- 10 km

Target parameters (ranked by Sobol total-order sensitivity to D_cond):
  1. Q_v           — activation energy for volume diffusion
  2. d_grain       — grain size
  3. q_basal       — basal heat flux (reconstructed from P_tidal + radiogenic)
  4. epsilon_0     — reference tidal strain rate
  5. T_surf        — surface temperature

Usage:
    python bayesian_refit_equatorial.py
    python bayesian_refit_equatorial.py --sigma-model 5.0
    python bayesian_refit_equatorial.py --sigma-sensitivity
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
import matplotlib.gridspec as gridspec
import csv

from pub_style import (
    apply_style, PAL, SINGLE_COL, DOUBLE_COL,
    label_panel, save_fig, add_minor_gridlines,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
OUTPUT_DIR = os.path.join(RESULTS_DIR, 'bayesian_refit')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures', 'bayesian_refit')

apply_style()

# ═══════════════════════════════════════════════════════════════════════════
# Equatorial modes
# ═══════════════════════════════════════════════════════════════════════════

MODES = [
    ("eq_baseline_andrade.npz",         "Baseline (1.0x)",         "baseline"),
    ("eq_moderate_andrade.npz",         "Moderate (1.2x)",         "moderate"),
    ("eq_depleted_andrade.npz",         "Depleted (0.67x)",        "depleted"),
    ("eq_depleted_strong_andrade.npz",  "Depleted strong (0.55x)", "depleted_strong"),
    ("eq_strong_andrade.npz",           "Strong (1.5x)",           "strong"),
]

JUNO_OBS = [
    (29.0, 10.0, "pure_water",    "Pure water 29±10 km"),
    (24.0, 10.0, "low_salinity",  "Low salinity 24±10 km"),
]

# Target parameters: (npz_key, display_name, unit, transform, display_range)
# transform converts raw NPZ values to display units
TARGET_PARAMS = [
    ("Q_v",       r"$Q_v$",                   "kJ/mol",              lambda x: x / 1e3,   (40, 80)),
    ("d_grain",   r"$d_\mathrm{grain}$",      "mm",                  lambda x: x * 1e3,   (0, 3.5)),
    ("q_basal",   r"$q_\mathrm{basal}$",      r"mW/m$^2$",          lambda x: x,          (0, 40)),
    ("epsilon_0", r"$\varepsilon_0$",          r"$\times 10^{-5}$",  lambda x: x * 1e5,   (0, 2.5)),
    ("T_surf",    r"$T_\mathrm{surf}$",       "K",                   lambda x: x,          (93, 122)),
]


# ═══════════════════════════════════════════════════════════════════════════
# Core Bayesian machinery
# ═══════════════════════════════════════════════════════════════════════════

def compute_log_weights(D_cond, D_obs, sigma_obs, sigma_model):
    """Gaussian log-likelihood weights for importance reweighting."""
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


def weighted_quantiles(values, weights, quantiles):
    """Compute multiple weighted quantiles efficiently."""
    idx = np.argsort(values)
    sorted_v = values[idx]
    sorted_w = weights[idx]
    cum_w = np.cumsum(sorted_w)
    return [float(np.interp(q, cum_w, sorted_v)) for q in quantiles]


# ═══════════════════════════════════════════════════════════════════════════
# PSIS diagnostics (Vehtari, Gelman & Gabry 2017)
# ═══════════════════════════════════════════════════════════════════════════

def psis_diagnostic(log_w):
    """
    Pareto-smoothed importance sampling (PSIS) stability diagnostic.

    Fits a generalized Pareto distribution (GPD) to the upper tail of the
    importance weights and reports the shape parameter k (Vehtari, Gelman &
    Gabry, 2017; Zhang & Stephens, 2009). This is the standard PSIS k-hat
    diagnostic; it is *not* capped or ESS-adjusted --- the raw MLE of the
    GPD shape parameter is returned.

    Tail size follows Vehtari et al.: M = min(0.2 * N, 3 * sqrt(N)).

    Interpretation of k-hat (Vehtari, Gelman & Gabry, 2017):
      k <= 0.5: reliable (importance sampling converges quickly)
      0.5 < k <= 0.7: acceptable (finite variance, slower convergence)
      0.7 < k <= 1.0: unreliable (infinite variance)
      k > 1.0: very unreliable (infinite mean)

    Returns
    -------
    k_hat : float
        GPD shape parameter estimated from the upper tail of the weights.
    diagnostic : str
        Human-readable assessment using the thresholds above.
    """
    from scipy import stats

    S = len(log_w)
    if S < 25:
        return float('nan'), "insufficient samples"

    # Stable normalisation
    log_w_stable = log_w - np.max(log_w)
    w = np.exp(log_w_stable)

    # Tail size (Vehtari et al. 2017 default)
    M = int(min(0.2 * S, 3.0 * np.sqrt(S)))
    M = max(M, 25)
    if M >= S:
        return float('nan'), "tail too large"

    # Upper tail of the (unnormalised) weights
    w_sorted = np.sort(w)
    tail = w_sorted[-M:]
    threshold = w_sorted[-M - 1] if S > M else 0.0
    excess = tail - threshold
    excess = excess[excess > 0]
    if len(excess) < 10:
        return float('nan'), "insufficient tail excesses"

    # GPD MLE via scipy. floc=0 enforces the threshold-shifted parameterisation.
    try:
        k_hat, _, _ = stats.genpareto.fit(excess, floc=0.0)
    except Exception:
        return float('nan'), "GPD fit failed"

    if k_hat <= 0.5:
        diagnostic = "good"
    elif k_hat <= 0.7:
        diagnostic = "acceptable"
    elif k_hat <= 1.0:
        diagnostic = "unreliable"
    else:
        diagnostic = "very unreliable"

    return float(k_hat), diagnostic


# ═══════════════════════════════════════════════════════════════════════════
# Parameter extraction
# ═══════════════════════════════════════════════════════════════════════════

def extract_params(data):
    """
    Extract target parameter arrays from NPZ data.

    Reconstructs q_basal from P_tidal + radiogenic heating.
    Returns dict of {param_name: array_in_display_units}.
    """
    A_surf = 3.063e13  # Europa surface area, m^2
    D_H2O = data['param_D_H2O']
    H_rad = data['param_H_rad']
    R_rock = 1.561e6 - D_H2O  # Rocky core radius
    M_rock = (4.0 / 3.0) * np.pi * (R_rock ** 3) * 3500.0
    q_rad = (H_rad * M_rock) / A_surf
    q_tidal = data['param_P_tidal'] / A_surf
    q_basal_mw = (q_rad + q_tidal) * 1e3  # mW/m^2

    return {
        "Q_v":       data['param_Q_v'] / 1e3,        # kJ/mol
        "d_grain":   data['param_d_grain'] * 1e3,     # mm
        "q_basal":   q_basal_mw,                       # mW/m^2
        "epsilon_0": data['param_epsilon_0'] * 1e5,    # x10^-5
        "T_surf":    data['param_T_surf'],             # K
    }


def shrinkage_ratio(prior_values, weights):
    """
    Compute prior-to-posterior shrinkage ratio.

    Shrinkage = 1 - (posterior_68_width / prior_68_width).
    0 = no shrinkage, 1 = perfectly constrained.
    Negative = posterior broader (pathological).
    """
    # Prior 68% interval
    p16_prior = np.percentile(prior_values, 15.87)
    p84_prior = np.percentile(prior_values, 84.13)
    prior_width = p84_prior - p16_prior

    if prior_width < 1e-15:
        return 0.0

    # Posterior 68% interval
    p16_post = weighted_percentile(prior_values, weights, 15.87)
    p84_post = weighted_percentile(prior_values, weights, 84.13)
    post_width = p84_post - p16_post

    return 1.0 - (post_width / prior_width)


# ═══════════════════════════════════════════════════════════════════════════
# Weighted KDE helper
# ═══════════════════════════════════════════════════════════════════════════

def weighted_kde(values, weights, n_pts=300, x_range=None):
    """Weighted KDE via resampling."""
    n_resample = min(15000, len(values))
    idx = np.random.choice(len(values), size=n_resample, p=weights, replace=True)
    resampled = values[idx]
    if len(np.unique(resampled)) < 5:
        return None, None
    kde = gaussian_kde(resampled)
    if x_range is not None:
        lo, hi = x_range
    else:
        lo = max(0, np.percentile(resampled, 0.5) - 2)
        hi = np.percentile(resampled, 99.5) + 2
    x = np.linspace(lo, hi, n_pts)
    return x, kde(x)


# ═══════════════════════════════════════════════════════════════════════════
# Single-mode posterior computation
# ═══════════════════════════════════════════════════════════════════════════

def compute_posterior(data, D_obs, sigma_obs, sigma_model):
    """
    Compute importance-reweighted posterior for one mode × one obs model.

    Returns dict with weights, ESS, PSIS k-hat, and per-parameter summaries.
    """
    D_cond = data['D_cond_km']
    n = len(D_cond)
    sigma_total = np.sqrt(sigma_obs**2 + sigma_model**2)

    # Prior predictive overlap
    prior_overlap = np.mean(
        (D_cond > D_obs - 2 * sigma_total) & (D_cond < D_obs + 2 * sigma_total)
    )

    # Log-weights and normalized weights
    log_w = compute_log_weights(D_cond, D_obs, sigma_obs, sigma_model)
    w = normalize_weights(log_w)
    ess = effective_sample_size(w)

    # PSIS diagnostic
    k_hat, k_diagnostic = psis_diagnostic(log_w)

    # Max-weight ratio (additional stability check)
    max_w_ratio = float(np.max(w) * n)

    # Extract parameters
    params = extract_params(data)

    # Per-parameter summaries
    param_summaries = {}
    for key, display_name, unit, _, _ in TARGET_PARAMS:
        vals = params[key]
        # Prior quantiles
        prior_med = float(np.median(vals))
        prior_16, prior_84 = float(np.percentile(vals, 15.87)), float(np.percentile(vals, 84.13))
        prior_2p5, prior_97p5 = float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

        # Posterior quantiles
        post_med = weighted_percentile(vals, w, 50)
        post_16 = weighted_percentile(vals, w, 15.87)
        post_84 = weighted_percentile(vals, w, 84.13)
        post_2p5 = weighted_percentile(vals, w, 2.5)
        post_97p5 = weighted_percentile(vals, w, 97.5)

        # Shrinkage
        shrink = shrinkage_ratio(vals, w)

        param_summaries[key] = {
            "display_name": display_name,
            "unit": unit,
            "prior_median": prior_med,
            "prior_68_lo": prior_16,
            "prior_68_hi": prior_84,
            "prior_95_lo": prior_2p5,
            "prior_95_hi": prior_97p5,
            "posterior_median": post_med,
            "posterior_68_lo": post_16,
            "posterior_68_hi": post_84,
            "posterior_95_lo": post_2p5,
            "posterior_95_hi": post_97p5,
            "shrinkage": shrink,
        }

    # D_cond summary
    dc_prior_med = float(np.median(D_cond))
    dc_post_med = weighted_percentile(D_cond, w, 50)
    dc_post_16 = weighted_percentile(D_cond, w, 15.87)
    dc_post_84 = weighted_percentile(D_cond, w, 84.13)
    dc_shrink = shrinkage_ratio(D_cond, w)

    return {
        "weights": w,
        "log_weights": log_w,
        "n_samples": n,
        "ess": ess,
        "ess_frac": ess / n,
        "k_hat": k_hat,
        "k_diagnostic": k_diagnostic,
        "max_w_ratio": max_w_ratio,
        "prior_overlap": prior_overlap,
        "sigma_total": sigma_total,
        "params": params,
        "param_summaries": param_summaries,
        "D_cond_prior_median": dc_prior_med,
        "D_cond_posterior_median": dc_post_med,
        "D_cond_posterior_68": (dc_post_16, dc_post_84),
        "D_cond_shrinkage": dc_shrink,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Figures
# ═══════════════════════════════════════════════════════════════════════════

def fig_dcond_prior_posterior_likelihood(data, result, D_obs, sigma_obs, sigma_model,
                                         mode_label, obs_label, tag):
    """
    Per-scenario figure: D_cond prior, posterior, and Juno likelihood.
    Unambiguously labeled — this is a posterior refit, not a visual overlay.
    """
    D_cond = data['D_cond_km']
    w = result['weights']
    sigma_total = result['sigma_total']
    ess = result['ess']
    k_hat = result['k_hat']

    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.3, SINGLE_COL * 0.85))

    x_grid = np.linspace(0, 70, 400)

    # Prior KDE
    kde_prior = gaussian_kde(D_cond)
    ax.fill_between(x_grid, kde_prior(x_grid), alpha=0.12, color=PAL.BLUE)
    ax.plot(x_grid, kde_prior(x_grid), color=PAL.BLUE, lw=1.0, label="Prior")

    # Posterior KDE (resampled)
    x_post, pdf_post = weighted_kde(D_cond, w, n_pts=400, x_range=(0, 70))
    if pdf_post is not None:
        ax.fill_between(x_post, pdf_post, alpha=0.25, color=PAL.RED)
        ax.plot(x_post, pdf_post, color=PAL.RED, lw=1.4, label="Posterior")

    # Juno likelihood (scaled for visibility)
    juno_pdf = np.exp(-0.5 * ((x_grid - D_obs) / sigma_total)**2)
    juno_pdf /= (sigma_total * np.sqrt(2 * np.pi))
    ax.plot(x_grid, juno_pdf, color=PAL.BLACK, lw=1.2, ls="--",
            label=f"Juno likelihood N({D_obs:.0f}, {sigma_total:.1f})")
    ax.axvline(D_obs, color=PAL.BLACK, lw=0.5, ls=":", alpha=0.4)

    ax.set_xlabel(r"Conductive lid $D_\mathrm{cond}$ (km)")
    ax.set_ylabel("Probability density")
    ax.set_xlim(0, 70)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=6, loc="upper right")
    add_minor_gridlines(ax, axis="y")

    ax.set_title(
        f"{mode_label} — {obs_label}\n"
        f"ESS = {ess:.0f}/{len(D_cond)}, "
        f"PSIS $\\hat{{k}}$ = {k_hat:.2f}",
        fontsize=7.5
    )

    fig.tight_layout()
    save_fig(fig, f"fig_refit_dcond_{tag}", FIGURES_DIR)


def fig_parameter_posteriors(data, result, mode_label, obs_label, tag):
    """
    Per-scenario figure: prior vs posterior for all target parameters.
    """
    w = result['weights']
    params = result['params']
    summaries = result['param_summaries']
    ess = result['ess']
    k_hat = result['k_hat']

    n_params = len(TARGET_PARAMS)
    fig, axes = plt.subplots(1, n_params, figsize=(DOUBLE_COL, DOUBLE_COL * 0.30))

    for i, (key, display_name, unit, _, x_range) in enumerate(TARGET_PARAMS):
        ax = axes[i]
        vals = params[key]
        shrink = summaries[key]['shrinkage']

        # Prior KDE
        x_grid = np.linspace(x_range[0], x_range[1], 300)
        try:
            kde_prior = gaussian_kde(vals)
            ax.fill_between(x_grid, kde_prior(x_grid), alpha=0.12, color=PAL.BLUE)
            ax.plot(x_grid, kde_prior(x_grid), color=PAL.BLUE, lw=0.8, label="Prior")
        except np.linalg.LinAlgError:
            pass

        # Posterior KDE
        x_post, pdf_post = weighted_kde(vals, w, n_pts=300, x_range=x_range)
        if pdf_post is not None:
            ax.fill_between(x_post, pdf_post, alpha=0.22, color=PAL.RED)
            ax.plot(x_post, pdf_post, color=PAL.RED, lw=1.1, label="Posterior")

        ax.set_xlabel(f"{display_name} ({unit})", fontsize=7)
        if i == 0:
            ax.set_ylabel("Density", fontsize=7)
        ax.set_xlim(x_range)
        ax.set_ylim(bottom=0)
        ax.tick_params(labelsize=6)

        # Shrinkage annotation
        ax.text(0.97, 0.95, f"S = {shrink:.0%}",
                transform=ax.transAxes, fontsize=6, ha='right', va='top',
                color=PAL.RED if shrink > 0.10 else '0.5')

        if i == 0:
            ax.legend(fontsize=5.5)

        label_panel(ax, chr(ord('a') + i), x=-0.15, y=1.05)

    fig.suptitle(
        f"Posterior parameter update — {mode_label}, {obs_label}\n"
        f"ESS = {ess:.0f}, PSIS $\\hat{{k}}$ = {k_hat:.2f}",
        fontsize=8, y=1.06
    )
    fig.tight_layout(w_pad=1.0)
    save_fig(fig, f"fig_refit_params_{tag}", FIGURES_DIR)


def fig_shrinkage_summary(all_results):
    """
    Summary figure: prior-to-posterior shrinkage across parameters and scenarios.
    One panel per observation model. Heatmap-style or grouped bar.
    """
    for obs_key, obs_label in [("pure_water", "Pure water 29±10"),
                                ("low_salinity", "Low salinity 24±10")]:
        # Collect shrinkage values
        mode_labels = []
        param_keys = [p[0] for p in TARGET_PARAMS]
        param_display = [p[1] for p in TARGET_PARAMS]
        shrinkage_matrix = []

        for mode_tag, results_dict in all_results.items():
            if obs_key not in results_dict:
                continue
            r = results_dict[obs_key]
            mode_labels.append(mode_tag.replace("_", " ").title())
            row = [r['param_summaries'][k]['shrinkage'] for k in param_keys]
            # Also add D_cond shrinkage
            shrinkage_matrix.append(row)

        if not shrinkage_matrix:
            continue

        shrinkage_matrix = np.array(shrinkage_matrix)
        n_modes = len(mode_labels)
        n_params = len(param_keys)

        fig, ax = plt.subplots(figsize=(DOUBLE_COL * 0.7, SINGLE_COL * 0.8))

        x = np.arange(n_params)
        width = 0.8 / n_modes
        colors = [PAL.BLUE, PAL.ORANGE, PAL.GREEN, PAL.RED, PAL.PURPLE]

        for i, (label, row) in enumerate(zip(mode_labels, shrinkage_matrix)):
            offset = (i - n_modes / 2 + 0.5) * width
            bars = ax.bar(x + offset, row * 100, width * 0.9,
                          color=colors[i % len(colors)], alpha=0.7,
                          edgecolor='0.3', lw=0.3, label=label)

        ax.set_xticks(x)
        ax.set_xticklabels([f"{p[1]}" for p in TARGET_PARAMS], fontsize=6.5)
        ax.set_ylabel("Shrinkage (%)", fontsize=8)
        ax.set_title(f"Prior → Posterior shrinkage — {obs_label}", fontsize=8)
        ax.axhline(0, color='0.5', lw=0.5)
        ax.axhline(10, color='0.7', lw=0.4, ls='--', alpha=0.5)
        ax.legend(fontsize=5.5, ncol=2, loc='upper right')
        add_minor_gridlines(ax, axis="y")

        fig.tight_layout()
        save_fig(fig, f"fig_refit_shrinkage_{obs_key}", FIGURES_DIR)


def fig_combined_dcond_posteriors(all_results, all_data):
    """
    Combined figure: D_cond prior/posterior/likelihood for Baseline, Moderate,
    Depleted side-by-side, for pure-water obs model.
    """
    obs_key = "pure_water"
    D_obs, sigma_obs = 29.0, 10.0
    target_modes = ["baseline", "moderate", "depleted"]

    available = [m for m in target_modes if m in all_results and obs_key in all_results[m]]
    if len(available) < 2:
        return

    n = len(available)
    fig, axes = plt.subplots(1, n, figsize=(DOUBLE_COL, DOUBLE_COL * 0.30))
    if n == 1:
        axes = [axes]

    for i, mode_tag in enumerate(available):
        ax = axes[i]
        r = all_results[mode_tag][obs_key]
        D_cond = all_data[mode_tag]['D_cond_km']
        w = r['weights']
        sigma_total = r['sigma_total']
        ess = r['ess']

        x_grid = np.linspace(0, 70, 400)

        # Prior
        kde_prior = gaussian_kde(D_cond)
        ax.fill_between(x_grid, kde_prior(x_grid), alpha=0.12, color=PAL.BLUE)
        ax.plot(x_grid, kde_prior(x_grid), color=PAL.BLUE, lw=0.8, label="Prior")

        # Posterior
        x_post, pdf_post = weighted_kde(D_cond, w, n_pts=400, x_range=(0, 70))
        if pdf_post is not None:
            ax.fill_between(x_post, pdf_post, alpha=0.25, color=PAL.RED)
            ax.plot(x_post, pdf_post, color=PAL.RED, lw=1.2, label="Posterior")

        # Likelihood
        juno_pdf = np.exp(-0.5 * ((x_grid - D_obs) / sigma_total)**2)
        juno_pdf /= (sigma_total * np.sqrt(2 * np.pi))
        ax.plot(x_grid, juno_pdf, color=PAL.BLACK, lw=1.0, ls="--", label="Juno likelihood")

        ax.set_xlabel(r"$D_\mathrm{cond}$ (km)", fontsize=7)
        if i == 0:
            ax.set_ylabel("Density", fontsize=7)
        ax.set_xlim(0, 65)
        ax.set_ylim(bottom=0)
        ax.set_title(f"{mode_tag.replace('_', ' ').title()} (ESS={ess:.0f})", fontsize=7)
        ax.tick_params(labelsize=6)

        if i == 0:
            ax.legend(fontsize=5, loc='upper right')
        label_panel(ax, chr(ord('a') + i), x=-0.15, y=1.05)

    fig.suptitle("Posterior predictive: $D_\\mathrm{cond}$ refit (pure water)", fontsize=8, y=1.04)
    fig.tight_layout(w_pad=1.5)
    save_fig(fig, "fig_refit_dcond_combined", FIGURES_DIR)


# ═══════════════════════════════════════════════════════════════════════════
# sigma_model sensitivity
# ═══════════════════════════════════════════════════════════════════════════

def sigma_sensitivity(data, sigmas=(1.0, 3.0, 5.0)):
    """
    Run sigma_model sensitivity test for one mode.
    Returns list of (sigma_model, ess, k_hat, D_cond_posterior_med, shrinkage_dict).
    """
    D_cond = data['D_cond_km']
    results = []
    for sm in sigmas:
        r = compute_posterior(data, D_obs=29.0, sigma_obs=10.0, sigma_model=sm)
        shrinks = {k: r['param_summaries'][k]['shrinkage'] for k in
                   [p[0] for p in TARGET_PARAMS]}
        results.append({
            'sigma_model': sm,
            'ess': r['ess'],
            'k_hat': r['k_hat'],
            'D_cond_post_med': r['D_cond_posterior_median'],
            'shrinkage': shrinks,
        })
    return results


# ═══════════════════════════════════════════════════════════════════════════
# CSV / NPZ output
# ═══════════════════════════════════════════════════════════════════════════

def write_summary_csv(all_results, filepath):
    """Write machine-readable posterior summary table."""
    fieldnames = [
        'scenario', 'obs_model', 'parameter', 'unit',
        'prior_median', 'prior_68_lo', 'prior_68_hi',
        'posterior_median', 'posterior_68_lo', 'posterior_68_hi',
        'posterior_95_lo', 'posterior_95_hi',
        'shrinkage', 'ess', 'k_hat', 'k_diagnostic',
    ]

    rows = []
    for mode_tag, obs_dict in all_results.items():
        for obs_key, r in obs_dict.items():
            for key in [p[0] for p in TARGET_PARAMS]:
                s = r['param_summaries'][key]
                rows.append({
                    'scenario': mode_tag,
                    'obs_model': obs_key,
                    'parameter': key,
                    'unit': s['unit'],
                    'prior_median': f"{s['prior_median']:.4f}",
                    'prior_68_lo': f"{s['prior_68_lo']:.4f}",
                    'prior_68_hi': f"{s['prior_68_hi']:.4f}",
                    'posterior_median': f"{s['posterior_median']:.4f}",
                    'posterior_68_lo': f"{s['posterior_68_lo']:.4f}",
                    'posterior_68_hi': f"{s['posterior_68_hi']:.4f}",
                    'posterior_95_lo': f"{s['posterior_95_lo']:.4f}",
                    'posterior_95_hi': f"{s['posterior_95_hi']:.4f}",
                    'shrinkage': f"{s['shrinkage']:.4f}",
                    'ess': f"{r['ess']:.1f}",
                    'k_hat': f"{r['k_hat']:.3f}",
                    'k_diagnostic': r['k_diagnostic'],
                })

            # Also add D_cond row
            D_cond = r.get('_D_cond', None)
            rows.append({
                'scenario': mode_tag,
                'obs_model': obs_key,
                'parameter': 'D_cond',
                'unit': 'km',
                'prior_median': f"{r['D_cond_prior_median']:.2f}",
                'prior_68_lo': '',
                'prior_68_hi': '',
                'posterior_median': f"{r['D_cond_posterior_median']:.2f}",
                'posterior_68_lo': f"{r['D_cond_posterior_68'][0]:.2f}",
                'posterior_68_hi': f"{r['D_cond_posterior_68'][1]:.2f}",
                'posterior_95_lo': '',
                'posterior_95_hi': '',
                'shrinkage': f"{r['D_cond_shrinkage']:.4f}",
                'ess': f"{r['ess']:.1f}",
                'k_hat': f"{r['k_hat']:.3f}",
                'k_diagnostic': r['k_diagnostic'],
            })

    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_posterior_npz(data, result, mode_tag, obs_key, output_dir):
    """Save posterior-resampled draws and weights."""
    w = result['weights']
    params = result['params']

    # Resample draws for downstream use
    n_resample = min(10000, len(w))
    idx = np.random.choice(len(w), size=n_resample, p=w, replace=True)

    save_dict = {
        'weights': w,
        'ess': np.array([result['ess']]),
        'k_hat': np.array([result['k_hat']]),
        'D_cond_posterior': data['D_cond_km'][idx],
        'D_cond_prior': data['D_cond_km'],
        'H_total_posterior': data['thicknesses_km'][idx],
    }
    for key in params:
        save_dict[f'{key}_posterior'] = params[key][idx]
        save_dict[f'{key}_prior'] = params[key]

    fname = f"posterior_{mode_tag}_{obs_key}.npz"
    np.savez_compressed(os.path.join(output_dir, fname), **save_dict)
    return fname


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Bayesian posterior refit against Juno MWR D_cond constraint."
    )
    parser.add_argument("--sigma-model", type=float, default=3.0,
                        help="Model discrepancy term (km). Default 3.0.")
    parser.add_argument("--sigma-sensitivity", action="store_true",
                        help="Run sigma_model sensitivity at 1, 3, 5 km.")
    parser.add_argument("--modes", nargs='*', default=None,
                        help="Subset of mode tags to run. Default: all.")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    np.random.seed(42)

    # Filter modes
    modes_to_run = MODES
    if args.modes:
        modes_to_run = [(f, l, t) for f, l, t in MODES if t in args.modes]

    all_results = {}  # {mode_tag: {obs_key: result_dict}}
    all_data = {}     # {mode_tag: npz_data}

    print("=" * 70)
    print("BAYESIAN POSTERIOR REFIT — Equatorial Juno Constraint")
    print(f"sigma_model = {args.sigma_model:.1f} km")
    print("=" * 70)

    for filename, mode_label, mode_tag in modes_to_run:
        filepath = os.path.join(RESULTS_DIR, filename)
        if not os.path.exists(filepath):
            print(f"\nSkipping {mode_label}: {filepath} not found")
            continue

        data = np.load(filepath)
        all_data[mode_tag] = data
        all_results[mode_tag] = {}

        print(f"\n{'#' * 60}")
        print(f"# {mode_label} (N = {len(data['D_cond_km']):,})")
        print(f"{'#' * 60}")

        for D_obs, sigma_obs, obs_key, obs_label in JUNO_OBS:
            print(f"\n  --- {obs_label} ---")

            r = compute_posterior(data, D_obs, sigma_obs, args.sigma_model)
            all_results[mode_tag][obs_key] = r

            # Print summary
            print(f"  Prior overlap (±2σ): {r['prior_overlap']:.1%}")
            print(f"  ESS: {r['ess']:.0f} / {r['n_samples']} ({r['ess_frac']:.1%})")
            print(f"  PSIS k-hat: {r['k_hat']:.3f} ({r['k_diagnostic']}), "
                  f"max-weight ratio: {r['max_w_ratio']:.1f}")
            print(f"  D_cond posterior median: {r['D_cond_posterior_median']:.1f} km "
                  f"(prior: {r['D_cond_prior_median']:.1f} km)")
            print(f"  D_cond shrinkage: {r['D_cond_shrinkage']:.0%}")

            print(f"\n  Parameter posteriors:")
            print(f"  {'Param':<12s} {'Prior med':>10s} {'Prior 68%':>18s} "
                  f"{'Post med':>10s} {'Post 68%':>18s} {'Shrink':>8s}")
            print(f"  {'-'*80}")
            for key in [p[0] for p in TARGET_PARAMS]:
                s = r['param_summaries'][key]
                print(f"  {key:<12s} "
                      f"{s['prior_median']:10.3f} "
                      f"[{s['prior_68_lo']:7.3f}, {s['prior_68_hi']:7.3f}] "
                      f"{s['posterior_median']:10.3f} "
                      f"[{s['posterior_68_lo']:7.3f}, {s['posterior_68_hi']:7.3f}] "
                      f"{s['shrinkage']:7.0%}")

            # Generate per-scenario figures
            tag = f"{mode_tag}_{obs_key}"
            fig_dcond_prior_posterior_likelihood(
                data, r, D_obs, sigma_obs, args.sigma_model,
                mode_label, obs_label, tag
            )
            fig_parameter_posteriors(data, r, mode_label, obs_label, tag)

            # Save posterior NPZ
            fname = write_posterior_npz(data, r, mode_tag, obs_key, OUTPUT_DIR)
            print(f"  Saved: {fname}")

    # sigma_model sensitivity
    if args.sigma_sensitivity:
        print(f"\n{'=' * 70}")
        print("SIGMA_MODEL SENSITIVITY (pure water, baseline)")
        print(f"{'=' * 70}")

        if 'baseline' in all_data:
            sens = sigma_sensitivity(all_data['baseline'])
            print(f"\n  {'σ_model':>8s} {'ESS':>8s} {'k_hat':>8s} {'D_cond post':>12s} ", end="")
            for key in [p[0] for p in TARGET_PARAMS]:
                print(f"  S({key})", end="")
            print()
            for row in sens:
                print(f"  {row['sigma_model']:8.1f} {row['ess']:8.0f} {row['k_hat']:8.3f} "
                      f"{row['D_cond_post_med']:12.1f}", end="")
                for key in [p[0] for p in TARGET_PARAMS]:
                    print(f"  {row['shrinkage'][key]:7.0%}", end="")
                print()

            # Also do moderate and depleted for comparison
            for mtag in ['moderate', 'depleted']:
                if mtag in all_data:
                    print(f"\n  --- {mtag} ---")
                    sens_m = sigma_sensitivity(all_data[mtag])
                    for row in sens_m:
                        print(f"  {row['sigma_model']:8.1f} {row['ess']:8.0f} {row['k_hat']:8.3f} "
                              f"{row['D_cond_post_med']:12.1f}", end="")
                        for key in [p[0] for p in TARGET_PARAMS]:
                            print(f"  {row['shrinkage'][key]:7.0%}", end="")
                        print()

    # Summary figures
    if len(all_results) >= 2:
        fig_shrinkage_summary(all_results)
        fig_combined_dcond_posteriors(all_results, all_data)

    # Write CSV
    csv_path = os.path.join(OUTPUT_DIR, "posterior_summary.csv")
    write_summary_csv(all_results, csv_path)
    print(f"\nSaved: {csv_path}")

    # Final diagnostic summary
    print(f"\n{'=' * 70}")
    print("DIAGNOSTIC SUMMARY")
    print(f"{'=' * 70}")
    for mode_tag, obs_dict in all_results.items():
        for obs_key, r in obs_dict.items():
            top_shrink = sorted(
                [(k, r['param_summaries'][k]['shrinkage']) for k in [p[0] for p in TARGET_PARAMS]],
                key=lambda x: -x[1]
            )
            meaningful = [(k, s) for k, s in top_shrink if s > 0.05]
            weak = [(k, s) for k, s in top_shrink if s <= 0.05]

            print(f"\n  {mode_tag} / {obs_key}:")
            print(f"    ESS = {r['ess']:.0f} ({r['ess_frac']:.0%}), "
                  f"PSIS k = {r['k_hat']:.3f} ({r['k_diagnostic']})")
            if meaningful:
                print(f"    Meaningfully constrained: "
                      + ", ".join(f"{k} ({s:.0%})" for k, s in meaningful))
            if weak:
                print(f"    Weakly/not constrained:   "
                      + ", ".join(f"{k} ({s:.0%})" for k, s in weak))

    print(f"\n{'=' * 70}")
    print(f"All outputs saved to:")
    print(f"  Results: {OUTPUT_DIR}")
    print(f"  Figures: {FIGURES_DIR}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
