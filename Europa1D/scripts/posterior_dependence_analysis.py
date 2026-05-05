#!/usr/bin/env python3
"""
Phase 1: Posterior covariance, identifiability, and information diagnostics.

Implements the refined Bayesian-Sobol framework (2026-03-25 roadmap):
  - Weighted posterior covariance and correlation matrices
  - Prior-normalized eigenanalysis (R matrix) for identifiability
  - KL divergence from importance weights (total + marginal)
  - Bootstrap confidence intervals on all quantities
  - Five diagnostic plots: correlation heatmap, pair plot, scree plot,
    loading plot, shrinkage waterfall

Works on any MC archive with param_* arrays and D_cond_km.
Importance weights are computed internally against Juno D_cond = 29 +/- 10 km.

Usage:
    python posterior_dependence_analysis.py midlat35_uniform_broad.npz
    python posterior_dependence_analysis.py eq_baseline_andrade.npz --tag equatorial
    python posterior_dependence_analysis.py archive.npz --n-bootstrap 2000
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import argparse
import json

import numpy as np
from scipy import stats
from scipy.linalg import sqrtm, inv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

from constants import Planetary


# ═══════════════════════════════════════════════════════════════════════════
# Juno observation model
# ═══════════════════════════════════════════════════════════════════════════

JUNO_D_OBS = 29.0       # km
JUNO_SIGMA_OBS = 10.0   # km
SIGMA_MODEL = 3.0        # km

# A priori decision thresholds (roadmap Table)
SHRINKAGE_THRESHOLDS = {'negligible': 0.10, 'weak': 0.30, 'moderate': 0.60}
CORRELATION_THRESHOLDS = {'negligible': 0.20, 'weak': 0.50, 'moderate': 0.70}
EIGENVALUE_THRESHOLDS = {'strong': 0.10, 'moderate': 0.50, 'weak': 0.90}
KL_THRESHOLDS = {'negligible': 0.05, 'weak': 0.20, 'moderate': 1.00}


# ═══════════════════════════════════════════════════════════════════════════
# Parameter extraction and transformation
# ═══════════════════════════════════════════════════════════════════════════

PARAM_LABELS = {
    'q_basal':   r'$q_\mathrm{basal}$ (mW/m$^2$)',
    'd_grain':   r'$\log_{10}\,d_\mathrm{grain}$ (m)',
    'Q_v':       r'$Q_v$ (kJ/mol)',
    'epsilon_0': r'$\log_{10}\,\varepsilon_0$',
    'T_surf':    r'$T_\mathrm{surf}$ (K)',
}

PARAM_SHORT = {
    'q_basal': 'q_bas',
    'd_grain': 'd_gr',
    'Q_v': 'Q_v',
    'epsilon_0': 'eps_0',
    'T_surf': 'T_s',
}


def extract_q_basal(data):
    """Reconstruct q_basal (W/m^2) from NPZ param arrays."""
    D_H2O = data['param_D_H2O']
    H_rad = data['param_H_rad']
    R_rock = Planetary.RADIUS - D_H2O
    M_rock = (4.0 / 3.0) * np.pi * (R_rock ** 3) * 3500.0
    q_rad = (H_rad * M_rock) / Planetary.AREA
    q_tidal = data['param_P_tidal'] / Planetary.AREA
    return q_rad + q_tidal


def extract_params(data):
    """Extract and transform the 5 target parameters to natural scale.

    Returns dict of {name: array} in transformed coordinates:
      q_basal:   mW/m^2 (linear, matches uniform prior)
      d_grain:   log10(meters) (matches lognormal prior)
      Q_v:       kJ/mol (linear, matches normal prior)
      epsilon_0: log10(dimensionless) (matches lognormal prior)
      T_surf:    K (linear, matches normal prior)
    """
    params = {}
    params['q_basal'] = extract_q_basal(data) * 1e3          # W/m^2 -> mW/m^2
    params['d_grain'] = np.log10(data['param_d_grain'])       # log10(m)
    params['Q_v'] = data['param_Q_v'] / 1e3                   # J/mol -> kJ/mol
    params['epsilon_0'] = np.log10(data['param_epsilon_0'])   # log10
    params['T_surf'] = data['param_T_surf']                   # K
    return params


def params_to_matrix(params, names):
    """Stack parameter arrays into (N, d) matrix."""
    return np.column_stack([params[n] for n in names])


# ═══════════════════════════════════════════════════════════════════════════
# Importance weighting
# ═══════════════════════════════════════════════════════════════════════════

def compute_weights(D_cond):
    """Gaussian importance weights for Juno likelihood."""
    sigma_total = np.sqrt(JUNO_SIGMA_OBS**2 + SIGMA_MODEL**2)
    log_w = -0.5 * ((D_cond - JUNO_D_OBS) / sigma_total) ** 2
    log_w -= np.max(log_w)
    w = np.exp(log_w)
    return w / w.sum()


def kish_ess(w):
    """Kish's effective sample size."""
    return 1.0 / np.sum(w**2)


def entropy_ess(w):
    """Entropy-based ESS = exp(H(w))."""
    w_pos = w[w > 0]
    return np.exp(-np.sum(w_pos * np.log(w_pos)))


def kl_from_weights(w):
    """KL(posterior || prior) = log(N) - log(ESS_entropy)."""
    N = len(w)
    return np.log(N) - np.log(entropy_ess(w))


# ═══════════════════════════════════════════════════════════════════════════
# Weighted statistics
# ═══════════════════════════════════════════════════════════════════════════

def weighted_mean(X, w):
    """Weighted mean of (N, d) matrix."""
    return np.sum(w[:, None] * X, axis=0)


def weighted_cov(X, w):
    """Weighted covariance matrix of (N, d) matrix."""
    mu = weighted_mean(X, w)
    delta = X - mu
    return (delta * w[:, None]).T @ delta


def weighted_corr(X, w):
    """Weighted Pearson correlation matrix."""
    C = weighted_cov(X, w)
    std = np.sqrt(np.diag(C))
    outer_std = np.outer(std, std)
    outer_std[outer_std == 0] = 1.0
    return C / outer_std


def weighted_spearman(X, w):
    """Weighted Spearman correlation via SIR resampling."""
    N_eff = int(kish_ess(w))
    N_resample = max(N_eff, 5000)
    idx = np.random.choice(len(w), size=N_resample, p=w, replace=True)
    X_resamp = X[idx]
    d = X_resamp.shape[1]
    R = np.eye(d)
    for i in range(d):
        for j in range(i + 1, d):
            rho, _ = stats.spearmanr(X_resamp[:, i], X_resamp[:, j])
            R[i, j] = rho
            R[j, i] = rho
    return R


def marginal_shrinkage(X, w):
    """Per-parameter shrinkage: 1 - Var_post/Var_prior."""
    d = X.shape[1]
    shrinkage = np.zeros(d)
    for i in range(d):
        var_prior = np.var(X[:, i])
        mu_post = np.sum(w * X[:, i])
        var_post = np.sum(w * (X[:, i] - mu_post)**2)
        shrinkage[i] = 1.0 - var_post / var_prior if var_prior > 0 else 0.0
    return shrinkage


def marginal_kl(X, w, n_bins=100):
    """Per-parameter marginal KL(posterior || prior) via histogram."""
    d = X.shape[1]
    kl_vals = np.zeros(d)
    for i in range(d):
        x = X[:, i]
        lo, hi = np.min(x), np.max(x)
        edges = np.linspace(lo - 1e-10, hi + 1e-10, n_bins + 1)
        # Prior histogram (unweighted)
        prior_counts, _ = np.histogram(x, bins=edges)
        prior_p = prior_counts / prior_counts.sum()
        # Posterior histogram (weighted)
        post_counts = np.zeros(n_bins)
        bin_idx = np.digitize(x, edges) - 1
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)
        for k in range(len(x)):
            post_counts[bin_idx[k]] += w[k]
        post_p = post_counts / post_counts.sum()
        # KL with smoothing to avoid log(0)
        eps = 1e-10
        prior_p = np.maximum(prior_p, eps)
        post_p = np.maximum(post_p, eps)
        # Re-normalize
        prior_p /= prior_p.sum()
        post_p /= post_p.sum()
        kl_vals[i] = np.sum(post_p * np.log(post_p / prior_p))
    return kl_vals


# ═══════════════════════════════════════════════════════════════════════════
# Prior-normalized eigenanalysis
# ═══════════════════════════════════════════════════════════════════════════

def prior_normalized_eigen(Sigma_post, Sigma_prior):
    """Compute R = Sigma_prior^{-1/2} Sigma_post Sigma_prior^{-1/2}.

    Returns eigenvalues (ascending) and eigenvectors in parameter space.
    """
    Sigma_prior_sqrt = sqrtm(Sigma_prior).real
    Sigma_prior_inv_sqrt = inv(Sigma_prior_sqrt)

    R = Sigma_prior_inv_sqrt @ Sigma_post @ Sigma_prior_inv_sqrt

    eigenvalues, eigenvectors_norm = np.linalg.eigh(R)

    # Sort ascending
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors_norm = eigenvectors_norm[:, idx]

    # Transform back to parameter space
    eigenvectors_param = Sigma_prior_sqrt @ eigenvectors_norm

    return eigenvalues, eigenvectors_param, R


# ═══════════════════════════════════════════════════════════════════════════
# Bootstrap
# ═══════════════════════════════════════════════════════════════════════════

def bootstrap_diagnostics(X, w, n_boot=1000, seed=42):
    """Bootstrap CIs via SIR resampling.

    Procedure (standard Bayesian bootstrap for importance sampling):
      1. SIR resample N indices according to weights w
      2. Compute unweighted statistics on the resampled data
      3. Prior covariance is held fixed (computed once from full sample)
      4. Repeat B times
    """
    rng = np.random.default_rng(seed)
    N, d = X.shape

    shrinkage_boot = np.zeros((n_boot, d))
    corr_boot = np.zeros((n_boot, d, d))
    eigenvalue_boot = np.zeros((n_boot, d))
    kl_total_boot = np.zeros(n_boot)
    kl_marginal_boot = np.zeros((n_boot, d))

    # Prior covariance is fixed (not bootstrapped)
    Sigma_prior = np.cov(X, rowvar=False)
    w_uniform = np.ones(N) / N

    for b in range(n_boot):
        # SIR: resample indices weighted by w, then treat as unweighted
        idx = rng.choice(N, size=N, p=w, replace=True)
        X_post = X[idx]

        # Shrinkage: posterior variance (from SIR) vs fixed prior variance
        for i in range(d):
            var_prior = np.var(X[:, i])
            var_post = np.var(X_post[:, i])
            shrinkage_boot[b, i] = 1.0 - var_post / var_prior if var_prior > 0 else 0.0

        # Correlation on posterior samples
        corr_boot[b] = np.corrcoef(X_post, rowvar=False)

        # Eigenanalysis: fixed prior, bootstrapped posterior
        Sigma_post_b = np.cov(X_post, rowvar=False)
        try:
            evals_b, _, _ = prior_normalized_eigen(Sigma_post_b, Sigma_prior)
            eigenvalue_boot[b] = evals_b
        except Exception:
            eigenvalue_boot[b] = np.nan

        # KL and marginal KL: bootstrap (X, w) pairs together so parameter
        # rows stay aligned with their weights. Previous version shuffled w
        # independently of X, which broke the pairing and artificially
        # narrowed the CIs on marginal_kl.
        idx_pair = rng.choice(N, size=N, replace=True)
        X_b = X[idx_pair]
        w_b = w[idx_pair]
        w_b = w_b / w_b.sum()
        kl_total_boot[b] = kl_from_weights(w_b)
        kl_marginal_boot[b] = marginal_kl(X_b, w_b)

    return {
        'shrinkage': shrinkage_boot,
        'corr': corr_boot,
        'eigenvalues': eigenvalue_boot,
        'kl_total': kl_total_boot,
        'kl_marginal': kl_marginal_boot,
    }


def ci_95(arr, axis=0):
    """2.5th and 97.5th percentiles."""
    return np.nanpercentile(arr, [2.5, 97.5], axis=axis)


# ═══════════════════════════════════════════════════════════════════════════
# Hypothesis discrimination
# ═══════════════════════════════════════════════════════════════════════════

def classify_hypothesis(shrinkage, eigenvalues, kl_total, corr_matrix):
    """Classify result against the four competing hypotheses."""
    max_shrink = np.max(shrinkage)
    n_strong_shrink = np.sum(shrinkage > 0.40)
    max_corr = np.max(np.abs(corr_matrix - np.eye(len(shrinkage))))
    min_eigenvalue = np.min(eigenvalues)
    eigenvalue_range = np.max(eigenvalues) / max(np.min(eigenvalues), 1e-10)

    verdicts = []

    # H3: Weak information
    if kl_total < KL_THRESHOLDS['negligible'] and max_shrink < 0.10:
        verdicts.append(('H3', 'Weak information',
                         f'KL={kl_total:.3f} nats, max shrinkage={max_shrink:.1%}'))

    # H1: Single-parameter dominance
    if n_strong_shrink == 1 and max_corr < 0.50:
        verdicts.append(('H1', 'Single-parameter dominance',
                         f'1 param >40% shrinkage, max |r|={max_corr:.2f}'))

    # H2: Trade-off manifold
    if max_corr > 0.50 and min_eigenvalue < 0.30:
        verdicts.append(('H2', 'Trade-off manifold',
                         f'max |r|={max_corr:.2f}, min eigenvalue={min_eigenvalue:.3f}'))

    # H4: Hierarchical identifiability
    if eigenvalue_range > 10 and max_shrink > 0.10:
        verdicts.append(('H4', 'Hierarchical identifiability',
                         f'eigenvalue range={eigenvalue_range:.0f}x, '
                         f'max shrinkage={max_shrink:.1%}'))

    if not verdicts:
        verdicts.append(('??', 'Ambiguous',
                         f'KL={kl_total:.3f}, max_shrink={max_shrink:.1%}, '
                         f'max_corr={max_corr:.2f}'))

    return verdicts


# ═══════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════

def setup_style():
    """Publication-quality matplotlib defaults."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 8,
        'axes.labelsize': 9,
        'axes.titlesize': 10,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 7,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })


def plot_correlation_heatmap(corr, names, outpath):
    """Lower-triangle Spearman correlation heatmap."""
    d = len(names)
    mask = np.triu(np.ones((d, d), dtype=bool))

    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    im = ax.imshow(np.ma.array(corr, mask=mask), cmap='RdBu_r',
                   vmin=-1, vmax=1, aspect='equal')

    # Annotate cells
    for i in range(d):
        for j in range(d):
            if not mask[i, j]:
                color = 'white' if abs(corr[i, j]) > 0.6 else 'black'
                ax.text(j, i, f'{corr[i,j]:.2f}', ha='center', va='center',
                        fontsize=8, color=color, fontweight='bold')

    short = [PARAM_SHORT.get(n, n) for n in names]
    ax.set_xticks(range(d))
    ax.set_xticklabels(short, rotation=45, ha='right')
    ax.set_yticks(range(d))
    ax.set_yticklabels(short)
    ax.set_title('Posterior Spearman correlation')

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, label='Spearman r')
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  Saved: {outpath}")


def plot_pair(X, w, names, outpath, top_pairs=3):
    """Prior-posterior pair plot for the most correlated parameter pairs."""
    d = X.shape[1]

    # Find top pairs by absolute posterior correlation
    corr = weighted_corr(X, w)
    pairs = []
    for i in range(d):
        for j in range(i + 1, d):
            pairs.append((abs(corr[i, j]), i, j))
    pairs.sort(reverse=True)
    top = pairs[:top_pairs]

    fig, axes = plt.subplots(1, len(top), figsize=(4 * len(top), 3.5))
    if len(top) == 1:
        axes = [axes]

    # SIR resample for posterior scatter
    N_eff = int(kish_ess(w))
    N_resamp = min(max(N_eff, 3000), 10000)
    idx_post = np.random.choice(len(w), size=N_resamp, p=w, replace=True)

    for ax, (r_val, i, j) in zip(axes, top):
        # Prior (gray)
        N_prior_show = min(3000, len(X))
        idx_prior = np.random.choice(len(X), size=N_prior_show, replace=False)
        ax.scatter(X[idx_prior, i], X[idx_prior, j],
                   s=1, alpha=0.15, color='gray', rasterized=True, label='Prior')

        # Posterior (colored)
        ax.scatter(X[idx_post, i], X[idx_post, j],
                   s=2, alpha=0.25, color='#0072B2', rasterized=True, label='Posterior')

        # Credible ellipses (50%, 90%)
        _draw_credible_ellipse(X[idx_post, i], X[idx_post, j], ax,
                               n_std=1.177, color='#D55E00', label='50%')  # chi2(2).ppf(0.5)
        _draw_credible_ellipse(X[idx_post, i], X[idx_post, j], ax,
                               n_std=2.146, color='#D55E00', label='90%',  # chi2(2).ppf(0.9)
                               linestyle='--')

        short = [PARAM_SHORT.get(n, n) for n in names]
        ax.set_xlabel(PARAM_LABELS.get(names[i], names[i]))
        ax.set_ylabel(PARAM_LABELS.get(names[j], names[j]))
        ax.set_title(f'{short[i]} vs {short[j]}  (r={r_val:.2f})')
        ax.legend(fontsize=6, loc='upper right', markerscale=3)

    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  Saved: {outpath}")


def _draw_credible_ellipse(x, y, ax, n_std, color, label='', linestyle='-'):
    """Draw a covariance ellipse on ax."""
    cov = np.cov(x, y)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    width = 2 * n_std * np.sqrt(eigvals[0])
    height = 2 * n_std * np.sqrt(eigvals[1])

    ellipse = Ellipse(xy=(np.mean(x), np.mean(y)),
                      width=width, height=height, angle=angle,
                      edgecolor=color, facecolor='none',
                      linewidth=1.5, linestyle=linestyle, label=label)
    ax.add_patch(ellipse)


def plot_scree(eigenvalues, eigenvalue_ci, names, outpath):
    """Scree plot of prior-normalized eigenvalues with threshold lines."""
    d = len(eigenvalues)
    x = np.arange(1, d + 1)

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(x, eigenvalues, color='#0072B2', alpha=0.7, width=0.6)

    # Bootstrap CIs
    if eigenvalue_ci is not None:
        err_lo = np.maximum(eigenvalues - eigenvalue_ci[0], 0)
        err_hi = np.maximum(eigenvalue_ci[1] - eigenvalues, 0)
        ax.errorbar(x, eigenvalues, yerr=[err_lo, err_hi],
                    fmt='none', color='black', capsize=3, linewidth=1)

    # Threshold lines
    ax.axhline(0.1, color='green', ls='--', lw=0.8, alpha=0.7, label='Strong (0.1)')
    ax.axhline(0.5, color='orange', ls='--', lw=0.8, alpha=0.7, label='Moderate (0.5)')
    ax.axhline(0.9, color='red', ls='--', lw=0.8, alpha=0.7, label='Weak (0.9)')

    ax.set_xlabel('Eigendirection')
    ax.set_ylabel(r'$\rho_i$ (posterior/prior variance ratio)')
    ax.set_title('Prior-normalized eigenvalues')
    ax.set_xticks(x)
    ax.set_ylim(0, max(1.1, np.max(eigenvalues) * 1.1))
    ax.legend(fontsize=6, title='Identifiability threshold')

    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  Saved: {outpath}")


def plot_loadings(eigenvectors, eigenvalues, names, outpath):
    """Loading plot for the leading 1-2 eigenvectors."""
    d = len(names)
    n_show = min(2, d)

    fig, axes = plt.subplots(1, n_show, figsize=(3.5 * n_show, 3))
    if n_show == 1:
        axes = [axes]

    short = [PARAM_SHORT.get(n, n) for n in names]
    colors = ['#0072B2', '#D55E00']

    for k, ax in enumerate(axes):
        loadings = eigenvectors[:, k]
        # Normalize to unit length for display
        loadings = loadings / np.linalg.norm(loadings)

        bars = ax.barh(range(d), loadings, color=colors[k], alpha=0.7)
        ax.set_yticks(range(d))
        ax.set_yticklabels(short)
        ax.axvline(0, color='gray', lw=0.5)
        ax.set_xlabel('Loading')
        ax.set_title(f'Direction {k+1}  ($\\rho$={eigenvalues[k]:.3f})')

        # Annotate values
        for i, (v, bar) in enumerate(zip(loadings, bars)):
            ha = 'left' if v >= 0 else 'right'
            offset = 0.02 if v >= 0 else -0.02
            ax.text(v + offset, i, f'{v:.2f}', ha=ha, va='center', fontsize=7)

    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  Saved: {outpath}")


def plot_shrinkage_waterfall(shrinkage, shrinkage_ci, names, outpath):
    """Shrinkage waterfall sorted by magnitude, with bootstrap CIs."""
    d = len(shrinkage)
    idx = np.argsort(shrinkage)[::-1]

    fig, ax = plt.subplots(figsize=(4.5, 3))
    x = np.arange(d)

    short = [PARAM_SHORT.get(names[i], names[i]) for i in idx]
    vals = shrinkage[idx]

    # Color by threshold
    colors = []
    for v in vals:
        if v > SHRINKAGE_THRESHOLDS['moderate']:
            colors.append('#009E73')   # strong (green)
        elif v > SHRINKAGE_THRESHOLDS['weak']:
            colors.append('#E69F00')   # moderate (amber)
        elif v > SHRINKAGE_THRESHOLDS['negligible']:
            colors.append('#56B4E9')   # weak (blue)
        else:
            colors.append('#999999')   # negligible (gray)

    ax.bar(x, vals * 100, color=colors, alpha=0.8, width=0.6)

    # Bootstrap CIs
    if shrinkage_ci is not None:
        ci_lo = shrinkage_ci[0, idx] * 100
        ci_hi = shrinkage_ci[1, idx] * 100
        err_lo = np.maximum(vals * 100 - ci_lo, 0)
        err_hi = np.maximum(ci_hi - vals * 100, 0)
        ax.errorbar(x, vals * 100, yerr=[err_lo, err_hi],
                    fmt='none', color='black', capsize=3, linewidth=1)

    # Threshold lines
    ax.axhline(10, color='gray', ls=':', lw=0.8, label='10% (negligible)')
    ax.axhline(30, color='orange', ls=':', lw=0.8, label='30% (weak)')

    ax.set_xticks(x)
    ax.set_xticklabels(short, rotation=45, ha='right')
    ax.set_ylabel('Marginal shrinkage (%)')
    ax.set_title('Prior-to-posterior shrinkage')
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=6)

    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  Saved: {outpath}")


# ═══════════════════════════════════════════════════════════════════════════
# Main analysis
# ═══════════════════════════════════════════════════════════════════════════

def run_analysis(npz_path, tag, n_bootstrap, out_dir):
    """Run the full Phase 1 analysis on one MC archive."""
    print(f"\n{'='*60}")
    print(f"  Phase 1: Posterior Dependence Analysis")
    print(f"  Archive: {npz_path}")
    print(f"  Tag: {tag}")
    print(f"{'='*60}")

    # Load data
    data = np.load(npz_path)
    D_cond = data['D_cond_km']
    H_total = data['thicknesses_km']
    N = len(D_cond)
    print(f"\n  Samples: {N}")

    # Extract parameters
    params = extract_params(data)
    names = list(params.keys())
    d = len(names)
    X = params_to_matrix(params, names)
    print(f"  Parameters: {names}")

    # Compute weights
    w = compute_weights(D_cond)
    ess_kish = kish_ess(w)
    ess_entropy = entropy_ess(w)
    kl_total = kl_from_weights(w)
    print(f"\n  ESS (Kish):    {ess_kish:.0f} / {N} ({100*ess_kish/N:.1f}%)")
    print(f"  ESS (entropy): {ess_entropy:.0f} / {N} ({100*ess_entropy/N:.1f}%)")
    print(f"  KL(post||prior): {kl_total:.4f} nats")

    # Weighted statistics
    Sigma_post = weighted_cov(X, w)
    Sigma_prior = np.cov(X, rowvar=False)
    corr_pearson = weighted_corr(X, w)

    np.random.seed(42)
    corr_spearman = weighted_spearman(X, w)

    shrinkage = marginal_shrinkage(X, w)
    kl_marginals = marginal_kl(X, w)

    print(f"\n  Marginal shrinkage:")
    for i, n in enumerate(names):
        short = PARAM_SHORT.get(n, n)
        print(f"    {short:>6s}: {shrinkage[i]:6.1%}  "
              f"(KL_marginal = {kl_marginals[i]:.4f} nats)")

    # Eigenanalysis
    eigenvalues, eigenvectors, R = prior_normalized_eigen(Sigma_post, Sigma_prior)

    print(f"\n  Prior-normalized eigenvalues (R):")
    for i, ev in enumerate(eigenvalues):
        reduction = (1 - ev) * 100
        label = ('STRONG' if ev < 0.1 else
                 'moderate' if ev < 0.5 else
                 'weak' if ev < 0.9 else
                 'negligible')
        print(f"    direction {i+1}: rho={ev:.4f}  "
              f"(variance reduction {reduction:.1f}%, {label})")

    print(f"\n  Leading eigenvector (most constrained direction):")
    v0 = eigenvectors[:, 0] / np.linalg.norm(eigenvectors[:, 0])
    for i, n in enumerate(names):
        short = PARAM_SHORT.get(n, n)
        print(f"    {short:>6s}: {v0[i]:+.3f}")

    # Posterior correlations
    print(f"\n  Posterior Spearman correlations (|r| > 0.2):")
    for i in range(d):
        for j in range(i + 1, d):
            r = corr_spearman[i, j]
            if abs(r) > 0.2:
                si = PARAM_SHORT.get(names[i], names[i])
                sj = PARAM_SHORT.get(names[j], names[j])
                print(f"    {si} vs {sj}: r = {r:+.3f}")

    # Hypothesis discrimination
    verdicts = classify_hypothesis(shrinkage, eigenvalues, kl_total, corr_spearman)
    print(f"\n  Hypothesis verdicts:")
    for hyp, desc, evidence in verdicts:
        print(f"    {hyp}: {desc}")
        print(f"         {evidence}")

    # Bootstrap
    print(f"\n  Running {n_bootstrap} bootstrap resamples ...")
    np.random.seed(42)
    boot = bootstrap_diagnostics(X, w, n_boot=n_bootstrap)

    shrinkage_ci = ci_95(boot['shrinkage'])
    eigenvalue_ci = ci_95(boot['eigenvalues'])
    kl_total_ci = ci_95(boot['kl_total'])
    kl_marginal_ci = ci_95(boot['kl_marginal'])

    print(f"  KL total: {kl_total:.4f}  "
          f"95% CI [{kl_total_ci[0]:.4f}, {kl_total_ci[1]:.4f}]")

    print(f"\n  Shrinkage with 95% CI:")
    for i, n in enumerate(names):
        short = PARAM_SHORT.get(n, n)
        print(f"    {short:>6s}: {shrinkage[i]:6.1%}  "
              f"[{shrinkage_ci[0,i]:6.1%}, {shrinkage_ci[1,i]:6.1%}]")

    print(f"\n  Eigenvalues with 95% CI:")
    for i, ev in enumerate(eigenvalues):
        print(f"    dir {i+1}: {ev:.4f}  "
              f"[{eigenvalue_ci[0,i]:.4f}, {eigenvalue_ci[1,i]:.4f}]")

    # Save numerical results
    os.makedirs(out_dir, exist_ok=True)

    # CSV: weighted correlation
    corr_csv = os.path.join(out_dir, f'{tag}_posterior_correlation.csv')
    header = ','.join(names)
    np.savetxt(corr_csv, corr_spearman, delimiter=',', header=header,
               fmt='%.4f', comments='')
    print(f"\n  Saved: {corr_csv}")

    # CSV: covariance
    cov_csv = os.path.join(out_dir, f'{tag}_posterior_covariance.csv')
    np.savetxt(cov_csv, Sigma_post, delimiter=',', header=header,
               fmt='%.6e', comments='')
    print(f"  Saved: {cov_csv}")

    # JSON: summary
    summary = {
        'archive': os.path.basename(npz_path),
        'tag': tag,
        'N': int(N),
        'ESS_kish': float(ess_kish),
        'ESS_entropy': float(ess_entropy),
        'KL_total_nats': float(kl_total),
        'KL_total_95ci': [float(kl_total_ci[0]), float(kl_total_ci[1])],
        'juno_D_obs_km': JUNO_D_OBS,
        'juno_sigma_obs_km': JUNO_SIGMA_OBS,
        'sigma_model_km': SIGMA_MODEL,
        'parameters': names,
        'marginal_shrinkage': {n: float(shrinkage[i]) for i, n in enumerate(names)},
        'marginal_shrinkage_95ci': {
            n: [float(shrinkage_ci[0, i]), float(shrinkage_ci[1, i])]
            for i, n in enumerate(names)
        },
        'marginal_KL_nats': {n: float(kl_marginals[i]) for i, n in enumerate(names)},
        'eigenvalues_R': [float(ev) for ev in eigenvalues],
        'eigenvalues_R_95ci': [
            [float(eigenvalue_ci[0, i]), float(eigenvalue_ci[1, i])]
            for i in range(d)
        ],
        'leading_eigenvector': {n: float(v0[i]) for i, n in enumerate(names)},
        'hypothesis_verdicts': [
            {'hypothesis': h, 'description': desc, 'evidence': ev}
            for h, desc, ev in verdicts
        ],
    }
    summary_path = os.path.join(out_dir, f'{tag}_dependence_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {summary_path}")

    # Plots
    setup_style()
    fig_dir = os.path.join(out_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    np.random.seed(42)

    plot_correlation_heatmap(
        corr_spearman, names,
        os.path.join(fig_dir, f'{tag}_correlation_heatmap.png'))

    plot_pair(
        X, w, names,
        os.path.join(fig_dir, f'{tag}_pair_plot.png'))

    plot_scree(
        eigenvalues, eigenvalue_ci, names,
        os.path.join(fig_dir, f'{tag}_scree_plot.png'))

    plot_loadings(
        eigenvectors, eigenvalues, names,
        os.path.join(fig_dir, f'{tag}_loading_plot.png'))

    plot_shrinkage_waterfall(
        shrinkage, shrinkage_ci, names,
        os.path.join(fig_dir, f'{tag}_shrinkage_waterfall.png'))

    print(f"\n{'='*60}")
    print(f"  Phase 1 COMPLETE: {tag}")
    print(f"  Results in: {os.path.abspath(out_dir)}")
    print(f"{'='*60}\n")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1: Posterior dependence and identifiability analysis")
    parser.add_argument("archive", help="Path to MC archive NPZ file")
    parser.add_argument("--tag", default=None,
                        help="Output tag (default: derived from filename)")
    parser.add_argument("--n-bootstrap", type=int, default=1000,
                        help="Number of bootstrap resamples (default 1000)")
    parser.add_argument("--out-dir", default=None,
                        help="Output directory (default: results/identifiability/)")
    args = parser.parse_args()

    # Resolve archive path
    archive_path = args.archive
    if not os.path.isabs(archive_path):
        # Try relative to results/
        results_base = os.path.join(os.path.dirname(__file__), '..', 'results')
        candidates = [
            archive_path,
            os.path.join(results_base, archive_path),
            os.path.join(results_base, 'midlat_juno', archive_path),
        ]
        for c in candidates:
            if os.path.exists(c):
                archive_path = c
                break

    if not os.path.exists(archive_path):
        print(f"ERROR: Archive not found: {archive_path}")
        sys.exit(1)

    tag = args.tag or os.path.splitext(os.path.basename(archive_path))[0]
    out_dir = args.out_dir or os.path.join(
        os.path.dirname(__file__), '..', 'results', 'identifiability')

    run_analysis(archive_path, tag, args.n_bootstrap, out_dir)


if __name__ == "__main__":
    main()
