#!/usr/bin/env python3
"""
Phase 3: Posterior sensitivity via Shapley effects.

Computes Shapley values for the posterior-predictive variance of D_cond,
correctly handling posterior dependence between parameters.

For d=5 parameters, requires only 2^5 = 32 coalition evaluations.

Also computes prior Sobol-like indices (from unweighted samples) for
comparison: how does calibration reshuffle parameter importance?

Usage:
    python posterior_sensitivity_shapley.py midlat35_uniform_broad.npz
    python posterior_sensitivity_shapley.py eq_baseline_andrade.npz --tag equatorial
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import argparse
import json
import itertools

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from constants import Planetary

JUNO_D_OBS = 29.0
JUNO_SIGMA_OBS = 10.0
SIGMA_MODEL = 3.0

PARAM_SHORT = {
    'q_basal': 'q_bas', 'd_grain': 'd_gr', 'Q_v': 'Q_v',
    'epsilon_0': 'eps_0', 'T_surf': 'T_s',
}

PARAM_LABELS = {
    'q_basal': r'$q_\mathrm{basal}$',
    'd_grain': r'$d_\mathrm{grain}$',
    'Q_v': r'$Q_v$',
    'epsilon_0': r'$\varepsilon_0$',
    'T_surf': r'$T_\mathrm{surf}$',
}


def extract_q_basal(data):
    D_H2O = data['param_D_H2O']
    H_rad = data['param_H_rad']
    R_rock = Planetary.RADIUS - D_H2O
    M_rock = (4.0 / 3.0) * np.pi * (R_rock ** 3) * 3500.0
    q_rad = (H_rad * M_rock) / Planetary.AREA
    q_tidal = data['param_P_tidal'] / Planetary.AREA
    return q_rad + q_tidal


def extract_params(data):
    params = {}
    params['q_basal'] = extract_q_basal(data) * 1e3
    params['d_grain'] = np.log10(data['param_d_grain'])
    params['Q_v'] = data['param_Q_v'] / 1e3
    params['epsilon_0'] = np.log10(data['param_epsilon_0'])
    params['T_surf'] = data['param_T_surf']
    return params


def compute_weights(D_cond):
    sigma_total = np.sqrt(JUNO_SIGMA_OBS**2 + SIGMA_MODEL**2)
    log_w = -0.5 * ((D_cond - JUNO_D_OBS) / sigma_total) ** 2
    log_w -= np.max(log_w)
    w = np.exp(log_w)
    return w / w.sum()


def kish_ess(w):
    return 1.0 / np.sum(w**2)


def coalition_value(X, Y, coalition_idx, n_bins=30):
    """Compute c(J) = Var(E[Y | X_J]) via binning.

    For each unique combination of binned values of X_J,
    compute the conditional mean of Y, then take the variance
    of those conditional means weighted by bin counts.
    """
    N = len(Y)

    if len(coalition_idx) == 0:
        return 0.0

    # Bin each coalition variable
    bin_assignments = np.zeros((N, len(coalition_idx)), dtype=int)
    for k, idx in enumerate(coalition_idx):
        x = X[:, idx]
        edges = np.percentile(x, np.linspace(0, 100, n_bins + 1))
        edges[0] -= 1e-10
        edges[-1] += 1e-10
        # Merge duplicate edges
        edges = np.unique(edges)
        if len(edges) < 2:
            bin_assignments[:, k] = 0
        else:
            bin_assignments[:, k] = np.clip(
                np.digitize(x, edges) - 1, 0, len(edges) - 2)

    # Multi-dimensional bin index
    # Hash bins into a single key per sample
    multipliers = np.ones(len(coalition_idx), dtype=int)
    for k in range(1, len(coalition_idx)):
        multipliers[k] = multipliers[k-1] * (bin_assignments[:, k-1].max() + 1)
    bin_keys = np.sum(bin_assignments * multipliers, axis=1)

    # Compute conditional means
    unique_keys = np.unique(bin_keys)
    cond_means = np.zeros(len(unique_keys))
    cond_counts = np.zeros(len(unique_keys))

    for i, key in enumerate(unique_keys):
        mask = bin_keys == key
        cond_means[i] = np.mean(Y[mask])
        cond_counts[i] = np.sum(mask)

    # Var(E[Y|X_J]) = weighted variance of conditional means
    total = np.sum(cond_counts)
    weights = cond_counts / total
    grand_mean = np.sum(weights * cond_means)
    var_cond_mean = np.sum(weights * (cond_means - grand_mean)**2)

    return var_cond_mean


def compute_shapley(X, Y, d, n_bins=30):
    """Compute Shapley effects for all d variables.

    Sh_i = (1/d) * sum over J subset {1..d}\\{i}
            [binom(d-1, |J|)]^{-1} * [c(J+{i}) - c(J)]

    For d=5: 2^5 = 32 coalition evaluations.
    """
    from math import comb

    # Precompute all coalition values
    all_subsets = {}
    for r in range(d + 1):
        for subset in itertools.combinations(range(d), r):
            key = frozenset(subset)
            all_subsets[key] = coalition_value(X, Y, list(subset), n_bins)

    total_var = np.var(Y)

    # Compute Shapley values
    shapley = np.zeros(d)
    for i in range(d):
        others = [j for j in range(d) if j != i]
        sh_i = 0.0
        for r in range(d):  # |J| from 0 to d-1
            for J in itertools.combinations(others, r):
                J_set = frozenset(J)
                J_plus_i = frozenset(J) | {i}
                marginal = all_subsets[J_plus_i] - all_subsets[J_set]
                weight = 1.0 / (d * comb(d - 1, r))
                sh_i += weight * marginal
        shapley[i] = sh_i

    return shapley, total_var


def bootstrap_shapley(X, Y, d, n_boot=200, n_bins=30, seed=42):
    """Bootstrap CIs on Shapley values."""
    rng = np.random.default_rng(seed)
    N = len(Y)
    shapley_boot = np.zeros((n_boot, d))

    for b in range(n_boot):
        idx = rng.choice(N, size=N, replace=True)
        sh_b, _ = compute_shapley(X[idx], Y[idx], d, n_bins)
        shapley_boot[b] = sh_b

    return shapley_boot


def plot_shapley_comparison(prior_sh, post_sh, prior_var, post_var,
                            names, post_ci, outpath):
    """Prior (unweighted) vs posterior Shapley comparison."""
    d = len(names)
    x = np.arange(d)
    width = 0.35

    # Normalize to percentages
    prior_pct = (prior_sh / prior_var) * 100 if prior_var > 0 else prior_sh * 0
    post_pct = (post_sh / post_var) * 100 if post_var > 0 else post_sh * 0
    post_ci_pct = (post_ci / post_var) * 100 if post_var > 0 else post_ci * 0

    # Sort by posterior importance
    idx = np.argsort(post_pct)[::-1]

    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    short = [PARAM_SHORT.get(names[i], names[i]) for i in idx]

    ax.bar(x - width/2, prior_pct[idx], width, label='Prior (uncalibrated)',
           color='#56B4E9', alpha=0.7)
    ax.bar(x + width/2, post_pct[idx], width, label='Posterior (calibrated)',
           color='#D55E00', alpha=0.7)

    # CIs on posterior
    err_lo = np.maximum(post_pct[idx] - post_ci_pct[0, idx], 0)
    err_hi = np.maximum(post_ci_pct[1, idx] - post_pct[idx], 0)
    ax.errorbar(x + width/2, post_pct[idx], yerr=[err_lo, err_hi],
                fmt='none', color='black', capsize=3, linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(short)
    ax.set_ylabel('Shapley effect (% of Var)')
    ax.set_title('Prior vs posterior parameter importance')
    ax.legend(fontsize=7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {outpath}")


def main():
    parser = argparse.ArgumentParser(
        description="Phase 3: Posterior Shapley effects")
    parser.add_argument("archive", help="MC archive NPZ")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--n-bootstrap", type=int, default=200)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    archive_path = args.archive
    if not os.path.isabs(archive_path):
        results_base = os.path.join(os.path.dirname(__file__), '..', 'results')
        for c in [archive_path,
                  os.path.join(results_base, archive_path),
                  os.path.join(results_base, 'midlat_juno', archive_path)]:
            if os.path.exists(c):
                archive_path = c
                break

    tag = args.tag or os.path.splitext(os.path.basename(archive_path))[0]
    out_dir = args.out_dir or os.path.join(
        os.path.dirname(__file__), '..', 'results', 'identifiability')

    print(f"\n{'='*60}")
    print(f"  Phase 3: Shapley Effects")
    print(f"  Archive: {archive_path}")
    print(f"{'='*60}")

    data = np.load(archive_path)
    D_cond = data['D_cond_km']
    H_total = data['thicknesses_km']
    params = extract_params(data)
    names = list(params.keys())
    X = np.column_stack([params[n] for n in names])
    w = compute_weights(D_cond)
    N = len(D_cond)
    d = len(names)

    # SIR resample for posterior
    N_eff = int(kish_ess(w))
    N_post = min(max(N_eff, 5000), 15000)
    print(f"  SIR resampling: {N_post} posterior samples (ESS={N_eff})")

    np.random.seed(42)
    idx_post = np.random.choice(N, size=N_post, p=w, replace=True)
    X_post = X[idx_post]
    D_cond_post = D_cond[idx_post]

    # Prior Shapley (unweighted)
    print(f"  Computing prior Shapley effects (32 coalitions) ...")
    prior_sh, prior_var = compute_shapley(X, D_cond, d)
    print(f"  Prior Var(D_cond) = {prior_var:.2f} km^2")
    print(f"  Prior Shapley (% of Var):")
    for i, n in enumerate(names):
        pct = (prior_sh[i] / prior_var * 100) if prior_var > 0 else 0
        print(f"    {PARAM_SHORT[n]:>6s}: {pct:5.1f}%")

    # Posterior Shapley (on SIR-resampled data)
    print(f"\n  Computing posterior Shapley effects ...")
    post_sh, post_var = compute_shapley(X_post, D_cond_post, d)
    print(f"  Posterior Var(D_cond) = {post_var:.2f} km^2")
    print(f"  Posterior Shapley (% of Var):")
    for i, n in enumerate(names):
        pct = (post_sh[i] / post_var * 100) if post_var > 0 else 0
        print(f"    {PARAM_SHORT[n]:>6s}: {pct:5.1f}%")

    # Variance reduction
    var_reduction = (1 - post_var / prior_var) * 100 if prior_var > 0 else 0
    print(f"\n  Total variance reduction: {var_reduction:.1f}%")

    # Bootstrap CIs on posterior Shapley
    print(f"  Bootstrapping ({args.n_bootstrap} resamples) ...")
    post_boot = bootstrap_shapley(X_post, D_cond_post, d,
                                  n_boot=args.n_bootstrap)
    post_ci = np.percentile(post_boot, [2.5, 97.5], axis=0)

    print(f"\n  Posterior Shapley with 95% CI:")
    for i, n in enumerate(names):
        pct = (post_sh[i] / post_var * 100) if post_var > 0 else 0
        ci_lo = (post_ci[0, i] / post_var * 100) if post_var > 0 else 0
        ci_hi = (post_ci[1, i] / post_var * 100) if post_var > 0 else 0
        print(f"    {PARAM_SHORT[n]:>6s}: {pct:5.1f}%  [{ci_lo:5.1f}, {ci_hi:5.1f}]")

    # Ranking change
    prior_rank = np.argsort(prior_sh)[::-1]
    post_rank = np.argsort(post_sh)[::-1]
    print(f"\n  Ranking change:")
    print(f"    Prior:     {' > '.join(PARAM_SHORT[names[i]] for i in prior_rank)}")
    print(f"    Posterior: {' > '.join(PARAM_SHORT[names[i]] for i in post_rank)}")

    # Save
    os.makedirs(out_dir, exist_ok=True)
    summary = {
        'archive': os.path.basename(archive_path),
        'tag': tag,
        'N_prior': int(N),
        'N_posterior_sir': int(N_post),
        'prior_var_dcond_km2': float(prior_var),
        'posterior_var_dcond_km2': float(post_var),
        'variance_reduction_pct': float(var_reduction),
        'prior_shapley': {n: float(prior_sh[i]) for i, n in enumerate(names)},
        'prior_shapley_pct': {
            n: float(prior_sh[i] / prior_var * 100) if prior_var > 0 else 0
            for i, n in enumerate(names)
        },
        'posterior_shapley': {n: float(post_sh[i]) for i, n in enumerate(names)},
        'posterior_shapley_pct': {
            n: float(post_sh[i] / post_var * 100) if post_var > 0 else 0
            for i, n in enumerate(names)
        },
        'posterior_shapley_95ci_pct': {
            n: [float(post_ci[0, i] / post_var * 100) if post_var > 0 else 0,
                float(post_ci[1, i] / post_var * 100) if post_var > 0 else 0]
            for i, n in enumerate(names)
        },
        'prior_ranking': [names[i] for i in prior_rank],
        'posterior_ranking': [names[i] for i in post_rank],
    }
    summary_path = os.path.join(out_dir, f'{tag}_shapley_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {summary_path}")

    # Plot
    fig_dir = os.path.join(out_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    plot_shapley_comparison(
        prior_sh, post_sh, prior_var, post_var,
        names, post_ci,
        os.path.join(fig_dir, f'{tag}_shapley_comparison.png'))

    print(f"\n  Phase 3 COMPLETE: {tag}\n")


if __name__ == "__main__":
    main()
