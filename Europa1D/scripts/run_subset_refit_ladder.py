#!/usr/bin/env python3
"""
Phase 2: Restricted-update calibration ladder.

Greedy forward-selection of parameter subsets scored by KL divergence.
For each subset S, only parameters in S are allowed to update from prior
to posterior; parameters NOT in S are resampled from the prior (breaking
posterior dependence).

This is a calibration-ablation experiment, NOT sensitivity analysis.

Usage:
    python run_subset_refit_ladder.py midlat35_uniform_broad.npz
    python run_subset_refit_ladder.py eq_baseline_andrade.npz --tag equatorial
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

# Juno observation
JUNO_D_OBS = 29.0
JUNO_SIGMA_OBS = 10.0
SIGMA_MODEL = 3.0

PARAM_SHORT = {
    'q_basal': 'q_bas', 'd_grain': 'd_gr', 'Q_v': 'Q_v',
    'epsilon_0': 'eps_0', 'T_surf': 'T_s',
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


def compute_log_likelihood(D_cond):
    """Per-sample Gaussian log-likelihood under the Juno constraint,
    shifted for numerical stability (max at 0)."""
    sigma_total = np.sqrt(JUNO_SIGMA_OBS**2 + SIGMA_MODEL**2)
    log_lik = -0.5 * ((D_cond - JUNO_D_OBS) / sigma_total) ** 2
    log_lik -= np.max(log_lik)
    return log_lik


def compute_weights(D_cond):
    log_lik = compute_log_likelihood(D_cond)
    w = np.exp(log_lik)
    return w / w.sum()


def entropy_ess(w):
    w_pos = w[w > 0]
    return np.exp(-np.sum(w_pos * np.log(w_pos)))


def kish_ess(w):
    return 1.0 / np.sum(w**2)


def kl_from_weights(w):
    N = len(w)
    return np.log(N) - np.log(entropy_ess(w))


def weighted_percentile(values, weights, pct):
    idx = np.argsort(values)
    sorted_v = values[idx]
    sorted_w = weights[idx]
    cum_w = np.cumsum(sorted_w)
    return float(np.interp(pct / 100.0, cum_w, sorted_v))


def restricted_update_weights(X, log_lik, names, allowed_set, k=None):
    """Compute true restricted-update posterior weights for a parameter subset.

    A restricted update over subset S asks: what would the posterior look
    like if the likelihood could only see parameters in S?  Formally:

        p_S(theta_S | D) ∝ p_S(theta_S) * L_S(theta_S),
        where L_S(theta_S) = E_{theta_{-S} ~ prior}[L(theta_S, theta_{-S})].

    Because we only have L sampled at the MC design points, L_S must be
    estimated nonparametrically.  We use a kNN conditional-mean emulator
    on the z-scored parameter subspace: for each prior sample i,
    L_S(theta_S^i) is estimated as the mean likelihood over the k nearest
    neighbours of theta_S^i in the |S|-dimensional subspace.

    Parameters
    ----------
    X : array (N, d)
        Prior parameter samples.
    log_lik : array (N,)
        Per-sample log-likelihood (max-shifted; any additive constant cancels
        out in normalisation).
    names : list[str]
        Parameter names in column order of X.
    allowed_set : set[str]
        Subset S of parameters allowed to drive the update.
    k : int or None
        kNN neighbourhood size.  Default: max(50, round(sqrt(N))).

    Returns
    -------
    w : array (N,)
        Normalised restricted-update posterior weights (sum to 1).

    Notes
    -----
    - For |S|=0 (empty), returns uniform prior weights (by construction).
    - For |S|=d, recovers a smoothed approximation to the full posterior
      (exact recovery would require k=1; smoothing is intentional because
      the kNN estimator integrates out local sampling noise).
    - Biases: small k → high variance in L_S; large k → oversmoothing and
      underestimated KL.  sqrt(N) is a standard default.
    """
    from scipy.spatial import cKDTree
    from scipy.special import logsumexp

    N, d = X.shape
    allowed_idx = [i for i, n in enumerate(names) if n in allowed_set]

    if len(allowed_idx) == 0:
        # Empty subset: no parameters drive the update → prior
        return np.ones(N) / N

    if k is None:
        k = max(50, int(round(np.sqrt(N))))
    k = min(k, N)

    # z-score the allowed subspace so each parameter contributes equally
    # to the kNN distance metric
    X_S = X[:, allowed_idx]
    mu = X_S.mean(axis=0)
    sd = X_S.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    X_S_std = (X_S - mu) / sd

    # kNN in allowed subspace (includes self-neighbour at rank 0)
    tree = cKDTree(X_S_std)
    _, nn_idx = tree.query(X_S_std, k=k)

    # Marginalised log-likelihood: log(mean(L_j for j in NN_S(i)))
    # = logsumexp(log_lik[NN(i)]) - log(k)
    log_lik_marg = logsumexp(log_lik[nn_idx], axis=1) - np.log(k)

    # Normalise to get restricted-update posterior weights
    log_w = log_lik_marg - logsumexp(log_lik_marg)
    return np.exp(log_w)


def compute_scorecard(X, D_cond, H_total, w, names, allowed_set):
    """Compute the full scorecard for a given allowed parameter subset."""
    N = len(w)
    allowed_idx = [i for i, n in enumerate(names) if n in allowed_set]

    # KL and ESS
    kl = kl_from_weights(w)
    ess = kish_ess(w)

    # D_cond posterior
    dc_med = weighted_percentile(D_cond, w, 50)
    dc_16 = weighted_percentile(D_cond, w, 15.87)
    dc_84 = weighted_percentile(D_cond, w, 84.13)

    # H_total posterior
    ht_med = weighted_percentile(H_total, w, 50)
    ht_16 = weighted_percentile(H_total, w, 15.87)
    ht_84 = weighted_percentile(H_total, w, 84.13)

    # Per-parameter shrinkage (only for allowed params)
    shrinkage = {}
    for i, n in enumerate(names):
        var_prior = np.var(X[:, i])
        mu_post = np.sum(w * X[:, i])
        var_post = np.sum(w * (X[:, i] - mu_post)**2)
        s = 1.0 - var_post / var_prior if var_prior > 0 else 0.0
        shrinkage[n] = s

    # Pairwise correlations for allowed params
    correlations = {}
    if len(allowed_idx) >= 2:
        N_resamp = min(int(ess), 8000)
        idx = np.random.choice(N, size=max(N_resamp, 2000), p=w, replace=True)
        for i_idx in range(len(allowed_idx)):
            for j_idx in range(i_idx + 1, len(allowed_idx)):
                i, j = allowed_idx[i_idx], allowed_idx[j_idx]
                r, _ = stats.spearmanr(X[idx, i], X[idx, j])
                key = f"{PARAM_SHORT[names[i]]}_{PARAM_SHORT[names[j]]}"
                correlations[key] = float(r)

    return {
        'subset': sorted(list(allowed_set)),
        'subset_size': len(allowed_set),
        'KL_nats': float(kl),
        'ESS_kish': float(ess),
        'ESS_frac': float(ess / N),
        'D_cond_median_km': float(dc_med),
        'D_cond_68ci_km': [float(dc_16), float(dc_84)],
        'H_total_median_km': float(ht_med),
        'H_total_68ci_km': [float(ht_16), float(ht_84)],
        'shrinkage': shrinkage,
        'correlations': correlations,
    }


def greedy_ladder(X, D_cond, H_total, log_lik, names, knn_k=None):
    """Build a genuine greedy forward-selection restricted-update ladder.

    At each step, for every candidate parameter j not yet in the selected set
    S, compute the restricted-update posterior p_{S ∪ {j}}(theta | D) via the
    kNN conditional-mean emulator (restricted_update_weights), and measure its
    KL from the prior.  Add the parameter j that maximises this restricted KL.

    This is a genuine information-theoretic ladder: each row reports the
    posterior you would obtain if only the parameters in S were allowed to
    drive the update, with the rest marginalised over the prior.  Unlike the
    previous implementation, ESS, KL, medians, and shrinkage differ between
    rows.
    """
    N, d = X.shape
    remaining = set(names)
    selected = []
    ladder = []

    # Step 0: empty set — by construction p_S = prior, KL = 0.
    ladder.append({
        'subset': [],
        'subset_size': 0,
        'KL_nats': 0.0,
        'ESS_kish': float(N),
        'ESS_frac': 1.0,
        'D_cond_median_km': float(np.median(D_cond)),
        'D_cond_68ci_km': [float(np.percentile(D_cond, 15.87)),
                           float(np.percentile(D_cond, 84.13))],
        'H_total_median_km': float(np.median(H_total)),
        'H_total_68ci_km': [float(np.percentile(H_total, 15.87)),
                            float(np.percentile(H_total, 84.13))],
        'shrinkage': {n: 0.0 for n in names},
        'correlations': {},
        'added_param': None,
        'marginal_kl_gain': 0.0,
        'cumulative_marginal_kl': 0.0,
    })

    prev_kl = 0.0
    for step in range(d):
        # Score each candidate: KL of the restricted-update posterior
        # obtained by allowing {selected} ∪ {candidate} to drive the update.
        best_param = None
        best_kl = -np.inf
        best_weights = None
        for cand in remaining:
            trial_set = set(selected) | {cand}
            w_trial = restricted_update_weights(
                X, log_lik, names, trial_set, k=knn_k)
            kl_trial = kl_from_weights(w_trial)
            if kl_trial > best_kl:
                best_kl = kl_trial
                best_param = cand
                best_weights = w_trial

        remaining.remove(best_param)
        selected.append(best_param)

        scorecard = compute_scorecard(
            X, D_cond, H_total, best_weights, names, set(selected))
        scorecard['added_param'] = best_param
        scorecard['marginal_kl_gain'] = float(best_kl - prev_kl)
        scorecard['cumulative_marginal_kl'] = float(best_kl)
        ladder.append(scorecard)
        prev_kl = best_kl

    return ladder


def plot_information_ladder(ladder, outpath):
    """Plot the information accumulation curve."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.8))

    sizes = [s['subset_size'] for s in ladder]
    kl_cum = [0.0] + [s['cumulative_marginal_kl'] for s in ladder[1:]]
    kl_gains = [0.0] + [s['marginal_kl_gain'] for s in ladder[1:]]
    labels = ['(none)'] + [PARAM_SHORT.get(s['added_param'], s['added_param'])
                           for s in ladder[1:]]

    # Panel A: cumulative KL
    ax1.plot(sizes, kl_cum, 'o-', color='#0072B2', linewidth=2, markersize=8)
    for i, (sz, kl, lab) in enumerate(zip(sizes, kl_cum, labels)):
        if i > 0:
            ax1.annotate(f'+{lab}', (sz, kl),
                        textcoords='offset points', xytext=(8, 5),
                        fontsize=7, color='#D55E00', fontweight='bold')
    ax1.set_xlabel('Number of free parameters |S|')
    ax1.set_ylabel('Cumulative marginal KL (nats)')
    ax1.set_title('(a) Information accumulation')
    ax1.set_xticks(sizes)
    ax1.axhline(0.05, color='gray', ls=':', lw=0.8, alpha=0.5)
    ax1.set_ylim(bottom=-0.005)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Panel B: marginal KL gains (bar chart)
    x = np.arange(len(sizes) - 1) + 1
    gains = kl_gains[1:]
    bar_labels = labels[1:]
    colors = ['#0072B2' if g > 0.01 else '#999999' for g in gains]
    ax2.bar(x, gains, color=colors, alpha=0.8, width=0.6)
    ax2.set_xticks(x)
    ax2.set_xticklabels(bar_labels, rotation=45, ha='right')
    ax2.set_ylabel('Marginal KL gain (nats)')
    ax2.set_title('(b) Per-parameter information gain')
    ax2.axhline(0.01, color='gray', ls=':', lw=0.8, alpha=0.5,
                label='0.01 nats threshold')
    ax2.legend(fontsize=6)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {outpath}")


def plot_posterior_predictive(ladder, D_cond, outpath):
    """D_cond prior vs posterior predictive for each subset."""
    fig, ax = plt.subplots(figsize=(5, 3.5))

    bins = np.linspace(0, 80, 80)

    # Prior
    ax.hist(D_cond, bins=bins, density=True, alpha=0.2, color='gray',
            edgecolor='gray', linewidth=0.3, label='Prior')

    # Juno constraint
    sigma_total = np.sqrt(JUNO_SIGMA_OBS**2 + SIGMA_MODEL**2)
    x_juno = np.linspace(0, 70, 300)
    juno_pdf = np.exp(-0.5 * ((x_juno - JUNO_D_OBS) / sigma_total)**2)
    juno_pdf /= sigma_total * np.sqrt(2 * np.pi)
    ax.plot(x_juno, juno_pdf, 'k--', lw=1.2, label=f'Juno ({JUNO_D_OBS} +/- {sigma_total:.0f} km)')

    # Full posterior (last step)
    full = ladder[-1]
    ax.axvline(full['D_cond_median_km'], color='#D55E00', lw=1.5, ls='-',
               label=f'Posterior median: {full["D_cond_median_km"]:.1f} km')
    ax.axvspan(full['D_cond_68ci_km'][0], full['D_cond_68ci_km'][1],
               alpha=0.15, color='#D55E00', label='Posterior 68% CI')

    ax.set_xlabel(r'$D_\mathrm{cond}$ (km)')
    ax.set_ylabel('Density')
    ax.set_xlim(0, 70)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=7)
    ax.set_title('Prior vs posterior predictive')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {outpath}")


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2: Restricted-update calibration ladder")
    parser.add_argument("archive", help="MC archive NPZ")
    parser.add_argument("--tag", default=None)
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
    print(f"  Phase 2: Restricted-Update Ladder")
    print(f"  Archive: {archive_path}")
    print(f"{'='*60}")

    data = np.load(archive_path)
    D_cond = data['D_cond_km']
    H_total = data['thicknesses_km']
    params = extract_params(data)
    names = list(params.keys())
    X = np.column_stack([params[n] for n in names])

    # Drop rows with non-finite D_cond or parameters (the MC archive
    # occasionally contains a single invalid sample)
    mask = np.isfinite(D_cond) & np.isfinite(H_total) & np.all(np.isfinite(X), axis=1)
    D_cond = D_cond[mask]
    H_total = H_total[mask]
    X = X[mask]

    log_lik = compute_log_likelihood(D_cond)
    w_full = compute_weights(D_cond)

    print(f"  Samples: {len(D_cond)}")
    print(f"  Full posterior KL: {kl_from_weights(w_full):.4f} nats")

    np.random.seed(42)
    ladder = greedy_ladder(X, D_cond, H_total, log_lik, names)

    # Print ladder
    print(f"\n  {'Step':<6} {'Added':<8} {'|S|':<4} {'Marg KL':<10} "
          f"{'Cum KL':<10} {'D_cond med':<12}")
    print(f"  {'-'*56}")
    for i, step in enumerate(ladder):
        added = PARAM_SHORT.get(step.get('added_param', ''), '-')
        cum_kl = step.get('cumulative_marginal_kl', 0.0)
        marg_kl = step.get('marginal_kl_gain', 0.0)
        dc = step['D_cond_median_km']
        print(f"  {i:<6} {added:<8} {step['subset_size']:<4} "
              f"{marg_kl:<10.4f} {cum_kl:<10.4f} {dc:<12.1f}")

    # Hypothesis discrimination from ladder shape
    gains = [s.get('marginal_kl_gain', 0) for s in ladder[1:]]
    total_gain = sum(gains)
    if total_gain < 0.05:
        verdict = "H3: Weak information (total marginal KL < 0.05)"
    elif gains[0] > 0.8 * total_gain:
        verdict = "H1: Single-parameter dominance (first param captures >80%)"
    elif sum(gains[:2]) > 0.8 * total_gain:
        verdict = "H2: Two-parameter trade-off (first 2 capture >80%)"
    else:
        verdict = "H4: Hierarchical identifiability (gradual accumulation)"

    print(f"\n  Ladder verdict: {verdict}")

    # Save
    os.makedirs(out_dir, exist_ok=True)
    ladder_path = os.path.join(out_dir, f'{tag}_ladder.json')
    with open(ladder_path, 'w') as f:
        json.dump({'ladder': ladder, 'verdict': verdict}, f, indent=2)
    print(f"  Saved: {ladder_path}")

    fig_dir = os.path.join(out_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    plot_information_ladder(
        ladder, os.path.join(fig_dir, f'{tag}_information_ladder.png'))
    plot_posterior_predictive(
        ladder, D_cond, os.path.join(fig_dir, f'{tag}_posterior_predictive.png'))

    print(f"\n  Phase 2 COMPLETE: {tag}\n")
    return ladder


if __name__ == "__main__":
    main()
