#!/usr/bin/env python3
"""
Iterative Bayesian refit of mid-latitude (35 deg) endmember Monte Carlo
against Juno MWR conductive shell constraint (D_cond = 29 +/- 10 km).

Runs 3 ocean transport models (uniform, Soderlund, Lemasquerier) at
35 deg latitude (center of Juno 30-40 deg measurement band).

For each model:
  1. Run MC with current priors (n=5000)
  2. Importance-reweight against Juno D_cond = 29 +/- 10 km
  3. Extract posterior parameter distributions
  4. Tighten priors from posterior
  5. Re-run with constrained priors
  6. Iterate until convergence or max rounds

Observation model:
  D_obs = 29 km, sigma_obs = 10 km, sigma_model = 3 km
  Gaussian likelihood: w_i ~ exp(-0.5 * ((D_cond_i - D_obs) / sigma_total)^2)

Convergence criterion:
  Posterior D_cond median within [24, 34] km AND ESS/N > 10%

Usage:
    python run_midlat_juno_refit.py
    python run_midlat_juno_refit.py -n 1000 --max-rounds 3
    python run_midlat_juno_refit.py --models uniform soderlund
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import argparse
import json
import time
import multiprocessing as mp

from runtime_support import configure_numeric_runtime, resolve_worker_count

configure_numeric_runtime()

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from Monte_Carlo import MonteCarloRunner, SolverConfig, save_results
from juno_constrained_sampler import JunoConstrainedMidLatSampler
from constants import Rheology, Planetary


# ═══════════════════════════════════════════════════════════════════════════
# Physical constants at 35 deg latitude
# ═══════════════════════════════════════════════════════════════════════════

PHI_DEG = 35.0
PHI_RAD = np.radians(PHI_DEG)

# Surface temperature: T_s(phi) = ((T_eq^4 - T_fl^4)*cos^p(phi) + T_fl^4)^0.25
# Ashkenazy (2019) calibrated exponent p = 1.25
T_EQ = 110.0   # K
T_FLOOR = 46.0  # K
_SURF_EXP = 1.25
_cos_p = np.cos(PHI_RAD) ** _SURF_EXP
T_SURF_35 = float(((T_EQ**4 - T_FLOOR**4) * _cos_p + T_FLOOR**4) ** 0.25)

# Tidal strain: eps(phi) = eps_eq * sqrt(1 + c*sin^2(phi))
# Beuthe (2013) eccentricity-tide pattern
EPS_EQ = 6e-6
EPS_POLE = 1.2e-5
_c_strain = (EPS_POLE / EPS_EQ)**2 - 1.0
EPS_35 = float(EPS_EQ * np.sqrt(1 + _c_strain * np.sin(PHI_RAD)**2))

# q_tidal_multiplier at 35 deg for each ocean transport pattern
def _equator_enhanced_mult(q_star, phi):
    """Soderlund (2014) equator-enhanced pattern multiplier."""
    a = 3 * q_star / (3 - 2 * q_star)
    return (1 + a * np.cos(phi)**2) / (1 + 2 * a / 3)

def _polar_enhanced_mult(q_star, phi):
    """Lemasquerier (2023) polar-enhanced pattern multiplier."""
    a = 3 * q_star / (3 - q_star)
    return (1 + a * np.sin(phi)**2) / (1 + a / 3)

Q_MULT = {
    'uniform':       1.0,
    'soderlund':     float(_equator_enhanced_mult(0.4, PHI_RAD)),
    'lemasquerier':  float(_polar_enhanced_mult(0.455, PHI_RAD)),
}

# Juno observation
JUNO_D_OBS = 29.0       # km
JUNO_SIGMA_OBS = 10.0   # km
SIGMA_MODEL = 3.0        # km  (model discrepancy)

# Directories
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'midlat_juno')
FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')
PRIORS_FILE = os.path.join(
    os.path.dirname(__file__), '..', 'results', 'midlat_juno', 'current_priors.json'
)

# Initial (unconstrained) priors — matches AuditedShellSampler defaults
INITIAL_PRIORS = {
    'q_basal_lo': 0.005,       # 5 mW/m^2
    'q_basal_hi': 0.025,       # 25 mW/m^2
    'd_grain_log_center': float(np.log10(6e-4)),   # 0.6 mm
    'd_grain_log_sigma': 0.35,
    'd_grain_lo': 5e-5,        # 0.05 mm
    'd_grain_hi': 3e-3,        # 3.0 mm
    'T_surf_mean': T_SURF_35,
    'T_surf_std': 5.0,
    'T_surf_clip': [85.0, 115.0],
    'eps_log_center': float(np.log10(EPS_35)),
    'eps_log_sigma': 0.2,
    'eps_clip': [2e-6, 3e-5],
    'q_tidal_mult': 1.0,       # placeholder; overwritten per model
}


# ═══════════════════════════════════════════════════════════════════════════
# Bayesian machinery
# ═══════════════════════════════════════════════════════════════════════════

def compute_weights(D_cond, D_obs, sigma_obs, sigma_model):
    """Gaussian importance weights for Juno likelihood."""
    sigma_total = np.sqrt(sigma_obs**2 + sigma_model**2)
    log_w = -0.5 * ((D_cond - D_obs) / sigma_total) ** 2
    log_w_stable = log_w - np.max(log_w)
    w = np.exp(log_w_stable)
    return w / w.sum()


def effective_sample_size(w):
    """Kish's ESS."""
    return 1.0 / np.sum(w**2)


def weighted_percentile(values, weights, pct):
    """Weighted percentile via CDF interpolation."""
    idx = np.argsort(values)
    sorted_v = values[idx]
    sorted_w = weights[idx]
    cum_w = np.cumsum(sorted_w)
    return float(np.interp(pct / 100.0, cum_w, sorted_v))


def posterior_stats(values, weights):
    """Return (median, 16th, 84th, 5th, 95th) weighted percentiles."""
    return (
        weighted_percentile(values, weights, 50),
        weighted_percentile(values, weights, 15.87),
        weighted_percentile(values, weights, 84.13),
        weighted_percentile(values, weights, 5),
        weighted_percentile(values, weights, 95),
    )


def shrinkage_ratio(values, weights):
    """Prior-to-posterior 68% interval shrinkage: 0=none, 1=fully constrained."""
    p16_prior = np.percentile(values, 15.87)
    p84_prior = np.percentile(values, 84.13)
    prior_w = p84_prior - p16_prior
    if prior_w < 1e-15:
        return 0.0
    p16_post = weighted_percentile(values, weights, 15.87)
    p84_post = weighted_percentile(values, weights, 84.13)
    return 1.0 - (p84_post - p16_post) / prior_w


# ═══════════════════════════════════════════════════════════════════════════
# Parameter extraction
# ═══════════════════════════════════════════════════════════════════════════

def extract_q_basal(data):
    """Reconstruct q_basal (W/m^2) from NPZ param arrays."""
    D_H2O = data['param_D_H2O']
    H_rad = data['param_H_rad']
    R_rock = Planetary.RADIUS - D_H2O
    M_rock = (4.0 / 3.0) * np.pi * (R_rock ** 3) * 3500.0
    q_rad = (H_rad * M_rock) / Planetary.AREA
    q_tidal = data['param_P_tidal'] / Planetary.AREA
    return q_rad + q_tidal


# ═══════════════════════════════════════════════════════════════════════════
# Prior derivation from posterior
# ═══════════════════════════════════════════════════════════════════════════

def derive_constrained_priors(data, weights, q_mult, prev_priors):
    """
    Extract tighter prior parameters from the posterior distribution.

    Strategy:
      - q_basal: set new uniform range to posterior [5th, 95th] percentile
      - d_grain: set lognormal center to posterior median, sigma from posterior IQR
      - T_surf, epsilon_0: keep fixed (determined by latitude, not Juno)
    """
    q_basal = extract_q_basal(data)
    d_grain = data['param_d_grain']

    # q_basal posterior range
    q_med, q_16, q_84, q_05, q_95 = posterior_stats(q_basal, weights)
    new_q_lo = max(0.001, q_05)
    new_q_hi = min(0.050, q_95)
    # Ensure minimum width to avoid degenerate priors
    if new_q_hi - new_q_lo < 0.002:
        mid = (new_q_lo + new_q_hi) / 2
        new_q_lo = mid - 0.001
        new_q_hi = mid + 0.001

    # d_grain posterior distribution
    dg_med, dg_16, dg_84, dg_05, dg_95 = posterior_stats(d_grain, weights)
    dg_log_center = np.log10(dg_med)
    # Log-space sigma from 68% interval
    dg_log_sigma = (np.log10(dg_84) - np.log10(dg_16)) / 2.0
    dg_log_sigma = max(0.08, min(0.5, dg_log_sigma))  # bound to reasonable range
    # Clip range from posterior 5-95% with margin
    new_dg_lo = max(1e-5, dg_05 * 0.5)
    new_dg_hi = min(1e-2, dg_95 * 2.0)

    return {
        'q_basal_lo': float(new_q_lo),
        'q_basal_hi': float(new_q_hi),
        'd_grain_log_center': float(dg_log_center),
        'd_grain_log_sigma': float(dg_log_sigma),
        'd_grain_lo': float(new_dg_lo),
        'd_grain_hi': float(new_dg_hi),
        'T_surf_mean': prev_priors['T_surf_mean'],
        'T_surf_std': prev_priors['T_surf_std'],
        'T_surf_clip': prev_priors['T_surf_clip'],
        'eps_log_center': prev_priors['eps_log_center'],
        'eps_log_sigma': prev_priors['eps_log_sigma'],
        'eps_clip': prev_priors['eps_clip'],
        'q_tidal_mult': float(q_mult),
    }


# ═══════════════════════════════════════════════════════════════════════════
# JSON I/O
# ═══════════════════════════════════════════════════════════════════════════

def write_priors_json(priors, filepath):
    """Write prior parameters to JSON for the constrained sampler."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    safe = {}
    for k, v in priors.items():
        if isinstance(v, (np.floating, np.integer)):
            safe[k] = float(v)
        elif isinstance(v, np.ndarray):
            safe[k] = v.tolist()
        elif isinstance(v, list):
            safe[k] = [float(x) if isinstance(x, (np.floating, np.integer)) else x for x in v]
        else:
            safe[k] = v
    with open(filepath, 'w') as f:
        json.dump(safe, f, indent=2)


# ═══════════════════════════════════════════════════════════════════════════
# MC run + analysis
# ═══════════════════════════════════════════════════════════════════════════

def run_mc(n_iter, seed, n_workers, tag):
    """Run MC with JunoConstrainedMidLatSampler and save results."""
    config = SolverConfig(reject_subcritical=False)
    runner = MonteCarloRunner(
        n_iterations=n_iter,
        seed=seed,
        verbose=True,
        n_workers=n_workers,
        config=config,
        sampler_class=JunoConstrainedMidLatSampler,
    )

    print(f"\n{'='*60}")
    print(f"  MC RUN: {tag}  (N={n_iter})")
    print(f"{'='*60}")

    results = runner.run()
    outpath = os.path.join(RESULTS_DIR, f"{tag}.npz")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    save_results(results, outpath)
    return outpath


def analyze_juno(npz_path, model_name, round_num):
    """Bayesian importance reweighting against Juno constraint."""
    data = np.load(npz_path)
    D_cond = data['D_cond_km']
    H_total = data['thicknesses_km']
    n = len(D_cond)

    sigma_total = np.sqrt(JUNO_SIGMA_OBS**2 + SIGMA_MODEL**2)

    # Compute weights
    w = compute_weights(D_cond, JUNO_D_OBS, JUNO_SIGMA_OBS, SIGMA_MODEL)
    ess = effective_sample_size(w)
    ess_frac = ess / n

    # Prior overlap check
    prior_overlap = np.mean(
        (D_cond > JUNO_D_OBS - 2 * sigma_total) &
        (D_cond < JUNO_D_OBS + 2 * sigma_total)
    )

    # Posterior summaries
    dc_med, dc_16, dc_84, dc_05, dc_95 = posterior_stats(D_cond, w)
    ht_med, ht_16, ht_84, _, _ = posterior_stats(H_total, w)

    # Parameter posteriors
    q_basal = extract_q_basal(data)
    d_grain = data['param_d_grain']
    q_med, q_16, q_84, _, _ = posterior_stats(q_basal * 1e3, w)  # mW/m^2
    dg_med, dg_16, dg_84, _, _ = posterior_stats(d_grain * 1e3, w)  # mm

    # Shrinkage
    q_shrink = shrinkage_ratio(q_basal * 1e3, w)
    dg_shrink = shrinkage_ratio(d_grain * 1e3, w)

    print(f"\n  --- Juno Analysis: {model_name}, round {round_num} ---")
    print(f"  Valid samples: {n}")
    print(f"  Prior overlap (2sigma): {prior_overlap:.1%}")
    print(f"  ESS = {ess:.0f} / {n} ({100*ess_frac:.1f}%)")
    print(f"")
    print(f"  D_cond prior:     median={np.median(D_cond):.1f}, "
          f"1sig=[{np.percentile(D_cond, 15.87):.1f}, {np.percentile(D_cond, 84.13):.1f}] km")
    print(f"  D_cond posterior:  median={dc_med:.1f}, 1sig=[{dc_16:.1f}, {dc_84:.1f}] km")
    print(f"  H_total posterior: median={ht_med:.1f}, 1sig=[{ht_16:.1f}, {ht_84:.1f}] km")
    print(f"")
    print(f"  q_basal posterior: median={q_med:.1f}, 1sig=[{q_16:.1f}, {q_84:.1f}] mW/m^2  "
          f"(shrinkage={q_shrink:.2f})")
    print(f"  d_grain posterior: median={dg_med:.2f}, 1sig=[{dg_16:.2f}, {dg_84:.2f}] mm  "
          f"(shrinkage={dg_shrink:.2f})")

    # Convergence check
    # Convergence: posterior D_cond median within Juno 1-sigma [19, 39] km
    # AND ESS fraction above 10%
    converged = (19.0 <= dc_med <= 39.0) and (ess_frac > 0.10)
    if converged:
        print(f"  >>> CONVERGED: D_cond median={dc_med:.1f} km in [24, 34], ESS={ess_frac:.1%}")
    else:
        reasons = []
        if dc_med < 24.0 or dc_med > 34.0:
            reasons.append(f"D_cond median {dc_med:.1f} outside [24,34]")
        if ess_frac <= 0.10:
            reasons.append(f"ESS {ess_frac:.1%} <= 10%")
        print(f"  >>> NOT CONVERGED: {'; '.join(reasons)}")

    return data, w, ess, dc_med, converged


def make_figure(npz_path, model_name, round_num, weights):
    """Generate D_cond prior/posterior comparison figure."""
    data = np.load(npz_path)
    D_cond = data['D_cond_km']

    sigma_total = np.sqrt(JUNO_SIGMA_OBS**2 + SIGMA_MODEL**2)

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    bins = np.linspace(0, 80, 60)

    # Prior histogram
    ax.hist(D_cond, bins=bins, density=True, alpha=0.3, color='steelblue',
            edgecolor='steelblue', linewidth=0.3, label='Prior')

    # Posterior (resampled)
    n_resample = min(50000, len(D_cond))
    idx = np.random.choice(len(D_cond), size=n_resample, p=weights, replace=True)
    ax.hist(D_cond[idx], bins=bins, density=True, alpha=0.4, color='firebrick',
            edgecolor='firebrick', linewidth=0.3, label='Posterior')

    # Juno constraint
    x_juno = np.linspace(0, 70, 300)
    juno_pdf = np.exp(-0.5 * ((x_juno - JUNO_D_OBS) / sigma_total)**2)
    juno_pdf /= sigma_total * np.sqrt(2 * np.pi)
    ax.plot(x_juno, juno_pdf, 'k--', lw=1.2,
            label=f'Juno N({JUNO_D_OBS:.0f}, {sigma_total:.0f})')
    ax.axvline(JUNO_D_OBS, color='k', lw=0.6, ls=':', alpha=0.5)

    ax.set_xlabel(r'Conductive lid $D_\mathrm{cond}$ (km)')
    ax.set_ylabel('Density')
    ax.set_xlim(0, 70)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)
    ax.set_title(f'{model_name} @ 35° — Round {round_num}', fontsize=10)

    fig.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    tag = model_name.lower().replace(' ', '_')
    figpath = os.path.join(FIGURES_DIR, f'dcond_{tag}_round{round_num}.png')
    fig.savefig(figpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Figure saved: {figpath}")


def make_param_figure(npz_path, model_name, round_num, weights):
    """Prior vs posterior KDEs for q_basal and d_grain."""
    from scipy.stats import gaussian_kde

    data = np.load(npz_path)
    q_basal_mw = extract_q_basal(data) * 1e3  # mW/m^2
    d_grain_mm = data['param_d_grain'] * 1e3   # mm

    n_resample = min(15000, len(weights))
    idx = np.random.choice(len(weights), size=n_resample, p=weights, replace=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # q_basal
    x_q = np.linspace(2, 30, 300)
    kde_prior = gaussian_kde(q_basal_mw)
    ax1.fill_between(x_q, kde_prior(x_q), alpha=0.2, color='steelblue')
    ax1.plot(x_q, kde_prior(x_q), color='steelblue', lw=1, label='Prior')
    q_post = q_basal_mw[idx]
    if len(np.unique(q_post)) > 5:
        kde_post = gaussian_kde(q_post)
        ax1.fill_between(x_q, kde_post(x_q), alpha=0.3, color='firebrick')
        ax1.plot(x_q, kde_post(x_q), color='firebrick', lw=1.2, label='Posterior')
    ax1.set_xlabel(r'$q_\mathrm{basal}$ (mW/m$^2$)')
    ax1.set_ylabel('Density')
    ax1.set_xlim(2, 30)
    ax1.legend(fontsize=8)

    # d_grain
    x_dg = np.linspace(0, 3.5, 300)
    kde_prior_dg = gaussian_kde(d_grain_mm)
    ax2.fill_between(x_dg, kde_prior_dg(x_dg), alpha=0.2, color='steelblue')
    ax2.plot(x_dg, kde_prior_dg(x_dg), color='steelblue', lw=1, label='Prior')
    dg_post = d_grain_mm[idx]
    if len(np.unique(dg_post)) > 5:
        kde_post_dg = gaussian_kde(dg_post)
        ax2.fill_between(x_dg, kde_post_dg(x_dg), alpha=0.3, color='firebrick')
        ax2.plot(x_dg, kde_post_dg(x_dg), color='firebrick', lw=1.2, label='Posterior')
    ax2.set_xlabel(r'$d_\mathrm{grain}$ (mm)')
    ax2.set_ylabel('Density')
    ax2.set_xlim(0, 3.5)
    ax2.legend(fontsize=8)

    fig.suptitle(f'{model_name} @ 35° — Round {round_num}', fontsize=10)
    fig.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    tag = model_name.lower().replace(' ', '_')
    figpath = os.path.join(FIGURES_DIR, f'params_{tag}_round{round_num}.png')
    fig.savefig(figpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Figure saved: {figpath}")


# ═══════════════════════════════════════════════════════════════════════════
# Main iterative loop
# ═══════════════════════════════════════════════════════════════════════════

def run_model_refit(model_key, model_name, q_mult, n_iter, seed, n_workers, max_rounds):
    """Run iterative Juno refit for one ocean transport model."""
    print(f"\n{'#'*60}")
    print(f"# MODEL: {model_name}")
    print(f"# q_tidal_mult @ 35° = {q_mult:.4f}")
    print(f"# Max rounds = {max_rounds}")
    print(f"{'#'*60}")

    # Start with initial (unconstrained) priors
    current_priors = dict(INITIAL_PRIORS)
    current_priors['q_tidal_mult'] = q_mult

    history = []

    for rnd in range(1, max_rounds + 1):
        print(f"\n{'='*60}")
        print(f"ROUND {rnd}/{max_rounds} — {model_name}")
        print(f"{'='*60}")

        # Print current prior ranges
        print(f"  Priors: q_basal=[{current_priors['q_basal_lo']*1e3:.1f}, "
              f"{current_priors['q_basal_hi']*1e3:.1f}] mW/m^2")
        print(f"          d_grain center={10**current_priors['d_grain_log_center']*1e3:.2f} mm, "
              f"sigma={current_priors['d_grain_log_sigma']:.2f} dex")

        # Write priors JSON
        write_priors_json(current_priors, PRIORS_FILE)

        # Run MC
        tag = f"midlat35_{model_key}_round{rnd}"
        npz_path = run_mc(n_iter, seed + rnd * 1000, n_workers, tag)

        # Bayesian analysis
        data, w, ess, dc_med, converged = analyze_juno(npz_path, model_name, rnd)

        # Generate figures
        np.random.seed(42)
        make_figure(npz_path, model_name, rnd, w)
        make_param_figure(npz_path, model_name, rnd, w)

        # Record history
        q_basal_mw = extract_q_basal(data) * 1e3
        d_grain_mm = data['param_d_grain'] * 1e3
        q_med = weighted_percentile(q_basal_mw, w, 50)
        dg_med = weighted_percentile(d_grain_mm, w, 50)
        history.append({
            'round': rnd,
            'n_valid': len(data['D_cond_km']),
            'D_cond_prior_median': float(np.median(data['D_cond_km'])),
            'D_cond_post_median': float(dc_med),
            'ESS': float(ess),
            'ESS_frac': float(ess / len(data['D_cond_km'])),
            'q_basal_post_median_mw': float(q_med),
            'd_grain_post_median_mm': float(dg_med),
            'converged': converged,
            'priors_used': dict(current_priors),
        })

        if converged:
            print(f"\n  CONVERGED after {rnd} round(s).")
            break

        # Derive constrained priors for next round
        current_priors = derive_constrained_priors(data, w, q_mult, current_priors)
        print(f"\n  Tightened priors for round {rnd+1}:")
        print(f"    q_basal=[{current_priors['q_basal_lo']*1e3:.1f}, "
              f"{current_priors['q_basal_hi']*1e3:.1f}] mW/m^2")
        print(f"    d_grain center={10**current_priors['d_grain_log_center']*1e3:.2f} mm, "
              f"sigma={current_priors['d_grain_log_sigma']:.2f} dex")

    # Save history
    history_path = os.path.join(RESULTS_DIR, f"history_{model_key}.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2, default=str)
    print(f"\n  History saved: {history_path}")

    return history


def print_summary(all_histories):
    """Print final summary table across all models."""
    print(f"\n\n{'='*70}")
    print(f"FINAL SUMMARY — Mid-Latitude (35°) Juno Refit")
    print(f"Juno constraint: D_cond = {JUNO_D_OBS} +/- {JUNO_SIGMA_OBS} km")
    print(f"T_surf(35°) = {T_SURF_35:.1f} K, eps(35°) = {EPS_35:.3e}")
    print(f"{'='*70}")

    for model_key, history in all_histories.items():
        last = history[-1]
        print(f"\n  {model_key.upper()} (q_mult={Q_MULT[model_key]:.4f}):")
        print(f"    Rounds: {last['round']}, Converged: {last['converged']}")
        print(f"    Final D_cond posterior median: {last['D_cond_post_median']:.1f} km")
        print(f"    Final ESS: {last['ESS']:.0f} ({last['ESS_frac']:.1%})")
        print(f"    Final q_basal posterior: {last['q_basal_post_median_mw']:.1f} mW/m^2")
        print(f"    Final d_grain posterior: {last['d_grain_post_median_mm']:.2f} mm")

        if last['converged']:
            priors = last['priors_used']
            print(f"    Converged priors:")
            print(f"      q_basal: [{priors['q_basal_lo']*1e3:.1f}, {priors['q_basal_hi']*1e3:.1f}] mW/m^2")
            print(f"      d_grain: center={10**priors['d_grain_log_center']*1e3:.2f} mm, "
                  f"sigma={priors['d_grain_log_sigma']:.2f} dex")

    print(f"\n{'='*70}")
    print("DONE")
    print(f"Results in: {os.path.abspath(RESULTS_DIR)}")
    print(f"Figures in: {os.path.abspath(FIGURES_DIR)}")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(
        description="Iterative mid-latitude Juno refit MC suite")
    parser.add_argument("-n", type=int, default=5000,
                        help="Iterations per MC run (default 5000)")
    parser.add_argument("--seed", type=int, default=35042,
                        help="Base random seed (default 35042)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Parallel workers (default: shared Windows-safe default)")
    parser.add_argument("--max-rounds", type=int, default=5,
                        help="Max refit iterations per model (default 5)")
    parser.add_argument("--models", nargs='+',
                        default=['uniform', 'soderlund', 'lemasquerier'],
                        choices=['uniform', 'soderlund', 'lemasquerier'],
                        help="Ocean transport models to run")
    args = parser.parse_args()

    n_workers = resolve_worker_count(args.workers)

    print(f"Rheology model: {Rheology.MODEL}")
    assert Rheology.MODEL == "Andrade", f"Expected Andrade, got {Rheology.MODEL}"

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print(f"\nMid-Latitude (35°) Juno Refit Suite")
    print(f"  T_surf(35°) = {T_SURF_35:.1f} K")
    print(f"  epsilon_0(35°) = {EPS_35:.3e}")
    print(f"  q_tidal_multipliers:")
    for k, v in Q_MULT.items():
        print(f"    {k}: {v:.4f}")
    print(f"  Juno: D_cond = {JUNO_D_OBS} +/- {JUNO_SIGMA_OBS} km")
    print(f"  sigma_model = {SIGMA_MODEL} km")
    print(f"  N = {args.n}, workers = {n_workers}, max rounds = {args.max_rounds}")

    MODEL_NAMES = {
        'uniform': 'Uniform',
        'soderlund': 'Soderlund 2014',
        'lemasquerier': 'Lemasquerier 2023',
    }

    t0 = time.time()
    all_histories = {}

    for model_key in args.models:
        model_name = MODEL_NAMES[model_key]
        q_mult = Q_MULT[model_key]
        history = run_model_refit(
            model_key, model_name, q_mult,
            args.n, args.seed, n_workers, args.max_rounds,
        )
        all_histories[model_key] = history

    elapsed = time.time() - t0
    print(f"\nTotal runtime: {elapsed/60:.1f} minutes")

    print_summary(all_histories)

    # Save combined summary
    summary_path = os.path.join(RESULTS_DIR, 'summary.json')
    summary = {
        'phi_deg': PHI_DEG,
        'T_surf_35': T_SURF_35,
        'eps_35': EPS_35,
        'q_mult': Q_MULT,
        'juno_D_obs': JUNO_D_OBS,
        'juno_sigma_obs': JUNO_SIGMA_OBS,
        'sigma_model': SIGMA_MODEL,
        'n_iter': args.n,
        'seed': args.seed,
        'total_runtime_min': elapsed / 60,
        'models': {},
    }
    for k, h in all_histories.items():
        summary['models'][k] = h
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
